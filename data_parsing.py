from datetime import datetime, timedelta
import dateutil.relativedelta
import pandas as pd
from PIL.ImageChops import constant
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import constants
import numpy as np
from data_getters.bls_data_getter import BLSDataFetcher
from data_getters.fred_data_getter import FREDDataFetcher

df = pd.read_csv('data/PAYEMS.csv', parse_dates=['DATE'], date_format='%Y-%m-%d', index_col='DATE')

bls_series_dict = {
    'Unemployment Rate': 'LNS14000000',
    'Consumer Price Index': 'CUUR0000SA0',
    'Employment Levels': 'CES0000000001',
    'Unemployment Rate (6-month)': 'LNS14000006',
    'Consumer Price Index (SA)': 'CUSR0000SA0',
    'Labor Force Participation': 'LNS11300000',
    'Consumer Price Index (L1E)': 'CUUR0000SA0L1E',
    'Employment Levels (SA)': 'CES3000000001',
    'CPI for Food': 'CES0500000003',
    'Labor Force Non-Participation': 'LNU04000000',
    'Producer Price Index': 'PCU327320327320',
    'PPI for Specific Industry': 'PCU33312033312014',
    'Employment Rate': 'LNS12000000',
    'CPI (L1E)': 'CUSR0000SA0L1E',
    'Consumer Price Index (M1)': 'CUUR0000SAM1',
    'Consumer Price Index (SEMC01)': 'CUUR0000SEMC01',
    'Employment Levels (Latest)': 'LNS12300000',
    'Labor Force Trends': 'LNS12035019',
    'Consumer Price Index (SEMC)': 'CUUR0000SEMC',
    'Consumer Price Index (SS5702)': 'CUUR0000SS5702',
    'Consumer Price Index (SS5703)': 'CUUR0000SS5703',
    'Consumer Price Index (SEMC02)': 'CUUR0000SEMC02',
    'CPI for Specific Products': 'CES0500000001',
    'Consumer Price Index (SAM1)': 'CUUR0100SAM1',
    'Labor Force Participation Rate': 'LNS11300000',
    'Average Weekly Hours - All Employees': 'CES0500000007',
    'Job Openings (JOLTS)': 'JTS110099000000000HIL',
    'Employment - Manufacturing': 'CES3000000001',
    'Employment - Construction': 'CES2000000001',
    'Employment - Retail Trade': 'CES4200000001',
    'Employment - Professional and Business Services': 'CES6000000001',
}

fred_series_dict = {
    'Initial Jobless Claims': 'ICSA',
    'Personal Consumption Expenditures (PCE)': 'PCEPI',
    'Real Gross Domestic Product (A191RL1Q225SBEA)': 'GDPC1',
    'Real Disposable Personal Income (DSPIC96)': 'DSPIC96'
}

bls_data_fetcher = BLSDataFetcher(api_key=constants.api_key, storage_dir=constants.bls_data_storage)
fred_data_fetcher = FREDDataFetcher(storage_dir=constants.fred_data_storage)
bls_data = bls_data_fetcher.get_series_data(bls_series_dict, start_year=constants.start_year, end_year=datetime.now().year)
fred_data = fred_data_fetcher.get_series_data(fred_series_dict, start_year=constants.start_year, end_year=datetime.now().year)
combined_data = pd.concat([bls_data, fred_data], axis=1)

# Resampling to 1 month frequency using last observation. eg if there is a weekly release I am using the last know observation before the end of the month
resampled_data = combined_data.resample('1ME', origin='end').last()

# forward filling data, e.g. using most recent GDP obervation for the future quarter
filled_data = resampled_data.ffill()

# getting pct change for all variables, I know it does not make sense for figures like Unemployment rate but I don't have time to handpick diffs vs pct_change
pct_changes = filled_data.pct_change()

# enhancing data-set by adding lags up to constants.total_lag (currently at 13 to cover up to a year of monthly lags for all variables)
lagged_dfs = {}
for item in pct_changes.columns:
    lagged_dfs.update({f'{item}_lag_{lag}': pct_changes[item].shift(lag) for lag in range(1, constants.total_lags)})
pct_changes = pd.concat([pct_changes, pd.DataFrame(lagged_dfs)], axis=1)


# adding a seasonality component to the month
pct_changes['month_sin'] = np.sin(2 * np.pi * pct_changes.index.month / 12)
pct_changes['month_cos'] = np.cos(2 * np.pi * pct_changes.index.month / 12)

# This is to backfill any initial missing values with 0 meaning 0 change
pct_changes = pct_changes.fillna(0)


# I will roll the pct_changes twice so that end of August data (available in September will be used to predict, September PAYEMS available in October
X = pct_changes.values[:-2].astype(np.float32)

y = df.pct_change().values[-len(X):].astype(np.float32)

combined_df = pd.DataFrame(X, columns=pct_changes.columns)
combined_df = pd.concat([pd.DataFrame(y, columns=['PAYEMS change']), combined_df], axis=1)
corr = combined_df.corr()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=constants.PCA_components)  # We want to reduce to 5 features
principal_components = pca.fit_transform(X_scaled)

# AAAaaa 1 hour before submission and getting the below error :( I am really regretting automating data pulling
# BLS API Error: ['Request could not be serviced, as the daily threshold for total number of requests allocated to the user with registration key ea501ce11f96437e8ea2c2fe3389d937 has been reached.']

# Creating a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(5)])
explained_variance = pca.explained_variance_ratio_

X_train, X_test, y_train, y_test = train_test_split(principal_components, y, test_size=0.2, random_state=42)

# Set up the hyperparameter grid for Elastic Net
param_dist = {
    'alpha': uniform(0.1, 10),  # Uniform distribution from 0.1 to 10
    'l1_ratio': uniform(0, 1)    # Uniform distribution from 0 to 1
}

n_runs = constants.number_of_search_runs

# Initialize GridSearchCV with Elastic Net
random_search = RandomizedSearchCV(
    ElasticNet(random_state=42),
    param_distributions=param_dist,
    n_iter=n_runs,  # Set the number of runs here
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

random_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = random_search.best_params_
best_score = -random_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validated Mean Squared Error:", best_score)

# Train the final model with the best parameters
best_model = ElasticNet(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Making predictions
y_pred = best_model.predict(X_test)

# Evaluating the model
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
