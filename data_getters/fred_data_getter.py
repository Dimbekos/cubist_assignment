import os

import pandas as pd

class FREDDataFetcher:
    def __init__(self, api_key=None, storage_dir='data/fred_data', storage_name='fred_data'):
        #todo add api data collection for Fred
        self.api_key = api_key
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        self.storage_dir = storage_dir
        self.storage_name = storage_name

        # Ensure the storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def get_series_data(self, series_dict, start_year, end_year):

        df = self.load_from_storage()

        if df is not None:
            return df

        data_frames = []

        start_date = pd.to_datetime(f'{start_year}-01-01')
        end_date = pd.to_datetime(f'{end_year}-12-31')

        for name,key in series_dict.items():
            df = pd.read_csv(self.storage_dir+key+'.csv', parse_dates=['DATE'], date_format='%Y-%m-%d', index_col='DATE')
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            data_frames.append(df)

        merged_data = pd.concat(data_frames, axis=1)
        merged_data.sort_index(inplace=True)
        merged_data.apply(pd.to_numeric, errors='coerce')
        self.save_to_storage(merged_data)
        return merged_data

    # Method to check if data is already stored
    def load_from_storage(self):
        file_path = os.path.join(self.storage_dir+self.storage_name)
        if os.path.exists(file_path):
            print(f"Loaded {self.storage_name} data from storage...")
            return pd.read_pickle(file_path)
        return None

    def save_to_storage(self, df):
        df.to_pickle(self.storage_dir+self.storage_name)
        print(f"Saved full dataset as {self.storage_dir+self.storage_name}")


