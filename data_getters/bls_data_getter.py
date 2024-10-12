import os
from turtledemo.penrose import start

import requests
import pandas as pd
import json
from datetime import datetime
import constants
from math import ceil


class BLSDataFetcher:
    def __init__(self, api_key, storage_dir='data/bls_data', storage_name='bls_data'):
        self.api_key = api_key
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        self.storage_dir = storage_dir
        self.storage_name = storage_name

        # Ensure the storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    # Method to fetch BLS data for a given series ID
    def fetch_bls_data(self, series_id, start_year, end_year):
        headers = {'Content-type': 'application/json'}
        data = json.dumps({
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": self.api_key
        })

        response = requests.post(self.base_url, data=data, headers=headers)
        json_data = response.json()

        # Check for errors in response
        if json_data.get('status') == 'REQUEST_SUCCEEDED':
            return json_data['Results']['series'][0]['data']
        else:
            raise Exception(f"BLS API Error: {json_data.get('message')}")

    # Method to format BLS data into a DataFrame
    def format_bls_data(self, series_name, bls_data):
        df = pd.DataFrame(bls_data)
        df['date'] = pd.to_datetime(df['year'] + df['period'].str[1:], format='%Y%m')
        df = df[['date', 'value']].set_index('date')
        df.columns = [series_name]
        df[series_name] = pd.to_numeric(df[series_name], errors='coerce')  # Convert to numeric
        return df

    # Main method to fetch data for multiple series
    def get_series_data(self, series_dict, start_year, end_year):
        """
        Fetches data for multiple series and returns a merged pandas DataFrame.

        :param series_dict: Dictionary of {series_name: series_id}
        :param start_year: Start year for the data
        :param end_year: End year for the data
        :return: pandas DataFrame with all series data
        """
        data_frames = []
        df = self.load_from_storage()

        if df is not None:
            return df

        for name, series_id in series_dict.items():
            print(f"Fetching data for {name} (Series ID: {series_id})...")
            runs = ceil((end_year - start_year) / constants.bls_data_year_limit)
            bls_data = []
            temp_start_year = start_year
            for i in range(runs):
                max_year_within_year_limit = min(temp_start_year + constants.bls_data_year_limit, end_year)
                bls_data.extend(self.fetch_bls_data(series_id, temp_start_year, max_year_within_year_limit))
                temp_start_year += constants.bls_data_year_limit
            df = self.format_bls_data(name, bls_data)
            df.drop_duplicates(inplace=True)
            data_frames.append(df)

            # Merge all DataFrames on the date index
        merged_data = pd.concat(data_frames, axis=1)
        merged_data.sort_index(inplace=True)
        df.apply(pd.to_numeric, errors='coerce')
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