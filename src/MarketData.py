"""
Code to snag market data and perform different analysis from the Bureau of Labor Statistics.
"""

import pandas as pd
from fredapi import Fred
from typing import AnyStr, Dict, List
import logging

from utils import logging_config


class FredQuery:
    """Allows ability to interace and call various Fred datapoints using the Fred API."""

    def __init__(self, key: str) -> None:
        self.api_key = key
        self.fred = self._login()

        self.market_data = pd.DataFrame()

    def _login(self):
        """Attempt to verify that the API key works by assigning an API key and then performing a query
        of one of the most common and popular series, the S&P500.

        If pull is successful, the API key is used. If it is not successful, the API key is not applied
        and the error is returned.
        """
        try:
            fred = Fred(self.api_key)
            fred.get_series("SP500")
            logging.info("Successfully Accessed API")
        except Exception as e:
            logging.error(f"[!ERROR!] Unable to Query Fred API. \n Error: \n {e}")

    def _format_bls_data(self, resample: str = None):
        self.market_data.index = pd.to_datetime(self.market_data.index)

        if type(resample) != type(None):
            self.market_data = self.market_data.resample(resample).mean()
            self.market_data.dropna(inplace=True)

    def get_market_data_df(
        self,
        series_data: Dict[AnyStr],
        rename_columns: bool = True,
        resample: str = None,
    ):
        """Retrieves relevant series from the FRED dataset and returns it as a
        pandas dataframe for easy analysis.

        Args:
            series_data (Dict[AnyStr]): Dictionary of series where KEY = desired series name and VALUE = series key from FRED
            rename_columns (bool): If the columns of the dataframe should be renamed with the custom names provided in the dictionary.
            resample (str): Pandas string resample, otherwise deafult is no resampling (None).

        Returns:
            pd.DataFrame: Pandas DataFrame of Series History.
        """

        keys_pulled = []
        for key, value in zip(series_data.keys(), series_data.values()):
            logging.info(key, value)
            try:
                data = self.fred.get_series(series_id=value)
                self.market_data = pd.concat((self.market_data, data), axis=1)
                keys_pulled.append(key)
            except Exception as e:
                error_message = f"Not able to retrieve series ID {value} named {key}. The pull has been skipped."
                logging.error(error_message)
                logging.error(e)
                print(error_message)

        if rename_columns:
            # name the columns the desired custom names
            self.market_data.columns = keys_pulled

        self._format_bls_data()

        return self.market_data

    def plot_market_data(self):
        pd.plotting.scatter_matrix(self.market_data, figsize=(15, 10))
