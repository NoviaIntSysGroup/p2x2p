import numpy as np
import pandas as pd

from datetime import datetime

from p2x2p.utils.utils import get_config


def get_mfrr_data(frac=False):
    """
    It reads the spot price data from the csv file specified in config
    and returns it as a dataframe.

    return: A dataframe with spot price data.
    """
    config = get_config()

    # Use only hourly data or a smaller data set with fractional activation per hour
    if frac:
        file_path = config["data_paths"]["mfrr_frac_fi"]
    else:
        file_path = config["data_paths"]["mfrr_fi"]
    mfrr_data = pd.read_csv(file_path)

    # Change the original dates to datetimes with time zone and daylight saving
    # The originals are in European/Helsinki time zone. So we are simply
    # adding that information explicitly.
    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(mfrr_data.loc[mfrr_data.index[0], 'date'])
    dt_end = datetime.fromisoformat(mfrr_data.loc[mfrr_data.index[-1], 'date'])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    mfrr_data["date"] = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')

    # Make sure that the energy quantities are numeric
    for col in mfrr_data.columns[1:]:
        mfrr_data[col] = pd.to_numeric(
            mfrr_data[col],
            errors="coerce"
        )
    
    # Replace the hours with a bidding error on the spot market to have zero volyme and price
    # https://yle.fi/a/74-20061943
    specific_datetime = pd.to_datetime('2023-11-24 15:00+02:00')
    matching_row_index = mfrr_data.index[mfrr_data['date'] == specific_datetime]
    if len(matching_row_index) > 0:
        mfrr_data.loc[matching_row_index[0]:matching_row_index[0] + 10, 'down_regulating_price_mFRR'] = 0
        mfrr_data.loc[matching_row_index[0]:matching_row_index[0] + 10, 'up_regulating_price_mFRR'] = 0
        mfrr_data.loc[matching_row_index[0]:matching_row_index[0] + 10, 'down_regulation_volyme_mFRR'] = 0
        mfrr_data.loc[matching_row_index[0]:matching_row_index[0] + 10, 'up_regulation_volyme_mFRR'] = 0

    return mfrr_data


def get_energy_data():

    config = get_config()

    file_path = config["data_paths"]["energy_data_fi"]
    energy_data = pd.read_csv(file_path)

    # Change the original dates to datetimes with time zone and daylight saving
    # The originals are in European/Helsinki time zone. So we are simply
    # adding that information explicitly.
    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(
        energy_data.loc[energy_data.index[0], 'date'])
    dt_end = datetime.fromisoformat(
        energy_data.loc[energy_data.index[-1], 'date'])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    energy_data["date"] = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')

    # Make sure that the energy quantities are numeric
    for col in energy_data.columns[1:]:
        energy_data[col] = pd.to_numeric(
            energy_data[col],
            errors="coerce"
        )

    return energy_data
