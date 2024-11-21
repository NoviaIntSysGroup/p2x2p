import numpy as np
import pandas as pd

from datetime import datetime

from p2x2p.utils.utils import get_config


def get_data():
    """
    It reads the spot price data from the csv file specified in config
    and returns it as a dataframe.

    return: A dataframe with spot price data.
    """
    config = get_config()

    # Spot price data first
    price_file_path = config["data_paths"]["elspot_fi"]
    spot_data = pd.read_csv(price_file_path)

    # Change the original dates to datetimes with time zone and daylight saving
    # The originals are in European/Helsinki time zone. So we are simply
    # adding that information explicitly.
    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(spot_data.loc[spot_data.index[0], 'date'])
    dt_end = datetime.fromisoformat(spot_data.loc[spot_data.index[-1], 'date'])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    spot_data["date"] = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')

    # Make sure that the spot price is numeric
    spot_data["elspot-fi"] = pd.to_numeric(
        spot_data["elspot-fi"],
        errors="coerce"
    )

    # Spot price predictions second
    predictions_file_path = config["data_paths"]["elspot_fi_predictions"]
    spot_predictions = pd.read_csv(predictions_file_path)
    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(spot_predictions.loc[spot_data.index[0], 'date'])
    dt_end = datetime.fromisoformat(spot_predictions.loc[spot_data.index[-1], 'date'])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    spot_predictions["date"] = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')

    # Merge the spot price and predictions dataframes
    spot_price_and_predictions = spot_data.merge(spot_predictions['predicted_price'], left_index=True, right_index=True, how='left')

    # Replace the hours with a bidding error to be 0 €/MWh instead of -500 €/MWh
    # https://yle.fi/a/74-20061943
    specific_datetime = pd.to_datetime('2023-11-24 15:00+02:00')
    matching_row_index = spot_price_and_predictions.index[spot_price_and_predictions['date'] == specific_datetime]
    spot_price_and_predictions.loc[matching_row_index[0]:matching_row_index[0]+10, 'elspot-fi'] = 0
    spot_price_and_predictions.loc[matching_row_index[0]:matching_row_index[0]+10, 'predicted_price'] = 0

    return spot_price_and_predictions
