import numpy as np
import pandas as pd

from datetime import datetime

from p2x2p.utils.utils import get_config


def get_data():
    
    config = get_config()

    file_path = config["data_paths"]["ttf_natural_gas"]
    ttf_data = pd.read_csv(file_path)

    # Change the original dates to datetimes with time zone and daylight saving
    # The originals are in European/Helsinki time zone. So we are simply
    # adding that information explicitly.

    # Convert the first and last hour to datetime
    dt_start = datetime.fromisoformat(ttf_data.loc[ttf_data.index[0], 'date'])
    dt_end = datetime.fromisoformat(ttf_data.loc[ttf_data.index[-1], 'date'])
    # Get the hourly timestamps for the interval
    n_hours = (dt_end.timestamp() - dt_start.timestamp()) / 3600 + 1
    timestamps = dt_start.timestamp() + np.arange(n_hours) * 3600
    ttf_data["date"] = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki')

    # Drop valuees that are not in the date interval 2016-2023
    ttf_data = ttf_data[
        (ttf_data["date"].dt.year >= 2016) &
        (ttf_data["date"].dt.year <= 2023)
    ]

    return ttf_data
