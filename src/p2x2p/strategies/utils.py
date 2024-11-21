import numpy as np

from p2x2p.data import spot, ttf, fingrid


DATA_FIELD_NAMES = ['spot_data', 'ttf_data', 'mfrr_data']
P2X_FIELD_NAMES = ['bid_spot_p2x', 'bid_mfrr_up_p2x', 'bid_mfrr_down_p2x']
X2P_FIELD_NAMES = ['bid_spot_x2p', 'bid_mfrr_up_x2p', 'bid_mfrr_down_x2p']
Y2P_FIELD_NAMES = ['bid_spot_y2p', 'bid_mfrr_up_y2p', 'bid_mfrr_down_y2p']
BID_FIELD_NAMES = P2X_FIELD_NAMES + X2P_FIELD_NAMES + Y2P_FIELD_NAMES
RUNNING_FIELD_NAMES = ['running_p2x', 'running_x2p', 'running_y2p', 'storage_x']


def load_data(params):

    if not isinstance(params, list):
        params = [params]

    # Get the spot data
    spot_data = spot.get_data()

    # Get the TTF data (if needed)
    if any(p['ttf'] for p in params):
        ttf_data = ttf.get_data()
    else:
        ttf_data = None

    # Get the mFRR data (if needed)
    if any(p['mfrr'] for p in params):
        mfrr_data = fingrid.get_mfrr_data()
    else:
        mfrr_data = None

    return {'spot_data': spot_data, 'ttf_data': ttf_data, 'mfrr_data': mfrr_data}


def filter_data(data, params):

    # Calculate the moving average for the spot price if needed
    if params['n_days']:
        data['spot_data']['mean_price'] = (
            data['spot_data']['elspot-fi'].ewm(int(24 * params['n_days'])).mean().values
        )

    # Filter out data for the selected year
    data_year = {
        key: df[df['date'].dt.year == params['year']] if df is not None else None for key, df in data.items()
    }

    return data_year


def get_data(params):

    data = load_data(params)
    data_year = filter_data(data, params)
    return data_year


def select_data_range(data, start, end):

    data_sample = {key: df.iloc[start:end] if df is not None else None for key, df in data.items()}
    return data_sample


def split_data(data, params):
    """Split the data into spot, ttf and mfrr data"""

    # Get the number of data points
    n = max([df.shape[0] if df is not None else 0 for key, df in data.items()])

    # Extract the data
    keys = data.keys()
    spot_data, ttf_data, mfrr_data = [data[key] if key in keys else None for key in DATA_FIELD_NAMES]

    # Check that the data is present and has the correct length
    if params['spot']:
        if spot_data is None:
            raise ValueError('spot_data is missing')
        if spot_data.shape[0] != n:
            raise ValueError('spot_data: incosistent data lengths')
    if params['ttf']:
        if ttf_data is None:
            raise ValueError('ttf_data is missing')
        if ttf_data.shape[0] != n:
            raise ValueError('ttf_data: incosistent data lengths')
    if params['mfrr']:
        if mfrr_data is None:
            raise ValueError('mfrr_data is missing')
        if mfrr_data.shape[0] != n:
            raise ValueError('mfrr_data: incosistent data lengths')

    return spot_data, ttf_data, mfrr_data


def get_number_of_hours(data):
    """Get the number of hours in the data"""
    n = max([df.shape[0] if df is not None else 0 for key, df in data.items()])
    return n


def initialize_strategy(data):
    """Initialize a strategy with zeros for all bid fields"""
    # Get the number of data points
    n = get_number_of_hours(data)
    strategy = {key: np.zeros(n) for key in BID_FIELD_NAMES + RUNNING_FIELD_NAMES}
    return strategy


def get_initial_storage_level(params):
    return params['storage_size']/2


def compute_actual_powers_from_bids(strategy):
    strategy['running_p2x'] = strategy['bid_spot_p2x'] + strategy['bid_mfrr_down_p2x'] - strategy['bid_mfrr_up_p2x']
    strategy['running_x2p'] = strategy['bid_spot_x2p'] + strategy['bid_mfrr_up_x2p'] - strategy['bid_mfrr_down_x2p']
    strategy['running_y2p'] = strategy['bid_spot_y2p'] + strategy['bid_mfrr_up_y2p'] - strategy['bid_mfrr_down_y2p']
    return strategy


def compute_storage_level_from_bids(strategy, params):
    flow = strategy['running_p2x'] * params['eff_p2x'] - strategy['running_x2p'] / params['eff_x2p']
    strategy['storage_x'] = np.cumsum(flow) + get_initial_storage_level(params)
    return strategy

def corrent_numerical_errors(strategy, storage_init, params):
    """
    Correct numerical errors in the strategy
    Iteratively calls to PuLP where variables are conveterted back and forth between numpy and PuLP
    seems to introduce small numerical erorrors that can accumulate over time. Consequently, the storage
    level can become negative or exceed the maximum storage level. This function corrects these errors.

    Parameters:
    strategy (dict): The strategy to correct

    Returns:
    strategy (dict): The corrected strategy
    """

    # Ensure that all power levels are within the limits
    for key in P2X_FIELD_NAMES + ['running_p2x']:
        strategy[key] = np.clip(strategy[key], 0, params['power_p2x'])
    for key in X2P_FIELD_NAMES + ['running_x2p']:
        strategy[key] = np.clip(strategy[key], 0, params['power_x2p'])
    for key in Y2P_FIELD_NAMES + ['running_y2p']:
        strategy[key] = np.clip(strategy[key], 0, params['power_x2p'])

    # Ensure that the storage level is within the limits
    storage_tmp = storage_init
    for i in range(strategy['running_p2x'].size):
        # Test the next step to see what the storage level would be
        delta_test = strategy['running_p2x'][i]*params['eff_p2x'] - strategy['running_x2p'][i]/params['eff_x2p']
        storage_test = storage_tmp + delta_test
        # storage_test = np.round(storage_tmp + delta_test, 6)
        # Check and correct if the storage is about to overflow
        if params['storage_size'] < storage_test:
            delta_p = (params['storage_size']-storage_test) / params['eff_p2x']
            if np.round(strategy['bid_spot_p2x'][i], 6) > 0:    # Round fist so that we don't correc if it is only nozero due to numerical errors
                strategy['bid_spot_p2x'][i] += delta_p
            elif strategy['bid_mfrr_down_p2x'][i] > 0:
                strategy['bid_mfrr_down_p2x'][i] += delta_p
        # Check and correct if the storage is about to underflow
        if 0 > storage_test:
            delta_p = -storage_test * params['eff_x2p']
            if np.round(strategy['bid_spot_x2p'][i], 6) > 0:    # Round fist so that we don't correc if it is only nozero due to numerical errors
                strategy['bid_spot_x2p'][i] -= delta_p
            elif strategy['bid_mfrr_up_x2p'][i] > 0:
                strategy['bid_mfrr_up_x2p'][i] -= delta_p
        # Update the storage level based on the corrected strategy
        strategy['running_p2x'][i] = strategy['bid_spot_p2x'][i] + strategy['bid_mfrr_down_p2x'][i] - strategy['bid_mfrr_up_p2x'][i]
        strategy['running_x2p'][i] = strategy['bid_spot_x2p'][i] + strategy['bid_mfrr_up_x2p'][i] - strategy['bid_mfrr_down_x2p'][i]
        storage_tmp += strategy['running_p2x'][i]*params['eff_p2x'] - strategy['running_x2p'][i]/params['eff_x2p']

    return strategy
