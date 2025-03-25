import json
import pickle
import hashlib
import pandas as pd

from tqdm import tqdm

from p2x2p.data import spot, ttf, fingrid
from p2x2p.utils.utils import get_config
from p2x2p.strategies import base_strategies, utils

TABLE_COLS = [
    'profit', 'spot_x_profit', 'spot_y_profit', 'mfrr_x_profit', 'mfrr_y_profit',
    'p2x_running_frac', 'x2p_running_frac', 'y2p_running_frac',
    'p2x_avg_price', 'x2p_avg_price', 'y2p_avg_price',
    'p2x_amount', 'x2p_amount', 'y2p_amount',
    ]
#TABLE_COLS = ['profit']
VALID_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
VALID_STRATEGIES = ['infinite_storage', 'no_horizon', 'infinite_horizon', 'moving_horizon']
IGNORE_PARAMS = {
    'infinite_storage': ['storage_size', 'n_days','k_p2x','k_x2p', 'horizon',
                         'balanced', 'ttf_premium', 'ttf', 'mfrr',
                         'lambda_p2x', 'lambda_x2p'],
    'no_horizon': ['horizon'],
    'infinite_horizon': ['n_days','k_p2x','k_x2p', 'horizon', 'balanced'],
    'moving_horizon': [],
}


def hash_params(params):
    """ Generate a hash for the parameters """
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_string.encode()).hexdigest()


def validate_params(params):
    """ Validate the parameters """
    if params['year'] not in VALID_YEARS:
        raise ValueError(f"Invalid year: {params['year']}")

    if params['name'] not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy name: {params['name']}")

    if params['horizon'] < 24:
        raise ValueError("The horizon has to be at least 24 hours")

    # Set params that don't affect the strategy to Null
    params_pruned = params.copy()
    for key in IGNORE_PARAMS[params['name']]:
        params_pruned[key] = None
    if not params_pruned['ttf']:
        params_pruned['ttf_premium'] = None

    return params_pruned
    

def compute_strategy(data, params):
    """ Get the strategy for a given set of params"""

    data_year = utils.filter_data(data, params)

    # Get the strategy
    if params['name'] == 'infinite_storage':
        strategy = base_strategies.get_infinite_storage_strategy(data_year, params)
    elif params['name'] == 'no_horizon':
        strategy = base_strategies.get_no_horizon_strategy(data_year, params)
    elif params['name'] == 'infinite_horizon':
        strategy = base_strategies.get_infinite_horizon_strategy(data_year, params)
    elif params['name'] == 'moving_horizon':
        strategy = base_strategies.get_moving_horizon_strategy(data_year, params)
    else:
        raise ValueError("Invalid strategy name")

    return strategy


def get_summary_for_strategies(params_list, overwrite=False):
    """
    Calculate the profit for a list of parameters. If the profit already exists in the cache, it is read from there.
    
    :param params_list: list of parameters
    :param overwrite: if True, the existing profits are overwritten
    
    :return: pandas dataframe with the profits
    """

    config = get_config()
    if not isinstance(params_list, list):
        params = [params_list]

    # Read the existing profits
    summery_file = config['project_paths']['strategies'] + 'summaries.csv'
    try:
        df = pd.read_csv(summery_file, index_col='hash_index')
    except:
        df = pd.DataFrame(columns=['hash_index'] + TABLE_COLS + list(params_list[0].keys()))
        df.set_index('hash_index', inplace=True)

    # Get hashes and check if we have cached results for all params in the list
    params_hash_indices = [hash_params(validate_params(p)) for p in params_list]
    if overwrite:
        params_missing_list = params_list
    else:
        params_missing_list = [p for p, hash in zip(params_list, params_hash_indices) if hash not in df.index]

    # load data and get results for parameter combinations not in the cache
    if params_missing_list:

        data = utils.load_data(params_missing_list)

        # Get results for parameter combinations not in the cache
        for params in params_missing_list:
            
            # Get the strategy for the current parameters
            params_hash_index = hash_params(validate_params(params))
            #print(f"Calculating the running strategy for params {str(params)}")
            strategy = compute_strategy(data, params)

            # Save the strategy to disk
            if params['save']:
                strategy_file = config['project_paths']['strategies'] + f'{str(params_hash_index)}.pkl'
                with open(strategy_file, 'wb') as f:
                    pickle.dump(strategy, f)

            # Add the profit and parameters to the dataframe
            new_row = {key: strategy[key] if key in strategy else None for key in TABLE_COLS}
            new_row.update(params)
            df.loc[params_hash_index] = new_row
            # Save the dataframe to disk
            df.to_csv(summery_file, index=True)

    # return df with index rows selected
    return df.loc[params_hash_indices]


def load_strategies(df):
    """
    Load the strategies from the dataframe

    :param df: dataframe with the strategies

    :return: list of strategies
    """

    config = get_config()
    strategies = []
    for index, _ in df.iterrows():
        strategy_file = config['project_paths']['strategies'] + f'{str(index)}.pkl'
        try:
            with open(strategy_file, 'rb') as f:
                strategy = pickle.load(f)
                strategies.append(strategy)
        except:
            print(f"Error loading strategy from file {strategy_file}")

    return strategies


def get_strategy(params, overwrite=False):
    df = get_summary_for_strategies([params], overwrite=overwrite)
    return load_strategies(df)[0]
