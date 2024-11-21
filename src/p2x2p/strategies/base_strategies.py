import numpy as np

from tqdm import tqdm

from p2x2p.strategies import utils, validate, evaluate
from p2x2p.strategies import spot_strategies as spot
from p2x2p.strategies import mfrr_strategies as mfrr


def get_infinite_storage_strategy(data, params):

    strategy = utils.initialize_strategy(data)

    # --- ADD SPOT BIDS ---
    if params['spot']:
        # Optimal allocation of spot bids given the spot market prices for the whole period
        strategy = spot.get_infinite_storage_strategy(data, strategy, params)

    # Compute the actual powers from the bids
    strategy = utils.compute_actual_powers_from_bids(strategy)
    # Validate that the strategy is feasible

    validate.validate_strategy(strategy, params)
    # Evaluate the strategy
    strategy = evaluate.evaluate_strategy(data, strategy, params)

    return strategy


def get_no_horizon_strategy(data, params):
    """
    Get the no horizon strategy for a given plant
    """

   # Check that the price contains 24 hours per day
    n_hours = utils.get_number_of_hours(data)
    assert n_hours%24 == 0, 'Price must contain 24 hours per day'
    n_days = int(n_hours / 24)

    # Initialize a strategy dictionary for the full duration
    strategy = utils.initialize_strategy(data)

    # Loop over all days
    for i in range(n_days):

        # Start index and range (horizon) that we want spot bids for
        spot_start_idx = i * 24
        spot_horizon = 24
        # Shift the start index 12 hours as this is when spot bids are handed in and roughly
        mfrr_offset = -12
        # Start index and range (horizon) that we want mFRR bids for
        mfrr_horizon = 24
        mfrr_start_idx = spot_start_idx + mfrr_offset
        # Look all settled spot bids (day-ahead and potential offset)
        lock_n_spot_bids = 24 - mfrr_offset

        if params['spot']:
            strategy = spot.get_no_horizon_strategy(data, strategy, spot_start_idx, spot_horizon, params)
            strategy = utils.compute_actual_powers_from_bids(strategy)
            strategy = utils.compute_storage_level_from_bids(strategy, params)

        if params['mfrr']:
            strategy = mfrr.get_no_horizon_strategy(data, strategy, mfrr_start_idx, mfrr_horizon, params, lock_n_spot_bids=lock_n_spot_bids)
            strategy = utils.compute_actual_powers_from_bids(strategy)
            strategy = utils.compute_storage_level_from_bids(strategy, params)

    # Complement with additional operations to ensure that the initial storage level is reached at the end
    if params['balanced'] and not params['mfrr']:
        strategy = spot.get_optimal_balanced_strategy(data, strategy, params)
        strategy = utils.compute_actual_powers_from_bids(strategy)
        strategy = utils.compute_storage_level_from_bids(strategy, params)
        
    # Get actual powers and validate the final strategy
    validate.validate_strategy(strategy, params)

    # Evaluate the strategy
    strategy = evaluate.evaluate_strategy(data, strategy, params)

    return strategy


def get_infinite_horizon_strategy(data, params):
    """
    Get the optimal strategy for an infinite horizon
    
    Parameters:
    spot_data (pd.DataFrame): The spot data
    ttf_data (pd.DataFrame): The TTF data
    params (dict): The plant parameters
    
    Returns:
    strategy (dict): The optimal strategy
    """

    start_idx = 0
    horizon = utils.get_number_of_hours(data)
    strategy = utils.initialize_strategy(data)

    # --- ADD SPOT BIDS ---
    if params['spot']:
        # Optimal allocation of spot bids given the spot market prices for the whole period
        strategy = spot.get_horizon_strategy(data, strategy, start_idx, horizon, params, balanced=True)
        strategy = utils.compute_actual_powers_from_bids(strategy)
        strategy = utils.compute_storage_level_from_bids(strategy, params)

    # --- ADD mFRR BIDS ---
    if params['mfrr']:
        # Optimal allocation of mFRR bids given the spot market bids and mFRR prices for the whole period
        strategy = mfrr.get_horizon_strategy(data, strategy, start_idx, horizon, params, balanced=True)
        strategy = utils.compute_actual_powers_from_bids(strategy)
        strategy = utils.compute_storage_level_from_bids(strategy, params)

    # Validate that the strategy is feasible
    validate.validate_strategy(strategy, params)
    
    # Evaluate the strategy
    strategy = evaluate.evaluate_strategy(data, strategy, params)

    return strategy


def get_moving_horizon_strategy(data, params):
    """
    Get the optimal strategy for a moving horizon of horizon_duration hours
    
    Parameters:
    spot_data (pd.DataFrame): The spot data
    params (dict): The plant parameters

    Returns:
    running_p2x (np.array): The optimal power to produce x
    running_x2p (np.array): The optimal power to produce electricity
    profit (float): The profit
    """

    # Number of days
    n_hours = utils.get_number_of_hours(data)
    assert n_hours%24 == 0, 'Price must contain 24 hours per day'
    n_days = int(n_hours / 24)

    # Initialize a strategy dictionary for the full duration
    strategy = utils.initialize_strategy(data)

    # Loop over all days
    for i in tqdm(range(n_days)):

        # Start index and range (horizon) that we want spot bids for
        spot_start_idx = i * 24
        spot_horizon = params['horizon']
        # Shift the start index 12 hours as this is when spot bids are handed in and roughly
        mfrr_offset = -12
        # Start index and range (horizon) that we want mFRR bids for
        mfrr_horizon = 24
        mfrr_start_idx = spot_start_idx + mfrr_offset
        # Look all settled spot bids (day-ahead and potential offset)
        lock_n_spot_bids = 24 - mfrr_offset

        # --- ADD SPOT BIDS ---
        if params['spot']:
            strategy = spot.get_horizon_strategy(data, strategy, spot_start_idx, spot_horizon, params)
            strategy = utils.compute_actual_powers_from_bids(strategy)
            strategy = utils.compute_storage_level_from_bids(strategy, params)

        # --- ADD mFRR BIDS ---
        # Ignore the first day as we do not have any spot bids to lock in
        if params['mfrr']:
            strategy = mfrr.get_horizon_strategy(data, strategy, mfrr_start_idx, mfrr_horizon, params, lock_n_spot_bids=lock_n_spot_bids)
            strategy = utils.compute_actual_powers_from_bids(strategy)
            strategy = utils.compute_storage_level_from_bids(strategy, params)

    # Complement with additional operations to ensure that the initial storage level is reached at the end
    if params['balanced'] and not params['mfrr']:
        strategy = spot.get_optimal_balanced_strategy(data, strategy, params)
        strategy = utils.compute_actual_powers_from_bids(strategy)
        strategy = utils.compute_storage_level_from_bids(strategy, params)

    # Validate the final strategy
    validate.validate_strategy(strategy, params)
        
    # Evaluate the strategy
    strategy = evaluate.evaluate_strategy(data, strategy, params)

    return strategy
