import os
import pulp
import inspect
import numpy as np

from p2x2p.strategies import infinite_storage, evaluate, utils


class suppress_output:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.saved_stdout = os.dup(1)
        self.saved_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.saved_stdout, 1)
        os.dup2(self.saved_stderr, 2)
        os.close(self.null_fd)
        os.close(self.saved_stdout)  # Close the saved stdout descriptor
        os.close(self.saved_stderr)  # Close the saved stderr descriptor

def get_infinite_storage_strategy(data, strategy, params):

    spot_data, _, _ = utils.split_data(data, params)

    # Extract the spot data
    price = spot_data['elspot-fi'].values
    # Sort the price indices in descending order
    price_sorted_args = np.argsort(price)[::-1]

    # Get the maximum fraction of the time that the plant can be in either mode
    max_x2p_frac = infinite_storage.get_max_x2p_frac(params)
    max_p2x_frac = 1 - max_x2p_frac

    # P2X2P profits
    durations, revenues, costs = infinite_storage.get_revenues_and_costs(spot_data, params)
    profits = revenues - costs
    # Check the utilization that maximizes the profit
    optimal_utilization = durations[np.argmax(profits)]

    # Get the indices for the most expensive and cheapest hours
    x2p_indices = price_sorted_args[durations<=max_x2p_frac*optimal_utilization]
    p2x_indices = price_sorted_args[durations>=(1-max_p2x_frac*optimal_utilization)]

    strategy['bid_spot_p2x'][p2x_indices] += params['power_p2x']
    strategy['bid_spot_x2p'][x2p_indices] += params['power_x2p']

    return  strategy

def get_no_horizon_strategy(data, strategy, start_idx, horizon, params):

    spot_data, ttf_data, _ = utils.split_data(data, params)

    # Determined the range to find bids for
    end_idx = start_idx + horizon
    if start_idx > 0:
        storage_init = strategy['storage_x'][start_idx-1]
    else:
        storage_init = utils.get_initial_storage_level(params)

     # Extract the spot data
    spot_price = spot_data['elspot-fi'].values[start_idx:end_idx]
    spot_baseline = spot_data['mean_price'].values[start_idx]
    spot_forecasts = spot_data['predicted_price'].values[start_idx:end_idx]
    # spot_forecasts = spot_data['elspot-fi'].values[start_idx:end_idx]
    # Charging and discharging price levels
    p2x_level = spot_baseline * params['k_p2x'] / (1+params['kappa_p2x'])
    x2p_level = spot_baseline * params['k_x2p'] / (1-params['kappa_x2p'])
    if params['ttf']:
        y2p_level = (ttf_data['TTF (Euro/MWh)'].values[start_idx]+params['ttf_premium']) / params['eff_x2p']
        y2p_level = y2p_level / (1-params['kappa_x2p'])
    else:
        y2p_level = np.inf

    # How many hours can each step be run during the coming day
    n_possible_hours_p2x = int((params['storage_size']-storage_init) / params['power_p2x'] / params['eff_p2x'])
    n_possible_hours_x2p = int(storage_init / params['power_x2p'] * params['eff_x2p'])

    # How many hours would we want to run
    # Sort based on the daily price first so that one can easily pick the cheapest and most expensive hours
    spot_fluctuations_sorted_idx = np.argsort(spot_forecasts)

    # Get indices for the hours that we want to run
    wanted_hours_p2x = spot_fluctuations_sorted_idx[:n_possible_hours_p2x]
    wanted_hours_x2p = np.flip(spot_fluctuations_sorted_idx)[:n_possible_hours_x2p]
    # Assume an infinite supply of Y so that it can be used during every hour
    wanted_hours_y2p = np.flip(spot_fluctuations_sorted_idx)

    # Set bid levels for each hour
    p2x_bid_limits = np.full(spot_price.size, -np.inf)
    p2x_bid_limits[wanted_hours_p2x] = p2x_level
    x2p_bid_limits = np.full(spot_price.size, np.inf)
    x2p_bid_limits[wanted_hours_x2p] = x2p_level
    y2p_bid_limits = np.full(spot_price.size, np.inf)
    y2p_bid_limits[wanted_hours_y2p] = y2p_level

    # Get indices for accepted bids
    accepted_hours_p2x = np.where(p2x_bid_limits > spot_price)[0]
    accepted_hours_x2p = np.where(x2p_bid_limits < spot_price)[0]
    accepted_hours_y2p = np.setdiff1d(np.where(y2p_bid_limits < spot_price)[0], accepted_hours_x2p)

    # Daily result arrays
    strategy['bid_spot_p2x'][accepted_hours_p2x+start_idx] += params['power_p2x']
    strategy['bid_spot_x2p'][accepted_hours_x2p+start_idx] += params['power_x2p']
    strategy['bid_spot_y2p'][accepted_hours_y2p+start_idx] += params['power_x2p']

    return strategy


def get_horizon_strategy(data, strategy, start_idx, horizon, params, balanced=False):
    """
    Get the optimal strategy for a given horizon
    """

    spot_data, ttf_data, _ = utils.split_data(data, params)

    # Determined the range to find bids for
    end_idx = start_idx + horizon
    if start_idx > 0:
        storage_init = strategy['storage_x'][start_idx - 1]
    else:
        storage_init = utils.get_initial_storage_level(params)

    # Extract the price data
    spot_price = spot_data['elspot-fi'].values[start_idx:end_idx]
    mean_price_level = spot_data['mean_price'].values[start_idx:end_idx]
    if params['ttf']:
        ttf_price = ttf_data['TTF (Euro/MWh)'].values[start_idx:end_idx] + params['ttf_premium']
    else:
        ttf_price = np.full(spot_price.size, np.inf)

    # Set the charge/discharge limits
    mean_price_init = mean_price_level[0]
    p2x_lim = mean_price_init * params['k_p2x']
    x2p_lim = mean_price_init * params['k_x2p']
    
    # Constant
    m = 1e6  # Linearization constant

    # Time steps
    timesteps = [f"t_{i}" for i in range(spot_price.size)]

    # MILP variables
    running_p2x_var = pulp.LpVariable.dicts("running_p2x", timesteps, lowBound=0, upBound=params['power_p2x'])
    running_x2p_var = pulp.LpVariable.dicts("running_x2p", timesteps, lowBound=0, upBound=params['power_x2p'])
    running_y2p_var = pulp.LpVariable.dicts("running_y2p", timesteps, lowBound=0, upBound=params['power_x2p'])
    storage_level_var = pulp.LpVariable.dicts("storage", timesteps, lowBound=0, upBound=params['storage_size'])
    # Expected added value from charging or discharging the storage
    delta_p2x_var = pulp.LpVariable("delta_p2x", lowBound=0)
    delta_x2p_var = pulp.LpVariable("delta_x2p", lowBound=0)
    # Binary variables needed for keeping the deltas >= 0
    y_p2x_var = pulp.LpVariable("y_p2x", lowBound=0, upBound=1, cat='Binary')
    y_x2p_var = pulp.LpVariable("y_x2p", lowBound=0, upBound=1, cat='Binary')
    
    # Problem
    prob = pulp.LpProblem("p2x2p", pulp.LpMaximize)

    # --- OBJECTIVE FUNCTION ---
    # Different objective functions depending on whether the TTF option is used or not
    if params['ttf']:
        # The objective function is the profit for the running strategy with TTF option added
        prob += (
            pulp.lpSum(
                [(1-params['kappa_x2p']) * spot_price[i] * running_x2p_var[t] + # Income from selling electricity produced from x
                 (1-params['kappa_x2p']) * spot_price[i] * running_y2p_var[t] - # Income from selling electricity produced from y
                 (1+params['kappa_p2x']) * spot_price[i] * running_p2x_var[t] - # Cost of producing x
                 ttf_price[i] * running_y2p_var[t] / params["eff_x2p"]          # Cost of used y
                 for i, t in enumerate(timesteps)]
                ) + delta_p2x_var - delta_x2p_var,       # Expected value change from charging or discharging the storage
            "Profit",
        )
    else:
        # The objective function is the profit for the running strategy
        prob += (
            pulp.lpSum(
                [(1-params['kappa_x2p']) * spot_price[i] * running_x2p_var[t] - # Income from selling electricity produced from x
                 (1+params['kappa_p2x']) * spot_price[i] * running_p2x_var[t]   # Cost of producing x
                 for i, t in enumerate(timesteps)]
                ) + delta_p2x_var - delta_x2p_var,      # Expected value change from charging or discharging the storage
            "Profit",
        )

    # --- CONSTRAINTS ---
    # Constraints to ensure that the total electricity production does not exceed the plant capacity
    if params['ttf']:
        for t in timesteps:
            prob += running_x2p_var[t] + running_y2p_var[t] <= params["power_x2p"]

    # Ensure that the storage level change corresponds to the amount produced or used during every hour
    prob += storage_level_var["t_0"] - params['eff_p2x']*running_p2x_var["t_0"] + 1/params['eff_x2p']*running_x2p_var["t_0"] - storage_init == 0, "storage_t_0"
    for i in range(1, len(timesteps)):
        current = timesteps[i]
        previous = timesteps[i-1]
        prob += storage_level_var[current] - params['eff_p2x']*running_p2x_var[current] + 1/params['eff_x2p']*running_x2p_var[current] - storage_level_var[previous] == 0, "storage_" + current
    if balanced:
        # Ensure that the storage level is the same at the beginning and at the end of the horizon
        prob += storage_level_var[timesteps[-1]] - storage_init == 0, "storage_t_end"

    # Estimating the cost of the storage level change
    prob += delta_p2x_var >= (storage_level_var[timesteps[-1]]-storage_init)/params['eff_p2x'] * p2x_lim
    prob += delta_p2x_var <= (storage_level_var[timesteps[-1]]-storage_init)/params['eff_p2x'] * p2x_lim + m*(1-y_p2x_var)
    prob += delta_p2x_var <= m*y_p2x_var
    prob += delta_x2p_var >= (storage_init-storage_level_var[timesteps[-1]])*params['eff_x2p'] * x2p_lim
    prob += delta_x2p_var <= (storage_init-storage_level_var[timesteps[-1]])*params['eff_x2p'] * x2p_lim + m*(1-y_x2p_var)
    prob += delta_x2p_var <= m*y_x2p_var
    
    # --- SOLVE AND EXTRACT VARIABLES VALUES ---
    # The HiGHS solver seems more robust so we use that one if available
    if 'HiGHS' in pulp.listSolvers(onlyAvailable=True):
        with suppress_output():
            prob.solve(pulp.apis.HiGHS())
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the optimization was successful
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Spot {inspect.currentframe().f_code.co_name}: Optimization status: {pulp.LpStatus[prob.status]}")
        print(f"Storage start: {storage_init}")

    # Extract found variables for the 24 first hours
    strategy['bid_spot_p2x'][start_idx:end_idx] = np.array([running_p2x_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_spot_x2p'][start_idx:end_idx] = np.array([running_x2p_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_mfrr_down_p2x'][start_idx:end_idx] = np.zeros(len(timesteps))
    strategy['bid_mfrr_down_x2p'][start_idx:end_idx] = np.zeros(len(timesteps))
    strategy['bid_mfrr_up_p2x'][start_idx:end_idx] = np.zeros(len(timesteps))
    strategy['bid_mfrr_up_x2p'][start_idx:end_idx] = np.zeros(len(timesteps))
    if params['ttf']:
        strategy['bid_spot_y2p'][start_idx:end_idx] = np.array([running_y2p_var[t].value() for t in timesteps[:horizon]])
        strategy['bid_mfrr_down_y2p'][start_idx:end_idx] = np.zeros(len(timesteps))
        strategy['bid_mfrr_up_y2p'][start_idx:end_idx] = np.zeros(len(timesteps))

    return strategy


def get_optimal_balanced_strategy(data, strategy, params): 
    """
    Get the optimal additional charge and discharge strategy for getting the storage back to the initial level
    """

    storage_init = utils.get_initial_storage_level(params)

    # Get the optimal charge or discharge strategy depending on wather the storage is over or under the initial level
    net_flux = np.sum(strategy['running_p2x']*params['eff_p2x']) - np.sum(strategy['running_x2p']/params['eff_x2p'])
    if net_flux < 0:
        strategy = _get_optimal_charge_strategy(data, strategy, storage_init, params)
    else:
        strategy = _get_optimal_discharge_strategy(data, strategy, storage_init, params)

    return strategy


def _get_optimal_charge_strategy(data, strategy, storage_init, params):
    """
    Get the optimal additional charge strategy for getting the storage back to the initial level
    """

    # Extract the spot data
    spot_data, _, _ = utils.split_data(data, params)
    price = spot_data['elspot-fi'].values

    # Time steps
    timesteps = [f"t_{i}" for i in range(price.size)]

    # Problem variables
    running_p2x_var = pulp.LpVariable.dicts("running_p2x", timesteps, lowBound=0, upBound=params["power_p2x"])
    storage_var = pulp.LpVariable.dicts("storage", timesteps, lowBound=0, upBound=params["storage_size"])

    # Problem
    prob = pulp.LpProblem("p2x2p", pulp.LpMaximize)

    # The objective function is added to 'prob' first
    # The profit for the running strategy
    prob += (
        pulp.lpSum(
            [-price[i] * running_p2x_var[t]   # Cost of producing x
             for i, t in enumerate(timesteps)]
            ), 
        "Profit"
        )

     # Constraints to ensure that the total x production does not exceed the plant capacity
    for i, t in enumerate(timesteps):
        prob += strategy['bid_spot_p2x'][i] + strategy['bid_mfrr_down_p2x'][i] + running_p2x_var[t] <= params['power_p2x']

    # Ensure that the storage level change corresponds to the amount produced or used during every hour
    prob += storage_var["t_0"] - params["eff_p2x"]*(strategy['running_p2x'][0]+running_p2x_var["t_0"]) + 1/params["eff_x2p"]*strategy['running_x2p'][0] - storage_init == 0, "storage_t_0"
    for i in range(1, len(timesteps)):
        current = timesteps[i]
        previous = timesteps[i-1]
        prob += storage_var[current] - params["eff_p2x"]*(strategy['running_p2x'][i]+running_p2x_var[current]) + 1/params["eff_x2p"]*strategy['running_x2p'][i]- storage_var[previous] == 0, "storage_" + current
    # Ensure that the storage level is the same at the beginning and at the end of the horizon
    prob += storage_var[timesteps[-1]] - storage_init == 0, "storage_t_end"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the optimization was successful
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"{inspect.currentframe().f_code.co_name}: Optimization status: {pulp.LpStatus[prob.status]}")

    # Extract found variables
    strategy['bid_spot_p2x'] += np.array([running_p2x_var[t].value() for t in timesteps])

    return strategy


def _get_optimal_discharge_strategy(data, strategy, storage_init, params):
    """
    Get the optimal additional discharge strategy for getting the storage back to the initial level
    """

    # Extract the spot data
    spot_data, _, _ = utils.split_data(data, params)
    price = spot_data['elspot-fi'].values

    # Time steps
    timesteps = [f"t_{i}" for i in range(price.size)]

    # Problem variables
    running_x2p_var = pulp.LpVariable.dicts("running_x2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    # Increase the upper bound for the storage variable to avoid numerical issues
    storage_var = pulp.LpVariable.dicts("storage", timesteps, lowBound=0, upBound=params["storage_size"])

    # Problem
    prob = pulp.LpProblem("p2x2p", pulp.LpMaximize)

    # The objective function is added to 'prob' first
    # The profit for the running strategy
    prob += (
        pulp.lpSum(
            [price[i] * running_x2p_var[t]   # Income from selling electricity produced from x
             for i, t in enumerate(timesteps)]
            ), 
        "Profit"
        )

    # Constraints to ensure that the total electricity production does not exceed the plant capacity
    for i, t in enumerate(timesteps):
        prob += strategy['bid_spot_x2p'][i] + strategy['bid_spot_y2p'][i] + strategy['bid_mfrr_up_x2p'][i] + strategy['bid_mfrr_up_y2p'][i] + running_x2p_var[t] <= params['power_x2p']

    # Ensure that the storage level change corresponds to the amount produced or used during every hour
    prob += storage_var["t_0"] - params["eff_p2x"]*strategy['running_p2x'][0] + 1/params["eff_x2p"]*(strategy['running_x2p'][0]+running_x2p_var['t_0']) - storage_init == 0, "storage_t_0"
    for i in range(1, len(timesteps)):
        current = timesteps[i]
        previous = timesteps[i-1]
        prob += storage_var[current] - params["eff_p2x"]*strategy['running_p2x'][i] + 1/params["eff_x2p"]*(strategy['running_x2p'][i]+running_x2p_var[current]) - storage_var[previous] == 0, "storage_" + current
    # Ensure that the storage level is the same at the beginning and at the end of the horizon
    prob += storage_var[timesteps[-1]] - storage_init == 0, "storage_t_end"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the optimization was successful
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"{inspect.currentframe().f_code.co_name}: Optimization status: {pulp.LpStatus[prob.status]}")

    # Extract found variables
    strategy['bid_spot_x2p'] += np.array([running_x2p_var[t].value() for t in timesteps])

    return strategy