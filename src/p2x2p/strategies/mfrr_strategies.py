import os
import pulp
import inspect
import numpy as np

from p2x2p.strategies.utils import split_data, get_initial_storage_level


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


def get_no_horizon_strategy(data, strategy, start_idx, horizon, params, lock_n_spot_bids=None):

    if lock_n_spot_bids is None:
        lock_n_spot_bids = horizon
    assert(lock_n_spot_bids >= horizon)

    # Split the data
    spot_data, ttf_data, mfrr_data = split_data(data, params)

    # Determined the range to find bids for
    if start_idx < 0:
        horizon += start_idx
        start_idx = 0
    end_idx = start_idx + horizon

    # Extract the price data
    mean_spot_price = spot_data['mean_price'].values[start_idx:end_idx]
    # mean_spot_price = spot_data['baseline'].values[start_idx:end_idx]
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values[start_idx:end_idx]
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values[start_idx:end_idx]
    mfrr_down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values[start_idx:end_idx]
    mfrr_up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values[start_idx:end_idx]
    if any('frac' in s for s in mfrr_data.columns):
        mfrr_down_frac = mfrr_data['down_regulation_frac_mFRR'].values[start_idx:end_idx]
        mfrr_up_frac = mfrr_data['up_regulation_frac_mFRR'].values[start_idx:end_idx]
    else:
        mfrr_down_frac = np.full(mean_spot_price.size, 1)
        mfrr_up_frac = np.full(mean_spot_price.size, 1)
    # Extract spot bids
    bid_spot_p2x = strategy['bid_spot_p2x'][start_idx:end_idx]
    bid_spot_x2p = strategy['bid_spot_x2p'][start_idx:end_idx]
    bid_spot_y2p = strategy['bid_spot_y2p'][start_idx:end_idx]

    # Charging and discharging price levels
    p2x_level = mean_spot_price[0] * params['k_p2x']
    x2p_level = mean_spot_price[0] * params['k_x2p']
    if params['ttf']:
        y2p_level = (ttf_data['TTF (Euro/MWh)'].values[start_idx]+params['ttf_premium'])/params['eff_x2p']
    else:
        y2p_level = np.inf

    # Result arrays
    mfrr_down_p2x = np.zeros(mean_spot_price.size)
    mfrr_up_p2x = np.zeros(mean_spot_price.size)
    mfrr_down_x2p = np.zeros(mean_spot_price.size)
    mfrr_up_x2p = np.zeros(mean_spot_price.size)
    mfrr_down_y2p = np.zeros(mean_spot_price.size)
    mfrr_up_y2p = np.zeros(mean_spot_price.size)

    # Storage levels from spot bids that constrain mFRR bids
    storage_level = strategy['storage_x'][start_idx:(start_idx+lock_n_spot_bids)]

    # Loop over all hours
    for i in range(mean_spot_price.size):

        # Down regulation
        if mfrr_down_volyme[i] < 0:

            market_power = -mfrr_down_volyme[i]                                         # Power available within the market

            # P2X
            if (mfrr_down_price[i] < p2x_level) and (bid_spot_p2x[i] < params['power_p2x']):
                max_x_in = params['storage_size'] - storage_level[i:].max()             # Max flux without overfilling the storage
                available_power = params['power_p2x'] - bid_spot_p2x[i]                 # Power available for P2X down regulation
                allowed_power = max_x_in / params['eff_p2x']                            # Max power without overfilling the storage
                used_power = min([available_power, allowed_power, market_power])        # Power used for P2X down regulation
                used_power *= mfrr_down_frac[i]                                         # Fraction of the hour activated
                flux = used_power * params['eff_p2x']                                   # Flux to the storage
                storage_level[i:] += flux                                               # Update the storage level             
                mfrr_down_p2x[i] = used_power                                           # Update the P2X down regulation

            # X2P
            if (mfrr_down_price[i] < x2p_level) and (bid_spot_x2p[i] > 0):
                max_x_in = params['storage_size'] - storage_level[i:].max()             # Max flux without overfilling the storage
                available_power = bid_spot_x2p[i]                                       # Power available for X2P down regulation
                allowed_power = max_x_in * params['eff_x2p']                            # Max power without overfilling the storage
                used_power = min([available_power, allowed_power, market_power])        # Power used for X2P down regulation
                used_power *= mfrr_down_frac[i]                                         # Fraction of the hour activated
                flux = used_power / params['eff_x2p']                                   # Flux from the storage
                storage_level[i:] += flux                                               # Update the storage level
                mfrr_down_x2p[i] = used_power                                           # Update the X2P down regulation

            # Y2P
            if (mfrr_down_price[i] < y2p_level) and (bid_spot_y2p[i] > 0):
                available_power = bid_spot_y2p[i] - mfrr_down_x2p[i]                    # Power available for Y2P down regulation
                used_power = min([available_power, market_power])                       # Power used for Y2P down regulation
                used_power *= mfrr_down_frac[i]                                         # Fraction of the hour activated
                mfrr_down_y2p[i] = used_power                                           # Update the Y2P down regulation

        # Up regulation
        if mfrr_up_volyme[i] > 0:

            market_power = mfrr_up_volyme[i]                                            # Power available within the market

            # P2X
            if (mfrr_up_price[i] > p2x_level) and (bid_spot_p2x[i] > 0):
                max_x_out = storage_level[i:].min()                                     # Max flux without emptying the storage
                available_power = bid_spot_p2x[i]                                       # Power available for P2X up regulation
                allowed_power = max_x_out / params['eff_p2x']                           # Max power without emptying the storage
                used_power = min([available_power, allowed_power, market_power])        # Power used for P2X up regulation
                used_power *= mfrr_up_frac[i]                                           # Fraction of the hour activated
                flux = -used_power * params['eff_p2x']                                  # Flux from the storage
                storage_level[i:] += flux                                               # Update the storage level
                mfrr_up_p2x[i] = used_power                                             # Update the P2X up regulation

            # X2P
            if (mfrr_up_price[i] > x2p_level) and (bid_spot_x2p[i] < params['power_x2p']):
                max_x_out = storage_level[i:].min()                                     # Max flux without emptying the storage  
                available_power = (params['power_x2p'] - 
                                   bid_spot_x2p[i] - 
                                   bid_spot_y2p[i])                                     # Power available for X2P up regulation
                allowed_power = max_x_out * params['eff_x2p']                           # Max power without emptying the storage
                used_power = min([available_power, allowed_power, market_power])        # Power used for X2P up regulation
                used_power *= mfrr_up_frac[i]                                           # Fraction of the hour activated
                flux = -used_power / params['eff_x2p']                                  # Flux to the storage
                storage_level[i:] += flux                                               # Update the storage level
                mfrr_up_x2p[i] = used_power                                             # Update the X2P up regulation

            # Y2P
            if (mfrr_up_price[i] > y2p_level) and (bid_spot_y2p[i] < params['power_x2p']):
                available_power = (params['power_x2p'] - 
                                   bid_spot_x2p[i] - 
                                   bid_spot_y2p[i] - 
                                   mfrr_up_x2p[i])                                      # Power available for Y2P up regulation
                used_power = min([available_power, market_power])                       # Power used for Y2P up regulation
                used_power *= mfrr_up_frac[i]                                           # Fraction of the hour activated
                mfrr_up_y2p[i] = used_power                                             # Update the Y2P up regulation

    strategy['bid_mfrr_down_p2x'][start_idx:end_idx] = mfrr_down_p2x
    strategy['bid_mfrr_down_x2p'][start_idx:end_idx] = mfrr_down_x2p
    strategy['bid_mfrr_down_y2p'][start_idx:end_idx] = mfrr_down_y2p
    strategy['bid_mfrr_up_p2x'][start_idx:end_idx] = mfrr_up_p2x
    strategy['bid_mfrr_up_x2p'][start_idx:end_idx] = mfrr_up_x2p
    strategy['bid_mfrr_up_y2p'][start_idx:end_idx] = mfrr_up_y2p

    return strategy


def get_horizon_strategy(data, strategy, start_idx, horizon, params, lock_n_spot_bids=None, balanced=False):
    """"""

    # Split the data
    spot_data, ttf_data, mfrr_data = split_data(data, params)

    # Determined the range to find bids for
    if start_idx < 0:
        horizon += start_idx
        start_idx = 0
    end_idx = start_idx + horizon

    # Extract the spot data
    spot_price = spot_data['elspot-fi'].values[start_idx:end_idx]
    mean_price_level = spot_data['mean_price'].values[start_idx:end_idx]
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values[start_idx:end_idx]
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values[start_idx:end_idx]
    mfrr_down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values[start_idx:end_idx]
    mfrr_up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values[start_idx:end_idx]
    if params['ttf']:
        ttf_price = ttf_data['TTF (Euro/MWh)'].values[start_idx:end_idx] + params['ttf_premium']
    else:
        ttf_price = np.full(spot_price.size, np.inf)

    # Set the charge/discharge limits
    mean_price_init = mean_price_level[0]
    p2x_lim = mean_price_init * params['k_p2x']
    x2p_lim = mean_price_init * params['k_x2p']

    # Determine the initial storage level
    if start_idx > 0:
        storage_init = strategy['storage_x'][start_idx - 1]
    else:
        storage_init = get_initial_storage_level(params)
    # Determine the final storage level from the spot bids
    if (lock_n_spot_bids is None) or (lock_n_spot_bids > horizon):
        storage_end = strategy['storage_x'][end_idx-1]
    else:
        storage_end = strategy['storage_x'][start_idx+lock_n_spot_bids-1]

    # Time steps
    timesteps = [f"t_{i}" for i in range(spot_price.size)]

    # Problem variables
    # Actual power
    running_p2x_var = pulp.LpVariable.dicts("running_p2x", timesteps, lowBound=0, upBound=params["power_p2x"])
    running_x2p_var = pulp.LpVariable.dicts("running_x2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    running_y2p_var = pulp.LpVariable.dicts("running_y2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    # Power bid to the mFRR down market
    mfrr_down_p2x_var = pulp.LpVariable.dicts("mfrr_down_p2x", timesteps, lowBound=0, upBound=params["power_p2x"])
    mfrr_down_x2p_var = pulp.LpVariable.dicts("mfrr_down_x2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    mfrr_down_y2p_var = pulp.LpVariable.dicts("mfrr_down_y2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    # Power bid to the mFRR up market
    mfrr_up_p2x_var = pulp.LpVariable.dicts("mfrr_up_p2x", timesteps, lowBound=0, upBound=params["power_p2x"])
    mfrr_up_x2p_var = pulp.LpVariable.dicts("mfrr_up_x2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    mfrr_up_y2p_var = pulp.LpVariable.dicts("mfrr_up_y2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    # Expected added value from charging or discharging the storage
    delta_p2x_var = pulp.LpVariable("delta_p2x", lowBound=0)
    delta_x2p_var = pulp.LpVariable("delta_x2p", lowBound=0)
    # Binary variables needed for keeping the deltas >= 0
    y_p2x_var = pulp.LpVariable("y_p2x", lowBound=0, upBound=1, cat='Binary')
    y_x2p_var = pulp.LpVariable("y_x2p", lowBound=0, upBound=1, cat='Binary')
    # Storage level
    storage_var = pulp.LpVariable.dicts("storage", timesteps, lowBound=0, upBound=params["storage_size"])
    # Freely adjustable spot bids
    spot_p2x_var = pulp.LpVariable.dicts("spot_p2x", timesteps, lowBound=0, upBound=params["power_p2x"])
    spot_x2p_var = pulp.LpVariable.dicts("spot_x2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    spot_y2p_var = pulp.LpVariable.dicts("spot_y2p", timesteps, lowBound=0, upBound=params["power_x2p"])
    # Fix the values of selected spot bids by setting the upper and lower bounds to be the wanted value
    if lock_n_spot_bids is not None:
        for i, t in enumerate(timesteps[:lock_n_spot_bids]):
            spot_p2x_var[t].lowBound = strategy['bid_spot_p2x'][i+start_idx]
            spot_p2x_var[t].upBound = strategy['bid_spot_p2x'][i+start_idx]
            spot_x2p_var[t].lowBound = strategy['bid_spot_x2p'][i+start_idx]
            spot_x2p_var[t].upBound = strategy['bid_spot_x2p'][i+start_idx]
            spot_y2p_var[t].lowBound = strategy['bid_spot_y2p'][i+start_idx]
            spot_y2p_var[t].upBound = strategy['bid_spot_y2p'][i+start_idx]
    # Adjust the bounds for the final storage level if there are more locked spot bids than hours to get bids for
    if (lock_n_spot_bids is not None) and (lock_n_spot_bids > horizon):
        space_up = params['storage_size'] - strategy['storage_x'][(end_idx-1):(start_idx+lock_n_spot_bids)].max()
        space_down = strategy['storage_x'][(end_idx-1):(start_idx+lock_n_spot_bids)].min()
        storage_var[timesteps[-1]].lowBound = storage_end - space_down
        storage_var[timesteps[-1]].upBound = storage_end + space_up

    # Linearization constant
    # We want the constant to be as small as possible to avoid numerical precision problems with the solver
    # We therefor take it to be the maximun of the smallest possible values that still allow the whole storage to be used during the horizon
    # If you still encounter numerical problems (unfeasible solution), try rescaling all prices downward so that 
    # x2p_lim and p2x_lim become smallar to thus also get a smaller m
    m = max([params['storage_size']*params['eff_x2p']*x2p_lim, 
             params['storage_size']/params['eff_p2x']*p2x_lim])
    # Problem
    prob = pulp.LpProblem("p2x2p", pulp.LpMaximize)

    # --- OBJECTIVE FUNCTION ---
    # Different objective functions depending on whether the TTF option is used or not
    if params['ttf']:
        # The objective function is the profit for the running strategy with TTF option added
        prob += (
            pulp.lpSum(
                [
                    spot_price[i] * spot_x2p_var[t] +  # Income from selling electricity produced from x on the spot market
                    spot_price[i] * spot_y2p_var[t] +  # Income from selling electricity produced from y on the spot market
                    mfrr_up_price[i] * mfrr_up_p2x_var[t] +  # Income from selling electricity by stopping x production
                    mfrr_up_price[i] * mfrr_up_x2p_var[t] +  # Income from selling electricity produced from x on the mFRR up market
                    mfrr_up_price[i] * mfrr_up_y2p_var[t] -  # Income from selling electricity produced from y on the mFRR up market
                    spot_price[i] * spot_p2x_var[t] -  # Cost of producing x by bying electricity from the spot market
                    ttf_price[i] * spot_y2p_var[t] / params["eff_x2p"] -  # Cost of used y used to produce electricity for the spot market
                    mfrr_down_price[i] * mfrr_down_p2x_var[t] -  # Cost of producing x by bying electricity from the mFRR market
                    mfrr_down_price[i] * mfrr_down_x2p_var[t] -  # Cost of stopping electricity production from x
                    mfrr_down_price[i] * mfrr_down_y2p_var[t] -  # Cost of stopping electricity production from y
                    ttf_price[i] * mfrr_up_y2p_var[t] / params["eff_x2p"]
                    # Cost of used y used to produce electricity for the mFRR market
                    for i, t in enumerate(timesteps)
                ]
            ) + delta_p2x_var - delta_x2p_var,
            "Profit"
        )
    else:
        # The objective function is the profit for the running strategy
        prob += (
            pulp.lpSum(
                [
                    spot_price[i] * spot_x2p_var[t] +  # Income from selling electricity produced from x on the spot market
                    mfrr_up_price[i] * mfrr_up_p2x_var[t] +  # Income from selling electricity by stopping x production
                    mfrr_up_price[i] * mfrr_up_x2p_var[t] -  # Income from selling electricity produced from x on the mFRR up market
                    spot_price[i] * spot_p2x_var[t] -  # Cost of producing x by bying electricity from the spot market
                    mfrr_down_price[i] * mfrr_down_p2x_var[t] -  # Cost of producing x by bying electricity from the mFRR market
                    mfrr_down_price[i] * mfrr_down_x2p_var[t]  # Cost of stopping electricity production from x
                    for i, t in enumerate(timesteps)
                ]
            ) + delta_p2x_var - delta_x2p_var,
            "Profit"
        )

    # --- CONSTRAINTS ---
    for i, t in enumerate(timesteps):

        # Constraints to define the actual power based on on spot and balancing market bids
        prob += spot_p2x_var[t] - mfrr_up_p2x_var[t] + mfrr_down_p2x_var[t] == running_p2x_var[t]
        prob += spot_x2p_var[t] + mfrr_up_x2p_var[t] - mfrr_down_x2p_var[t] == running_x2p_var[t]

        # Ensure that we can't consume more than the maximum power
        prob += spot_p2x_var[t] + mfrr_down_p2x_var[t] <= params["power_p2x"]
        # Ensure that we can only cut consumption if consuming
        prob += spot_p2x_var[t] - mfrr_up_p2x_var[t] >= 0
        # Ensure that we can't produce more than the maximum power
        prob += spot_x2p_var[t] + spot_y2p_var[t] + mfrr_up_x2p_var[t] + mfrr_up_y2p_var[t] <= params["power_x2p"]
        # Ensure that we can only cut production if producing
        prob += spot_x2p_var[t] - mfrr_down_x2p_var[t] >= 0

        # Ensure that we can't buy or sell more on the mFRR market than the traded volume
        prob += mfrr_down_p2x_var[t] + mfrr_down_x2p_var[t] <= -mfrr_down_volyme[i]
        prob += mfrr_up_p2x_var[t] + mfrr_up_x2p_var[t] <= mfrr_up_volyme[i]

        if params['ttf']:
            # Constraints to define the actual power based on on spot and balancing market bids
            prob += spot_y2p_var[t] + mfrr_up_y2p_var[t] - mfrr_down_y2p_var[t] == running_y2p_var[t]
            # Ensure that we can only cut production if producing
            prob += spot_y2p_var[t] - mfrr_down_y2p_var[t] >= 0

    # Ensure that the storage level change corresponds to the amount produced or used during every hour
    prob += storage_var["t_0"] - params['eff_p2x'] * running_p2x_var["t_0"] + 1 / params['eff_x2p'] * running_x2p_var["t_0"] == storage_init, "storage_t_0"
    for i in range(1, len(timesteps)):
        current = timesteps[i]
        previous = timesteps[i - 1]
        prob += storage_var[current] - params['eff_p2x'] * running_p2x_var[current] + 1 / params['eff_x2p'] * running_x2p_var[current] == storage_var[previous], "storage_" + current
    if balanced:
        # Ensure that the storage level is the same at the beginning and at the end of the horizon
        prob += storage_var[timesteps[-1]] == storage_end, "storage_t_end"

    # Estimating the cost of the storage level change
    prob += delta_p2x_var >= (storage_var[timesteps[-1]] - storage_end) / params['eff_p2x'] * p2x_lim
    prob += delta_p2x_var <= (storage_var[timesteps[-1]] - storage_end) / params['eff_p2x'] * p2x_lim + m * (1 - y_p2x_var), "delta_p2x_lower_lim"
    prob += delta_p2x_var <= m * y_p2x_var, "delta_p2x_upper_lim"
    prob += delta_x2p_var >= (storage_end - storage_var[timesteps[-1]]) * params['eff_x2p'] * x2p_lim
    prob += delta_x2p_var <= (storage_end - storage_var[timesteps[-1]]) * params['eff_x2p'] * x2p_lim + m * (1 - y_x2p_var), "delta_x2p_lower_lim"
    prob += delta_x2p_var <= m * y_x2p_var, "delta_x2p_upper_lim"

    # --- SOLVE AND EXTRACT VARIABLES VALUES ---
    # The HiGHS solver seems more robust so we use that one if available
    if 'HiGHS' in pulp.listSolvers(onlyAvailable=True):
        with suppress_output():
            prob.solve(pulp.apis.HiGHS())
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the optimization was successful
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"mFRR {inspect.currentframe().f_code.co_name}: Optimization status: {pulp.LpStatus[prob.status]}")
        print(f"Storage start: {storage_init}")
        print(f"Storage end: {storage_end}")

    # Extract the restults into a running strategy
    strategy['bid_spot_p2x'][start_idx:end_idx] = np.array([spot_p2x_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_spot_x2p'][start_idx:end_idx] = np.array([spot_x2p_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_mfrr_down_p2x'][start_idx:end_idx] = np.array([mfrr_down_p2x_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_mfrr_down_x2p'][start_idx:end_idx] = np.array([mfrr_down_x2p_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_mfrr_up_p2x'][start_idx:end_idx] = np.array([mfrr_up_p2x_var[t].value() for t in timesteps[:horizon]])
    strategy['bid_mfrr_up_x2p'][start_idx:end_idx] = np.array([mfrr_up_x2p_var[t].value() for t in timesteps[:horizon]])
    # Extract the y2p variables if the TTF option is used
    if params['ttf']:
        strategy['bid_spot_y2p'][start_idx:end_idx] = np.array([spot_y2p_var[t].value() for t in timesteps[:horizon]])
        strategy['bid_mfrr_down_y2p'][start_idx:end_idx] = np.array([mfrr_down_y2p_var[t].value() for t in timesteps[:horizon]])
        strategy['bid_mfrr_up_y2p'][start_idx:end_idx] = np.array([mfrr_up_y2p_var[t].value() for t in timesteps[:horizon]])

    return strategy
