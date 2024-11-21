
import numpy as np

from p2x2p.strategies import best_params

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
P2X_VS_X2P_DELTA = 2.5
TTF_PREMIUMS = [0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
STORAGE_MULTIPLES = 2**np.arange(-2., 4)


def get_strategy_abbreviation(params):

    if params['name'] == 'infinite_horizon':
        return 'Inf. hor.'
    elif params['name'] == 'moving_horizon':
        return f"{params['horizon']/24:1.0f}d hor."
    elif params['name'] == 'no_horizon':
        return 'No hor.'
    elif params['name'] == 'infinite_storage':
        return 'Inf. stor. & inf. hor.'
    else:
        raise ValueError(f"Unknown strategy name: {params['name']}")


def set_price_limits_and_opp_costs(params):
    if params['name'] == 'no_horizon':
        params = best_params.set_lower_bound_price_limits(params)
        if params['mfrr']:
            params = best_params.set_lower_bound_opportunity_costs(params)
    elif params['name'] == 'moving_horizon':
        params = best_params.set_upper_bound_price_limits(params)
        if params['mfrr']:
            params = best_params.set_upper_bound_opportunity_costs(params)
    return params
    

def get_stratety_params(params, inf_hor=True):

    # Profits per strategy
    params_list = []
    names = []

    # Infinite horizon
    if inf_hor:
        params_infinite_horizon = params.copy()
        params_infinite_horizon['name'] = 'infinite_horizon'
        params_list.append(params_infinite_horizon)
        names.append(get_strategy_abbreviation(params_infinite_horizon))

    # Moving horizon
    horizons = [i*24 for i in range(7, 0, -1)]
    #horizons = [i * 24 for i in [7, 3, 2, 1]]
    for horizon in horizons:
        params_moving_horizon = params.copy()
        params_moving_horizon['name'] = 'moving_horizon'
        params_moving_horizon = set_price_limits_and_opp_costs(params_moving_horizon)
        params_moving_horizon['horizon'] = horizon
        params_list.append(params_moving_horizon)
        names.append(get_strategy_abbreviation(params_moving_horizon))

    # No horizon
    params_no_horizon = params.copy()
    params_no_horizon['name'] = 'no_horizon'
    params_no_horizon = set_price_limits_and_opp_costs(params_no_horizon)
    params_list.append(params_no_horizon)
    names.append(get_strategy_abbreviation(params_no_horizon))

    return params_list, names


def get_yearly_margin_params(params):

    years = range(2017, 2024)
    strategies = ['no_horizon', 'moving_horizon']

    params_list = []
    for year in years:
        for strategy in strategies:
            params_tmp = params.copy()
            params_tmp['year'] = year
            params_tmp['name'] = strategy
            params_tmp = set_price_limits_and_opp_costs(params_tmp)
            params_list.append(params_tmp)

    return params_list, years


def get_ttf_premium_params(params):

    params_list = []
    for premium in TTF_PREMIUMS:
        params_tmp = params.copy()
        params_tmp['ttf'] = True
        params_tmp['ttf_premium'] = premium
        params_tmp = set_price_limits_and_opp_costs(params_tmp)
        params_list.append(params_tmp)

    return params_list


def get_p2x_vs_x2p_params(params):

    # Define p2x and x2p values to evaluate
    power_tot = params['power_p2x'] + params['power_x2p']
    p2x = np.arange(P2X_VS_X2P_DELTA, power_tot, P2X_VS_X2P_DELTA)
    x2p = power_tot - p2x

    # Create the parameters list
    params_list = []
    for i in range(len(p2x)):
        params_i = params.copy()
        params_i['power_p2x'] = np.round(p2x[i], 1)
        params_i['power_x2p'] = np.round(x2p[i], 1)
        params_list.append(params_i)

    return params_list


def get_storage_size_params(params):

    # Create the parameters list
    params_list = []
    for size_multiplier in STORAGE_MULTIPLES:
        params_i = params.copy()
        params_i['storage_size'] = np.round(params['storage_size'] * size_multiplier)
        params_list.append(params_i)

    return params_list