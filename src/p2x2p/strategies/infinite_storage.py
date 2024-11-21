import numpy as np


def get_max_x2p_frac(params):
    """
    Calculate the maximum fraction of time that the plant can be in X2P mode.
    
    :param params: dictionary with the parameters
    
    :return: maximum fraction of time in X2P mode
    """
    p2x2p_efficiency = params['eff_p2x']*params['eff_x2p']
    conversion_ratio = p2x2p_efficiency * (params['power_p2x']/params['power_x2p'])
    max_x2p_frac = 1/(1/conversion_ratio+1)
    return max_x2p_frac


def get_revenues_and_costs(spot_data, params):
    """
    Calculate the revenues and costs for the infinite storage strategy

    :param spot_data: pandas dataframe with the spot data
    :param params: dictionary with the parameters

    :return: durations, revenues, costs
    """

    # Extract the spot price data
    price = spot_data['elspot-fi'].values

    # Define a normalized duration vector
    durations = np.arange(price.size) / (price.size-1)

    # Get the maximum fraction of the time that the plant can be in either mode
    max_x2p_frac = get_max_x2p_frac(params)
    max_p2x_frac = 1 - max_x2p_frac

    # Sort the price in descending order
    price_sorted = np.sort(price)[::-1]

    # P2X2P revenues
    revenues = price_sorted[durations<=max_x2p_frac]
    revenues = np.cumsum(revenues)*params['power_x2p']
    revenues = np.interp(durations, np.arange(revenues.size)/(revenues.size-1), revenues)

    # P2X2P costs
    costs = np.flip(price_sorted)[durations<=max_p2x_frac]
    costs = np.cumsum(costs)*params['power_p2x']
    costs = np.interp(durations, np.arange(costs.size)/(costs.size-1), costs)

    return durations, revenues, costs
