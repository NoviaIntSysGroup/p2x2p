import numpy as np
import pandas as pd

from p2x2p.strategies.utils import split_data


def get_running_frac(running, max_power):
    """
    Compute the fraction of the time that the plant operation is running
    
    Parameters:
    running (np.array): The power time series
    
    Returns:
    running_frac (float): The fraction of the time that the plant is running
    """
    running_frac = np.sum(running) / running.size / max_power
    return running_frac


def get_avg_price(cash_flow, running):
    """
    Compute the average price for a given running profile
    
    Parameters:
    price (np.array): The price time series
    running (np.array): The power time series
    
    Returns:
    avg_price (float): The average price per unit of power
    """
    if np.sum(running) == 0:
        avg_price = 0.0
    else:
        avg_price = np.sum(cash_flow) / np.sum(running)
    return avg_price


def get_production(running, eff):
    """
    Compute the production for a given running profile
    
    Parameters:
    running (np.array): The power time series
    eff (float): The efficiency of the operation
    
    Returns:
    production (float): The production
    """
    production = np.sum(running) * eff
    return production


def get_usage(running, eff):
    """
    Compute the usage for a given running profile
    
    Parameters:
    running (np.array): The power time series
    eff (float): The efficiency of the operation
    
    Returns:
    usage (float): The usage
    """
    usage = np.sum(running) / eff
    return usage


def evaluate_strategy(data, strategy, params):

    spot_data, ttf_data, mfrr_data = split_data(data, params)

    # P2X2P costs and revenues from the spot market
    spot_price = spot_data['elspot-fi'].values
    strategy['spot_p2x_cost'] = np.sum(spot_price * strategy['bid_spot_p2x'])
    strategy['spot_x2p_revenue'] = np.sum(spot_price * strategy['bid_spot_x2p'])
    # P2X2P costs and revenues from the mFRR market
    if params['mfrr']:
        mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
        mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values
        # P2X costs and revenues
        strategy['mfrr_p2x_cost'] = np.sum(mfrr_down_price * strategy['bid_mfrr_down_p2x'])
        strategy['mfrr_p2x_revenue'] = np.sum(mfrr_up_price * strategy['bid_mfrr_up_p2x'])
        # X2P costs and revenues
        strategy['mfrr_x2p_cost'] = np.sum(mfrr_down_price * strategy['bid_mfrr_down_x2p'])
        strategy['mfrr_x2p_revenue'] = np.sum(mfrr_up_price * strategy['bid_mfrr_up_x2p'])
        # Profits from cancelled hours
        strategy['mfrr_p2x_profit'] = np.sum((mfrr_up_price-spot_price) * strategy['bid_mfrr_up_p2x'])
        strategy['mfrr_x2p_profit'] = np.sum((spot_price-mfrr_down_price) * strategy['bid_mfrr_down_x2p'])
    else:
        # All values are zero if not used
        strategy['mfrr_p2x_cost'] = 0
        strategy['mfrr_p2x_revenue'] = 0
        strategy['mfrr_x2p_cost'] = 0
        strategy['mfrr_x2p_revenue'] = 0
        strategy['mfrr_p2x_profit'] = 0
        strategy['mfrr_x2p_profit'] = 0

    # Y (TTF)
    if params['ttf']:
        ttf_price = ttf_data['TTF (Euro/MWh)'].values + params['ttf_premium']
        # Y2P costs and revenues from the spot market
        strategy['spot_y_cost'] = np.sum(ttf_price * strategy['bid_spot_y2p']) / params['eff_x2p']
        strategy['spot_y2p_revenue'] = np.sum(spot_price * strategy['bid_spot_y2p'])
        # Y2P costs and revenues from the mFRR market
        if params['mfrr']:
            mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
            mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values
            strategy['mfrr_y2p_cost'] = (
                    np.sum(mfrr_down_price * strategy['bid_mfrr_down_y2p']) +           # down regulation
                    np.sum(ttf_price * strategy['bid_mfrr_up_y2p']) / params['eff_x2p'] # Y cost for up regulation
            )
            strategy['mfrr_y2p_revenue'] = np.sum(mfrr_up_price * strategy['bid_mfrr_up_y2p'])
        else:
            # All values are zero if not used
            strategy['mfrr_y2p_cost'] = 0
            strategy['mfrr_y2p_revenue'] = 0
    else:
        # All values are zero if not used
        strategy['spot_y_cost'] = 0
        strategy['spot_y2p_revenue'] = 0
        strategy['mfrr_y2p_cost'] = 0
        strategy['mfrr_y2p_revenue'] = 0

    # Profits
    strategy['spot_x_profit'] = strategy['spot_x2p_revenue'] - strategy['spot_p2x_cost']
    strategy['spot_y_profit'] = strategy['spot_y2p_revenue'] - strategy['spot_y_cost']
    strategy['mfrr_x_profit'] = (
            strategy['mfrr_p2x_revenue'] + strategy['mfrr_x2p_revenue'] # Revenues
            - strategy['mfrr_p2x_cost'] - strategy['mfrr_x2p_cost']     # Costs
    )
    strategy['mfrr_y_profit'] = strategy['mfrr_y2p_revenue'] - strategy['mfrr_y2p_cost']

    # Running fractions
    strategy['spot_p2x_running_frac'] = get_running_frac(strategy['bid_spot_p2x'] - strategy['bid_mfrr_up_p2x'], params['power_p2x'])
    strategy['spot_x2p_running_frac'] = get_running_frac(strategy['bid_spot_x2p'] - strategy['bid_mfrr_down_x2p'], params['power_x2p'])
    strategy['spot_y2p_running_frac'] = get_running_frac(strategy['bid_spot_y2p'] - strategy['bid_mfrr_down_y2p'], params['power_x2p'])
    strategy['mfrr_p2x_running_frac'] = get_running_frac(strategy['bid_mfrr_down_p2x'], params['power_p2x'])
    strategy['mfrr_x2p_running_frac'] = get_running_frac(strategy['bid_mfrr_up_x2p'], params['power_x2p'])
    strategy['mfrr_y2p_running_frac'] = get_running_frac(strategy['bid_mfrr_up_y2p'], params['power_x2p'])
    strategy['p2x_running_frac'] = get_running_frac(strategy['running_p2x'], params['power_p2x'])
    strategy['x2p_running_frac'] = get_running_frac(strategy['running_x2p'], params['power_x2p'])
    strategy['y2p_running_frac'] = get_running_frac(strategy['running_y2p'], params['power_x2p'])

    # Production and usage
    strategy['spot_p2x_amount'] = get_production(strategy['bid_spot_p2x'] - strategy['bid_mfrr_up_p2x'], params['eff_p2x'])
    strategy['spot_x2p_amount'] = get_usage(strategy['bid_spot_x2p'] - strategy['bid_mfrr_down_x2p'], params['eff_x2p'])
    strategy['spot_y2p_amount'] = get_usage(strategy['bid_spot_y2p'] - strategy['bid_mfrr_down_y2p'], params['eff_x2p'])
    strategy['mfrr_p2x_amount'] = get_production(strategy['bid_mfrr_down_p2x'], params['eff_p2x'])
    strategy['mfrr_x2p_amount'] = get_usage(strategy['bid_mfrr_up_x2p'], params['eff_x2p'])
    strategy['mfrr_y2p_amount'] = get_usage(strategy['bid_mfrr_up_y2p'], params['eff_x2p'])
    strategy['p2x_amount'] = get_production(strategy['running_p2x'], params['eff_p2x'])
    strategy['x2p_amount'] = get_usage(strategy['running_x2p'], params['eff_x2p'])
    strategy['y2p_amount'] = get_usage(strategy['running_y2p'], params['eff_x2p'])

    # Average prices
    p2x_cash_flow = strategy['spot_p2x_cost'] + strategy['mfrr_p2x_cost'] - strategy['mfrr_p2x_revenue']
    strategy['p2x_avg_price'] = get_avg_price(p2x_cash_flow, strategy['running_p2x'])
    x2p_cash_flow = strategy['spot_x2p_revenue'] + strategy['mfrr_x2p_revenue'] - strategy['mfrr_x2p_cost']
    strategy['x2p_avg_price'] = get_avg_price(x2p_cash_flow, strategy['running_x2p'])
    y2p_cash_flow = strategy['spot_y2p_revenue'] + strategy['mfrr_y2p_revenue'] - strategy['mfrr_y2p_cost']
    strategy['y2p_avg_price'] = get_avg_price(y2p_cash_flow, strategy['running_y2p'])

    # Final profit
    strategy['x_value'] = (strategy['p2x_amount'] - strategy['x2p_amount']) * params['eff_x2p'] * strategy['x2p_avg_price']
    strategy['profit'] = (strategy['spot_x_profit'] + strategy['spot_y_profit'] +
                          strategy['mfrr_x_profit'] + strategy['mfrr_y_profit'] + strategy['x_value'])

    return strategy


def print_strategy_summary(strategy):

    print("Spot day-ahead market:")
    df = pd.DataFrame({
        "P2X": [f"{strategy['spot_p2x_running_frac'] * 1e2:.2f} %", f"{strategy['spot_p2x_amount']:.2f} MWh"],
        "X2P": [f"{strategy['spot_x2p_running_frac'] * 1e2:.2f} %", f"{strategy['spot_x2p_amount']:.2f} MWh"],
        "Y2P": [f"{strategy['spot_y2p_running_frac'] * 1e2:.2f} %", f"{strategy['spot_y2p_amount']:.2f} MWh"],
    }, index=["Usage", "Quantity  "])
    print(df)

    print("\nmFRR balancing market:")
    df = pd.DataFrame({
        "P2X": [f"{strategy['mfrr_p2x_running_frac'] * 1e2:.2f} %", f"{strategy['mfrr_p2x_amount']:.2f} MWh"],
        "X2P": [f"{strategy['mfrr_x2p_running_frac'] * 1e2:.2f} %", f"{strategy['mfrr_x2p_amount']:.2f} MWh"],
        "Y2P": [f"{strategy['mfrr_y2p_running_frac'] * 1e2:.2f} %", f"{strategy['mfrr_y2p_amount']:.2f} MWh"],
    }, index=["Usage", "Quantity  "])
    print(df)

    print("\nBoth markets:")
    df = pd.DataFrame({
        "P2X": [f"{strategy['p2x_running_frac'] * 1e2:.2f} %", f"{strategy['p2x_amount']:.2f} MWh",
                f"{strategy['p2x_avg_price']:.1f} €/MWh"],
        "X2P": [f"{strategy['x2p_running_frac'] * 1e2:.2f} %", f"{strategy['x2p_amount']:.2f} MWh",
                f"{strategy['x2p_avg_price']:.1f} €/MWh"],
        "Y2P": [f"{strategy['y2p_running_frac'] * 1e2:.2f} %", f"{strategy['y2p_amount']:.2f} MWh",
                f"{strategy['y2p_avg_price']:.1f} €/MWh"],
    }, index=["Usage", "Quantity", "Avg. price"])
    print(df)
    print("\n")

    print(f"Spot X profit:\t{strategy['spot_x_profit']:10.2f} €")
    print(f"Spot Y profit:\t{strategy['spot_y_profit']:10.2f} €")
    print(f"mFRR X profit:\t{strategy['mfrr_x_profit']:10.2f} €")
    print(f"mFRR Y profit:\t{strategy['mfrr_y_profit']:10.2f} €")
    print(f"Storage value:\t{strategy['x_value']:10.2f} €")
    print(f"Total profit:\t{strategy['profit']:10.2f} €\n")