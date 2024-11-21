import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
colors = px.colors.qualitative.Plotly

from p2x2p.utils.utils import get_config
from p2x2p.strategies.utils import split_data


def get_dates_for_fill(dates):
    delta = dates[:2].diff().iloc[1]
    dates_repeated = dates.repeat(2).reset_index(drop=True)
    dates_last = pd.Series([dates.iloc[-1]] * 2) + delta
    dates_fill = pd.concat([dates_repeated, dates_last], ignore_index=True)
    return dates_fill


def get_running_times_for_fill(running, y_range):
    running_fill = np.zeros(2 * running.size + 1)
    running_fill[1::2] = (running > 0) * [y_range[1] - y_range[0]] + y_range[0]
    running_fill[2::2] = (running > 0) * [y_range[1] - y_range[0]] + y_range[0]
    running_fill[0] = y_range[0]
    running_fill[-1] = y_range[0]
    return running_fill


def get_fill_trace(dates_fill, running_fill, color, label):
    trace = go.Scatter(
        x=dates_fill,
        y=running_fill,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='toself',
        fillcolor=color,
        name=label
    )
    return trace


def convert_hex_to_rgba(hex_val, alpha=0.):
    """
    It takes a hexadecimal color value and returns the corresponding RGB value

    :param hex_val: The hexadecimal value of the color you want to convert
    :return: A numpy array of the RGB values of the hex color code.
    """
    r, g, b = [int(hex_val.strip('#')[i:i + 2], 16) for i in (0, 2, 4)]
    return f'rgba({r}, {g}, {b}, {alpha})'


def plot_running_strategy(data, strategy, params):

    spot_data, ttf_data, mfrr_data = split_data(data, params)

    # Extract the spot data
    price = spot_data['elspot-fi'].values
    dates = spot_data['date']

    # Figure specific variables
    y_pad = 0.05 * (price.max() - price.min())
    y_range = [min([0, price.min()]) - y_pad, price.max() + y_pad]

    config = get_config()
    p2x_color = convert_hex_to_rgba(config['figures']['colors']['p2x'], 0.5)
    x2p_color = convert_hex_to_rgba(config['figures']['colors']['x2p'], 0.5)
    y2p_color = convert_hex_to_rgba(config['figures']['colors']['y2p'], 0.5)

    dates_fill = get_dates_for_fill(dates)
    p2x_fill = get_running_times_for_fill(strategy['running_p2x'], y_range)
    x2p_fill = get_running_times_for_fill(strategy['running_x2p'], y_range)
    y2p_fill = get_running_times_for_fill(strategy['running_y2p'], y_range)

    # Create a figure
    fig = go.Figure()
    # Add data traces
    fig.add_trace(get_fill_trace(dates_fill, p2x_fill, p2x_color, 'P2X'))
    fig.add_trace(get_fill_trace(dates_fill, x2p_fill, x2p_color, 'X2P'))
    fig.add_trace(get_fill_trace(dates_fill, y2p_fill, y2p_color, 'Y2P'))
    fig.add_trace(go.Scatter(x=dates, y=price, mode='lines', line=dict(color='black'), name='elspot-fi'))
    # Show the plot
    fig.update_xaxes(title="Date"),
    fig.update_yaxes(title="Spot price (€/MWh)", range=y_range),
    fig.show()

    return fig


def plot_storage_level(data, strategy, params):

    spot_data, ttf_data, mfrr_data = split_data(data, params)

    # Extract the spot data
    dates = spot_data['date']

    delta = dates[:2].diff().iloc[1]
    dates_last = pd.Series([dates.iloc[-1]]) + delta
    dates_storage = pd.concat([dates, dates_last], ignore_index=True)

    flux = strategy['running_p2x'] * params['eff_p2x'] - strategy['running_x2p'] / params['eff_x2p']
    storage_level = np.ones(strategy['running_p2x'].size + 1) * params['storage_size'] / 2
    storage_level[1:] += np.cumsum(flux)

    # Figure specific variables
    y_pad = 0.05 * params['storage_size']
    y_range = [0 - y_pad, params['storage_size'] + y_pad]

    config = get_config()
    p2x_color = convert_hex_to_rgba(config['figures']['colors']['p2x'], 0.5)
    x2p_color = convert_hex_to_rgba(config['figures']['colors']['x2p'], 0.5)

    dates_fill = get_dates_for_fill(dates)
    p2x_fill = get_running_times_for_fill(strategy['running_p2x'], y_range)
    x2p_fill = get_running_times_for_fill(strategy['running_x2p'], y_range)

    # Create a figure
    fig = go.Figure()
    # Add data traces
    fig.add_trace(get_fill_trace(dates_fill, p2x_fill, p2x_color, 'P2X'))
    fig.add_trace(get_fill_trace(dates_fill, x2p_fill, x2p_color, 'X2P'))
    fig.add_trace(go.Scatter(x=dates_storage, y=storage_level, mode='lines', line=dict(color='black'), name='Level'))
    # Show the plot
    fig.update_xaxes(title="Date"),
    fig.update_yaxes(title="Storage X (MWh)", range=y_range),
    fig.show()

    return fig