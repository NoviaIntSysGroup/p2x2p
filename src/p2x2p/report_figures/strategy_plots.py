import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as patches

from p2x2p.utils.utils import get_config
from p2x2p.report_figures import utils, mpl_helper
from p2x2p.strategies.utils import filter_data, split_data


def plot_running_strategy(spot_data, strategy, params, ttf_data=None):

    # Extract the yearly spot data
    spot_data_year = spot_data[spot_data['date'].dt.year == params['year']]
    spot_price = spot_data_year['elspot-fi'].values
    spot_dates = spot_data_year['date']  # keep dates in pandas format to keep timezone info
    if ttf_data is not None:
        ttf_data_year = ttf_data[ttf_data['date'].dt.year == params['year']]
        ttf_price = ttf_data_year['TTF (Euro/MWh)'].values
        ttf_dates = ttf_data_year['date']

    config = get_config()
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    y2p_color = config['figures']['colors']['y2p']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config)

    # Set limits and ticks
    delta_x = pd.Timedelta(days=5)
    delta_y = (spot_price.max() - spot_price.min())*0.05
    x_lim = [spot_dates.iloc[0]-delta_x, spot_dates.iloc[-1]+delta_x]
    y_lim = [spot_price.min()-delta_y, spot_price.max()+delta_y]
    x_ticks = [spot_dates.iloc[0]+pd.DateOffset(months=2*i) for i in range(10)]
    x_tick_labels = [date.strftime('%Y-%m') for date in x_ticks]  # Matplotlib stuggles to format the dates correctly when a time zone is included
    y_range = y_lim[1] - y_lim[0]

    # Plot the strategy
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].fill_between(spot_dates, y_lim[0], (strategy['running_x2p']>0)*y_range+y_lim[0], fc=x2p_color, alpha=0.5, label=f"X2P ({strategy['x2p_running_frac']*100:2.1f} %)")
    axs[0].fill_between(spot_dates, y_lim[0], (strategy['running_p2x']>0)*y_range+y_lim[0], fc=p2x_color, alpha=0.5, label=f"P2X ({strategy['p2x_running_frac']*100:2.1f} %)")
    if params['ttf']:
        axs[0].fill_between(spot_dates, y_lim[0], (strategy['running_y2p']>0)*y_range+y_lim[0], fc=y2p_color, alpha=0.5, label=f"Y2P ({strategy['y2p_running_frac']*100:2.1f} %)")
    axs[0].plot(spot_dates, spot_price, color='black', label='Spot price')
    if params['ttf']:
        axs[0].plot(ttf_dates, ttf_price+params['ttf_premium'], color='gray', label=f"TTF+{params['ttf_premium']:1.0f} €/MWh")

    axs[0].legend(loc='upper left', title=utils.get_strategy_abbreviation(params))
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim, ylim=y_lim, ylabel='Price (€/MWh)')

    return fig, axs


def plot_running_strategy_zoom_horizon(spot_data, strategy, params):

    # Extract the yearly spot data
    spot_data_year = spot_data[spot_data['date'].dt.year == params['year']]
    dates = spot_data_year['date']  # keep dates in pandas format to keep timezone info
    spot_price = spot_data_year['elspot-fi'].values
    spot_price_mean = spot_data_year['mean_price'].values
    spot_price_mean_daily = spot_price_mean[::24]
    spot_price_mean_daily = np.repeat(spot_price_mean_daily, 24)

    # Set the horizon
    if params['name'] == 'no_horizon':
        horizon = 0
        extend_limit = 24
    else:
        horizon = params['horizon']
        extend_limit = 0

    # Define the figure and plotting regions
    config = get_config()
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config)
    hor_sep_cm = 1.2
    zoom_color = config['figures']['colors']['zoom']
    
    # Set the style and get the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, grid_size=[1, 2], hor_ver_sep_cm=[hor_sep_cm, 0], rel_col_widths=[0.645, 0.355])

    # LEFT PLOT FIRST
    # Determine how many hours to include to get mathing ticks with a full page wide figure
    ax_width_cm = axs[0].get_position().width*fig_size_cm[0]
    frac_included = ax_width_cm/plot_rect_cm[2]
    n_included = int(frac_included*dates.size)
    n_included = n_included - n_included%24

    # Extend spot_price_mean_daily to match the length of the horizon
    spot_price_mean_daily = spot_price_mean_daily[:(n_included+horizon+extend_limit)]
    if horizon > 0:
        spot_price_mean_daily[n_included:(n_included+horizon+extend_limit)] = spot_price_mean_daily[n_included]

    # Set the limits and ticks
    delta_x = pd.Timedelta(days=5)
    delta_y = (spot_price.max() - spot_price.min())*0.05
    x_lim = [dates.iloc[0]-delta_x, dates.iloc[n_included-1]+delta_x]
    y_lim = [spot_price.min()-delta_y, spot_price.max()+delta_y]
    x_ticks = [dates.iloc[0]+pd.DateOffset(months=2*i) for i in range(10)]
    x_tick_labels = [date.strftime('%Y-%m') for date in x_ticks]  # Matplotlib stuggles to format the dates correctly when a time zone is included
    y_range = y_lim[1] - y_lim[0]

    # Define the zoomed in region
    zoomed_xy = (dates.iloc[n_included-1-4*24], -25)
    zoomed_width = pd.Timedelta(days=8)
    zoomed_height = 200

    # Left plot showcasing the strategy and the spot price
    axs[0].fill_between(dates[:n_included], y_lim[0], (strategy['running_x2p'][:n_included]>0)*y_range+y_lim[0], fc=x2p_color, alpha=0.5, label=f"X2P ({strategy['x2p_running_frac']*100:2.1f} %)")
    axs[0].fill_between(dates[:n_included], y_lim[0], (strategy['running_p2x'][:n_included]>0)*y_range+y_lim[0], fc=p2x_color, alpha=0.5, label=f"P2X ({strategy['p2x_running_frac']*100:2.1f} %)")
    axs[0].plot(dates[:n_included], spot_price[:n_included], 'k-', label='Spot price')
    axs[0].legend(loc='upper left', title=utils.get_strategy_abbreviation(params))
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim, ylim=y_lim, ylabel='Price (€/MWh)')

    # Plot the expected mean and the limits on a separate axis to get a second legend
    ax_ontop = mpl_helper.get_axes_ontop(fig, axs[0])
    ax_ontop.plot(dates[:n_included], spot_price_mean[:n_included], '-', c='gray', label='Exp. mean')
    ax_ontop.plot(dates[:n_included], spot_price_mean_daily[:n_included]*params['k_p2x'], ':', c='gray', label=r'${\pi}^\mathrm{P2X}$ & ${\pi}^\mathrm{X2P}$')
    ax_ontop.plot(dates[:n_included], spot_price_mean_daily[:n_included]*params['k_x2p'], ':', c='gray')
    ax_ontop.legend(loc='upper right')
    ax_ontop.set(xlim=x_lim, ylim=y_lim)
    rect = patches.Rectangle(zoomed_xy, zoomed_width, zoomed_height, linewidth=1, edgecolor=zoom_color, facecolor='none')
    ax_ontop.add_patch(rect)

    # RIGHT PLOT
    # Set the limits and ticks
    x_lim_zoom = [zoomed_xy[0], zoomed_xy[0]+zoomed_width]
    y_lim_zoom = [zoomed_xy[1], zoomed_xy[1]+zoomed_height]
    x_ticks = [zoomed_xy[0]+pd.DateOffset(days=2*i) for i in range(10)]
    x_tick_labels = [date.strftime('%d-%m') for date in x_ticks]
    x_tick_labels[2] = 'Now'

    # Zoomed in region to highlight the strategy and the known future prices
    n_included_horizon = n_included+horizon
    n_included_horizon_lim = n_included_horizon + extend_limit
    axs[1].fill_between(dates[:n_included], y_lim[0], (strategy['running_x2p'][:n_included]>0)*y_range+y_lim[0], fc=x2p_color, alpha=0.5)
    axs[1].fill_between(dates[:n_included], y_lim[0], (strategy['running_p2x'][:n_included]>0)*y_range+y_lim[0], fc=p2x_color, alpha=0.5)
    axs[1].plot(dates[:n_included_horizon], spot_price[:n_included_horizon], 'k-')
    axs[1].plot(dates[:n_included], spot_price_mean[:n_included], '-', c='gray', label='Exp. mean')
    axs[1].plot(dates[:n_included_horizon_lim], spot_price_mean_daily[:n_included_horizon_lim]*params['k_p2x'], ':', c='gray', label=r'${\pi}^\mathrm{P2X}$ & ${\pi}^\mathrm{X2P}$')
    axs[1].plot(dates[:n_included_horizon_lim], spot_price_mean_daily[:n_included_horizon_lim]*params['k_x2p'], ':', c='gray')
    # Day ahead and horizon
    day_ahead_x = [zoomed_xy[0]+pd.DateOffset(days=i) for i in [4, 5]]
    horizon_x = [zoomed_xy[0]+pd.DateOffset(days=i) for i in [4, 4+horizon/24]]
    axs[1].fill_between(day_ahead_x, y_lim[0], y_range, fc=0.9*np.ones(3))
    axs[1].text(zoomed_xy[0]+pd.DateOffset(hours=4*24+12), y_lim_zoom[1]*0.8, 'Day\nahead', ha='center', va='top', ma='right', rotation=90, fontsize=6)
    if horizon > 0:
        axs[1].annotate('', xy=(horizon_x[1], y_lim_zoom[1]*0.85), xytext=(horizon_x[0], y_lim_zoom[1]*0.85), arrowprops=dict(arrowstyle="<->", lw=1))
        axs[1].text(zoomed_xy[0]+pd.DateOffset(days=4+horizon/48), y_lim_zoom[1]*0.87, 'Horizon', ha='center', va='bottom', fontsize=6)

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    axs[1].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim_zoom, ylim=y_lim_zoom, yticks=[])
    for spine in axs[1].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(zoom_color)

    # CONNECTING LINES BETWEEN THE ZOOM REGIONS AND THE LEFT AXES
    fig.canvas.draw()   # Render iamge first to avoid transformation errors in the next step
    rect_bbox_figure = rect.get_window_extent().transformed(fig.transFigure.inverted())
    line_top = mlines.Line2D([rect_bbox_figure.x1, axs[1].get_position().x0], [rect_bbox_figure.y1, axs[1].get_position().y1], color=zoom_color, transform=fig.transFigure, figure=fig)
    line_bottom = mlines.Line2D([rect_bbox_figure.x1, axs[1].get_position().x0], [rect_bbox_figure.y0, axs[1].get_position().y0], color=zoom_color, transform=fig.transFigure, figure=fig)
    # Add the line to the figure, not the axis
    fig.lines.append(line_top)
    fig.lines.append(line_bottom)

    return fig, axs


def plot_running_strategy_zoom(data, strategy, params):

        # Filter the data
    data_year = filter_data(data, params)
    spot_data, _, mfrr_data = split_data(data_year, params)

    # Extract relevant data columns
    dates = spot_data['date']  # keep dates in pandas format to keep timezone info
    spot_price = spot_data['elspot-fi'].values
    spot_price_mean = spot_data['mean_price'].values
    spot_price_mean_daily = spot_price_mean[::24]
    spot_price_mean_daily = np.repeat(spot_price_mean_daily, 24)
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values

    # Define the figure and plotting regions
    config = get_config()
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config)
    hor_sep_cm = 1.2
    zoom_color = config['figures']['colors']['zoom']
    
    # Set the style and get the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, grid_size=[1, 2], hor_ver_sep_cm=[hor_sep_cm, 0], rel_col_widths=[0.645, 0.355])

    # LEFT PLOT FIRST
    # Determine how many hours to include to get mathing ticks with a full page wide figure
    ax_width_cm = axs[0].get_position().width*fig_size_cm[0]
    frac_included = ax_width_cm/plot_rect_cm[2]
    n_included = int(frac_included*dates.size)
    n_included = n_included - n_included%24

    # Set the limits and ticks
    delta_x = pd.Timedelta(days=5)
    delta_y = (spot_price.max() - spot_price.min())*0.05
    x_lim = [dates.iloc[0]-delta_x, dates.iloc[n_included-1]+delta_x]
    y_lim = [spot_price.min()-delta_y, spot_price.max()+delta_y]
    x_ticks = [dates.iloc[0]+pd.DateOffset(months=2*i) for i in range(10)]
    x_tick_labels = [date.strftime('%Y-%m') for date in x_ticks]  # Matplotlib stuggles to format the dates correctly when a time zone is included
    y_range = y_lim[1] - y_lim[0]

    # Define the zoomed in region
    zoomed_xy = (dates.iloc[n_included-1-40*24], -30)
    zoomed_width = pd.Timedelta(days=8)
    zoomed_height = 250

    # Left plot showcasing the strategy and the spot price
    axs[0].fill_between(dates[:n_included], y_lim[0], (strategy['running_x2p'][:n_included]>0)*y_range+y_lim[0], fc=x2p_color, alpha=0.5, label=f"X2P ({strategy['x2p_running_frac']*100:2.1f} %)")
    axs[0].fill_between(dates[:n_included], y_lim[0], (strategy['running_p2x'][:n_included]>0)*y_range+y_lim[0], fc=p2x_color, alpha=0.5, label=f"P2X ({strategy['p2x_running_frac']*100:2.1f} %)")
    axs[0].plot(dates[:n_included], spot_price[:n_included], 'k-', label='Spot price')
    axs[0].legend(loc='upper left', title=utils.get_strategy_abbreviation(params))
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim, ylim=y_lim, ylabel='Price (€/MWh)')

    # Plot the expected mean and the limits on a separate axis to get a second legend
    ax_ontop = mpl_helper.get_axes_ontop(fig, axs[0])
    ax_ontop.plot(dates[:n_included], spot_price_mean[:n_included], '-', c='gray', label='Exp. mean')
    ax_ontop.plot(dates[:n_included], spot_price_mean_daily[:n_included]*params['k_p2x'], ':', c='gray', label=r'${\pi}^\mathrm{P2X}$ & ${\pi}^\mathrm{X2P}$')
    ax_ontop.plot(dates[:n_included], spot_price_mean_daily[:n_included]*params['k_x2p'], ':', c='gray')
    ax_ontop.legend(loc='upper right')
    ax_ontop.set(xlim=x_lim, ylim=y_lim)
    rect = patches.Rectangle(zoomed_xy, zoomed_width, zoomed_height, linewidth=1, edgecolor=zoom_color, facecolor='none')
    ax_ontop.add_patch(rect)

    # RIGHT PLOT
    # Set the limits and ticks
    x_lim_zoom = [zoomed_xy[0], zoomed_xy[0]+zoomed_width]
    y_lim_zoom = [zoomed_xy[1], zoomed_xy[1]+zoomed_height]
    x_ticks = [zoomed_xy[0]+pd.DateOffset(days=2*i) for i in range(10)]
    x_tick_labels = [date.strftime('%d-%m') for date in x_ticks]

    # Zoomed in region to highlight the strategy and the known future prices
    axs[1].fill_between(dates, y_lim[0], (strategy['running_x2p']>0)*y_range+y_lim[0], fc=x2p_color, alpha=0.5)
    axs[1].fill_between(dates, y_lim[0], (strategy['running_p2x']>0)*y_range+y_lim[0], fc=p2x_color, alpha=0.5)
    if params['mfrr']:
        axs[1].plot(dates, mfrr_up_price, '-', c=up_color, label=mfrr_up_label)
        axs[1].plot(dates, mfrr_down_price, '-', c=down_color, label=mfrr_down_label)
    axs[1].plot(dates, spot_price, 'k-')
    axs[1].plot(dates, spot_price_mean, '-', c='gray')
    axs[1].plot(dates, spot_price_mean_daily*params['k_p2x'], ':', c='gray')
    axs[1].plot(dates, spot_price_mean_daily*params['k_x2p'], ':', c='gray')

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    axs[1].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim_zoom, ylim=y_lim_zoom, yticks=[])
    for spine in axs[1].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(zoom_color)
    if params['mfrr']:
        axs[1].legend(loc='upper right')

    # CONNECTING LINES BETWEEN THE ZOOM REGIONS AND THE LEFT AXES
    fig.canvas.draw()   # Render iamge first to avoid transformation errors in the next step
    rect_bbox_figure = rect.get_window_extent().transformed(fig.transFigure.inverted())
    line_top = mlines.Line2D([rect_bbox_figure.x1, axs[1].get_position().x0], [rect_bbox_figure.y1, axs[1].get_position().y1], color=zoom_color, transform=fig.transFigure, figure=fig)
    line_bottom = mlines.Line2D([rect_bbox_figure.x1, axs[1].get_position().x0], [rect_bbox_figure.y0, axs[1].get_position().y0], color=zoom_color, transform=fig.transFigure, figure=fig)
    # Add the line to the figure, not the axis
    fig.lines.append(line_top)
    fig.lines.append(line_bottom)

    return fig, axs



def plot_strategy_breakdown_euro(strategy, params):

    config = get_config()

    y_lim = [-5.5, 0.5]
    bar_height = 0.5
    price_scaling = 1e-6
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    up_color = config['figures']['colors']['mfrr_up']
    down_color = config['figures']['colors']['mfrr_down']
    new_color = config['figures']['colors']['mfrr_new']
    p2x_color_alpha = mpl_helper.hex_to_rgba(p2x_color, 0.5)
    x2p_color_alpha = mpl_helper.hex_to_rgba(x2p_color, 0.5)
    up_color_alpha = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(up_color, alpha=0.5))
    down_color_alpha= mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(down_color, alpha=0.5))
    new_color_alpha= mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(new_color, alpha=0.5))
    net_color = 0.75*np.ones(3)
    net_color_dark = 0.25*np.ones(3)
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    spot_p2x_euro = -strategy['spot_p2x_cost']*price_scaling
    spot_x2p_euro = strategy['spot_x2p_revenue']*price_scaling
    axs[0].barh(0, spot_p2x_euro, height=bar_height, fc=p2x_color_alpha, ec='none', label='P2X')
    axs[0].barh(0, spot_x2p_euro, height=bar_height, fc=x2p_color_alpha, ec='none', label='X2P')

    mfrr_down_p2x_euro = -strategy['mfrr_p2x_cost']*price_scaling
    mfrr_down_x2p_euro = -strategy['mfrr_x2p_cost']*price_scaling
    axs[0].barh(-1, mfrr_down_p2x_euro, height=bar_height, fc=p2x_color_alpha, ec='none')
    axs[0].barh(-1, mfrr_down_x2p_euro, left=mfrr_down_p2x_euro, height=bar_height, fc=x2p_color_alpha, ec='none')

    mfrr_up_p2x_euro = strategy['mfrr_p2x_revenue']*price_scaling
    mfrr_up_x2p_euro = strategy['mfrr_x2p_revenue']*price_scaling
    axs[0].barh(-2, mfrr_up_p2x_euro, height=bar_height, fc=p2x_color_alpha, ec='none')
    axs[0].barh(-2, mfrr_up_x2p_euro, left=mfrr_up_p2x_euro, height=bar_height, fc=x2p_color_alpha, ec='none')

    # All costs
    axs[0].barh(-3, spot_p2x_euro, height=bar_height, fc=p2x_color_alpha, ec='none')
    axs[0].barh(-3, mfrr_down_p2x_euro, left=spot_p2x_euro, height=bar_height, fc=p2x_color_alpha, ec='none')
    axs[0].barh(-3, mfrr_down_x2p_euro, left=spot_p2x_euro+mfrr_down_p2x_euro, height=bar_height, fc=x2p_color_alpha, ec='none')
    # All revenues
    axs[0].barh(-3, spot_x2p_euro, height=bar_height, fc=x2p_color_alpha, ec='none')
    axs[0].barh(-3, mfrr_up_p2x_euro, left=spot_x2p_euro, height=bar_height, fc=p2x_color_alpha, ec='none')
    axs[0].barh(-3, mfrr_up_x2p_euro, left=spot_x2p_euro+mfrr_up_p2x_euro, height=bar_height, fc=x2p_color_alpha, ec='none')

    axs[0].barh(-4, strategy['profit']*price_scaling, height=bar_height, fc=net_color_dark, ec='none', label='mFRR')
    axs[0].barh(-4, strategy['spot_x_profit']*price_scaling, height=bar_height, fc=net_color, ec='none', label='Spot')

    # cancel_p2x_euro = strategy['mfrr_p2x_profit']*price_scaling
    # cansel_x2p_euro = strategy['mfrr_x2p_profit']*price_scaling
    # axs[0].barh(-5, strategy['mfrr_x_profit']*price_scaling, left=strategy['spot_x_profit']*price_scaling, height=bar_height, fc=new_color_alpha, ec='none')
    # axs[0].barh(-5, cancel_p2x_euro, left=strategy['spot_x_profit']*price_scaling, height=bar_height, fc=up_color_alpha, ec='none')
    # axs[0].barh(-5, cansel_x2p_euro, left=strategy['spot_x_profit']*price_scaling+cancel_p2x_euro, height=bar_height, fc=down_color_alpha, ec='none')

    axs[0].legend(loc='upper right', ncol=2, title=utils.get_strategy_abbreviation(params))
    axs[0].axvline(0, color='black', lw=1)
    # axs[0].axhline(-4.5, color='black', linestyle=':', lw=1)
    # axs[0].set(xlabel='Cost|Revenue (M€)', ylim=[-5.5, 0.5], yticks=np.arange(-5, 1), yticklabels=['Source', 'Net', 'Sum', mfrr_up_label, mfrr_down_label, 'Spot'])
    axs[0].set(xlabel='Cost|Revenue (M€)', ylim=[-4.5, 0.5], yticks=np.arange(-4, 1), yticklabels=['Net', 'Sum', mfrr_up_label, mfrr_down_label, 'Spot'])

    return fig, axs


def plot_strategy_breakdown_h2(strategy):

    config = get_config()
    params = config['default_params']

    y_lim = [-4.5, 0.5]
    bar_height = 0.5
    h2_sclaing = 1e-3
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    p2x_color_alpha = mpl_helper.hex_to_rgba(p2x_color, 0.5)
    x2p_color_alpha = mpl_helper.hex_to_rgba(x2p_color, 0.5)
    net_color = 0.5*np.ones(3)
    net_color_dark = 0.25*np.ones(3)
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    spot_p2x_h2 = strategy['bid_spot_p2x'].sum()*params['eff_p2x']*h2_sclaing
    spot_x2p_h2 = -strategy['bid_spot_x2p'].sum()/params['eff_x2p']*h2_sclaing
    axs[0].barh(0, spot_p2x_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color, label='P2X')
    axs[0].barh(0, spot_x2p_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color, label='X2P')

    mfrr_down_p2x_h2 = strategy['bid_mfrr_down_p2x'].sum()*params['eff_p2x']*h2_sclaing
    mfrr_down_x2p_h2 = strategy['bid_mfrr_down_x2p'].sum()/params['eff_x2p']*h2_sclaing
    axs[0].barh(-1, mfrr_down_p2x_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color)
    axs[0].barh(-1, mfrr_down_x2p_h2, left=mfrr_down_p2x_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color)

    mfrr_up_p2x_h2 = -strategy['bid_mfrr_up_p2x'].sum()*params['eff_p2x']*h2_sclaing
    mfrr_up_x2p_h2 = -strategy['bid_mfrr_up_x2p'].sum()/params['eff_x2p']*h2_sclaing
    axs[0].barh(-2, mfrr_up_p2x_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color)
    axs[0].barh(-2, mfrr_up_x2p_h2, left=mfrr_up_p2x_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color)

    # All costs
    axs[0].barh(-3, spot_p2x_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color)
    axs[0].barh(-3, mfrr_down_p2x_h2, left=spot_p2x_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color)
    axs[0].barh(-3, mfrr_down_x2p_h2, left=spot_p2x_h2+mfrr_down_p2x_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color)
    # All revenues
    axs[0].barh(-3, spot_x2p_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color)
    axs[0].barh(-3, mfrr_up_p2x_h2, left=spot_x2p_h2, height=bar_height, fc=p2x_color_alpha, ec=p2x_color)
    axs[0].barh(-3, mfrr_up_x2p_h2, left=mfrr_up_p2x_h2+spot_x2p_h2, height=bar_height, fc=x2p_color_alpha, ec=x2p_color)

    axs[0].barh(-4, 0, height=bar_height, fc=net_color_dark, ec=net_color_dark)
    axs[0].barh(-4, 0, height=bar_height, fc=net_color, ec=net_color_dark)

    axs[0].legend(loc='upper right')
    axs[0].axvline(0, color='black', lw=1)
    axs[0].set(xlabel='Used|Produced ($\mathrm{H}_2$ GWh)', ylim=y_lim, yticks=np.arange(-4, 1), yticklabels=['Net', 'Sum', mfrr_up_label, mfrr_down_label, 'Spot'])

    return fig, axs
