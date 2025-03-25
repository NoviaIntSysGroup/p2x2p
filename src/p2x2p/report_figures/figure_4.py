import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from p2x2p.utils.utils import get_config
from p2x2p.data import spot, ttf, fingrid
from p2x2p.report_figures import strategy_plots, utils, mpl_helper
from p2x2p.strategies import cache
from p2x2p.strategies.utils import select_data_range, split_data, filter_data


def plot_all_panels(overwrite=False):

    # Get the configuration
    config = get_config()
    # Get the default parameters
    params = config['default_params'].copy()

    # Get the price data
    spot_data = spot.get_data()
    spot_data['mean_price'] = spot_data['elspot-fi'].ewm(int(24*params['n_days'])).mean().values
    ttf_data = ttf.get_data()
    mfrr_data = fingrid.get_mfrr_data()
    data = {'spot_data': spot_data, 'ttf_data': ttf_data, 'mfrr_data': mfrr_data}

    plot_mfrr_data(data)
    plot_spot_mfrr_violin(data)
    plot_opportunity_costs_grid_search(params, overwrite)
    plot_opportunity_costs_comparison(params, overwrite)
    # plot_possible_profits(data, overwrite)

    plot_profit_per_strategy(overwrite)
    plot_profit_per_year(overwrite)

    plot_mfrr_strategy_breakdown_euro(params, overwrite)
    plot_mfrr_x2p_revenue_per_hour(data, params, overwrite)

    plot_strategy(data, params, mfrr=False, overwrite=overwrite)
    plot_strategy(data, params, mfrr=True, overwrite=overwrite)


def plot_mfrr_data(data):

    config = get_config()
    params = config['default_params']
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    day = 53
    n_days = 1
    shift = 0   # Summer time shift
    data_year = filter_data(data, params)
    data_selection = select_data_range(data_year, 24*day+shift, 24*(day+n_days)+shift)
    spot_data, _, mfrr_data = split_data(data_selection, params)

    # Prepare data for plotting
    dates = spot_data['date']  # keep dates in pandas format to keep timezone info
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values
    # Repeat the data to get the step-like plot
    delta = dates[:2].diff().iloc[1]
    dates_first = pd.Series([dates.iloc[0]])
    dates_repeated = dates.iloc[1:].repeat(2).reset_index(drop=True)
    dates_last = pd.Series([dates.iloc[-1]]) + delta
    dates_rep = pd.concat([dates_first, dates_repeated, dates_last], ignore_index=True)
    spot_price_rep = np.repeat(spot_price, 2)
    mfrr_down_price_rep = np.repeat(mfrr_down_price, 2)
    mfrr_up_price_rep = np.repeat(mfrr_up_price, 2)

    x_ticks = [dates.iloc[0]+pd.DateOffset(hours=6*i) for i in range(5)]
    x_tick_labels = [date.strftime('%H:%M') for date in x_ticks] 
    x_label = x_ticks[0].strftime('%d.%m.%Y')

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Plot data
    axs[0].fill_between(dates_rep, spot_price_rep, mfrr_up_price_rep, color=up_color, alpha=0.5, edgecolor='none')
    axs[0].fill_between(dates_rep, spot_price_rep, mfrr_down_price_rep, color=down_color, alpha=0.5, edgecolor='none')
    axs[0].plot(dates_rep, mfrr_up_price_rep, color=up_color, label=mfrr_up_label)
    axs[0].plot(dates_rep, mfrr_down_price_rep, color=down_color, label=mfrr_down_label)
    axs[0].plot(dates_rep, spot_price_rep, color='black', label='Spot price')
    # Configure axes and legend
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel('Price (€/MWh)')
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_daily_price", 'pdf')


def plot_spot_mfrr_violin(data):

    # Get the figure configuration
    config = get_config()
    params = config['default_params']

    data_year = filter_data(data, params)
    spot_data, _, mfrr_data = split_data(data_year, params)
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values
    mfrr_down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values
    mfrr_up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values
    mfrr_price = spot_price.copy()
    mfrr_price[mfrr_down_volyme<0] = mfrr_down_price[mfrr_down_volyme<0]
    mfrr_price[mfrr_up_volyme>0] = mfrr_up_price[mfrr_up_volyme>0]

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.2)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    
    x_values = np.arange(2)
    parts = axs[0].violinplot([spot_price*1e-3, mfrr_price*1e-3], x_values, points=1000, widths=0.75)
    for pc in parts['bodies']:
        pc.set_facecolor(np.ones(3)*0.5)
        pc.set_alpha(1)
    parts['cmins'].set_color(np.zeros(3))
    parts['cmaxes'].set_color(np.zeros(3))
    parts['cbars'].set_color(np.zeros(3))
    axs[0].set_ylabel('Price (k€/MWh)')
    axs[0].set(xticks=x_values)
    axs[0].set_xticklabels(['Spot', 'mFRR'], rotation=45)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"spot_vs_mfrr_dist", 'pdf')


def plot_opportunity_costs_grid_search(params, overwrite=False):

    config = get_config()
    profit_scaling = 1e-6
    c_lim = [0.4, 1.4]

    # Get the summary for the spot only strategies
    params_tmp = params.copy()
    params_tmp['name'] = 'moving_horizon'
    params_tmp['horizon'] = 1*24
    params_tmp['mfrr'] = True
    params_tmp['ttf'] = False
    params_tmp = utils.set_price_limits_and_opp_costs(params_tmp)

    # Get profits over the whole grid search area
    kappa_p2x_vals = np.round(np.linspace(-0.8, 0.8, 17), 1)
    kappa_x2p_vals = np.round(np.linspace(-0.8, 0.8, 17), 1)
    params_costs = []
    for kappa_p2x in kappa_p2x_vals:
        for kappa_x2p in kappa_x2p_vals:
            params_tmp = params_tmp.copy()
            params_tmp['kappa_p2x'] = kappa_p2x
            params_tmp['kappa_x2p'] = kappa_x2p
            params_costs.append(params_tmp)

    # Scale and filter profits
    df = cache.get_summary_for_strategies(params_costs, overwrite=overwrite)
    profits = df['profit'].values * profit_scaling
    profits = np.reshape(profits, (len(kappa_p2x_vals), len(kappa_x2p_vals)))
    # print(profits.max(), profits.min())
    filtered_profits = profits.copy()
    filtered_profits[profits < c_lim[0]] = np.nan

    # Extract no cost profit and maximum profit
    # get indices for where kappa is zero
    zero_kappa_p2x = np.where(kappa_p2x_vals == 0)[0][0]
    zero_kappa_x2p = np.where(kappa_x2p_vals == 0)[0][0]
    no_cost_profit = profits[zero_kappa_p2x, zero_kappa_x2p]
    max_profit = profits.max()
    # Find the row and column of the maximum value
    flat_max_index = np.argmax(profits)
    row_max_val, col_max_val = np.unravel_index(flat_max_index, filtered_profits.shape)
    # print(f'Best kappa_p2x: {kappa_p2x_vals[row_max_val]}')
    # print(f'Best kappa_x2p: {kappa_x2p_vals[col_max_val]}')

    # Set up the figure
    mpl_helper.set_article_style()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.3)
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Colored pixel plot for the proifts
    c = axs[0].pcolor(kappa_x2p_vals, kappa_p2x_vals, filtered_profits, vmin=c_lim[0], vmax=c_lim[1], cmap='gist_heat')

    # Add text and markers for the no cost and maximum profit points
    x_shift = -0.2
    y_shift = 0.05
    axs[0].plot(0, 0, 'o', mec='k', mfc='none')
    axs[0].text(x_shift, y_shift, f"{no_cost_profit:.2f} M€", ha='center', va='bottom')
    axs[0].plot(kappa_x2p_vals[col_max_val], kappa_p2x_vals[row_max_val], '^', mec='k', mfc='none')
    axs[0].text(kappa_x2p_vals[col_max_val]+x_shift, kappa_p2x_vals[row_max_val]+y_shift, f"{max_profit:.2f} M€", ha='center', va='bottom')
    # Add an empty legend to only show the strategy name, as in other figures
    axs[0].legend(loc='upper left', title=utils.get_strategy_abbreviation(params_tmp))

    # Colorbar
    cbar = fig.colorbar(c, ax=axs[0], ticks=c_lim)
    cbar.set_label('Profit (M€)', labelpad=-10)
    # Add labels
    axs[0].set_xlabel('Opp. cost ($\kappa^\mathrm{X2P}$)')
    axs[0].set_ylabel('Opp. cost ($\kappa^\mathrm{P2X}$)', labelpad=1)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"opportunity_costs_" + utils.get_strategy_abbreviation(params_tmp) + "_ttf" + f"_{params_tmp['ttf_premium']}", 'pdf')


def plot_opportunity_costs_comparison(params, overwrite=False):

    # GEt strategy with and without opportunity costs
    params_tmp = params.copy()
    params_tmp['name'] = 'moving_horizon'
    params_tmp['horizon'] = 1*24
    params_tmp['mfrr'] = True
    params_tmp = utils.set_price_limits_and_opp_costs(params_tmp)
    strategy_with_cost = cache.get_strategy(params_tmp, overwrite)
    # print_strategy_summary(strategy_with_cost)
    params_tmp['kappa_p2x'] = 0.
    params_tmp['kappa_x2p'] = 0.
    strategy_no_cost = cache.get_strategy(params_tmp, overwrite)
    # print_strategy_summary(strategy_no_cost)

    # Get the figure configuration
    config = get_config()
    bar_width = 0.4
    delta = 0.25
    marker_height = 25
    scaling = 1e-3
    x_values = np.array([0, 1.25])
    x_lim = [-0.75, x_values[1]+1-0.25]

    # Set up the figure
    mpl_helper.set_article_style()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.2)
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Hydrogen production
    axs[0].bar(x_values[0]-delta, strategy_no_cost['spot_p2x_amount']*scaling, bar_width, color=0.75*np.ones(3), ec='none', label='Spot')
    axs[0].bar(x_values[0]-delta, strategy_no_cost['mfrr_p2x_amount']*scaling, bar_width, bottom=strategy_no_cost['spot_p2x_amount']*scaling, color=0.25*np.ones(3), ec='none', label='mFRR')
    axs[0].bar(x_values[0]+delta, strategy_with_cost['spot_p2x_amount']*scaling, bar_width, color=0.75*np.ones(3), ec='none')
    axs[0].bar(x_values[0]+delta, strategy_with_cost['mfrr_p2x_amount']*scaling, bar_width, bottom=strategy_with_cost['spot_p2x_amount']*scaling, color=0.25*np.ones(3), ec='none')
    # Hydrogen consumption
    axs[0].bar(x_values[1]-delta, strategy_no_cost['spot_x2p_amount']*scaling, bar_width, color=0.75*np.ones(3), ec='none')
    axs[0].bar(x_values[1]-delta, strategy_no_cost['mfrr_x2p_amount']*scaling, bar_width, bottom=strategy_no_cost['spot_x2p_amount']*scaling, color=0.25*np.ones(3), ec='none')
    axs[0].bar(x_values[1]+delta, strategy_with_cost['spot_x2p_amount']*scaling, bar_width, color=0.75*np.ones(3), ec='none')
    axs[0].bar(x_values[1]+delta, strategy_with_cost['mfrr_x2p_amount']*scaling, bar_width, bottom=strategy_with_cost['spot_x2p_amount']*scaling, color=0.25*np.ones(3), ec='none')
    # Markers for the the strategy with and without opportunity costs
    axs[0].plot(x_values-delta, marker_height*np.ones(2), 'o', mec='k', mfc='none')
    axs[0].plot(x_values+delta, marker_height*np.ones(2), '^', mec='k', mfc='none')
    # Set labels and ticks
    axs[0].legend(loc='upper left')
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=['Prod.', 'Cons.'], ylim=[0, 43])
    axs[0].tick_params(axis='x', rotation=45, pad=-1)
    axs[0].xaxis.set_zorder(5)
    axs[0].set_ylabel('Hydrogen (GWh)')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"hydrogen_flow_" + utils.get_strategy_abbreviation(params_tmp), 'pdf')


def plot_possible_profits(data, overwrite=False):

    # Get the figure configuration
    config = get_config()
    profit_scaling = 1e-6
    params = config['default_params']
    down_color = config['figures']['colors']['mfrr_down']
    down_color_alpha= mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(down_color, alpha=0.5))
    up_color = config['figures']['colors']['mfrr_up']
    up_color_alpha= mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(up_color, alpha=0.5))
    new_color = config['figures']['colors']['mfrr_new']
    new_color_alpha= mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(new_color, alpha=0.5))
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Get the profit for an infinite horizon strategy on only the mFRR market
    params_mfrr_only = params.copy()
    params_mfrr_only['mfrr'] = True
    params_mfrr_only['spot'] = False
    params_mfrr_only['ttf'] = False
    params_mfrr_only['name'] = 'infinite_horizon'
    df = cache.get_summary_for_strategies([params_mfrr_only], overwrite=overwrite)
    mfrr_only_profit = df['profit'].values[0] * profit_scaling

    # Extract relevant data
    data_year = filter_data(data, params)
    spot_data, _, mfrr_data = split_data(data_year, params)
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values
    mfrr_down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values
    mfrr_up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values
    # Get the potential profit from canceling bids placed on the spot market
    possible_p2x_up_volyme = np.clip(mfrr_up_volyme, 0, params['power_p2x']) 
    possible_p2x_down_volyme = np.clip(-mfrr_down_volyme, 0, params['power_p2x']) 
    possible_x2p_up_volyme = np.clip(mfrr_up_volyme, 0, params['power_x2p'])
    possible_x2p_down_volyme = np.clip(-mfrr_down_volyme, 0, params['power_x2p'])
    p2x_up_profit = np.sum(possible_p2x_up_volyme*(mfrr_up_price-spot_price)) * profit_scaling
    p2x_down_profit = np.sum(possible_p2x_down_volyme*(spot_price-mfrr_down_price)) * profit_scaling
    x2p_up_profit = np.sum(possible_x2p_up_volyme*(mfrr_up_price-spot_price)) * profit_scaling
    x2p_down_profit = np.sum(possible_x2p_down_volyme*(spot_price-mfrr_down_price))  * profit_scaling

    bar_width = 0.8
    x_values = np.arange(2)
    x_lim = [-0.75, x_values.size -0.25]
    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.2)
    plot_rect_cm[0] += 0.5*fig_size_cm[0]
    fig_size_cm[0] = fig_size_cm[0]*1.5
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Text
    line_sep = 0.08
    cancel_text = [('Cancel', 'black'), ('bids', 'black'), (f"P2X{mfrr_up_label.strip('mFRR')}", up_color), (f"X2P{mfrr_down_label.strip('mFRR')}", down_color)]
    for i, (text, color) in enumerate(cancel_text):
        fig.text(0.25, 0.9-line_sep*i, text, ha='center', va='top', color=color)
    new_text = [('New', 'black'), ('bids', 'black'), (f"P2X{mfrr_down_label.strip('mFRR')}", new_color), (f"X2P{mfrr_up_label.strip('mFRR')}", new_color)]
    for i, (text, color) in enumerate(new_text):
        fig.text(0.25, 0.55-line_sep*i, text, ha='center', va='top', color=color)
    # fig.text(0.25, 0.95, f"Cancel\nbids\nP2X{mfrr_up_label.strip('mFRR')}", ha='center', va='top')

    # Plots
    axs[0].bar(0, p2x_up_profit, bar_width, fc=up_color_alpha, ec='none', zorder=2)
    axs[0].bar(0, p2x_up_profit+x2p_down_profit, bar_width, fc=down_color_alpha, ec='none', zorder=1)
    # axs[0].bar(1, x2p_up_profit, bar_width, fc='gray', ec=up_color_alpha, zorder=2, hatch='/////', linewidth=0)
    # axs[0].bar(1, x2p_up_profit+p2x_down_profit, bar_width, fc='gray', ec=down_color_alpha, zorder=1, hatch='\\\\\\\\\\', linewidth=0)
    axs[0].bar(1, x2p_up_profit+p2x_down_profit, bar_width, fc=new_color_alpha, ec='none', zorder=1)
    # axs[0].bar(1, mfrr_only_profit, bar_width, fc='black', ec='none')    
    # axs[0].legend(loc='upper right')
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=['Cancel', 'New'])
    axs[0].tick_params(axis='x', rotation=45, pad=-1)
    axs[0].xaxis.set_zorder(5)
    axs[0].set_ylabel('Profit (M€)')

    # Print the profit values
    print(f"Profit from canceling P2X bids: {p2x_up_profit:.2f} M€")
    print(f"Profit from canceling X2P bids: {x2p_down_profit:.2f} M€")
    print(f"Profit from new bids: {(x2p_up_profit+p2x_down_profit):.2f} M€")

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"possible_profit_mfrr", 'pdf')


def plot_profit_per_strategy(overwrite=False):

    config = get_config()
    margin_color_spot = config['figures']['colors']['margin']
    margin_color_mfrr = config['figures']['colors']['margin3']

    # Get the summary for the spot only strategies
    params = config['default_params'].copy()
    params_list, names = utils.get_stratety_params(params, inf_hor=False)
    df_spot = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_mfrr = params.copy()
    params_mfrr['mfrr'] = True
    params_list_mfrr, _ = utils.get_stratety_params(params_mfrr, inf_hor=False)
    df_mfrr = cache.get_summary_for_strategies(params_list_mfrr, overwrite=overwrite)

    # Scale the profits to M€
    profits = df_spot['profit'].values
    profits = profits*1e-6
    profits_mfrr = df_mfrr['profit'].values
    profits_mfrr = profits_mfrr*1e-6

    # Prepare the plot parameters
    bar_width = 0.4
    x_lim = [-0.75, profits.size -0.25]
    y_lim = [0, profits_mfrr.max()*1.05]
    default_horizon = params['horizon']
    default_horizon_idx = np.where((df_spot['name']=='moving_horizon') & (df_spot['horizon'] == default_horizon))[0][0]
    x_values = np.arange(profits.size)

    # Create the figure
    mpl_helper.set_article_style()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Plot the profits
    axs[0].fill_between(x_lim, profits[-1], profits[default_horizon_idx], color=margin_color_spot, alpha=0.5, zorder=1)
    axs[0].fill_between(x_lim, profits_mfrr[-1], profits_mfrr[default_horizon_idx], color=margin_color_mfrr, alpha=0.5, zorder=1)
    axs[0].bar(x_values-0.2, profits, bar_width, fc=0.75*np.ones(3), ec=0.25*np.ones(3), zorder=3, label='Spot\n$\mathrm{H}_2$')
    axs[0].bar(x_values+0.2, profits_mfrr, bar_width, fc=0.25*np.ones(3), ec=0.25*np.ones(3), zorder=2, label='Spot+mFRR\n$\mathrm{H}_2$')
    # for x, name in zip(x_values, names):
    #     axs[0].text(x, y_lim[1]*0.075, name, ha='center', va='bottom', ma='left', rotation=90, fontsize=6)

    # Legends, labels, ticks, etc
    axs[0].legend(ncol=2, loc='lower left')
    # axs[0].set_xlabel(f"Degree of knowledge")
    axs[0].set_ylabel(f"Profit {params['year']} (M€)")
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=names, ylim=y_lim)
    axs[0].tick_params(axis='x', rotation=45, pad=-1)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profit_per_strategy_mfrr", 'pdf')


def plot_profit_per_year(overwrite=False):

    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Convert margn colors to RGBA to avoid transparency issues
    margin_color_spot = mpl_helper.hex_to_rgba(config['figures']['colors']['margin'], alpha=0.5)
    margin_color_mfrr = mpl_helper.hex_to_rgba(config['figures']['colors']['margin3'], alpha=0.5)

    # Get the summary for the spot only strategies
    params = config['default_params'].copy()
    params_list, years = utils.get_yearly_margin_params(params)
    df_spot = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_mfrr = params.copy()
    params_mfrr['mfrr'] = True
    params_list_mfrr, _ = utils.get_yearly_margin_params(params_mfrr)
    df_mfrr = cache.get_summary_for_strategies(params_list_mfrr, overwrite=overwrite)

    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Spot only
    axs[0].fill_between(years, df_spot[df_spot['name']=='no_horizon']['profit']*1e-6, df_spot[df_spot['name']=='moving_horizon']['profit']*1e-6, color=margin_color_spot, label='Spot\n$\mathrm{H}_2$')
    # Spot+mFRR
    axs[0].fill_between(years, df_mfrr[df_mfrr['name']=='no_horizon']['profit']*1e-6, df_mfrr[df_mfrr['name']=='moving_horizon']['profit']*1e-6, color=margin_color_mfrr, label='Spot+mFRR\n$\mathrm{H}_2$')

    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('Profit (M€/year)')
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set_xlabel('Year')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profits_per_year_mfrr", 'pdf')


def plot_mfrr_strategy_breakdown_euro(params, overwrite=False):

    config = get_config()

    params_tmp = params.copy()
    params_tmp['mfrr'] = True
    params_tmp = utils.set_price_limits_and_opp_costs(params_tmp)

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_tmp], overwrite)
    strategy = cache.load_strategies(df)[0]

    # Get the plot
    fig, _ = strategy_plots.plot_strategy_breakdown_euro(strategy, params_tmp)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_infinite_horizon_bar_plot_euro", 'pdf')

    return strategy


def plot_mfrr_x2p_revenue_per_hour(data, params, overwrite=False):

    config = get_config()
    price_scaling = 1e-3
    x2p_color = config['figures']['colors']['x2p']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Filter the data
    data_year = filter_data(data, params)
    _, _, mfrr_data = split_data(data_year, params)

    params_tmp = params.copy()
    params_tmp['mfrr'] = True
    params_tmp = utils.set_price_limits_and_opp_costs(params_tmp)

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_tmp], overwrite)
    strategy = cache.load_strategies(df)[0]

    # Prepare data to plot
    mfrr_x2P_revenu_per_hour = strategy['bid_mfrr_up_x2p']*mfrr_data['up_regulating_price_mFRR']*price_scaling
    mfrr_x2P_revenu_per_hour = np.flip(np.sort(mfrr_x2P_revenu_per_hour))
    mfrr_x2P_revenu_per_hour = mfrr_x2P_revenu_per_hour[mfrr_x2P_revenu_per_hour>0]
    cum_rev_frac = np.cumsum(mfrr_x2P_revenu_per_hour) / np.sum(mfrr_x2P_revenu_per_hour)
    hours = np.arange(mfrr_x2P_revenu_per_hour.size) + 1
    
    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].fill_between(hours, 0, mfrr_x2P_revenu_per_hour, facecolor=x2p_color, alpha=0.5, label=r'X2P ' + mfrr_up_label)
    axs[0].plot(np.nan, np.nan, 'k-', label='Cum. rev. (%)')
    axs[0].legend(loc='lower right', title=utils.get_strategy_abbreviation(params_tmp))
    axs[0].set(xlabel='Hours', ylabel='Revenue (k€/h)')

    x_lim = axs[0].get_xlim()
    show_rev_fracs = np.array([0.25, 0.5, 0.75])
    show_h = np.interp(show_rev_fracs, cum_rev_frac, hours)
    ax_ontop = mpl_helper.get_axes_ontop(fig, axs[0])
    ax_ontop.plot(hours, cum_rev_frac, 'k-')
    for i in range(show_rev_fracs.size):
        #ax_ontop.plot([show_h[i], show_h[i]], [0, show_rev_fracs[i]], ':', c='gray')
        ax_ontop.plot([show_h[i], show_h[i]+80], show_rev_fracs[i]*np.ones(2), ':', c='gray')
        ax_ontop.text(show_h[i]+80, show_rev_fracs[i], f'{show_rev_fracs[i]*100:2.0f} %', ha='right')
    ax_ontop.set(xlim=x_lim, ylim=[0, 1])

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_infinite_horizon_x2p", 'pdf')


def plot_strategy(data, params, mfrr=False, overwrite=False):

    config = get_config()
    params_plot = params.copy()
    params_plot['mfrr'] = mfrr
    params_plot = utils.set_price_limits_and_opp_costs(params_plot)

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_plot], overwrite)
    strategy = cache.load_strategies(df)[0]

    # Get the plot
    fig, axs = strategy_plots.plot_running_strategy_zoom(data, strategy, params_plot)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"running_strategy_mfrr_{mfrr:1.0f}", 'pdf')