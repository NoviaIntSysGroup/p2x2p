import numpy as np
from matplotlib.ticker import MaxNLocator

from p2x2p.data import spot
from p2x2p.report_figures import utils, mpl_helper
from p2x2p.utils.utils import get_config
from p2x2p.strategies import cache, infinite_storage
from p2x2p.report_figures.strategy_plots import plot_running_strategy, plot_running_strategy_zoom_horizon


def plot_all_panels(overwrite=False):

    # Get the configuration
    config = get_config()

    # Get the default parameters
    params = config['default_params'].copy()
    params['year'] = 2023

    # Get the spot data
    spot_data = spot.get_data()
    spot_data['mean_price'] = spot_data['elspot-fi'].ewm(int(24*params['n_days'])).mean().values
    spot_data_year = spot_data[spot_data['date'].dt.year == params['year']]

    plot_profits_per_strategy(params, overwrite)
    plot_profits_per_year(params, overwrite)
    plot_infinite_storage(spot_data_year, params)
    plot_infinite_horizon(spot_data_year, params, overwrite)
    plot_no_horizon(spot_data_year, params, overwrite)
    plot_moving_horizon(spot_data_year, params, overwrite)


def plot_profits_per_strategy(params, overwrite=False):

    config = get_config()
    margin_color = config['figures']['colors']['margin']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Profits per strategy
    params_list = []

    # Infinite storage
    params_infinite_storage = params.copy()
    params_infinite_storage['name'] = 'infinite_storage'
    params_list.append(params_infinite_storage)

    # Add all other strategies
    params_list += utils.get_stratety_params(params)[0]

    # Define shades and names for each strategy
    shades = np.array([0.75, 0.6, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.2])
    methods = ['Dur. curve', 'LP', 'MILP', 'Grid search']
    names = [utils.get_strategy_abbreviation(params) for params in params_list]

    df = cache.get_summary_for_strategies(params_list, overwrite)

    # get n random values in decreasing order
    profits = df['profit'].values
    profits = profits*1e-6
    x_values = np.arange(profits.size)

    x_lim = [-0.75, profits.size -0.25]
    y_lim = [0, profits.max()*1.05]

    mpl_helper.set_article_style()
    bar_width = 0.8
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].plot(x_lim, profits[-1]*np.ones(2), ':', color='black', zorder=2)
    axs[0].plot(x_lim, profits[-4]*np.ones(2), ':', color='black', zorder=2)
    axs[0].fill_between(x_lim, profits[-1], profits[-4], color=margin_color, alpha=0.5, zorder=1)
    # bars = axs[0].bar(range(profits.size), profits, bar_width, fc=0.75*np.ones(3), ec=0.25*np.ones(3), zorder=3)
    # for x, name, bar in zip(x_values, names, bars):
    #     bar.set_facecolor(np.ones(3)*shades[x])
    #     axs[0].text(x, y_lim[1]*0.05, name, ha='center', va='bottom', ma='right', rotation=90, fontsize=6)
    # Get unique elements and their first occurrence indices
    _, indices = np.unique(shades, return_index=True)
    unique_shades = shades[np.sort(indices)]
    for i, shade in enumerate(unique_shades):
        idx = np.where(shades == shade)[0]
        axs[0].bar(x_values[idx], profits[idx], bar_width, fc=shade*np.ones(3), ec=0.25*np.ones(3), zorder=3, label=methods[i])

    axs[0].text(0, y_lim[1]*0.05, names[0], ha='center', va='bottom', ma='right', rotation=90, fontsize=6)
    # axs[0].set_xlabel(f"Degree of knowledge and storage")
    axs[0].set_ylabel(f"Profit {params['year']} (M€)")
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=['']+names[1:], ylim=y_lim)
    axs[0].tick_params(axis='x', rotation=45, pad=-1)
    axs[0].legend(ncol=2, loc='upper right')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + 'profit_per_strategy', 'pdf')


def plot_profits_per_year(params, overwrite=False):

    config = get_config()
    margin_color = config['figures']['colors']['margin']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    params_list, _ = utils.get_yearly_margin_params(params)

    df = cache.get_summary_for_strategies(params_list, overwrite)
    df_no_horizon = df[df['name'] == 'no_horizon']
    df_3_day_horizon = df[df['name'] == 'moving_horizon']

    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].plot(df_no_horizon['year'], df_no_horizon['profit']*1e-6, 'ko:', label='No horizon')
    axs[0].plot(df_3_day_horizon['year'], df_3_day_horizon['profit']*1e-6, 'ko:', label='3 day horizon')
    axs[0].fill_between(df_no_horizon['year'], df_no_horizon['profit']*1e-6, df_3_day_horizon['profit']*1e-6, color=margin_color, alpha=0.5)
    axs[0].set_ylabel('Profit (M€/year)')
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set_xlabel('Year')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + 'profits_per_year', 'pdf')


def plot_infinite_storage(spot_data, params):

    params_infinite_storage = params.copy()
    params_infinite_storage['name'] = 'infinite_storage'

    # Extract the yearly spot data
    spot_data_year = spot_data[spot_data['date'].dt.year == params_infinite_storage['year']]
    price = spot_data_year['elspot-fi'].values
    # Sort the price in descending order
    price_sorted = np.sort(price)[::-1]

    # Get the maximum fraction of the time that the plant can be in either mode
    max_x2p_frac = infinite_storage.get_max_x2p_frac(params_infinite_storage)
    max_p2x_frac = 1 - max_x2p_frac

    # P2X2P profits
    durations, revenues, costs = infinite_storage.get_revenues_and_costs(spot_data, params_infinite_storage)
    profits = revenues - costs
    # Check the utilization that maximizes the profit
    optimal_utilization = durations[np.argmax(profits)]
    x2p_frac = max_x2p_frac*optimal_utilization
    p2x_frac = max_p2x_frac*optimal_utilization

    # Define x and y values for the revenue fill plot
    x_fill_revenue = durations[durations<=max_x2p_frac*optimal_utilization]
    x_fill_revenue = np.concatenate([x_fill_revenue, [x_fill_revenue[-1], x_fill_revenue[0]]])
    y_fill_revenue = price_sorted[durations<=max_x2p_frac*optimal_utilization]
    y_fill_revenue = np.concatenate([y_fill_revenue, [0, 0]])
    # Define x and y values for the cost fill plot
    x_fill_costs = durations[durations>=(1-max_p2x_frac*optimal_utilization)]
    x_fill_costs = np.concatenate([x_fill_costs, [x_fill_costs[-1], x_fill_costs[0]]])
    y_fill_costs = price_sorted[durations>=(1-max_p2x_frac*optimal_utilization)]
    y_fill_costs = np.concatenate([y_fill_costs, [0, 0]])

    config = get_config()
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Duration graph
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].fill(x_fill_revenue, y_fill_revenue, color=x2p_color, label=f"X2P ({x2p_frac*100:2.1f} %)", alpha=0.5)
    axs[0].fill(x_fill_costs, y_fill_costs, color=p2x_color, label=f"P2X ({p2x_frac*100:2.1f} %)", alpha=0.5)
    axs[0].plot(durations, price_sorted, color='black', label=f"Spot price {spot_data_year['date'].dt.year.iloc[0]}")
    axs[0].legend(loc='upper right', title=utils.get_strategy_abbreviation(params_infinite_storage))
    axs[0].set_xlabel('Utilization')
    axs[0].set_ylabel('Price (€/MWh)')
    axs[0].set(xlim=[-1e-2, 1+1e-2], ylim=[-25, 350])

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + 'infinite_storage_duration', 'pdf')

    # Profit graph
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].plot(durations, revenues*1e-6, color=x2p_color, label='Revenue')
    axs[0].plot(durations, costs*1e-6, color=p2x_color, label='Cost')
    axs[0].plot(durations, profits*1e-6, color='black', label='Profit')
    axs[0].plot(durations[np.argmax(profits)], profits[np.argmax(profits)]*1e-6, 'ko', ms=6)
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel('Utilization')
    axs[0].set_ylabel('Million €')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + 'infinite_storage_profit', 'pdf')


def plot_infinite_horizon(spot_data, params, overwrite=False):

    params_infinite_horizon = params.copy()
    params_infinite_horizon['name'] = 'infinite_horizon'

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_infinite_horizon], overwrite)
    strategy = cache.load_strategies(df)[0]
    
    # Plot the strategy
    config = get_config()
    fig, axs = plot_running_strategy(spot_data, strategy, params_infinite_horizon)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "infinite_horizon", 'pdf')


def plot_no_horizon(spot_data, params, overwrite=False):

    params_no_horizon = params.copy()
    params_no_horizon['name'] = 'no_horizon'
    params_no_horizon = utils.set_price_limits_and_opp_costs(params_no_horizon)

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_no_horizon], overwrite)
    strategy = cache.load_strategies(df)[0]

    # Plot the strategy
    config = get_config()
    fig, axs = plot_running_strategy_zoom_horizon(spot_data, strategy, params_no_horizon)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "no_horizon", 'pdf')


def plot_moving_horizon(spot_data, params, overwrite=False):

    params_moving_horizon = params.copy()
    params_moving_horizon['name'] = 'moving_horizon'
    params_moving_horizon = utils.set_price_limits_and_opp_costs(params_moving_horizon)

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_moving_horizon], overwrite)
    strategy = cache.load_strategies(df)[0]

    # Plot the strategy
    config = get_config()
    fig, axs = plot_running_strategy_zoom_horizon(spot_data, strategy, params_moving_horizon)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "moving_horizon", 'pdf')
