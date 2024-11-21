import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from p2x2p.utils.utils import get_config
from p2x2p.data import spot, fingrid, ttf
from p2x2p.report_figures import utils, mpl_helper
from p2x2p.strategies import cache
from p2x2p.strategies.utils import split_data, filter_data
from p2x2p.report_figures.strategy_plots import plot_running_strategy


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

    plot_mfrr_up_vs_ttf(data, params)
    plot_ttf_potential(data, params)

    plot_profit_per_strategy(overwrite)
    plot_profit_per_year(overwrite)

    plot_profit_split_per_strategy(overwrite)
    plot_profit_split_per_premium(overwrite)

    plot_strategy(spot_data, ttf_data, params, premium=None, overwrite=overwrite)
    plot_strategy(spot_data, ttf_data, params, premium=0., overwrite=overwrite)


def plot_mfrr_up_vs_ttf(data, params):

    config = get_config()
    y2p_color = config['figures']['colors']['y2p']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_up_label = config['latex']['mfrr']['up']
    scaling = 1e-3

    # Extract the yearly spot data
    data_year = filter_data(data, params)
    spot_data_year, ttf_data_year, mfrr_data_year = split_data(data_year, params)
    spot_price = spot_data_year['elspot-fi'].values * scaling
    spot_dates = spot_data_year['date']  # keep dates in pandas format to keep timezone info
    ttf_price = ttf_data_year['TTF (Euro/MWh)'].values * scaling
    mfrr_up_price = mfrr_data_year['up_regulating_price_mFRR'].values * scaling
    
    # Prepare data for the plot
    ttf_price_margin = ttf_price/params['eff_x2p']
    potential_y2p_frac = (mfrr_up_price>ttf_price_margin).mean()

    # Set limits and ticks
    delta_x = pd.Timedelta(days=5)
    delta_y = (mfrr_up_price.max() - mfrr_up_price.min())*0.05
    x_lim = [spot_dates.iloc[0]-delta_x, spot_dates.iloc[-1]+delta_x]
    y_lim = [mfrr_up_price.min()-delta_y, mfrr_up_price.max()+delta_y]
    x_ticks = [spot_dates.iloc[0]+pd.DateOffset(months=4*i) for i in range(10)]
    x_tick_labels = [date.strftime('%Y-%m') for date in x_ticks]  # Matplotlib stuggles to format the dates correctly when a time zone is included
    y_range = y_lim[1] - y_lim[0]

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Plot data
    axs[0].fill_between(ttf_data_year['date'], y_lim[0], (mfrr_up_price>ttf_price_margin)*y_range + y_lim[0], fc=y2p_color, alpha=0.5, label=f'Y2P {potential_y2p_frac*100:2.1f} %')
    axs[0].plot(spot_data_year['date'], mfrr_up_price, color=up_color, label=mfrr_up_label)
    axs[0].plot(spot_data_year['date'], spot_price, color='black', label='Spot price')
    axs[0].plot(ttf_data_year['date'], ttf_price, color='gray', label='TTF price')
    axs[0].plot(ttf_data_year['date'], ttf_price_margin, '--', color='gray', label='Fuel cost')
    # Configure axes and legend
    axs[0].legend(loc='upper left')
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, xlim=x_lim, ylim=y_lim, ylabel='Price (k€/MWh)')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_up_vs_ttf_prices", 'pdf')


def plot_ttf_potential(data, params):

    config = get_config()
    y2p_color = config['figures']['colors']['y2p']
    up_color = config['figures']['colors']['mfrr_up']
    premiums = utils.TTF_PREMIUMS
    premiums = np.append(premiums, params['ttf_premium'])

    # Extract the yearly spot data
    data_year = filter_data(data, params)
    spot_data_year, ttf_data_year, mfrr_data_year = split_data(data_year, params)
    spot_price = spot_data_year['elspot-fi'].values
    spot_dates = spot_data_year['date']  # keep dates in pandas format to keep timezone info
    ttf_price = ttf_data_year['TTF (Euro/MWh)'].values
    mfrr_up_price = mfrr_data_year['up_regulating_price_mFRR'].values
    mfrr_up_volyme = mfrr_data_year['up_regulation_volyme_mFRR'].values
    possible_power = np.clip(mfrr_up_volyme, 0, params['power_x2p'])        # Clip the possible power to the maximum power

    # Compute the potential Y2P fractions and profits
    # Assume that everything that can be sold on the spot market is sold there first
    potential_y2p_spot_fracs = []
    potential_y2p_mfrr_fracs = []
    potential_y2p_spot_profits = []
    potential_y2p_mfrr_profits = []
    for premium in premiums:
        # Get price information
        ttf_price_premium = ttf_price + premium                             # Add the premium to the TTF price
        ttf_price_margin_premium = ttf_price_premium/params['eff_x2p']      # Compute the TTF price margin
        price_difference_spot = spot_price - ttf_price_margin_premium       # Compute the price difference
        price_difference_mfrr = mfrr_up_price - ttf_price_margin_premium    # Compute the price difference
        # Get running hours
        spot_running_hours = price_difference_spot > 0
        mfrr_running_hours = price_difference_mfrr > 0
        mfrr_running_hours[spot_running_hours] = False
        # Compute the potential fractions and profits
        potential_y2p_spot_fracs.append(spot_running_hours.mean())
        potential_y2p_mfrr_fracs.append(mfrr_running_hours.mean())
        potential_y2p_spot_profits.append(
            np.sum(price_difference_spot[spot_running_hours]*params['power_x2p'])
            )
        potential_y2p_mfrr_profits.append(
            np.sum(price_difference_mfrr[mfrr_running_hours]*possible_power[mfrr_running_hours])
            )

    # Convert to numpy arrays and sum up the results
    potential_y2p_spot_fracs = np.array(potential_y2p_spot_fracs)
    potential_y2p_mfrr_fracs = np.array(potential_y2p_mfrr_fracs)
    potential_y2p_spot_profits = np.array(potential_y2p_spot_profits)
    potential_y2p_mfrr_profits = np.array(potential_y2p_mfrr_profits)
    potential_y2p_fracs = potential_y2p_spot_fracs + potential_y2p_mfrr_fracs
    potential_y2p_profits = potential_y2p_spot_profits + potential_y2p_mfrr_profits

    # Set the text padding
    text_x_pad = 0.025*max(premiums)
    text_y_pad = 0.025*potential_y2p_fracs[0]

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Plot the potential Y2P fractions
    axs[0].plot(premiums[:-1], potential_y2p_fracs[:-1], color=y2p_color)
    axs[0].axvline(params['ttf_premium'], c='k', ls=':')
    # Add marker and text with the potential profit in euros for two points
    for idx in [0, len(potential_y2p_fracs)-1]:
        potential_profit = f"{potential_y2p_fracs[idx]*100:2.1f} %; {potential_y2p_profits[idx]*1e-6:1.2f} M€"
        axs[0].plot(premiums[idx], potential_y2p_fracs[idx], 'o', color=up_color, mfc='none')
        axs[0].text(premiums[idx]+text_x_pad, potential_y2p_fracs[idx]+text_y_pad, potential_profit, va='bottom', ha='left')
    # Set labels
    axs[0].set_xlabel("TTF premium (€/MWh)")
    axs[0].set_ylabel(f"Y2P frac. {params['year']}")
    # Green premium text
    y_lim = axs[0].get_ylim()
    axs[0].text(premiums[-1]+text_x_pad, y_lim[1], 'Green\npremium', va='top', ha='left')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"ttf_potential_mfrr_up", 'pdf')


def plot_profit_per_strategy(overwrite=False):

    config = get_config()

    # Convert margn colors to RGBA to avoid transparency issues
    margin_color_ttf = mpl_helper.hex_to_rgba(config['figures']['colors']['margin2'], alpha=0.5)
    margin_color_mfrr = mpl_helper.hex_to_rgba(config['figures']['colors']['margin3'], alpha=0.5)

    # Get the summary for the spot only strategies
    params_mfrr = config['default_params'].copy()
    params_mfrr['mfrr'] = True
    params_list_mfrr, names = utils.get_stratety_params(params_mfrr, inf_hor=False)
    df_mfrr = cache.get_summary_for_strategies(params_list_mfrr, overwrite=overwrite)

    # Get the summary for the spot, TTF and mFRR strategies
    params_all = params_mfrr.copy()
    params_all['mfrr'] = True
    params_all['ttf'] = True
    params_list_all, _ = utils.get_stratety_params(params_all, inf_hor=False)
    df_all = cache.get_summary_for_strategies(params_list_all, overwrite=overwrite)

    # Scale the profits to M€
    profits_mfrr = df_mfrr['profit'].values
    profits_mfrr = profits_mfrr*1e-6
    profits_all = df_all['profit'].values
    profits_all = profits_all*1e-6

    # Prepare the plot parameters
    bar_width = 0.4
    x_lim = [-0.75, profits_mfrr.size -0.25]
    y_lim = [0, profits_all.max()*1.05]
    default_horizon = params_mfrr['horizon']
    default_horizon_idx = np.where((df_mfrr['name']=='moving_horizon') & (df_mfrr['horizon'] == default_horizon))[0][0]
    x_values = np.arange(profits_mfrr.size)

    # Create the figure
    mpl_helper.set_article_style()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Plot the profits
    axs[0].fill_between(x_lim, profits_mfrr[-1], profits_mfrr[default_horizon_idx], color=margin_color_mfrr, edgecolor='none', zorder=1)
    axs[0].fill_between(x_lim, profits_all[-1], profits_all[default_horizon_idx], color=margin_color_mfrr, edgecolor=margin_color_ttf, hatch='|||', lw=1, zorder=1)
    axs[0].bar(x_values-0.2, profits_mfrr, bar_width, fc=0.75*np.ones(3), ec=0.25*np.ones(3), zorder=3, label='Spot+mFRR\n$\mathrm{H}_2$')
    axs[0].bar(x_values+0.2, profits_all, bar_width, fc=0.25*np.ones(3), ec=0.25*np.ones(3), zorder=2, label='Spot+mFRR\n$\mathrm{H}_2$ & $\mathrm{CH}_4$')
    # for x, name in zip(x_values, names):
    #     axs[0].text(x, y_lim[1]*0.075, name, ha='center', va='bottom', ma='left', rotation=90, fontsize=6)

    # Legends, labels, ticks, etc
    axs[0].legend(ncol=2, loc='lower left')
    # axs[0].set_xlabel(f"Degree of knowledge")
    axs[0].set_ylabel(f"Profit {params_mfrr['year']} (M€)")
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=names, ylim=y_lim)
    axs[0].tick_params(axis='x', rotation=45, pad=-1)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profit_per_strategy_mfrr_ttf", 'png')


def plot_profit_per_year(overwrite=False):

    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Convert margn colors to RGBA to avoid transparency issues
    margin_color_spot = mpl_helper.hex_to_rgba(config['figures']['colors']['margin'], alpha=0.5)
    margin_color_ttf = mpl_helper.hex_to_rgba(config['figures']['colors']['margin2'], alpha=0.5)
    margin_color_mfrr = mpl_helper.hex_to_rgba(config['figures']['colors']['margin3'], alpha=0.5)

    # Get the summary for the spot only strategies
    params_spot = config['default_params'].copy()
    params_list_spot, years = utils.get_yearly_margin_params(params_spot)
    df_spot = cache.get_summary_for_strategies(params_list_spot, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_ttf = params_spot.copy()
    params_ttf['ttf'] = True
    params_list_ttf, _ = utils.get_yearly_margin_params(params_ttf)
    df_ttf = cache.get_summary_for_strategies(params_list_ttf, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_mfrr = params_spot.copy()
    params_mfrr['mfrr'] = True
    params_list_mfrr, _ = utils.get_yearly_margin_params(params_mfrr)
    df_mfrr = cache.get_summary_for_strategies(params_list_mfrr, overwrite=overwrite)

    # Get the summary for the spot, ttf, and mFRR strategies
    params_all = params_spot.copy()
    params_all['mfrr'] = True
    params_all['ttf'] = True
    params_list_all, _ = utils.get_yearly_margin_params(params_all)
    df_all = cache.get_summary_for_strategies(params_list_all, overwrite=overwrite)

    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Spot only
    axs[0].fill_between(years, df_spot[df_spot['name']=='no_horizon']['profit']*1e-6, df_spot[df_spot['name']=='moving_horizon']['profit']*1e-6, color=margin_color_spot, edgecolor='none', label='Spot\n$\mathrm{H}_2$')
    # Spot+TTF
    axs[0].fill_between(years, df_ttf[df_ttf['name']=='no_horizon']['profit']*1e-6, df_ttf[df_ttf['name']=='moving_horizon']['profit']*1e-6, color=margin_color_ttf, edgecolor='none', label='Spot\n$\mathrm{H}_2$ & $\mathrm{CH}_4$')
    # Spot+mFRR
    axs[0].fill_between(years, df_mfrr[df_mfrr['name']=='no_horizon']['profit']*1e-6, df_mfrr[df_mfrr['name']=='moving_horizon']['profit']*1e-6, color=margin_color_mfrr, edgecolor='none', label='Spot+mFRR\n$\mathrm{H}_2$')
    # Spot+TTF+mFRR
    axs[0].fill_between(years, df_all[df_all['name']=='no_horizon']['profit']*1e-6, df_all[df_all['name']=='moving_horizon']['profit']*1e-6, color=margin_color_mfrr, edgecolor=margin_color_ttf, hatch='|||', lw=1, label='Spot+mFRR\n$\mathrm{H}_2$ & $\mathrm{CH}_4$')

    axs[0].legend(loc='upper left', ncol=2)
    axs[0].set_ylabel('Profit (M€/year)')
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set_xlabel('Year')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profits_per_year_mfrr_ttf", 'png')


def plot_profit_split_per_strategy(overwrite=False):

    config = get_config()
    params = config['default_params'].copy()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    h2_color_spot = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split1'], alpha=0.5))
    h2_color_mfrr = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split1'], alpha=0.25))
    ch4_color_spot = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split2'], alpha=0.5))
    ch4_color_mfrr = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split2'], alpha=0.25))
    dark_gray = 0.25*np.ones(3)

    # Get the summary for the spot, TTF and mFRR strategies
    params = config['default_params'].copy()
    params['ttf'] = True
    params['mfrr'] = True
    params_list, names = utils.get_stratety_params(params, inf_hor=False)
    df = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

    # get n random values in decreasing order
    profits_h2_spot = df['spot_x_profit'].values
    profits_h2_spot = profits_h2_spot*1e-6
    profits_ch4_spot = df['spot_y_profit'].values
    profits_ch4_spot = profits_ch4_spot*1e-6
    profits_h2_mfrr = df['mfrr_x_profit'].values
    profits_h2_mfrr = profits_h2_mfrr*1e-6
    profits_ch4_mfrr = df['mfrr_y_profit'].values
    profits_ch4_mfrr = profits_ch4_mfrr*1e-6
    x_values = np.arange(profits_h2_spot.size)

    # Set the bar width and the limits
    bar_width = 0.8
    x_lim = [-0.75, profits_h2_spot.size -0.25]
    y_lim = [0, (profits_h2_spot+profits_h2_mfrr+profits_ch4_spot+profits_ch4_mfrr).max()*1.05]

    # Create the figure
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # Plot the profits
    axs[0].bar(range(profits_h2_spot.size), profits_h2_spot, bar_width, fc=h2_color_spot, ec=dark_gray, zorder=4, label="$\mathrm{H}_2$ spot")
    axs[0].bar(range(profits_h2_spot.size), profits_h2_spot+profits_h2_mfrr, bar_width, fc=h2_color_mfrr, ec=dark_gray, zorder=3, label="$\mathrm{H}_2$ mFRR")
    axs[0].bar(range(profits_h2_spot.size), profits_h2_spot+profits_h2_mfrr+profits_ch4_spot, bar_width, fc=ch4_color_spot, ec=dark_gray, zorder=2, label='$\mathrm{CH}_4$ spot')
    axs[0].bar(range(profits_h2_spot.size), profits_h2_spot+profits_h2_mfrr+profits_ch4_spot+profits_ch4_mfrr, bar_width, fc=ch4_color_mfrr, ec=dark_gray, zorder=1, label='$\mathrm{CH}_4$ mFRR')

    # Legends, labels, ticks, etc
    # axs[0].legend(ncol=2, loc='upper right', title=f"Premium: {params['ttf_premium']:2.0f} €/MWh")
    axs[0].legend(ncol=2, loc='lower left')
    # axs[0].legend(loc='lower left')
    # axs[0].text(-bar_width/2, 0.95*y_lim[1], f"Premium: {params['ttf_premium']:2.0f} €/MWh", ha='left', va='top', ma='left')
    # axs[0].set_xlabel(f"Degree of knowledge")
    axs[0].set_ylabel(f"Profit {params['year']} (M€)")
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0].set(xlim=x_lim, xticks=x_values, xticklabels=names, ylim=y_lim)
    axs[0].tick_params(axis='x', rotation=45, pad=-1)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profit_split_per_strategy_ttf_mfrr", 'pdf')


def plot_profit_split_per_premium(overwrite=False): 

    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Convert the split colors to rgba and then to rgb
    h2_color_spot = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split1'], alpha=0.5))
    h2_color_mfrr = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split1'], alpha=0.25))
    ch4_color_spot = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split2'], alpha=0.5))
    ch4_color_mfrr = mpl_helper.rgba_to_rgb(mpl_helper.hex_to_rgba(config['figures']['colors']['split2'], alpha=0.25))
    dark_gray = 0.25*np.ones(3)

    # Get results for wanted parameters
    params = config['default_params'].copy()
    params['mfrr'] = True
    params_list = utils.get_ttf_premium_params(params)
    df = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

    profits_h2_spot = df['spot_x_profit'].values
    profits_h2_spot = profits_h2_spot*1e-6
    profits_ch4_spot = df['spot_y_profit'].values
    profits_ch4_spot = profits_ch4_spot*1e-6
    profits_h2_mfrr = df['mfrr_x_profit'].values
    profits_h2_mfrr = profits_h2_mfrr*1e-6
    profits_ch4_mfrr = df['mfrr_y_profit'].values
    profits_ch4_mfrr = profits_ch4_mfrr*1e-6
    premiums = df['ttf_premium'].values
    y_lim = [0, 1.2*(profits_h2_spot+profits_ch4_spot+profits_h2_mfrr+profits_ch4_mfrr).max()]
    text_x_pad = 0.025*max(premiums)

    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    # Hydrogen and methane derived profits
    axs[0].fill_between(premiums, 0, profits_h2_spot, color=h2_color_spot, ec=dark_gray, label="$\mathrm{H}_2$ spot")
    axs[0].fill_between(premiums, profits_h2_spot, profits_h2_spot+profits_h2_mfrr, color=h2_color_mfrr, ec=dark_gray, label="$\mathrm{H}_2$ mFRR")
    axs[0].fill_between(premiums, profits_h2_spot+profits_h2_mfrr, profits_h2_spot+profits_h2_mfrr+profits_ch4_spot, color=ch4_color_spot, ec=dark_gray, label="$\mathrm{HC}_4$ spot")
    axs[0].fill_between(premiums, profits_h2_spot+profits_h2_mfrr+profits_ch4_spot, profits_h2_spot+profits_h2_mfrr+profits_ch4_spot+profits_ch4_mfrr, color=ch4_color_mfrr, ec=dark_gray, label="$\mathrm{HC}_4$ mFRR")
    axs[0].axvline(params['ttf_premium'], c='k', ls=':')
    axs[0].text(0, y_lim[1], f"Strategy: {utils.get_strategy_abbreviation(params)}", ha='left', va='top', ma='left')
    axs[0].text(params['ttf_premium']+text_x_pad, y_lim[1], 'Green\npremium', va='top', ha='left')

    # Legends, labels, ticks, etc
    # axs[0].legend(ncol=2, loc='lower right')
    # axs[0].legend(ncol=2, loc='upper right', title=f"Strategy: {params['horizon']/24:1.0f}d hor.")
    # axs[0].legend(loc='upper right')
    axs[0].set(ylim=y_lim)
    axs[0].set_xlabel("TTF premium (€/MWh)")
    axs[0].set_ylabel(f"Profit {params['year']} (M€)")
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"profit_split_per_premium_ttf_mfrr", 'pdf')


def plot_strategy(spot_data, ttf_data, params, premium=None, overwrite=False):

    params_tmp = params.copy()
    params_tmp['ttf'] = True
    params_tmp['mfrr'] = True
    params_tmp = utils.set_price_limits_and_opp_costs(params_tmp)

    if premium is not None:
        params_tmp['ttf_premium'] = premium

    # Get the strategy and the profit
    df = cache.get_summary_for_strategies([params_tmp], overwrite)
    strategy = cache.load_strategies(df)[0]
    
    # Plot the strategy
    config = get_config()
    leg_title = f"TTF {params_tmp['ttf_premium']:1.0f} €/MWh"
    fig, axs = plot_running_strategy(spot_data, strategy, params_tmp, ttf_data=ttf_data)
    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"moving_horizon_mfrr_ttf_{params_tmp['ttf_premium']:1.0f}", 'pdf')
    