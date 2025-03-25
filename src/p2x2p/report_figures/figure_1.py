import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

from p2x2p.utils.utils import get_config
from p2x2p.report_figures import mpl_helper
from p2x2p.data import spot, fingrid


def plot_all_panels():

    # Get the configuration
    config = get_config()
    params = config['default_params']
    p2x_color = config['figures']['colors']['p2x']
    x2p_color = config['figures']['colors']['x2p']

    # Get the spot and energy data
    spot_data = spot.get_data()
    spot_data['mean_price'] = spot_data['elspot-fi'].ewm(int(24*params['n_days'])).mean().values
    spot_data_year = spot_data[spot_data['date'].dt.year == params['year']]
    energy_data = fingrid.get_energy_data()

    plot_prod_and_cap_data(energy_data)
    plot_prod_and_price_corr(spot_data, energy_data)
    plot_spot_data(spot_data_year)
    plot_spot_data_density(spot_data_year, p2x_color, x2p_color)
    plot_spot_data_violin(spot_data)


def add_ol3_line(axs, plot_rect_cm):

    # Parameters
    pad = 0.1   
    ol3_start_data = pd.Timestamp('2023-04-16')

    # Add the OL3 line and the label
    x_lim = axs.get_xlim()
    y_lim = axs.get_ylim()
    x_range = np.diff(x_lim)[0]/plot_rect_cm[2]
    y_range = np.diff(y_lim)[0]/plot_rect_cm[3]
    axs.axvline(x=ol3_start_data, color='black', linestyle=':', zorder=0)
    axs.text(
        ol3_start_data+pd.Timedelta(days=pad*x_range), 
        y_lim[0]+pad*y_range, 
        'OL3', ha='left', va='bottom'
        )


def plot_prod_and_cap_data(energy_data):

    months = []
    monthly_avg_production =[]
    monthly_avg_capacity = []

    for year in energy_data['date'].dt.year.unique():
        for month in range(1, 13):
            months.append(pd.Timestamp(year=year, month=month, day=1))
            monthly_avg_production.append(
                energy_data.loc[
                    (energy_data['date'].dt.year == year) & 
                    (energy_data['date'].dt.month == month), 'production_total_hourly'].mean())
            monthly_avg_capacity.append(
                energy_data.loc[
                    (energy_data['date'].dt.year == year) & 
                    (energy_data['date'].dt.month == month), 'capacity_wind_hourly'].mean())
        
    monthly_avg_production = np.array(monthly_avg_production)
    monthly_avg_capacity = np.array(monthly_avg_capacity)

    # Get the figure configuration
    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(
        config, scale_w=0.5)
    x_ticks = [months[0] + pd.DateOffset(years=2*i) for i in range(5)]
    x_tick_labels = [date.strftime('%Y') for date in x_ticks]

    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].plot(months, 
                monthly_avg_production/1e3, 
                'k-', label='Avg. tot. prod.')
    axs[0].plot(months,
                 monthly_avg_capacity/1e3, 
                 color='gray', label='Inst. wind cap.')
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, ylim=(0, 11))
    axs[0].legend(loc='lower left')
    axs[0].set_ylabel('Prod. | Cap. (GW)')

    add_ol3_line(axs[0], plot_rect_cm)

    mpl_helper.save_figure(
        fig, 
        config['project_paths']['mpl_figures'] + "tot_prod_and_wind_cap", 'pdf')
    

def plot_prod_and_price_corr(spot_data, energy_data):

    months = []
    monthly_avg_corr =[]

    for year in energy_data['date'].dt.year.unique():
        for month in range(1, 13):
            months.append(pd.Timestamp(year=year, month=month, day=1))
            monthly_prod_tmp = energy_data.loc[
                    (energy_data['date'].dt.year == year) & 
                    (energy_data['date'].dt.month == month), :]
            monthly_spot_tmp = spot_data.loc[
                    (spot_data['date'].dt.year == year) & 
                    (spot_data['date'].dt.month == month), :]
            
            monthly_avg_corr.append(
                    monthly_prod_tmp['production_wind_hourly'].corr(
                        monthly_spot_tmp['elspot-fi']))
            
    monthly_avg_corr = np.array(monthly_avg_corr)

    # Get the figure configuration
    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(
        config, scale_w=0.5)
    x_ticks = [months[0] + pd.DateOffset(years=2*i) for i in range(5)]
    x_tick_labels = [date.strftime('%Y') for date in x_ticks]

    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].plot(
        months, 
        monthly_avg_corr, 
        'k.', label='spot price vs. wind prod.')
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, ylim=[-0.8, 0])
    axs[0].legend(loc='lower left')
    axs[0].set_ylabel('Corr. coeff.')

    add_ol3_line(axs[0], plot_rect_cm)

    mpl_helper.save_figure(
        fig, 
        config['project_paths']['mpl_figures'] + 
        "wind_prod_and_price_corr", 'pdf')


def plot_spot_data(spot_data_year):

    dates = spot_data_year['date']
    spot_price = spot_data_year['elspot-fi'].values

    # Get the figure configuration
    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    x_ticks = [dates.iloc[0] + pd.DateOffset(months=3*i) for i in range(5)]
    x_tick_labels = [date.strftime('%Y-%m') for date in x_ticks]

    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].plot(dates, spot_price, label=f"Spot price", color='black')
    axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels)
    axs[0].set(ylabel='Price (€/MWh)')
    axs[0].legend(loc='best')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "spot_price", 'pdf')


def plot_spot_data_density(spot_data_year, buy_color, sell_color):

    spot_price = spot_data_year['elspot-fi'].values

    # Create a gaussian_kde object
    kde = gaussian_kde(spot_price)
    x = np.linspace(min(spot_price), max(spot_price), 100)
    density = kde(x)

    # Get the figure configuration
    config = get_config()
    buy_lim = 10
    sell_lim = 20
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].fill_between(x[:buy_lim+1], density[:buy_lim+1], fc=buy_color, alpha=0.5, label='Buy (P2X)')
    axs[0].fill_between(x[buy_lim:sell_lim+1], density[buy_lim:sell_lim+1], fc='k', alpha=0.5, label='Idle')
    axs[0].fill_between(x[sell_lim:], density[sell_lim:], fc=sell_color, alpha=0.5, label='Sell (X2P)')
    axs[0].plot(x, density, color='black')
    axs[0].set(xlabel='Price (€/MWh)', ylabel='Prob. density', yticks=[])
    axs[0].legend(loc='best', title=str(spot_data_year['date'].dt.year.unique()[0]))

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "spot_price_density", 'pdf')


def plot_spot_data_violin(spot_data):

    available_years = spot_data['date'].dt.year.unique()
    price_per_year = [spot_data[spot_data['date'].dt.year == year]['elspot-fi'].values for year in available_years]

    # Get the figure configuration
    config = get_config()
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)

    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    parts = axs[0].violinplot(price_per_year, available_years)
    for pc in parts['bodies']:
        pc.set_facecolor(np.ones(3)*0.5)
        pc.set_alpha(1)
    parts['cmins'].set_color(np.zeros(3))
    parts['cmaxes'].set_color(np.zeros(3))
    parts['cbars'].set_color(np.zeros(3))
    axs[0].set(ylabel='Price (€/MWh)', yticks=np.arange(0, 1000, 300))

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + "spot_price_violin", 'pdf')
    