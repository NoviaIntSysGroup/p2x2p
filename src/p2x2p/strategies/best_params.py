import json
import hashlib
import numpy as np
import pandas as pd

from p2x2p.strategies import cache
from p2x2p.report_figures import mpl_helper, utils
from p2x2p.utils.utils import get_config

# Define which parameters that require should trigger a new grid search
PRICE_LIMIT_PARAMS = ['n_days', 'k_p2x', 'k_x2p']
PRICE_LIMIT_AFFECTING_PARAMS = ['year', 'power_p2x', 'power_x2p', 'eff_p2x', 'eff_x2p', 'storage_size']
OPPORTUNITY_COST_PARAMS = ['kappa_p2x', 'kappa_x2p']
OPPORTUNITY_COST_AFFECTING_PARAMS = ['year', 'power_p2x', 'power_x2p', 'eff_p2x', 'eff_x2p', 'storage_size', 'ttf', 'ttf_premium']


def hash_params(params):
    """ Generate a hash for the parameters """
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_string.encode()).hexdigest()


def get_price_limit_affecting_params(params):
    """ Get the parameters that affect the price limits """
    return {key: params[key] for key in PRICE_LIMIT_AFFECTING_PARAMS}


def get_opportunity_cost_affecting_params(params):
    """ Get the parameters that affect the opportunity costs """
    return {key: params[key] for key in OPPORTUNITY_COST_AFFECTING_PARAMS}


def set_upper_bound_price_limits(params, overwrite=False):
    """ Set the best price limit for the given parameters """
    best_params = get_ideal_price_limit_params(params, overwrite=overwrite)
    params.update(best_params)
    return params


def set_lower_bound_price_limits(params, overwrite=False):
    """ Set the best price limit for the given parameters and the previous year"""
    params_copy = params.copy()
    params_copy['year'] = params['year'] - 1
    best_params = get_ideal_price_limit_params(params_copy, overwrite=overwrite)
    params.update(best_params)
    return params


def set_upper_bound_opportunity_costs(params, overwrite=False):
    """ Set the best opportunity costs for the given parameters """

    best_params = get_ideal_opportunity_costs(params, overwrite=overwrite)
    params.update(best_params)
    return params


def set_lower_bound_opportunity_costs(params, overwrite=False):
    """ Set the best opportunity costs for the given parameters and the previous year """

    params_copy = params.copy()
    params_copy['year'] = params['year'] - 1
    best_params = get_ideal_opportunity_costs(params_copy, overwrite=overwrite)
    params.update(best_params)
    return params


# PRICE LIMITS
def precompute_best_price_limits(overwrite=False):
    """ Precompute the best price limit parameters for the given parameters """

    config = get_config()
    params = config['default_params'].copy()

    # Precompute the best price limit parameters for each year with default parameters
    years = range(2016, 2024)
    for year in years:
        params['year'] = year
        best_params = get_ideal_price_limit_params(params, overwrite=overwrite, save_img=True)
        print(f"Best parameters for year {year}: {best_params}")

    # Precompute the best price limit parameters for various storage sizes for the previous two years
    years = range(2022, 2024)
    for year in years:
        params['year'] = year
        params_storage_all = utils.get_storage_size_params(params)
        for params_storage in params_storage_all:
            best_params = get_ideal_price_limit_params(params_storage, overwrite=overwrite)
            print(f"Best parameters for year {year} and storage size {params_storage['storage_size']}: {best_params}")

    # Precompute the best price limit parameters for various P2X and X2P powers for the previous two years
    for year in years:
        params['year'] = year
        params_p2x_x2p_all = utils.get_p2x_vs_x2p_params(params)
        for params_p2x_x2p in params_p2x_x2p_all:
            best_params = get_ideal_price_limit_params(params_p2x_x2p, overwrite=overwrite)
            print(f"Best parameters for year {year} and  P2X {params_p2x_x2p['power_p2x']} and X2P {params_p2x_x2p['power_x2p']}: {best_params}")


def get_ideal_price_limit_params(params, overwrite=False, save_img=False):
    """ Get the best price limit parameters for the given parameters """

    config = get_config()

    # Load or create a DataFrame for keeping a records of the best price limit parameters
    best_params_file = config['data_paths']['best_price_limits']
    try:
        best_param_values = pd.read_csv(best_params_file, index_col='hash_index')
    except:
        print(f"Creating new file: {best_params_file}")
        best_param_values = pd.DataFrame(columns=['hash_index'] + PRICE_LIMIT_AFFECTING_PARAMS + PRICE_LIMIT_PARAMS)
        best_param_values.set_index('hash_index', inplace=True)

    # Get the parameters that affect the price limits
    affecting_params = get_price_limit_affecting_params(params)
    hash_index = hash_params(affecting_params)
    # Check if the best parameters are already in the DataFrame
    if (hash_index in best_param_values.index) and not overwrite:
        best_params = best_param_values.loc[hash_index, PRICE_LIMIT_PARAMS].to_dict()
    # If not, compute the best parameters
    else:
        # Compute the best parameters values if they are not already in the DataFrame
        best_params = search_for_optimal_price_limits(params, overwrite=overwrite, save_img=save_img)
         # Add the found best_params to the param_values DataFrame and resave it
        best_param_values.loc[hash_index] = affecting_params | best_params
        best_param_values.to_csv(best_params_file, index=True)

    # Return the best price limit parameters
    return best_params


def search_for_optimal_price_limits(params, overwrite=False, save_img=False):

    # Copy and reset parameters that should not be used during the grid search
    params_copy = params.copy()
    params_copy['name'] = 'no_horizon'
    params_copy['mfrr'] = False
    params_copy['ttf'] = False
    params_copy['kappa_p2x'] = 0.
    params_copy['kappa_x2p'] = 0.

    print(f'Grid search for optimal price limits...')
    n_days, k_p2x, k_x2p = grid_search_for_optimal_price_limits(params_copy, overwrite=overwrite, save_img=save_img)

    return {'n_days': n_days, 'k_p2x': k_p2x, 'k_x2p': k_x2p}


def grid_search_for_optimal_price_limits(params, overwrite=False, save_img=False):

        config = get_config()

        # Define the range for each parameter
        n_days_range = np.round(np.arange(5., 31, 5))
        k_p2x_range = np.round(np.arange(0.2, 0.75, 0.1), 1)
        k_x2p_range = np.round(np.arange(1.5, 2.05, 0.1), 1)

        params_list = []
        for n_days in n_days_range:
            for k_p2x in k_p2x_range:
                for k_x2p in k_x2p_range:
                    params_tmp = params.copy()
                    params_tmp['n_days'] = n_days
                    params_tmp['k_p2x'] = k_p2x
                    params_tmp['k_x2p'] = k_x2p
                    params_tmp['save'] = False
                    params_list.append(params_tmp)

        df = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

        profits = np.array(df['profit']).reshape([n_days_range.size, k_p2x_range.size, k_x2p_range.size]) * 1e-6
        max_profit = profits.max()
        levels_percent = np.arange(0.50, 1.01, 0.05)
        levels = levels_percent * max_profit

        # Extract the est values for n_days
        profit_n_days = profits.max(axis=1).max(axis=1)
        best_n_days_idx = np.argmax(profit_n_days)
        best_n_days = n_days_range[best_n_days_idx]
        # Extract the best values for k_p2x and k_x2p
        row_max_val, col_max_val = np.unravel_index(np.argmax(profits[best_n_days_idx, :, :]),
                                                    profits[best_n_days_idx, :, :].shape)
        best_k_p2x = k_p2x_range[row_max_val]
        best_k_x2p = k_x2p_range[col_max_val]

        if save_img:
            # Set up the figure window
            mpl_helper.set_article_style()
            hor_ver_sep_cm = [0, 1.25]
            fig_size_cm, plot_rect_cm = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5, scale_h=2)
            fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, [2, 1], hor_ver_sep_cm)

            # TOP PANEL
            # Plot levels
            for level, level_p in zip(levels[::2], levels_percent[::2]):
                axs[0].plot([n_days_range[0], n_days_range[-1] + 5], np.ones(2) * level, 'k:')
                axs[0].text(n_days_range[-1] + 5, level, f"{level_p * 100:2.0f} %", va='bottom', ha='right')
            # Profits on top
            axs[0].plot(n_days_range, profit_n_days, 'ko-',
                        label=f"Storage size: {params['storage_size']:1.0f} MWh")
            # Highlight the default value for n_days
            axs[0].plot(best_n_days, profit_n_days[best_n_days_idx], 'ko-', ms=8, mfc='none')
            # Labels and title
            axs[0].legend(loc='lower left')
            axs[0].set_xlabel('$n_\mathrm{days}$')
            axs[0].set_ylabel('Total profit (M€)')

            # BOTTOM PANEL
            # Create a contour plot highlighting the best values for k_p2x and k_x2p
            contour = axs[1].contourf(k_x2p_range, k_p2x_range, profits[best_n_days_idx, :, :], levels=levels,
                                    vmax=max_profit * 0.95, cmap='gist_heat')
            # axs[1].plot(params['k_x2p'], params['k_p2x'], 'o', color='black', ms=6, mfc='none')
            axs[1].plot(best_k_x2p, best_k_p2x, 'o', color='black', ms=6, mfc='none')
            axs[1].set_xlabel('$k_\mathrm{X2P}$')
            axs[1].set_ylabel('$k_\mathrm{P2X}$')
            # Add a color bar
            cbar = fig.colorbar(contour)
            cbar.set_ticks([level for level in levels[::2]])
            cbar.set_ticklabels([f"{int(level * 100)} %" for level in levels_percent[::2]])

            # Save the figure
            strategy_name = params['name']
            mpl_helper.save_figure(fig, config['project_paths'][
                'mpl_figures'] + f'price_limit_grid_search_{params["year"]}_{strategy_name}', 'pdf')

        return [best_n_days, best_k_p2x, best_k_x2p]


# OPPORTUNITY COSTS
def precompute_best_opportunity_costs(overwrite=False):
    """ Precompute the best price limit parameters for the given parameters """

    config = get_config()
    params = config['default_params'].copy()
    params['mfrr'] = True

    # Precompute the best opportunity cost parameters for each year without TTF
    years = range(2016, 2024)
    for ttf in [False, True]:
        params['ttf'] = ttf
        for strategy in ['no_horizon', 'moving_horizon']:
            params['name'] = strategy
            for year in years:
                params['year'] = year
                params = set_upper_bound_price_limits(params, overwrite=overwrite)
                best_params = get_ideal_opportunity_costs(params, overwrite=overwrite, save_img=True)
                print(f"{params['name']}, best parameters for year {year} with TTF {ttf}: {best_params}")

    # Precompute the best opportunity cost parameters for various TTF premiums for the previous two years
    years = range(2022, 2024)
    params['ttf'] = True
    params['name'] = 'no_horizon'
    for year in years:
        params['year'] = year
        params_ttf_all = utils.get_ttf_premium_params(params)
        for params_ttf in params_ttf_all:
            params_ttf = set_upper_bound_price_limits(params_ttf, overwrite=overwrite)
            best_params = get_ideal_opportunity_costs(params_ttf, overwrite=overwrite)
            print(f"{params['name']}, best parameters for year {year} with TTF premium {params_ttf['ttf_premium']}: {best_params}")

    # Precompute the best opportunity cost parameters for various storage sizes for the previous two years
    years = range(2022, 2024)
    for ttf in [False, True]:
        params['ttf'] = ttf
        for strategy in ['no_horizon', 'moving_horizon']:
            params['name'] = strategy
            for year in years:
                params['year'] = year
                params_size_all = utils.get_storage_size_params(params)
                for params_size in params_size_all:
                    params_size = set_upper_bound_price_limits(params_size, overwrite=overwrite)
                    best_params = get_ideal_opportunity_costs(params_size, overwrite=overwrite)
                    print(f"{params_size['name']}, best parameters for year {year} with storage size {params_size['storage_size']}: {best_params}")

    # Precompute the best opportunity cost parameters for various P2X and X2P powers for the previous two years
    years = range(2022, 2024)
    for ttf in [False, True]:
        params['ttf'] = ttf
        for strategy in ['no_horizon', 'moving_horizon']:
            params['name'] = strategy
            for year in years:
                params['year'] = year
                params_p2x_x2p_all = utils.get_p2x_vs_x2p_params(params)
                for params_p2x_x2p in params_p2x_x2p_all:
                    params_p2x_x2p = set_upper_bound_price_limits(params_p2x_x2p, overwrite=overwrite)
                    best_params = get_ideal_opportunity_costs(params_p2x_x2p, overwrite=overwrite)
                    print(f"{params_p2x_x2p['name']}, best parameters for year {year}, TTF {ttf}, P2X {params_p2x_x2p['power_p2x']}, and X2P {params_p2x_x2p['power_x2p']}: {best_params}")


def get_ideal_opportunity_costs(params, overwrite=False, save_img=False):
    """ Get the best opportunity cost parameters for the given parameters """

    config = get_config()

    # Load or create a DataFrame for keeping a records of the best price limit parameters
    if params['name'] == 'no_horizon':
        best_params_file = config['data_paths']['best_opp_costs_no_horizon']
    else:
        best_params_file = config['data_paths']['best_opp_costs_moving_horizon']
    try:
        best_param_values = pd.read_csv(best_params_file, index_col='hash_index')
    except:
        print(f"Creating new file: {best_params_file}")
        best_param_values = pd.DataFrame(columns=['hash_index'] + OPPORTUNITY_COST_AFFECTING_PARAMS + OPPORTUNITY_COST_PARAMS)
        best_param_values.set_index('hash_index', inplace=True)

    # Get the parameters that affect the price limits
    affecting_params = get_opportunity_cost_affecting_params(params)
    hash_index = hash_params(affecting_params)
    # Check if the best parameters are already in the DataFrame
    if (hash_index in best_param_values.index) and not overwrite:
        best_params = best_param_values.loc[hash_index, OPPORTUNITY_COST_PARAMS].to_dict()
    # If not, compute the best parameters
    else:
        # Compute the best parameters values if they are not already in the DataFrame
        best_params = search_for_opportunity_costs(params, overwrite=overwrite, save_img=save_img)
         # Add the found best_params to the param_values DataFrame and resave it
        best_param_values.loc[hash_index] = affecting_params | best_params
        best_param_values.to_csv(best_params_file, index=True)

    # Return the best price limit parameters
    return best_params


def search_for_opportunity_costs(params, overwrite=False, save_img=False):

    print(f'Grid search for optimal opportunity costs...')
    if params['name'] == 'no_horizon':
        wide_search = False
    else:
        wide_search = False
    kappa_p2x, kappa_x2p = grid_search_for_optimal_opportunity_costs(params, overwrite=overwrite, save_img=save_img, wide_search=wide_search)

    return {'kappa_p2x': kappa_p2x, 'kappa_x2p': kappa_x2p}


def grid_search_for_optimal_opportunity_costs(params, overwrite=False, save_img=False, wide_search=False):

        config = get_config()

        # Define the range for each parameter
        if wide_search:
            kappa_p2x_range = np.round(np.linspace(-0.8, 0.8, 17), 1)
            kappa_x2p_range = np.round(np.linspace(-0.8, 0.8, 17), 1)
        else:
            kappa_p2x_range = np.round(np.linspace(-0.5, 0., 6), 1)
            kappa_x2p_range = np.round(np.linspace(0, 0.5, 6), 1)

        params_list = []
        for k_p2x in kappa_p2x_range:
            for k_x2p in kappa_x2p_range:
                params_tmp = params.copy()

                params_tmp['kappa_p2x'] = k_p2x
                params_tmp['kappa_x2p'] = k_x2p
                params_tmp['save'] = False
                params_list.append(params_tmp)

        df = cache.get_summary_for_strategies(params_list, overwrite=overwrite)
        profits = df['profit'].values * 1e-6
        profits = np.reshape(profits, (len(kappa_p2x_range), len(kappa_x2p_range)))
        max_profit = profits.max()

        filter_lim = 0.5
        filtered_profits = profits.copy() / max_profit
        filtered_profits[filtered_profits < filter_lim] = np.nan
        levels_percent = np.arange(filter_lim, 1.01, 0.05)

        # Extract the best values for kappa_p2x and kappa_x2p
        row_max_val, col_max_val = np.unravel_index(np.argmax(profits), profits.shape)
        best_kappa_p2x = kappa_p2x_range[row_max_val]
        best_kappa_x2p = kappa_x2p_range[col_max_val]

        if save_img:
            # Set up the figure window
            mpl_helper.set_article_style()
            fig_size_cm, plot_rect_cm = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
            fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

            # Create a pseudo colored matrix plot
            c = axs[0].pcolor(kappa_x2p_range, kappa_p2x_range, filtered_profits, vmin=filter_lim, vmax=1.0, cmap='gist_heat')
            axs[0].plot(best_kappa_x2p, best_kappa_p2x, 'o', color='black', ms=6, mfc='none')

            # Add a color bar
            cbar = fig.colorbar(c)
            cbar.set_ticks([level for level in levels_percent[::2]])
            cbar.set_ticklabels([f"{int(level * 100)} %" for level in levels_percent[::2]])

            axs[0].set_xlabel('$\kappa_\mathrm{X2P}$')
            axs[0].set_ylabel('$\kappa_\mathrm{P2X}$')

            # Save the figure
            strategy_name = params['name']
            mpl_helper.save_figure(fig, config['project_paths'][
                'mpl_figures'] + f'opportunity_cost_grid_search_{params["year"]}_{strategy_name}', 'pdf')

        return [best_kappa_p2x, best_kappa_x2p]
