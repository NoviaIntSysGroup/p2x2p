import numpy as np

from p2x2p.utils.utils import get_config
from p2x2p.report_figures import mpl_helper, utils
from p2x2p.strategies import cache


def plot_all_panels(overwrite=False):

    config = get_config()
    params = config['default_params']
    y_lim = [0, 3]

    params['ttf'] = False
    params['mfrr'] = False
    margin_colors = [config['figures']['colors']['margin']]
    plot_p2x_vs_x2p_usage(params, margin_colors, y_lim, legend=True, last=False, overwrite=overwrite)
    plotflow_vs_storage_size(params, margin_colors, y_lim, legend=True, last=False, overwrite=overwrite)
    params['ttf'] = True
    params['mfrr'] = False
    margin_colors = [config['figures']['colors']['margin2']]
    plot_p2x_vs_x2p_usage(params, margin_colors, y_lim, last=False, overwrite=overwrite)
    plotflow_vs_storage_size(params, margin_colors, y_lim, last=False, overwrite=overwrite)
    params['ttf'] = False
    params['mfrr'] = True
    margin_colors = [config['figures']['colors']['margin3']]
    plot_p2x_vs_x2p_usage(params, margin_colors, y_lim, last=False, overwrite=overwrite)
    plotflow_vs_storage_size(params, margin_colors, y_lim, last=False, overwrite=overwrite)
    params['ttf'] = True
    params['mfrr'] = True
    margin_colors = [config['figures']['colors']['margin3'], config['figures']['colors']['margin2']]
    plot_p2x_vs_x2p_usage(params, margin_colors, y_lim, overwrite=overwrite)
    plotflow_vs_storage_size(params, margin_colors, y_lim, overwrite=overwrite)


def plot_p2x_vs_x2p_usage(params, margin_colors, y_lim, legend=False, last=True, overwrite=False):

    config = get_config()
    scaling = 1e-6
    power_tot = params['power_p2x'] + params['power_x2p']
    strategy_names = ['infinite_horizon', 'moving_horizon', 'no_horizon']
    margin_colors = [mpl_helper.hex_to_rgba(c, alpha=0.5) for c in margin_colors]
    fill_color = margin_colors[0]
    edge_color = None if len(margin_colors) < 2 else margin_colors[1]
    source_keys = ['spot', 'ttf', 'mfrr']
    sources = [key for key in source_keys if params[key]]

    # Get the results for each strategy
    names = []
    profits = []
    p2x_values = []
    for i, strategy_name in enumerate(strategy_names):

        # Get the parameters
        params_tmp = params.copy()
        params_tmp['name'] = strategy_name
        params_list = utils.get_p2x_vs_x2p_params(params_tmp)

        # Get the profits
        df_tmp = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

        # Extract the needed data
        names.append(utils.get_strategy_abbreviation(params_tmp))
        profits.append(df_tmp['profit'].values * scaling)
        p2x_values.append(df_tmp['power_p2x'].values)

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    if not last:
        fig_size_cm[1] -= 0.5
        plot_rect_cm[1] -= 0.5
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].axvline(x=params_tmp['power_p2x'], color='black', linestyle='dotted')
    # Plot the profit margin
    if len(margin_colors) > 1:
        axs[0].fill_between(p2x_values[0], profits[-1], profits[-2], color=fill_color, edgecolor=edge_color, hatch='|||', lw=1)
    else:
        axs[0].fill_between(p2x_values[0], profits[-1], profits[-2], color=fill_color)
    # Plot the profits
    for i, name in enumerate(names):
        axs[0].plot(p2x_values[i], profits[i], '-', c='k', marker=config['figures']['markers'][i], label=name)

    # Define ticks
    x_ticks = np.arange(5, 30, 5)
    x_tick_labels = [f'{x}|{int(power_tot)-x}' for x in x_ticks]
    if last:
        axs[0].set(xticks=x_ticks, xticklabels=x_tick_labels, xlabel='P2X | X2P power [MW]')
    else:
        axs[0].set(xticks=x_ticks, xticklabels=[])
    axs[0].set(ylim=y_lim, ylabel=f"Profit {params['year']} (M€)")
    if legend:
        axs[0].legend(loc='upper right')
    
    file_endings = '_'.join(sources)
    # Save as png if we have hatches as the pdf format struggles with transparency then
    if len(margin_colors) > 1:
        mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"p2x_vs_x2p_{file_endings}", 'png')
    else:
        mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"p2x_vs_x2p_{file_endings}", 'pdf')


def plotflow_vs_storage_size(params, margin_colors, y_lim, legend=False, last=True, overwrite=False):

    # Get the figure configuration
    config = get_config()
    scaling = 1e-6
    min_flux = min(params['power_p2x']*params['eff_p2x'], params['power_x2p']/params['eff_x2p'])
    strategy_names = ['infinite_horizon', 'moving_horizon', 'no_horizon']
    margin_colors = [mpl_helper.hex_to_rgba(c, alpha=0.5) for c in margin_colors]
    fill_color = margin_colors[0]
    edge_color = None if len(margin_colors) < 2 else margin_colors[1]
    source_keys = ['spot', 'ttf', 'mfrr']
    sources = [key for key in source_keys if params[key]]

    # Get the results for each strategy
    names = []
    profits = []
    storage_sizes = []
    for i, strategy_name in enumerate(strategy_names):

        # Get the parameters
        params_tmp = params.copy()
        params_tmp['name'] = strategy_name
        params_list = utils.get_storage_size_params(params_tmp)

        # Get the profits
        df_tmp = cache.get_summary_for_strategies(params_list, overwrite=overwrite)

        # Extract the needed data
        names.append(utils.get_strategy_abbreviation(params_tmp))
        profits.append(df_tmp['profit'].values * scaling)
        storage_sizes.append(df_tmp['storage_size'].values)
  
    storage_sizes_norm = [sizes / min_flux / 24 for sizes in storage_sizes]

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    if not last:
        fig_size_cm[1] -= 0.5
        plot_rect_cm[1] -= 0.5
        
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].axvline(x=params_tmp['storage_size'] / min_flux / 24, color='black', linestyle='dotted')
    # Plot the profit margin
    if len(margin_colors) > 1:
        axs[0].fill_between(storage_sizes_norm[0], profits[-1], profits[-2], color=fill_color, edgecolor=edge_color, hatch='|||', lw=1)
    else:
        axs[0].fill_between(storage_sizes_norm[0], profits[-1], profits[-2], color=fill_color)
    # Plot the profits
    for i, name in enumerate(names):
        axs[0].plot(storage_sizes_norm[i], profits[i], '-', c='k', marker=config['figures']['markers'][i], label=name)
        
    # Define ticks
    x_ticks = np.arange(0, 16, 3)
    if last:
        axs[0].set(xticks=x_ticks, xlabel='Normalized storage size [days]')
    else:
        axs[0].set(xticks=x_ticks, xticklabels=[])
    # axs[0].set(xscale='log')
    axs[0].set(ylim=y_lim, ylabel=f"Profit {params['year']} (M€)")
    # axs[0].set(ylim=y_lim)
    if legend:
        axs[0].legend(loc='upper right')

    file_endings = '_'.join(sources)
    if len(margin_colors) > 1:
        mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"flow_vs_storage_size_{file_endings}", 'png')
    else:
        mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"flow_vs_storage_size_{file_endings}", 'pdf')
