import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as patches

from p2x2p.utils.utils import get_config
from p2x2p.report_figures import mpl_helper
from p2x2p.data import spot, fingrid


def plot_all_panels():

    config = get_config()
    params = config['default_params']

    # Get the spot data
    spot_data = spot.get_data()
    spot_data['mean_price'] = spot_data['elspot-fi'].ewm(int(24*params['n_days'])).mean().values
    mfrr_data = fingrid.get_mfrr_data()

    plot_monthly_factions(mfrr_data)
    plot_sequence_lengths(mfrr_data)
    plot_mfrr_vs_spot(spot_data, mfrr_data)
    plot_mfrr_vs_spot_avg(spot_data, mfrr_data)
    plot_mfrr_vol(mfrr_data)
    plot_mfrr_price_vs_vol(mfrr_data)


def get_n_consecutive_probs(boolean_sequence, n_values):
    # Pad the sequence to handle edge cases correctly
    padded_boolean_sequence = np.pad(boolean_sequence, (1, 1), mode='constant', constant_values=False)
    # Check when changes occur and get start and end indices
    changes = np.diff(padded_boolean_sequence.astype(int))
    starts = np.where(changes == 1)[0]  # Start indices
    ends = np.where(changes == -1)[0]  # End indices
    lengths = ends - starts

    # Count the number of sequences for each sequence length
    counts = []
    for n in n_values:
        count = np.sum(lengths == n)
        counts.append(count)
    counts = np.array(counts)
    # Normalize to get a probability distribution
    probs = counts / counts.sum()

    return probs


def get_n_consecutive_probs_theory(p_wanted, n_values):
    # Count the probability for sequences of varying length
    # assuming independent samples.
    p_other = 1 - p_wanted
    probs = []
    for n in n_values:
        probs.append(p_wanted**n * p_other**2)
    probs = np.array(probs)
    # Normalize to get a probability distribution
    probs = probs / probs.sum()

    return probs


def plot_monthly_factions(mfrr_data):

    config = get_config()

    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Monthly fracs
    monthly_groups = mfrr_data.resample('ME', on='date')
    monthly_data = []
    # Loop through each month's data
    for name, group in monthly_groups:
        month_data = {
            'Month': name,
            'up_frac': np.mean(group['up_regulation_volyme_mFRR'] > 0),
            'down_frac': np.mean(group['down_regulation_volyme_mFRR'] < 0),
            'up_volyme': group['up_regulation_volyme_mFRR'].sum(),
            'down_volyme': group['down_regulation_volyme_mFRR'].sum(),
        }
        monthly_data.append(month_data)

    # Create a DataFrame from the list of dictionaries
    monthly_data = pd.DataFrame(monthly_data)

    # Get the figure configuration
    fig_size_cm, plot_rect_cm = mpl_helper.get_fig_and_plot_size(config)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    h_sep = config['figures']['margins']['left'] + config['figures']['margins']['right']
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, grid_size=[1, 2], hor_ver_sep_cm=[h_sep, 0])

    axs[0].plot(monthly_data['Month'], monthly_data['up_frac'], color=up_color, label=mfrr_up_label)
    axs[0].plot(monthly_data['Month'], monthly_data['down_frac'], color=down_color, label=mfrr_down_label)
    axs[0].set(ylabel='Monthly frac.')

    axs[1].plot(monthly_data['Month'], monthly_data['up_volyme'] * 1e-3, color=up_color, label=mfrr_up_label)
    axs[1].plot(monthly_data['Month'], monthly_data['down_volyme'] * 1e-3, color=down_color, label=mfrr_down_label)
    axs[1].set(ylabel='Volume (GWh)')

    for i in range(2):
        axs[i].legend(loc='upper left')
        axs[i].xaxis.set_major_locator(mdates.YearLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Get labels and set visibility to False for every second label
        labels = axs[i].get_xticklabels()
        for label in labels[1::2]:
            label.set_visible(False)

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_monthly_fracs", 'pdf')


def plot_sequence_lengths(mfrr_data):

    config = get_config()

    n_values = np.arange(1, 41)
    up_regulation = mfrr_data['up_regulation_volyme_mFRR'].values > 0
    p_up = np.mean(up_regulation)
    p_up_n_theory = get_n_consecutive_probs_theory(p_up, n_values)
    p_up_n = get_n_consecutive_probs(up_regulation, n_values)

    down_regulation = mfrr_data['down_regulation_volyme_mFRR'] < 0
    p_down = np.mean(up_regulation)
    p_down_n_theory = get_n_consecutive_probs_theory(p_down, n_values)
    p_down_n = get_n_consecutive_probs(down_regulation, n_values)

    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    down_color_alpha = mpl_helper.hex_to_rgba(down_color, 0.5)
    up_color_alpha = mpl_helper.hex_to_rgba(up_color, 0.5)
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Get the figure configuration
    fig_size_cm, plot_rect_cm = mpl_helper.get_fig_and_plot_size(config)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    h_sep = config['figures']['margins']['left'] + config['figures']['margins']['right']
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, grid_size=[1, 2], hor_ver_sep_cm=[h_sep, 0])

    n_max = 10
    axs[0].bar(n_values[:n_max], p_up_n[:n_max], 1, fc=up_color_alpha, ec='none', label=mfrr_up_label + ' data')
    axs[0].plot(n_values[:n_max], p_up_n_theory[:n_max], color=up_color, label='Ind. process')
    axs[0].set(xlabel='n consecutive hours', ylabel='Prob. dens.')
    axs[0].legend(loc='upper right')

    axs[1].bar(n_values[:n_max], p_down_n[:n_max], 1, fc=down_color_alpha, ec='none', label=mfrr_down_label + ' data')
    axs[1].plot(n_values[:n_max], p_down_n_theory[:n_max], color=down_color, label='Ind. process')
    axs[1].set(xlabel='n consecutive hours', ylabel='Prob. dens.')
    axs[1].legend(loc='upper right')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_sequence_lengths", 'pdf')


def plot_mfrr_vs_spot(spot_data, mfrr_data):

    config = get_config()

    mfrr_scaling = 1e-3
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    down_color_alpha = mpl_helper.hex_to_rgba(down_color, 0.5)
    up_color_alpha = mpl_helper.hex_to_rgba(up_color, 0.5)
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Extract data
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values

    delta_up = mfrr_up_price-spot_price
    delta_down = mfrr_down_price-spot_price

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    # LEFT PLOT
    axs[0].plot(spot_price[delta_up>0], delta_up[delta_up>0]*mfrr_scaling, 'o', c=up_color_alpha, mec='none', label=mfrr_up_label)
    axs[0].plot(spot_price[delta_down<0], delta_down[delta_down<0]*mfrr_scaling, 'o', c=down_color_alpha, mec='none', label=mfrr_down_label)
    axs[0].set(xlabel='Spot price (€/MWh)', ylabel='$\Delta_\mathrm{Spot}$ (k€/MWh)')
    axs[0].legend(loc='upper left')


    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_vs_spot", 'png')


def get_bin_means_and_sems(data_x, data_y, bin_edges):

    bin_indices = np.digitize(data_x, bin_edges)

    bin_means = []
    bin_sems = []
    bin_fracs = []
    bin_fracs_sems = []
    for idx in range(1, bin_edges.size):
        y_tmp = data_y[bin_indices == idx]
        y_tmp_nonzero = y_tmp[y_tmp != 0]
        if y_tmp_nonzero.size <= 2:
            bin_means.append(np.nan)
            bin_sems.append(np.nan)
            bin_fracs.append(np.nan)
            bin_fracs_sems.append(np.nan)
        else:
            p_tmp = y_tmp_nonzero.size/y_tmp.size
            bin_means.append(y_tmp_nonzero.mean())
            bin_sems.append(y_tmp_nonzero.std()/np.sqrt(y_tmp_nonzero.size))
            bin_fracs.append(p_tmp)
            bin_fracs_sems.append(np.sqrt(p_tmp*(1-p_tmp)/y_tmp.size))

    bin_means = np.array(bin_means)
    bin_sems = np.array(bin_sems)

    return bin_means, bin_sems, bin_fracs, bin_fracs_sems


def plot_mfrr_vs_spot_avg(spot_data, mfrr_data):

    config = get_config()

    mfrr_scaling = 1e-3
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Extract data
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values

    delta_up = (mfrr_up_price-spot_price)*mfrr_scaling
    delta_down = (mfrr_down_price-spot_price)*mfrr_scaling

    bin_width = 50
    bin_edges = np.arange(-75, 1000, bin_width)
    bin_centers = bin_edges[:-1] + bin_width/2
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    up_bin_means, up_bin_sems, up_bin_fracs, up_bin_fracs_sems = get_bin_means_and_sems(spot_price, delta_up, bin_edges)
    down_bin_means, down_bin_sems, down_bin_fracs, down_bin_fracs_sems = get_bin_means_and_sems(spot_price, delta_down, bin_edges)

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)

    axs[0].errorbar(bin_centers, up_bin_means, yerr=up_bin_sems, fmt='o', color=up_color, label=mfrr_up_label)
    axs[0].errorbar(bin_centers, down_bin_means, yerr=down_bin_sems, fmt='o', color=down_color, label=mfrr_down_label)
    axs[0].legend(loc='upper left')
    axs[0].set(xlabel='Spot price (€/MWh)', ylabel='$\Delta_\mathrm{Spot}$ (k€/MWh)')
    axs[0].yaxis.labelpad = -3

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_vs_spot_avg", 'pdf')


def plot_mfrr_vs_spot_zoom(spot_data, mfrr_data):

    config = get_config()

    mfrr_scaling = 1e-3
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    zoom_color = config['figures']['colors']['zoom']
    down_color_alpha = mpl_helper.hex_to_rgba(down_color, 0.5)
    up_color_alpha = mpl_helper.hex_to_rgba(up_color, 0.5)
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Extract data
    spot_price = spot_data['elspot-fi'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values

    delta_up = mfrr_up_price-spot_price
    delta_down = mfrr_down_price-spot_price

    # Define the zoomed in region
    zoomed_xy = (-50, -800*mfrr_scaling)
    zoomed_width = 700
    zoomed_height = 2000*mfrr_scaling

    # Get the figure configuration
    hor_sep_cm = 1.2
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm, grid_size=[1, 2], hor_ver_sep_cm=[hor_sep_cm, 0], rel_col_widths=[0.5, 0.5])

    # LEFT PLOT
    axs[0].plot(spot_price[delta_up>0], delta_up[delta_up>0]*mfrr_scaling, 'o', c=up_color_alpha, mec='none', label=mfrr_up_label)
    axs[0].plot(spot_price[delta_down<0], delta_down[delta_down<0]*mfrr_scaling, 'o', c=down_color_alpha, mec='none', label=mfrr_down_label)
    rect = patches.Rectangle(zoomed_xy, zoomed_width, zoomed_height, linewidth=1, edgecolor=zoom_color, facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set(xlabel='Spot price (€/MWh)', ylabel='$\Delta_\mathrm{Spot}$ (k€/MWh)')
    axs[0].legend(loc='upper left')

    # RIGHT PLOT
    x_mid = zoomed_xy[0]+zoomed_width/2
    x_lim_zoom = [zoomed_xy[0], zoomed_xy[0]+zoomed_width]
    y_lim_zoom = [zoomed_xy[1], zoomed_xy[1]+zoomed_height]
    y_ticks_zoom = [-500*mfrr_scaling, 0, 500*mfrr_scaling]
    rotation_text = np.arctan(
        (1/np.diff(y_lim_zoom)[0]*mfrr_scaling*axs[1].get_position().height*fig_size_cm[1]) /
        (1/np.diff(x_lim_zoom)[0]*axs[1].get_position().width*fig_size_cm[0])
        )*180/np.pi

    axs[1].plot(spot_price[delta_up>0], delta_up[delta_up>0]*mfrr_scaling, 'o', c=up_color_alpha, mec='none')
    axs[1].plot(spot_price[delta_down<0], delta_down[delta_down<0]*mfrr_scaling, 'o', c=down_color_alpha, mec='none')
    axs[1].text(x_mid, -x_mid*mfrr_scaling, mfrr_down_label + r'$ = 0$ €/MWh', ha='center', va='top', rotation=-rotation_text, rotation_mode='anchor', bbox={'facecolor': 'white', 'ec':'none', 'alpha': 0.5, 'pad': 0})
    axs[1].text(x_mid, (1000-x_mid)*mfrr_scaling, mfrr_up_label + r'$ = 1000$ €/MWh', ha='center', va='bottom', rotation=-rotation_text, rotation_mode='anchor', bbox={'facecolor': 'white', 'ec':'none', 'alpha': 0.5, 'pad': 0})
    axs[1].plot(x_lim_zoom, -1*np.array(x_lim_zoom)*mfrr_scaling, ':', c='black')
    axs[1].plot(x_lim_zoom, (-1*np.array(x_lim_zoom)+1000)*mfrr_scaling, ':', c='black')
    axs[1].set(xlim=x_lim_zoom, ylim=y_lim_zoom, yticks=y_ticks_zoom)
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

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_vs_spot_zoom", 'png')


def plot_mfrr_vol(mfrr_data):

    config = get_config()

    # Extract data
    down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values
    up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values

    max_volyme = 400
    bin_edges = np.arange(0, max_volyme+1, 25)
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].hist(up_volyme[up_volyme>0], bin_edges, density=True, color=up_color, alpha=0.5, label=mfrr_up_label)
    axs[0].hist(down_volyme[down_volyme<0], -np.flip(bin_edges), density=True, color=down_color, alpha=0.5, label=mfrr_down_label)
    axs[0].legend(loc='upper right')
    axs[0].set(xlabel='Volume (MWh/h)', ylabel='Prob. density', yticks=[])

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_overview_vol", 'pdf')


def plot_mfrr_price_vs_vol(mfrr_data):

    config = get_config()

    mfrr_scaling = 1e-3
    down_color = config['figures']['colors']['mfrr_down']
    up_color = config['figures']['colors']['mfrr_up']
    mfrr_down_label = config['latex']['mfrr']['down']
    mfrr_up_label = config['latex']['mfrr']['up']

    # Extract data
    down_volyme = mfrr_data['down_regulation_volyme_mFRR'].values
    up_volyme = mfrr_data['up_regulation_volyme_mFRR'].values
    mfrr_down_price = mfrr_data['down_regulating_price_mFRR'].values
    mfrr_up_price = mfrr_data['up_regulating_price_mFRR'].values

    # Get the figure configuration
    fig_size_cm, plot_rect_cm  = mpl_helper.get_fig_and_plot_size(config, scale_w=0.5)
    # Set up the figure and axes
    mpl_helper.set_article_style()
    fig, axs = mpl_helper.get_figure_win(fig_size_cm, plot_rect_cm)
    axs[0].plot(up_volyme[up_volyme>0], mfrr_up_price[up_volyme>0]*mfrr_scaling, 'o', color=up_color, mec='none', alpha=0.5, label=mfrr_up_label)
    axs[0].plot(down_volyme[down_volyme<0], mfrr_down_price[down_volyme<0]*mfrr_scaling, 'o', color=down_color, mec='none', alpha=0.5, label=mfrr_down_label)
    axs[0].legend(loc='upper left')
    axs[0].set(xlabel='Volume (MWh/h)', ylabel='mFRR (k€/MWh)')

    mpl_helper.save_figure(fig, config['project_paths']['mpl_figures'] + f"mfrr_overview_price_vs_vol", 'png')
