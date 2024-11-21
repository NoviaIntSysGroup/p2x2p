import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_article_style():
    """It sets the figure style intended for articles and reports"""
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.85
    mpl.rcParams['legend.edgecolor'] = (1, 1, 1, 0.5)
    mpl.rcParams['legend.handlelength'] = 1.0
    mpl.rcParams['legend.handletextpad'] = 0.7
    mpl.rcParams['legend.labelspacing'] = 0.4
    mpl.rcParams['legend.columnspacing'] = 1.0


def set_notebook_style():
    """It sets the figure style intended for notebooks"""
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 5


def get_default_colors():
    """
    It returns a list of colors that matplotlib uses by default

    :return: A list of colors.
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_pastel_colors():
    """
    It returns a list of 8 colors from the "Pastel2" color map

    :return: A list of 8 colors.
    """
    cmap = mpl.cm.get_cmap("Pastel2")
    colors = cmap(np.arange(8))
    return colors


def hex_to_rgba(hex_val, alpha=0):
    """
    It takes a hexadecimal color value and returns the corresponding RGB value

    :param hex_val: The hexadecimal value of the color you want to convert
    :return: A numpy array of the RGB values of the hex color code.
    """
    rgb = [int(hex_val.strip('#')[i:i + 2], 16)/255.0 for i in (0, 2, 4)]
    if alpha:
        rgb.append(alpha)
    return np.array(rgb)


def rgba_to_rgb(rgba_color, background_color=(1, 1, 1)):
    # Unpack the RGBA and background RGB values
    r, g, b, a = rgba_color
    rb, gb, bb = background_color

    # Calculate the new RGB values
    r_new = r * a + rb * (1 - a)
    g_new = g * a + gb * (1 - a)
    b_new = b * a + bb * (1 - a)

    return [r_new, g_new, b_new]


def get_figure_win(fig_size_cm,
                   plot_rect_cm,
                   grid_size=None,
                   hor_ver_sep_cm=None,
                   projection=None,
                   rel_col_widths=None,
                   rel_row_heights=None):
    """
    It creates a figure window with a grid of axes, where the size of the figure window and the axes are defined in
    centimeters

    :param fig_size_cm: the size of the figure window in cm
    :param plot_rect_cm: the rectangle in which the grid of axes will be placed
    :param grid_size: the number of rows and columns in the grid
    :param hor_ver_sep_cm: horizontal and vertical separation between axes (in cm)
    :param projection: the projection of the axes. If you want a 3D plot, you can set this to '3d'
    :param rel_col_widths: relative widths of the columns
    :param rel_row_heights: The relative heights of the rows in the grid
    :return: A figure and a list of axes.
    """
    if hor_ver_sep_cm is None:
        hor_ver_sep_cm = [0, 0]
    if grid_size is None:
        grid_size = [1, 1]
    if rel_col_widths is None:
        rel_col_widths = [1. / grid_size[1] for _ in range(grid_size[1])]
    if rel_row_heights is None:
        rel_row_heights = [1. / grid_size[0] for _ in range(grid_size[0])]

    # Convert the fig size to inches
    fig_size = [l / 2.54 for l in fig_size_cm]
    # Define the plotting rectangle containing the grid with all axes
    # cm values have to be converted to relative units (relative to the figure window)
    plot_rect = [plot_rect_cm[0] / fig_size_cm[0],
                 plot_rect_cm[1] / fig_size_cm[1],
                 plot_rect_cm[2] / fig_size_cm[0],
                 plot_rect_cm[3] / fig_size_cm[1], ]
    # Horizontal and vertical separation between axes (in relative units)
    hor_ver_sep = [hor_ver_sep_cm[i] / fig_size_cm[i] for i in range(2)]

    # Compute axes sizes in relative units
    axes_widths = [(plot_rect[2] - (grid_size[1] - 1) * hor_ver_sep[0]) * width for width in rel_col_widths]
    axes_heights = [(plot_rect[3] - (grid_size[0] - 1) * hor_ver_sep[1]) * height for height in rel_row_heights]

    # Create the figure window together with all axes in the grid
    fig = plt.figure(figsize=fig_size)
    axs = []
    # Initialize the bottom left y-coordinate
    axes_corner_y = plot_rect[1] + plot_rect[3]
    for i, row in enumerate(range(grid_size[0] - 1, -1, -1)):

        # Initialize the bottom left x-coordinate
        axes_corner_x = plot_rect[0]
        # Update the bottom left y-coordinate
        axes_corner_y -= axes_heights[i]

        for col in range(grid_size[1]):
            # Create an axis
            projection_tmp = projection[i*grid_size[0]+col] if projection else None
            ax_rect_tmp = [axes_corner_x, axes_corner_y, axes_widths[col], axes_heights[i]]
            axs.append(plt.axes(ax_rect_tmp, facecolor='none', projection=projection_tmp, aspect='auto'))
            axs[-1].set_position(ax_rect_tmp)

            # Update the bottom left x-coordinate
            axes_corner_x += axes_widths[col] + hor_ver_sep[0]

            if projection_tmp == '3d':
                axs[-1].w_xaxis.set_pane_color([1, 1, 1, 0])
                axs[-1].w_yaxis.set_pane_color([1, 1, 1, 0])
                axs[-1].w_zaxis.set_pane_color([1, 1, 1, 0])

        # Update the bottom left y-coordinate
        axes_corner_y -= hor_ver_sep[1]

    return fig, axs


def get_fig_and_plot_size(config, scale_w=1, scale_h=1):
        """
        It returns the figure size and the plot rectangle in centimeters

        :param config: The configuration dictionary
        :param scale_w: The width scaling factor (optional)
        :param scale_h: The height scaling factor (optional)
        :return: A list containing the figure size and the plot rectangle
        """
        
        # Define the figure size
        fig_size_cm = [
            config['figures']['page_width_cm']*scale_w, 
            config['figures']['panel_height_cm']*scale_h
            ]
        
        # Define the plot rectangle containing all axes
        plot_rect_cm  = [
            config['figures']['margins']['left'], 
            config['figures']['margins']['bottom'], 
            fig_size_cm[0] - config['figures']['margins']['left'] - config['figures']['margins']['right'], 
            fig_size_cm[1] - config['figures']['margins']['top'] - config['figures']['margins']['bottom']
            ]
        
        return fig_size_cm, plot_rect_cm


def get_axes_ontop(fig, ax):
    """
    It creates an axes object on top of the given axes object

    :param fig: The figure object
    :param ax: The axes object on top of which the new axes object will be created
    :return: The axes object on top of the given axes object
    """
    pos = ax.get_position()
    ax_overlay = fig.add_axes(pos)
    ax_overlay.set_facecolor('none')
    ax_overlay.spines['top'].set_visible(False)
    ax_overlay.spines['bottom'].set_visible(False)
    ax_overlay.spines['left'].set_visible(False)
    ax_overlay.spines['right'].set_visible(False)
    ax_overlay.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    return ax_overlay


def add_label_to_axes(ax, label):
    """
    It takes an axes object and a label, and prints the label in the top left corner of the axes

    :param ax: The axes object to which the label will be added
    :param label: The text to be printed
    """
    # Specify the label position from the top left corner of the axes
    label_pos_cm = [-1.2, 0.2]

    # Get the figure size in cm and convert the label pos to relative units
    fig_size_cm = ax.figure.get_size_inches() * 2.54
    label_pos = [label_pos_cm[i] / fig_size_cm[i] for i in range(2)]

    # Define the location for the label and print it
    ax_rect = ax.get_position()
    x_pos = ax_rect.x0 + label_pos[0]
    y_pos = ax_rect.y1 + label_pos[1]
    fig = plt.gcf()
    fig.text(x_pos, y_pos, label, weight='bold', fontsize='large', ha='left', va='top')


def save_figure(fig, file_name, format, dpi=600):
    """
    It saves a figure to a file

    :param fig: the figure object
    :param file_name: The name of the file to save the figure to
    :param format: the format of the figure to be saved
    :param dpi: dots per inch. This is the resolution of the image, defaults to 600 (optional)
    """
    figure_path = file_name + '.' + format
    if format == 'png':
        fig.savefig(figure_path, dpi=dpi)
    else:
        fig.savefig(figure_path)
