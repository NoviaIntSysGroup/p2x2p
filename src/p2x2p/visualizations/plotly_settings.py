import plotly.express as px

# Plotly settings
DPI = 600
PX_PER_INCH = 72  # Plotly appears to use 72 dpi, instead of 96 dpi
CM_PER_INCH = 2.54
FONTSIZE = 8
PNG_SCALING = DPI / PX_PER_INCH # scaling to get a dpi of 300 when saving images
PX_PER_CM = PX_PER_INCH / CM_PER_INCH


# Figure settings
fig_config = {
  'toImageButtonOptions': {
    'scale':PNG_SCALING
  }
}


def get_plotly_settings():

    # Layout settings
    margin = dict(
        l=1.2 * PX_PER_CM,
        r=0.5 * PX_PER_CM,
        t=0.6 * PX_PER_CM,
        b=1 * PX_PER_CM,
        pad=0
    )
    fig_settings = {
        "height": 6 * PX_PER_CM,  #
        "width": 16 * PX_PER_CM,
        "plot_bgcolor": "rgba(255, 255, 255, 0)",
        "paper_bgcolor": "rgba(255, 255, 255, 0)",
        "margin": margin,
        # "legend": dict(
        #     x=1, y=1,
        #     xanchor="left",
        #     bgcolor="rgba(255, 255, 255, 0.5)",
        #     font=dict(size=FONTSIZE),
        #     itemwidth=300,
        #     tracegroupgap=10,
        # ),
        "font": dict(size=FONTSIZE),
        "font_family": "arial",
        "title": dict(font=dict(size=FONTSIZE)),
    }

    # Axis settings
    ax_settings = {
        "showgrid": True,
        "showticklabels": True,
        "zeroline": False,
        "showline": False,
        "zerolinecolor": "rgba(0, 0, 0, 0.5)",
        "gridcolor": "rgba(0, 0, 0, 0.5)",
        "linecolor": "rgba(0, 0, 0, 1)",
        "griddash": "dot",
        "automargin": False,
        "titlefont": dict(size=FONTSIZE),
        "title_standoff": 0.1 * PX_PER_CM,
    }

    # Default discrete colors in plotly
    colors = px.colors.qualitative.Plotly

    return fig_settings, ax_settings, colors

