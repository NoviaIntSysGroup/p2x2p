import textwrap

from p2x2p.utils.utils import get_config
from p2x2p.report_figures import utils
from p2x2p.strategies import cache


def print_table(overwrite=False):

    # Get the configuration
    config = get_config()
    params = config['default_params'].copy()

    # Get the summary for the spot only strategies
    params_spot = params.copy()
    params_list_spot = utils.get_yearly_margin_params(params_spot)[0]
    df_spot = cache.get_summary_for_strategies(
        params_list_spot, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_ttf = params_spot.copy()
    params_ttf['ttf'] = True
    params_list_ttf = utils.get_yearly_margin_params(params_ttf)[0]
    df_ttf = cache.get_summary_for_strategies(
        params_list_ttf, overwrite=overwrite)

    # Get the summary for the spot and mFRR strategies
    params_mfrr = params_spot.copy()
    params_mfrr['mfrr'] = True
    params_list_mfrr = utils.get_yearly_margin_params(params_mfrr)[0]
    df_mfrr = cache.get_summary_for_strategies(
        params_list_mfrr, overwrite=overwrite)

    # Get the summary for the spot, ttf, and mFRR strategies
    params_all = params_spot.copy()
    params_all['mfrr'] = True
    params_all['ttf'] = True
    params_list_all = utils.get_yearly_margin_params(params_all)[0]
    df_all = cache.get_summary_for_strategies(
        params_list_all, overwrite=overwrite)
    
    table = textwrap.dedent(r"""
    \begin{table}[width=.9\linewidth,cols=9,pos=h]
    \caption{Summary of profits and yearly activity percentages for the lower and upper profit bounds (2023)}
    \label{tab_1}
    \begin{tabular*}{\tblwidth}{@{} LCCCCCCCC @{}}
    & \multicolumn{2}{c}{Profit region} 
    & \multicolumn{2}{c}{P2X active} 
    & \multicolumn{2}{c}{X2P active}
    & \multicolumn{2}{c}{Y2P active} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
    & Lower & Upper 
    & Lower & Upper
    & Lower & Upper
    & Lower & Upper  \\
    & No hor. & 3d hor. 
    & No hor. & 3d hor.
    & No hor. & 3d hor.
    & No hor. & 3d hor.  \\
    \midrule
    """)
    dfs = [df_spot, df_ttf, df_mfrr, df_all]
    names = [
        r"Spot; $\mathrm{H}_2$", 
        r"Spot; $\mathrm{H}_2$ \& $\mathrm{CH}_4$", 
        r"Spot+mFRR; $\mathrm{H}_2$", 
        r"Spot+mFRR; $\mathrm{H}_2$ \& $\mathrm{CH}_4$"]

    for df_tmp, name_tmp in zip(dfs, names): 
        df_lower = df_tmp[
            (df_tmp['year'] == params['year']) & 
            (df_tmp['name'] == 'no_horizon')]
        df_upper = df_tmp[
            (df_tmp['year'] == params['year']) & 
            (df_tmp['name'] == 'moving_horizon')]
        row = name_tmp
        row += f" & {df_lower['profit'].values[0]/1e6:.2f}" + r" M\euro"
        row += f" & {df_upper['profit'].values[0]/1e6:.2f}" + r" M\euro"
        row += f" & {df_lower['p2x_running_frac'].values[0]*100:.1f} \%"
        row += f" & {df_upper['p2x_running_frac'].values[0]*100:.1f} \%"
        row += f" & {df_lower['x2p_running_frac'].values[0]*100:.1f} \%"
        row += f" & {df_upper['x2p_running_frac'].values[0]*100:.1f} \%"
        # row += " & NaN"
        # row += " & NaN"
        row += f" & {df_lower['y2p_running_frac'].values[0]*100:.1f} \%"
        row += f" & {df_upper['y2p_running_frac'].values[0]*100:.1f} \%"
        row +=  r" \\"
        row +=  "\n"
        table += row

    table += r"\bottomrule" + "\n"
    table += r"\end{tabular*}" + "\n"
    table += r"\end{table}"

    print(table)
