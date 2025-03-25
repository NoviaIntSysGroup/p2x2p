# Evaluating running strategies for a P2X2P plant
This GitHub repository contains the code used to produce all the figure panels in the manuscript: "Evaluating running strategies for a P2X2P plant".

## Installation

Create a new conda environment with Python and install the p2x2p package using pip in editable mode.
```bash
$ conda create -n p2x2p python=3.11
$ conda activate p2x2p
$ git clone git@github.com:NoviaIntSysGroup/p2x2p.git
$ cd p2x2p
$ pip install -e .
```

## Data sources
We are only permitted to share part of the data used (the mFRR and production data ontained from Fingrid). Below are descriptions of the data included and instructions for obtaining your own copies of the data that can't be shared.
* mFRR price and volume data for 2016 to 2023 are in the folder `\data\mfrr_fi_price_and_volym_2016_2023.csv`.
* Production data for Finland, including installed wind power capacity, for 2016 to 2023 are in the folder: `\data\energy_data_fi_2016_2023.csv`.
* TTF price data can be obtained with the notebook: `\notebooks\ttf_price_svg_to_csv.ipynb`.
* Day-ahead spot prices can be obtained with the notebook `\notebooks\spot_price_api_to_csv.ipynb`. Usage requires free ENTSO-E transparency platform registration to obtain an API key, more detailed instructions are included in the notebook.
* Day-ahead spot predictions for 2017 to 2023 are in the folder `\data\spot_prices_fi_2016_2023_predictions.csv`. Observe that these are yearly predictions for a model trained on the previous year's data. The absolute value of the predicted price is thus badly scaled, but the model does a good job of estimating the most expensive hours or the cheapest hours each day. More details for the used forecasting model are available [here](https://github.com/NoviaIntSysGroup/spot-price-forecast).


## Usage
* The notebook `\notebooks\demo_usage.ipynb` illustrates the parameters available when computing yearly profits and shows how to determine yearly profits for arbitrary parameter values.
* The notebook `\notebooks\create_figure_panels.ipynb` recreates all the figure panels used in the manuscript.
The final manuscript figures are assembled using Tikz (requires a LaTeX installation) and can be recreated by running `\report\update_tikz_figures.sh`
