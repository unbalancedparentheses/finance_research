# Finance Research

Research notebooks and scripts for options backtesting, tail hedging, and multi-asset carry strategies.

Built on the [options_portfolio_backtester](https://github.com/unbalancedparentheses/options_backtester) library.

## Setup

```bash
# Install the backtester library (editable)
pip install -e ../options_backtester

# Install research dependencies
pip install -r requirements.txt
```

## Data

Data files are gitignored due to size. To populate:

```bash
# Fetch SPY/options data (OptionsDX)
python data/fetch_data.py
python data/fetch_signals.py

# Databento CME data (requires API key)
# Parquets go in data/databento/
```

## Structure

- `notebooks/` — Jupyter notebooks for analysis and research
- `scripts/` — Backtesting scripts, parameter sweeps, and benchmarks
- `research/` — Research notes and figures
- `data/databento/` — CME futures and options data (Databento parquets)
- `data/processed/` — Processed CSVs (options.csv, stocks.csv, signals.csv)
- `data/raw/` — Raw OptionsDX downloads
- `data/fetch_data.py` — Fetch stock data from Tiingo
- `data/fetch_signals.py` — Fetch VIX/signal data
- `data/convert_optionsdx.py` — Convert OptionsDX raw data to processed CSVs

## Running

```bash
# Start Jupyter
jupyter notebook notebooks/

# Run a script
python scripts/calm_period_experiment.py
```

## Key Notebooks

- `spitznagel_case.ipynb` — Spitznagel tail-hedging analysis (SPY + OTM puts)
- `equity_spitznagel.ipynb` — ES futures + puts (CME data)
- `multi_asset_carry.ipynb` — FX carry with tail hedging (6 pairs)
- `commodity_carry.ipynb` — Gold, crude, copper, natgas carry
- `treasury_spitznagel.ipynb` — Treasury futures with tail hedging
- `leverage_analysis.ipynb` — Kelly-optimal leverage analysis
