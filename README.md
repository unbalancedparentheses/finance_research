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

## Future Work: Deepening the Spitznagel Research

The current Spitznagel analysis is compelling but needs a higher standard of proof, tighter claim discipline, and more adversarial testing to move from a strong argument to great research.

### Broader Validation

The core result relies on one asset, one implementation family, and one historical window. It needs to be tested across multiple indices, longer histories, different volatility regimes, and multiple option markets. If the conclusion survives that, it stops looking like a clever backtest and starts looking like a durable finding.

### Causal Decomposition

The strongest claim is that AQR tests the wrong question because both framing and strike selection differ. To prove that cleanly, we need an explicit decomposition holding other variables constant:

- Near-ATM vs deep OTM, holding funding constant
- No-leverage vs overlay, holding strikes constant
- Rebalance frequency, holding everything else constant

This turns the argument from "this seems to be the driver" into "here is the marginal contribution of each driver."

### Execution Realism

Include conservative slippage, bid-ask spread assumptions by delta bucket, liquidity filters, and stress-period fill assumptions. If the effect survives realistic frictions, the result becomes much harder to dismiss.

### Stricter Wording

A few lines currently blur "shown" and "inferred." For each claim:

- State what the table directly proves
- State what is a plausible interpretation
- Avoid stronger historical claims unless actually demonstrated

### Out-of-Sample and Benchmark Discipline

- Compare against alternative tail hedges, especially trend-following
- Use walk-forward or pre-registered parameter choices rather than best-in-sample
- Show robustness under less favorable crash assumptions
- Include a benchmark that accounts for the external capital source in the overlay framing
