# Finance Research

Empirical research on tail hedging, volatility risk premium, and multi-asset carry strategies. The central question: does Spitznagel's approach to tail hedging actually work, and if so, why does AQR's research say otherwise?

Published writeup: [The Tail Hedge Debate: Spitznagel Is Right, AQR Is Answering the Wrong Question](https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/)

Built on the [options_portfolio_backtester](https://github.com/unbalancedparentheses/options_backtester) library.

## Setup

```bash
pip install -e ../options_backtester
pip install -r requirements.txt
```

## Data

Data files are gitignored due to size.

```bash
# SPY/options data (OptionsDX)
python data/fetch_data.py
python data/fetch_signals.py

# Databento CME data (requires API key)
# Parquets go in data/databento/
```

See [`data/README.md`](data/README.md) for details on data sources and formats.

## Research

### [`spitznagel_spy/`](research/spitznagel_spy/) — Spitznagel Thesis on SPY

The core research thread. Tests whether deep OTM puts improve geometric compounding via variance drain reduction. SPY options data 2008-2025 from Tiingo/OptionsDX.

### [`fx_carry_hedged/`](research/fx_carry_hedged/) — FX Carry + Tail Hedge

FX carry trades with OTM put protection. Databento CME futures data 2010-2026. Progresses from single AUD/JPY pair to multi-pair portfolio construction with leverage sweeps.

### [`cross_asset_spitznagel/`](research/cross_asset_spitznagel/) — Spitznagel Structure Across Asset Classes

Applies the leveraged + OTM puts framework to equity (ES), treasuries (ZN/ZB), US-UK bond carry, and commodities (gold, crude, copper, natgas) using CME futures from Databento.

Each directory has a `README.md` with findings and a `run.py` that reproduces all charts and tables.

## Scripts

`scripts/` contains parameter sweeps, validation scripts, and shared helpers. Each script has a docstring explaining what it does. Key ones:

- `backtest_runner.py` — shared data loading and charting helpers
- `databento_helpers.py` — shared Databento data loading and option parsing
- `spitznagel_sweep.py` — full parameter sweep across delta, DTE, allocation
- `walk_forward_report.py` — walk-forward out-of-sample validation
- `verify_blog_numbers.py` / `verify_aqr_numbers.py` — reproduce published numbers

## References

See [`REFERENCES.md`](REFERENCES.md) for an annotated literature review (~25 papers) and book recommendations.

For a general introduction to finance and economics for programmers, see this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f).
