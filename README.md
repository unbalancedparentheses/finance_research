# Finance Research

Does Spitznagel's approach to tail hedging actually work, and if so, why does AQR's research say otherwise?

AQR tests the wrong portfolio construction. They reduce equity to fund puts — of course that loses, you're selling your best asset to buy insurance. Spitznagel keeps 100% equity and adds deep OTM puts on top as a small leveraged overlay (~0.5% of portfolio per year). Every configuration we tested beats SPY on both return and risk-adjusted metrics across 17 years of real options data.

The mechanism is variance drain. Volatility costs ~1.4%/yr in geometric compounding drag (theoretical: sigma^2/2 = 2%/yr at 20% vol). Deep OTM puts reduce crash severity, which reduces variance drain by more than the premium costs. The payoff is convex: 0.5% annual premium produces ~5% annual excess return — 100x what linear leverage would deliver at the same notional.

Published writeup: [The Tail Hedge Debate: Spitznagel Is Right, AQR Is Answering the Wrong Question](https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/)

Built on the [options_portfolio_backtester](https://github.com/unbalancedparentheses/options_backtester) library.

## Key Results

**SPY + deep OTM puts (0.5% budget, leveraged overlay):** 16.0% CAGR vs 11.1% SPY, Sharpe 0.90 vs 0.55, max drawdown -47% vs -52%. All 36 parameter combinations in a grid search beat SPY. The result holds in both halves of an out-of-sample split and during the 2010-2019 bull market (no crash > -20%).

**FX carry + tail hedge:** Hedging improves Sharpe across all 7 currency pairs tested (AUD, GBP, CAD, EUR, CHF, MXN vs JPY). Best single pair: AUD/JPY at 0.82 Sharpe hedged vs 0.62 unhedged. But none beat SPY + puts on a risk-adjusted basis (best carry portfolio: 1.03 Sharpe vs SPY+puts: 1.88). The value is diversification, not replacement.

**Cross-asset Spitznagel:** The structure works on equity (ES) and treasuries (ZN/ZB), fails on commodities. Commodity futures have persistent negative roll yield (gold -5.4%/yr, natgas -22.3%/yr) — without positive base carry to fund the hedge cost, the structure breaks down. The key insight: Spitznagel requires a positive-carry asset.

## Caveats

The edge concentrates around 3 crashes in 17 years (2008 GFC, 2020 COVID, 2022 bear market). The puts lose money even during crashes — total put P&L is negative (-$1.6M on $2M premium). The portfolio benefit is second-order: reduced variance drain on geometric compounding, not direct option profits. If crashes of -30%+ stop happening once per decade, the strategy stops working.

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

Each directory has a `README.md` with findings and a `run.py` that reproduces all charts and tables.

- [`spitznagel_spy/`](research/spitznagel_spy/) — Core thesis: SPY + deep OTM puts. 17.9 years of real options data from Tiingo/OptionsDX. Parameter sweeps, out-of-sample validation, calm-period analysis.
- [`fx_carry_hedged/`](research/fx_carry_hedged/) — FX carry trades with OTM put protection. 7 pairs vs JPY, portfolio construction, leverage sweeps. Databento CME futures 2010-2026.
- [`cross_asset_spitznagel/`](research/cross_asset_spitznagel/) — Spitznagel structure on ES, treasuries, US-UK bond carry, and commodities. Databento CME futures.

## References

See [`REFERENCES.md`](REFERENCES.md) for an annotated literature review (~25 papers) and book recommendations.
