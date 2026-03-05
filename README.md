# Finance Research

Empirical research on derivatives, tail hedging, volatility risk premium, and multi-asset carry strategies. We backtest real strategies on real options and futures data, no synthetic pricing, no Black-Scholes assumptions.

The work so far focuses on Spitznagel's tail hedging thesis and whether it generalizes across asset classes, but the repo is a broader home for quantitative finance research using the [options_portfolio_backtester](https://github.com/unbalancedparentheses/options_backtester) library.

## Research

### [Spitznagel Thesis on SPY](research/spitznagel_spy/)

Does a small allocation to deep OTM puts improve long-run geometric compounding? Spitznagel says yes, the puts reduce variance drain (sigma^2/2) by more than they cost. AQR says no, puts are expensive insurance that drags returns.

AQR is right in their no-leverage framing: reducing equity to fund puts always underperforms SPY. But this is not what Spitznagel proposes. He keeps 100% equity and adds puts on top as a leveraged overlay costing ~0.5%/yr. We tested both framings on 17.9 years of real SPY options data (24.7M rows, 2008-2025):

- Every configuration beats SPY on both return and max drawdown in the leveraged framing. 0.5% budget: 13.5% CAGR vs 11.1%, max DD -50% vs -52%. At 3.3%: 26.2% CAGR, -39% DD.
- ATM puts outperform deep OTM in the leveraged overlay (14.3% vs 13.5% at 0.5%, 33.2% vs 26.2% at 3.3%). Both lose in the no-leverage framing.
- All 36 parameter combinations in a grid search beat SPY. Median excess +2.31%/yr. Not parameter-picking, the strategy is robust (spread/median ratio 2.2x).
- The result holds in both halves of an out-of-sample split (+1.96% first half, +2.92% second half) and during the 2010-2019 bull market where the 0.5% strategy improves Sharpe from 0.90 to 1.09.
- The puts make money during crashes (+$148k) but lose in calm markets (-$1.5M). Total put P&L is negative, but the crash payoffs reduce drawdowns and the variance drain reduction improves geometric compounding.
- The edge concentrates around 3 crashes in 17 years. If crashes of -30%+ stop happening once per decade, the strategy stops working.

Published writeup: [The Tail Hedge Debate: Spitznagel Is Right, AQR Is Answering the Wrong Question](https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/)

### [FX Carry + Tail Hedge](research/fx_carry_hedged/)

Can you apply the same structure to FX carry trades? Borrow JPY at ~0%, buy high-yield currencies, hedge tail risk with OTM puts on CME futures.

We tested 7 pairs vs JPY (AUD, GBP, CAD, EUR, CHF, MXN, NZD) on Databento CME data 2010-2026, progressing from a single AUD/JPY pair to a fully diversified portfolio with leverage sweeps:

- Hedging improves Sharpe across all pairs. Best single pair: AUD/JPY at 0.82 Sharpe hedged vs 0.62 unhedged.
- The dual-leg hedge (AUD puts + JPY calls) covers both crash modes: AUD weakness and JPY strength.
- Diversification across 6 pairs reduces correlations from 0.63 to 0.45 and improves risk-adjusted returns.
- But none beat SPY + puts (best carry portfolio: 1.03 Sharpe vs SPY+puts: 1.88). The equity premium is a better engine than carry. The value of FX carry is diversification, not replacement.

### [Spitznagel Structure Across Asset Classes](research/cross_asset_spitznagel/)

Does the leveraged + OTM puts structure work beyond equities? We tested it on S&P 500 (ES), US Treasuries (ZN/ZB), US-UK bond carry, and commodities (gold, crude, copper, natgas) using CME futures from Databento:

- Works well on equity and treasuries. Treasury + puts is naturally anti-correlated with equity + puts, making it an excellent diversifier.
- Fails on commodities. All four have persistent negative roll yield (gold -5.4%/yr, natgas -22.3%/yr). Without positive base carry to fund the hedge cost, the structure breaks down.
- The key insight: Spitznagel requires a positive-carry asset. Equity risk premium > FX carry >> commodity contango.
- A combined portfolio (ES + FX carry + bond carry, each with their own puts) achieves better risk-adjusted returns than any individual strategy.

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

## References

See [`REFERENCES.md`](REFERENCES.md) for an annotated literature review (~25 papers).

For a general introduction to finance and economics for programmers, see this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f).

### Recommended Books

**Introductory** — Option Volatility and Pricing (Natenberg), Options, Futures, and Other Derivatives (Hull), Trading Options Greeks (Passarelli)

**Intermediate** — Trading Volatility (Bennett), Volatility Trading (Sinclair)

**Advanced** — Dynamic Hedging (Taleb), The Volatility Surface (Gatheral), The Volatility Smile (Derman & Miller)
