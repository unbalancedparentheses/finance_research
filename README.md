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

The core research thread. Tests whether deep OTM puts improve geometric compounding via variance drain reduction. Uses SPY options data 2008-2025 from Tiingo/OptionsDX.

- [`summary.md`](research/spitznagel_spy/summary.md) — Hand-written findings with full data tables
- [`spitznagel_case.md`](research/spitznagel_spy/spitznagel_case.md) — Authoritative analysis: AQR framing vs Spitznagel overlay, parameter sweeps, out-of-sample, calm periods
- [`cross_asset_notes.md`](research/spitznagel_spy/cross_asset_notes.md) — Detailed notes on applying the structure to rates, FX, credit, commodities, EM debt. Includes data sourcing guide.

### [`spy_options_strategies/`](research/spy_options_strategies/) — SPY Options Strategy Analysis

Tests specific options structures on SPY against academic claims. Same backtester and data as the Spitznagel thread.

- [`findings.md`](research/spy_options_strategies/findings.md) — Comprehensive findings: puts as hedge, calls for momentum, macro signal timing
- [`paper_comparison.md`](research/spy_options_strategies/paper_comparison.md) — 10 strategies tested against Carr & Wu, Whaley, Israelov, etc.
- [`volatility_premium.md`](research/spy_options_strategies/volatility_premium.md) — Variance risk premium analysis
- [`strategies.md`](research/spy_options_strategies/strategies.md) — 4-strategy showcase (OTM put hedge, call momentum, straddle, strangle)
- [`trade_analysis.md`](research/spy_options_strategies/trade_analysis.md) — Trade-level P&L, greeks at entry, crash breakdown

### [`fx_carry_hedged/`](research/fx_carry_hedged/) — FX Carry + Tail Hedge

FX carry trades with OTM put protection. Uses Databento CME futures data 2010-2026. Progression: single pair -> multi-pair -> portfolio construction -> leverage analysis.

- [`fx_carry_real.md`](research/fx_carry_hedged/fx_carry_real.md) — AUD/JPY carry with dual-leg hedge (AUD puts + JPY calls)
- [`multi_asset_carry.md`](research/fx_carry_hedged/multi_asset_carry.md) — 7 FX pairs vs JPY with monthly OTM puts
- [`carry_portfolio.md`](research/fx_carry_hedged/carry_portfolio.md) — Portfolio construction: equal-weight, risk-parity, min-variance, max-Sharpe
- [`leverage_analysis.md`](research/fx_carry_hedged/leverage_analysis.md) — Kelly-optimal leverage, blow-up frontier, put budget sensitivity

### [`cross_asset_spitznagel/`](research/cross_asset_spitznagel/) — Spitznagel Structure Across Asset Classes

Applies the leveraged + OTM puts framework to non-equity asset classes using CME futures from Databento.

- [`equity_spitznagel.md`](research/cross_asset_spitznagel/equity_spitznagel.md) — ES futures + puts (1x-10x leverage)
- [`treasury_spitznagel.md`](research/cross_asset_spitznagel/treasury_spitznagel.md) — ZN/ZB Treasury futures + puts
- [`bond_carry_usuk.md`](research/cross_asset_spitznagel/bond_carry_usuk.md) — US-UK bond carry (ZN vs Gilt) + OZN options
- [`commodity_carry.md`](research/cross_asset_spitznagel/commodity_carry.md) — Gold, crude, copper, natgas carry + puts
- [`combined_portfolio.md`](research/cross_asset_spitznagel/combined_portfolio.md) — Capstone: ES + FX Carry + Bond Carry combined

## Scripts

Parameter sweeps, verification, and analysis scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `spitznagel_sweep.py` | Full parameter sweep across delta, DTE, allocation |
| `sweep_otm.py` | OTM depth sweep |
| `sweep_volatility.py` | Volatility regime sweep |
| `sweep_leverage.py` | Leverage level sweep |
| `sweep_iv_signal.py` | IV-based signal sweep |
| `sweep_allocation.py` | Allocation percentage sweep |
| `sweep_beat_spy.py` | Configurations that beat SPY |
| `sweep_comprehensive.py` | All-parameter combinatorial sweep |
| `walk_forward_report.py` | Walk-forward out-of-sample validation |
| `verify_blog_numbers.py` | Verify published article numbers |
| `verify_aqr_numbers.py` | Reproduce AQR's published results |
| `verify_atm_vs_otm.py` | ATM vs OTM direct comparison |
| `calm_period_experiment.py` | Performance during calm markets |
| `generate_blog_tables.py` | Generate tables for the published article |
| `benchmark_rust_vs_python.py` | Backtester engine performance comparison |
| `parallel_sweep.py` | Parallelized parameter sweep |
| `nb_style.py` | Shared FT-inspired matplotlib styling |
| `export_fx_results.py` | Export FX carry results to CSV |

## Structure

```
research/
  spitznagel_spy/           Core tail hedge thesis (SPY, Tiingo data 2008-2025)
  spy_options_strategies/   Options strategy analysis (SPY, same data)
  fx_carry_hedged/          FX carry + tail hedge (Databento CME data 2010-2026)
  cross_asset_spitznagel/   Spitznagel structure on futures (Databento CME data)
scripts/                    Backtesting scripts, sweeps, and verification
data/
  databento/                CME futures and options data (Databento parquets)
  processed/                Processed CSVs (options.csv, stocks.csv, signals.csv)
  raw/                      Raw OptionsDX downloads
  fetch_data.py             Fetch stock data from Tiingo
  fetch_signals.py          Fetch VIX/signal data
  convert_optionsdx.py      Convert OptionsDX raw to processed CSVs
REFERENCES.md               Annotated literature review (~50 papers)
```

## Future Work: Deepening the Spitznagel Research

The current analysis is compelling but needs a higher standard of proof, tighter claim discipline, and more adversarial testing to move from a strong argument to great research.

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

### Delta Selection and Source Attribution

There is no clean public primary source where Taleb or Universa state a single exact recommended delta. What is publicly verifiable:

- **Confirmed**: Taleb/Universa use very far OTM puts, small premium budget, constant protection. Patterson describes Universa as "constantly" buying far OTM puts aimed at roughly a 20% S&P decline in one month, but no exact delta is given ([Tim Ferriss transcript, 2023](https://tim.blog/2023/09/08/nassim-nicholas-taleb-scott-patterson-transcript/)).
- **Secondary inference**: A widely cited writeup estimates Universa uses roughly 0.01-delta puts, 70-90 DTE, around 30-35% OTM, but presents this as inference, not official specification ([Grey Enlightenment, 2016](https://greyenlightenment.com/2016/10/04/tail-hedging-part-2/)).
- **Qualitative only**: Spitznagel's own book discussions describe puts as "extremely far out of the money" with ~0.5% monthly spend, without a verified delta ([Founders podcast transcript](https://podscripts.co/podcasts/founders/70-mark-spitznagel-the-dao-of-capital)).

The article's test range of -0.10 to -0.02 delta should be presented as our chosen test range, not as a precise published Taleb/Spitznagel rule. If keeping a numeric range, frame it as: "we test across this spectrum" rather than "Taleb recommends X." Secondary commentary often pegs Universa even more extreme (~0.01 delta), which is beyond our current test range and worth exploring.

### Out-of-Sample and Benchmark Discipline

- Compare against alternative tail hedges, especially trend-following
- Use walk-forward or pre-registered parameter choices rather than best-in-sample
- Show robustness under less favorable crash assumptions
- Include a benchmark that accounts for the external capital source in the overlay framing

## References

See [`REFERENCES.md`](REFERENCES.md) for an annotated literature review covering protective put overlays, covered calls, volatility strategies, tail hedging, carry, ergodicity economics, and backtesting methodology.

For a general introduction to finance and economics for programmers, see this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f).

### Key Books

**Introductory** — Option Volatility and Pricing (Natenberg), Options, Futures, and Other Derivatives (Hull), Trading Options Greeks (Passarelli)

**Intermediate** — Trading Volatility (Bennett), Volatility Trading (Sinclair)

**Advanced** — Dynamic Hedging (Taleb), The Volatility Surface (Gatheral), The Volatility Smile (Derman & Miller)
