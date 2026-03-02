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

Each topic lives in its own folder under `research/` with markdown writeups, code, and images.

### Tail Hedging (Spitznagel vs AQR)

- [`research/spitznagel/`](research/spitznagel/) — Core analysis: SPY + deep OTM puts ([summary](research/spitznagel/summary.md), [full analysis](research/spitznagel/spitznagel_case_rerun.md))
- [`research/equity_spitznagel/`](research/equity_spitznagel/) — Same analysis on ES futures + puts (CME data)
- [`research/beyond_spitznagel/`](research/beyond_spitznagel/) — Can you profit from pure vol selling + tail protection without equity? (No)
- [`research/paper_comparison/`](research/paper_comparison/) — Direct comparison with AQR/Israelov published numbers
- [`research/trade_analysis/`](research/trade_analysis/) — Individual trade-level analysis

### Multi-Asset Carry

- [`research/multi_asset_carry/`](research/multi_asset_carry/) — FX carry with tail hedging (6 pairs, base + executed)
- [`research/fx_carry/`](research/fx_carry/) — FX carry with realistic transaction costs (AUD/USD, AUD/JPY, dual-leg hedge)
- [`research/commodity_carry/`](research/commodity_carry/) — Gold, crude, copper, natgas carry
- [`research/bond_carry/`](research/bond_carry/) — US/UK bond carry strategies
- [`research/carry_portfolio/`](research/carry_portfolio/) — Combined carry portfolio construction
- [`research/combined_portfolio/`](research/combined_portfolio/) — Multi-asset portfolio aggregation

### Treasury and Fixed Income

- [`research/treasury_spitznagel/`](research/treasury_spitznagel/) — Treasury futures with tail hedging
- [`research/gold_sp500/`](research/gold_sp500/) — Gold/S&P500 relationship analysis

### Volatility and Options

- [`research/volatility_premium/`](research/volatility_premium/) — Volatility risk premium analysis
- [`research/iron_condor/`](research/iron_condor/) — Iron condor strategy backtests
- [`research/strategies/`](research/strategies/) — Options strategy comparison

### Portfolio Construction

- [`research/leverage_analysis/`](research/leverage_analysis/) — Kelly-optimal leverage analysis
- [`research/ivy_portfolio/`](research/ivy_portfolio/) — Ivy portfolio replication
- [`research/findings/`](research/findings/) — Summary of key empirical findings
- [`research/results/`](research/results/) — Consolidated results and tables

### Exploration and Tooling

- [`research/exploration/`](research/exploration/) — Convexity scanner exploration
- [`research/quickstart/`](research/quickstart/) — Quick demo of the backtester
- [`research/comparison_with_bt/`](research/comparison_with_bt/) — Comparison with the `bt` library

### Cross-Asset Research Notes

- [`research/research.md`](research/research.md) — Detailed notes on the Spitznagel structure across asset classes: rates, FX carry, credit, commodities, EM debt, VIX. Includes data sourcing guide and cost breakdown.

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
research/               Research writeups (markdown + code + images)
  spitznagel/           Core tail hedge analysis (SPY + OTM puts)
  equity_spitznagel/    ES futures + puts (CME data)
  beyond_spitznagel/    Pure vol barbell test
  paper_comparison/     Academic paper comparison
  multi_asset_carry/    FX carry with tail hedging
  fx_carry/             Real FX options backtests
  commodity_carry/      Commodity carry strategies
  bond_carry/           US/UK bond carry
  treasury_spitznagel/  Treasury futures + tail hedging
  carry_portfolio/      Combined carry portfolio
  combined_portfolio/   Multi-asset aggregation
  leverage_analysis/    Kelly-optimal leverage
  volatility_premium/   VRP analysis
  iron_condor/          Iron condor backtests
  strategies/           Options strategy comparison
  ivy_portfolio/        Ivy portfolio replication
  gold_sp500/           Gold/equity analysis
  findings/             Key empirical findings
  results/              Consolidated results
  exploration/          Convexity scanner exploration
  quickstart/           Backtester demo
  comparison_with_bt/   bt library comparison
  trade_analysis/       Trade-level analysis
  research.md           Cross-asset research notes
scripts/                Backtesting scripts, sweeps, and verification
data/
  databento/            CME futures and options data (Databento parquets)
  processed/            Processed CSVs (options.csv, stocks.csv, signals.csv)
  raw/                  Raw OptionsDX downloads
  fetch_data.py         Fetch stock data from Tiingo
  fetch_signals.py      Fetch VIX/signal data
  convert_optionsdx.py  Convert OptionsDX raw to processed CSVs
REFERENCES.md           Annotated literature review (~50 papers)
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
