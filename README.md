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

### Delta Selection and Source Attribution

There is no clean public primary source where Taleb or Universa state a single exact recommended delta. What is publicly verifiable:

- **Confirmed**: Taleb/Universa use very far OTM puts, small premium budget, constant protection. Patterson describes Universa as "constantly" buying far OTM puts aimed at roughly a 20% S&P decline in one month, but no exact delta is given ([Tim Ferriss transcript, 2023](https://tim.blog/2023/09/08/nassim-nicholas-taleb-scott-patterson-transcript/)).
- **Secondary inference**: A widely cited writeup estimates Universa uses roughly 0.01-delta puts, 70–90 DTE, around 30–35% OTM, but presents this as inference, not official specification ([Grey Enlightenment, 2016](https://greyenlightenment.com/2016/10/04/tail-hedging-part-2/)).
- **Qualitative only**: Spitznagel's own book discussions describe puts as "extremely far out of the money" with ~0.5% monthly spend, without a verified delta ([Founders podcast transcript](https://podscripts.co/podcasts/founders/70-mark-spitznagel-the-dao-of-capital)).

The article's test range of -0.10 to -0.02 delta should be presented as our chosen test range, not as a precise published Taleb/Spitznagel rule. If keeping a numeric range, frame it as: "we test across this spectrum" rather than "Taleb recommends X." Secondary commentary often pegs Universa even more extreme (~0.01 delta), which is beyond our current test range and worth exploring.

### Out-of-Sample and Benchmark Discipline

- Compare against alternative tail hedges, especially trend-following
- Use walk-forward or pre-registered parameter choices rather than best-in-sample
- Show robustness under less favorable crash assumptions
- Include a benchmark that accounts for the external capital source in the overlay framing

## Recommended Reading

For complete novices in finance and economics, this [post](https://notamonadtutorial.com/how-to-earn-your-macroeconomics-and-finance-white-belt-as-a-software-developer-136e7454866f) gives a comprehensive introduction.

### Books

**Introductory**
- Option Volatility and Pricing 2nd Ed. - Natemberg, 2014
- Options, Futures, and Other Derivatives 10th Ed. - Hull 2017
- Trading Options Greeks 2nd Ed. - Passarelli 2012

**Intermediate**
- Trading Volatility - Bennet 2014
- Volatility Trading 2nd Ed. - Sinclair 2013

**Advanced**
- Dynamic Hedging - Taleb 1997
- The Volatility Surface: A Practitioner's Guide - Gatheral 2006
- The Volatility Smile - Derman & Miller 2016

### Papers

- [Volatility: A New Return Driver?](http://static.squarespace.com/static/53974e3ae4b0039937edb698/t/53da6400e4b0d5d5360f4918/1406821376095/Directional%20Volatility%20Research.pdf)
- [Easy Volatility Investing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2255327)
- [Everybody's Doing It: Short Volatility Strategies and Shadow Financial Insurers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071457)
- [Volatility-of-Volatility Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2497759)
- [The Distribution of Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2828744)
- [Safe Haven Investing Part I - Not all risk mitigation is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart1_RiskMitigation.pdf)
- [Safe Haven Investing Part II - Not all risk is created equal](https://www.universa.net/UniversaResearch_SafeHavenPart2_NotAllRisk.pdf)
- [Safe Haven Investing Part III - Those wonderful tenbaggers](https://www.universa.net/UniversaResearch_SafeHavenPart3_Tenbaggers.pdf)
- [Insurance makes wealth grow faster](https://arxiv.org/abs/1507.04655)
- [Ergodicity economics](https://ergodicityeconomics.files.wordpress.com/2018/06/ergodicity_economics.pdf)
- [The Rate of Return on Everything, 1870-2015](https://economics.harvard.edu/files/economics/files/ms28533.pdf)
- [Volatility and the Alchemy of Risk](https://static1.squarespace.com/static/5581f17ee4b01f59c2b1513a/t/59ea16dbbe42d6ff1cae589f/1508513505640/Artemis_Volatility+and+the+Alchemy+of+Risk_2017.pdf)
- [Variance Risk Premia - Carr & Wu, 2009](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=577222)
- [Portfolio Selection - Markowitz, 1952](https://www.jstor.org/stable/2975974)
- [The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market - Thorp, 2006](https://www.edwardothorp.com/wp-content/uploads/2016/11/TheKellyCriterionAndTheStockMarket.pdf)
- [Tail Risk Hedging: Creating Robust Portfolios for Volatile Markets - Bhansali, 2014](https://www.amazon.com/Tail-Risk-Hedging-Creating-Portfolios/dp/0071791760)

### Backtesting Methodology

- [The Backtest Overfitting Problem - Bailey et al., 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [Pseudo-Mathematics and Financial Charlatanism - Bailey et al., 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)
- [Advances in Financial Machine Learning - de Prado, 2018](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) (chapters on backtesting, cross-validation, bet sizing)
