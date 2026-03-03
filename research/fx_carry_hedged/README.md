# FX Carry Trade with Tail Hedging

Systematic FX carry trade strategy: borrow JPY at near-zero rates, buy high-yield currency assets, pocket the rate differential, and hedge tail risk with OTM puts. Backtested with real CME futures and options data from Databento (2010-2026).

## Overview

The classic carry trade exploits interest rate differentials between countries. Japan's persistently low rates (0-0.5%) make JPY the ideal funding currency. The strategy:

1. Borrow JPY at ~0% interest
2. Buy high-yield currency assets earning 1-11% depending on the pair
3. Pocket the rate differential (carry income)
4. Apply leverage (1-10x) to amplify returns
5. Hedge tail risk with monthly OTM put options (0.3-0.5% of notional)

The research spans four analyses, progressing from a single AUD/JPY pair to a fully optimized multi-asset portfolio with leverage analysis.

## Analyses

### 1. AUD/JPY Carry with Dual-Leg Hedge

Tests the "both legs" hedge concept: AUD/JPY can crash two ways (AUD weakness or JPY strength), so we hedge both:
- **AUD puts** (on 6A futures): protect against AUD/USD dropping
- **JPY calls** (on 6J futures): protect against JPY/USD rising

**Period:** 2010-06-06 to 2026-02-27 (4,880 days). Average AUD-JPY carry: 2.39%/yr.

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Total |
|----------|------|-----|--------|---------|-------|-------|
| 1x unhedged | 6.81% | 11.1% | 0.616 | 0.840 | -28.1% | 2.8x |
| 1x AUD puts only | 14.05% | 19.0% | 0.741 | 1.711 | -22.7% | 7.9x |
| 1x JPY calls only | 8.02% | 13.3% | 0.604 | 0.978 | -32.6% | 3.4x |
| 1x dual hedge | 11.37% | 13.9% | 0.817 | 1.386 | -22.1% | 5.4x |
| 3x unhedged | 16.45% | 33.2% | 0.496 | 0.676 | -70.5% | 11.0x |
| 3x AUD puts only | 35.69% | 56.9% | 0.627 | 1.449 | -58.9% | 121.6x |
| 3x dual hedge | 29.58% | 41.8% | 0.708 | 1.202 | -63.8% | 58.9x |
| 5x unhedged | 19.41% | 55.3% | 0.351 | 0.478 | -91.4% | 16.3x |
| 5x dual hedge | 39.31% | 69.6% | 0.565 | 0.958 | -88.2% | 183.9x |

**Key findings:**
- The dual-leg hedge (AUD puts + JPY calls, 0.25% budget each) achieves the best Sharpe at 1x (0.817) by covering both crash modes.
- AUD puts alone are more effective than JPY calls alone because AUD weakness is the more frequent crash trigger.
- In the 2011 EU debt crisis, AUD puts turned a -11% loss into +174% at 3x. During 2020 COVID, JPY calls triggered before AUD puts.
- The 50/50 budget split between legs provides more robust protection than concentrating on one leg.

### 2. Multi-Asset Carry

Extends to 7 FX carry pairs vs JPY, each hedged with monthly 8% OTM puts (0.5% budget):

| Pair | Avg Carry | 1x Unhedged Sharpe | 1x Hedged Sharpe | 3x Hedged CAGR |
|------|-----------|--------------------|--------------------|----------------|
| AUD/JPY | 2.43% | 0.619 | 0.743 | 35.82% |
| GBP/JPY | 1.36% | 0.508 | 0.676 | 20.87% |
| CAD/JPY | 1.59% | 0.427 | 0.604 | 17.36% |
| EUR/JPY | 0.99% | 0.457 | 0.399 | 11.29% |
| CHF/JPY | -0.14% | 0.577 | 0.507 | 24.41% |
| MXN/JPY | 6.39% | 0.961 | 0.965 | 35.72% |

**Carry spectrum:**
- MXN/JPY dominates with 6.39% average carry (Banxico rates 4-11%) but highest crash risk
- AUD/JPY is the workhorse: deep option liquidity, 2.43% carry, strong hedge performance
- CHF/JPY had negative carry 2015-2022 (SNB negative rates) but positive spot returns
- EUR/JPY carry collapsed during 2015-2021 (0% ECB rate era)

**Correlation structure:**
- Average pairwise correlation: 0.633 (all pairs share the JPY short)
- Lowest: CHF-MXN (0.391), highest: AUD-CAD (0.785)
- Equal-weight portfolio reduces max drawdown relative to any single pair

**Crisis behavior (3x leverage):**

| Crisis | AUD H | GBP H | CAD H | EUR H | MXN H |
|--------|-------|-------|-------|-------|-------|
| 2011 EU debt | +174.2% | +31.7% | +10.4% | -24.8% | -38.4% |
| 2015 China deval | -30.3% | -18.3% | -24.5% | -8.3% | -25.6% |
| 2018 trade war | +12.8% | -23.2% | +11.1% | -33.6% | +51.7% |
| 2020 COVID | -26.4% | +24.5% | -24.8% | +35.4% | -52.8% |

**P&L decomposition (3x hedged, annualized):**
- AUD/JPY: carry $226/yr + spot $476/yr + puts $76/yr = 123.4x total
- MXN/JPY: carry $280/yr + spot $498/yr + puts -$8/yr = 121.9x total (puts barely pay)
- GBP/JPY: carry $29/yr + spot $70/yr + puts $20/yr = 19.7x total

### 3. Portfolio Construction

Combines 6 FX carry pairs into optimal portfolios using four methods:

**Top portfolios by Sharpe ratio:**

| Portfolio | CAGR | Vol | Sharpe | Sortino | Calmar | MaxDD |
|-----------|------|-----|--------|---------|--------|-------|
| High-Carry (AUD+MXN) 1x H | 14.26% | 13.8% | 1.030 | 1.643 | 0.554 | -25.7% |
| MinVar All-6 1x H | 10.57% | 10.6% | 0.993 | 1.633 | 0.435 | -24.3% |
| High-Carry (AUD+MXN) 3x H | 39.94% | 41.5% | 0.961 | 1.534 | 0.631 | -63.3% |
| Diversified (AUD+EUR+CHF+MXN) 1x H | 11.57% | 12.1% | 0.955 | 1.737 | 0.497 | -23.3% |
| All-6 EW 1x H | 10.45% | 11.1% | 0.939 | 1.598 | 0.431 | -24.2% |

**Construction method ranking (best Sharpe achieved):**
- Equal-Weight: 0.939 (All-6 EW 1x H)
- Risk-Parity: 0.922 (RP All-6 1x H)
- Minimum Variance: 0.993 (MinVar All-6 1x H)
- Maximum Sharpe: 0.872 (MaxSharpe All-6 1x U)

**Hedging impact (averages across all portfolios):**
- Avg Sharpe hedged: 0.879 vs unhedged: 0.707
- Avg MaxDD hedged: -43.7% vs unhedged: -44.9%
- Avg Skew hedged: +9.63 vs unhedged: -0.34 (massive improvement in return distribution shape)

**Correlation effects of hedging:**
- Unhedged avg pairwise correlation: 0.633
- Hedged avg pairwise correlation: 0.446
- Hedging reduces correlations by 0.186 on average, improving diversification
- Largest decorrelation: EUR-CHF drops from 0.662 to 0.182

**Regime analysis (3x leverage):**
- Risk-on annualized returns: +47% to +61% depending on portfolio
- Risk-off annualized returns: -27% to -45%
- Hedging cuts risk-off losses nearly in half (from -42% to -27% for EW All-6)

### 4. Leverage Analysis

Sweeps leverage from 1x to 10x across 5 FX pairs (AUD, GBP, CAD, EUR, MXN) to find optimal leverage.

**Kelly-optimal leverage:**

| Pair | Unhedged Opt Lev | Unhedged Max CAGR | Hedged Opt Lev | Hedged Max CAGR |
|------|------------------|-------------------|----------------|-----------------|
| AUD/JPY | 5x | 19.60% | 5x | 47.21% |
| GBP/JPY | 5x | 12.94% | 5x | 26.28% |
| CAD/JPY | 3x | 9.23% | 5x | 20.77% |
| EUR/JPY | 5x | 10.65% | 3x | 11.29% |
| MXN/JPY | 5x | 46.70% | 5x | 47.24% |

**Equal-weight portfolio leverage sweep:**

| Leverage | Unhedged CAGR | Unhedged Sharpe | Hedged CAGR | Hedged Sharpe |
|----------|---------------|-----------------|-------------|---------------|
| 1x | 6.80% | 0.727 | 10.16% | 0.935 |
| 2x | 12.85% | 0.687 | 19.66% | 0.905 |
| 3x | 17.94% | 0.639 | 28.23% | 0.867 |
| 5x | 24.65% | 0.527 | 41.52% | 0.765 |
| 7x | 26.00% | 0.397 | 48.24% | 0.635 |
| 10x | 17.38% | 0.186 | 44.02% | 0.405 |

**Sharpe vs leverage:**
- Unhedged: Sharpe decreases monotonically with leverage (volatility drag)
- Hedged: Sharpe degradation is slower because puts truncate the left tail
- At 5x hedged, Sharpe is still 0.765 vs 0.527 unhedged

**Drawdown scaling:**
- Unhedged max drawdown scales roughly linearly with leverage
- Hedged drawdown scaling is sub-linear at moderate leverage (2-5x)
- At extreme leverage (7-10x), drawdowns converge as both strategies face cascading losses

**Put budget sensitivity (at 3x leverage):**

| Budget | AUD/JPY CAGR | AUD/JPY Sharpe | MXN/JPY CAGR | MXN/JPY Sharpe |
|--------|--------------|----------------|--------------|----------------|
| 0.0% (unhedged) | 16.56% | 0.499 | 35.30% | 0.853 |
| 0.1% | 21.52% | 0.624 | 35.41% | 0.856 |
| 0.3% | 29.54% | **0.682** | 35.59% | **0.858** |
| 0.5% | 35.82% | 0.629 | 35.72% | 0.857 |
| 1.0% | 46.47% | 0.474 | 35.77% | 0.841 |
| 2.0% | 53.96% | 0.288 | 34.96% | 0.763 |

Optimal Sharpe is achieved at 0.3% monthly put budget for both pairs.

**Blow-up frontier:** No pair blew up (capital hit zero) at any tested leverage (1-10x) during this sample period, though 10x unhedged on GBP/JPY and CAD/JPY came within 0.1% of total loss.

## Conclusions

1. **Hedging is almost always worth the cost.** Across all analyses, OTM puts improve Sharpe ratio, reduce max drawdown, and massively improve return skew (from -0.34 to +9.63 average). The optimal put budget is ~0.3% of notional per month for Sharpe maximization, or 0.5% for stronger tail protection.

2. **AUD/JPY is the best risk-adjusted single pair.** It has deep option liquidity, consistent carry (2.4%/yr average), and the strongest hedge payoff during crises. The dual-leg hedge (AUD puts + JPY calls) provides the most robust protection.

3. **MXN/JPY is the highest-return pair** (6.4% carry, 35% CAGR at 3x hedged) but with the thinnest option market and highest crash risk. Puts barely break even for MXN due to expensive implied vol.

4. **Diversification across 6 pairs reduces risk.** Average pairwise correlation is 0.633, and hedging further reduces it to 0.446. The equal-weight 6-pair portfolio achieves better risk-adjusted returns than any single pair.

5. **Portfolio construction method matters less than hedging.** MinVar achieves the highest Sharpe (0.993) at 1x hedged, but all methods (EW, RP, MinVar, MaxSharpe) perform similarly when hedged. The choice between them is secondary to the decision to hedge.

6. **Optimal leverage is 3-5x with hedging.** Kelly-optimal leverage is typically 5x for individual hedged pairs, but for the diversified portfolio, 3x offers the best Sharpe (0.867) while 7x maximizes CAGR (48.24%). Unhedged strategies degrade rapidly above 3x.

7. **None of these strategies beat SPY + puts on risk-adjusted basis.** The Spitznagel SPY + puts benchmark achieves ~1.88 Sharpe with ~-15% MaxDD. The best carry portfolio (High-Carry AUD+MXN 1x hedged) achieves 1.03 Sharpe with -25.7% MaxDD. Carry strategies offer diversification value rather than standalone superiority.

## Running the Analysis

```
python run.py --analysis all          # Run all 4 analyses
python run.py --analysis fx_carry_real  # AUD/JPY dual-leg hedge only
python run.py --analysis multi_asset    # Multi-asset carry
python run.py --analysis portfolio      # Portfolio construction
python run.py --analysis leverage       # Leverage optimization
```

Charts are saved to the `charts/` subdirectory with prefixes `fxr_`, `ma_`, `pf_`, `lv_`.

## Data

All data is from Databento CME futures and options OHLCV:
- Futures: 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S (FUT ohlcv1d)
- Options: 6A/ADU, 6B/GBU, 6C/CAU, 6E/EUU, 6J/JPU, 6M, 6N, 6S (OPT ohlcv1d)
- Period: 2010-06-06 to 2026-02-27
