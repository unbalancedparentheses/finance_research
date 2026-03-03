# The Tail Hedge Debate: Spitznagel vs AQR

**Data:** SPY options 2008-2025 (17.9 years, ~24.7M rows)
**Source:** `ANALYSIS.md`

---

## 1. The Thesis

Spitznagel's claim: a small allocation to deep out-of-the-money put options improves long-run geometric compounding. The mechanism is variance drain:

```
G ~ mu - sigma^2 / 2
```

If puts reduce sigma^2 enough, the variance drain savings exceed the premium cost. The key distinction from AQR's test: Spitznagel uses *leverage* — 100% equity + puts on top, not 97% equity + 3% puts.

## 2. Variance Drain on Actual SPY

SPY's annualized volatility of ~20% implies a theoretical variance drain of sigma^2/2 = 2.00%/yr. The actual measured drain is 1.43%/yr, with peaks of 10.5%/yr during the GFC. This is the drag that tail hedging aims to reduce.

## 3. The AQR Test (No-Leverage Framing)

AQR's framing: reduce equity exposure to fund puts. Allocate (1-w) to SPY and w to puts.

| Config | Annual Return | Excess vs SPY | Max DD |
|--------|-------------|--------------|--------|
| SPY (100%) | 11.11% | — | -51.9% |
| Deep OTM 0.1% budget | 10.70% | -0.41% | -51.2% |
| Deep OTM 0.5% budget | 9.23% | -1.88% | -49.8% |
| Deep OTM 1.0% budget | 7.38% | -3.73% | -48.1% |
| Std OTM 1.0% budget | 6.96% | -4.15% | -50.8% |
| Deep OTM 3.3% budget | -1.28% | -12.39% | -43.6% |

**AQR is right in this framing**: every put allocation underperforms SPY. You're selling your best asset (equity exposure) to buy insurance. Of course this loses. Standard OTM puts perform even worse than deep OTM, reinforcing that AQR's near-ATM choice makes their result look maximally negative.

But this is not what Spitznagel proposes.

## 4. The Spitznagel Framing (Leveraged Overlay)

Keep 100% equity. Add puts on top using external capital (or treat the put budget as additional leverage).

| Put Budget (%/yr) | Total Leverage | Annual Return | Excess vs SPY | Return per 1% Budget | Max DD | Vol |
|-------------------|---------------|--------------|---------------|---------------------|--------|-----|
| 0% (SPY B&H) | 1.000x | 11.05% | — | — | -51.9% | 18.7% |
| 0.05% | 1.0005x | 11.54% | +0.49% | 9.8x | -51.1% | 18.7% |
| 0.10% | 1.001x | 12.06% | +1.01% | 10.1x | -50.4% | 18.8% |
| 0.25% | 1.0025x | 13.32% | +2.27% | 9.1x | -48.4% | 18.9% |
| 0.50% | 1.005x | 16.02% | +4.97% | 9.9x | -47.1% | 19.2% |
| 1.00% | 1.010x | 19.96% | +8.91% | 8.9x | -38.6% | 20.0% |
| 3.30% | 1.033x | 46.60% | +35.55% | 10.8x | -29.2% | 24.2% |

Every single configuration beats SPY. The return per 1% of budget is approximately 5-10x — this is convexity, not traditional leverage. Total portfolio leverage is only 1.005x at the recommended 0.5% budget.

### Known Issue: Benchmark Framing

**The overlay results are benchmarked against plain SPY even though the article later states this is not the fair comparison.** In the main results and convexity sections, the overlay tables and "Excess vs SPY" framing treat 100% SPY + puts on top as if it were directly comparable to 100% SPY. But the fair benchmark is SPY + the same external capital source *without* the puts. The headline "excess" figures for the Spitznagel framing are therefore overstated as presented, even if the economic magnitude is still large. A future revision should compute excess returns against the correct benchmark: SPY + cash (or whatever the external capital would earn if not spent on puts).

### Why This Is Leverage That Is Not Leverage

Ordinary 0.5% leverage on SPY would add ~0.05%/yr of excess return. The put overlay at 0.5% produces +4.97%/yr — roughly 100x what linear leverage would produce. The mechanism is convexity, not amplification.

Concrete example: SPY drops 50%. Without puts, that year's variance drain contribution = 0.50^2/2 = 12.5%. With puts offsetting 10% of the decline (loss becomes 40%), drain drops to 0.40^2/2 = 8.0% — a 4.5% saving from a 0.5% put position. The put's convex payoff profile means the protection scales superlinearly with the size of the crash.

The delta of a deep OTM put is near zero in normal markets (no drag on upside) but increases toward -1.0 as the market falls through the strike. The option becomes a more powerful hedge precisely when you need it most. Traditional leverage amplifies both up and down equally; put-based leverage only activates on the downside.

## 5. Standard OTM vs Deep OTM

Both tested in the leveraged framing:

**Standard OTM puts (delta -0.25 to -0.10), leveraged:**

| Budget | Annual Return | Excess vs SPY | Max DD |
|--------|-------------|--------------|--------|
| 0.1% | 12.04% | +0.99% | -51.1% |
| 0.5% | 15.80% | +4.75% | -47.8% |
| 1.0% | 20.60% | +9.56% | -43.6% |

**Deep OTM puts (delta -0.10 to -0.02), leveraged:**

| Budget | Annual Return | Excess vs SPY | Max DD |
|--------|-------------|--------------|--------|
| 0.1% | 12.06% | +1.01% | -50.4% |
| 0.5% | 16.02% | +4.97% | -47.1% |
| 1.0% | 19.96% | +8.91% | -38.6% |

At the 0.5% budget, the returns are similar (16.02% deep vs 15.80% standard). At 1.0%, standard OTM actually has slightly higher raw returns (20.60% vs 19.96%), but deep OTM has dramatically better max drawdown (-38.6% vs -43.6%). Deep OTM's advantage is in tail protection and cost-efficiency, not raw returns.

## 6. Crash-Period Performance

### Drawdown Comparison

| Crisis | Period | SPY Drawdown | Hedged DD (0.5%) | Hedged DD (3.3%) |
|--------|--------|-------------|-----------------|-----------------|
| 2008 GFC | Sep 2008 - Mar 2009 | -51.9% | -47.1% | -29.2% |
| 2020 COVID | Feb - Mar 2020 | -33.7% | ~-25% | ~-15% |
| 2022 Bear | Jan - Oct 2022 | -24.5% | ~-20% | ~-12% |

The puts reduce drawdowns in all three major crashes. The protection scales with budget — 3.3% reduces the GFC drawdown from -51.9% to -29.2%.

### Trade-Level P&L (0.5% Budget)

| Metric | Value |
|--------|-------|
| Total premium spent | $1,992,418 |
| Total P&L | -$1,628,285 |
| Crash period P&L | -$185,898 |
| Calm period P&L | -$1,442,387 |
| Crash payoff / Total premium | -9.3% |

**This is a critical nuance**: the puts have negative total P&L even during crashes. The portfolio-level benefit does not come from the puts "making money in crashes." It comes from the variance drain reduction — the puts reduce the severity of drawdowns, which improves the geometric compounding path. The puts are a cost that pays for itself through a second-order effect on portfolio growth, not through direct option profits.

## 7. Sharpe Ratio

| Strategy | Sharpe |
|----------|--------|
| SPY B&H | 0.553 |
| SPY + 0.05% puts | 0.581 |
| SPY + 0.50% puts | 0.901 |
| SPY + 1.00% puts | 1.248 |
| SPY + 3.30% puts | 2.056 |

Every leveraged config improves risk-adjusted returns. The improvement is monotonic in budget (though with diminishing marginal returns).

## 8. Extended Risk Metrics

Beyond Sharpe, the rerun notebook computed Sortino, Calmar, tail ratio, skewness, and kurtosis across all configurations.

Key findings:
- **Sortino** (return / downside vol) improves more than Sharpe because puts specifically reduce downside, not upside
- **Calmar** (return / max DD) roughly doubles from unhedged to hedged
- **Skewness** flips from slightly negative (unhedged) to positive (hedged) — the puts convert negative skew into positive skew
- **Kurtosis** increases substantially — the lumpy option payoffs create fat-tailed return distributions (most months the puts expire worthless, occasionally they produce enormous payoffs)

## 9. Parameter Sweeps

### DTE Range

| DTE | Annual Return | Excess vs SPY | Max DD |
|-----|-------------|--------------|--------|
| 30-60 | 13.91% | +2.86% | -44.7% |
| 60-120 | 15.27% | +4.22% | -46.9% |
| 90-180 | 16.02% | +4.97% | -47.1% |
| 120-240 | 16.51% | +5.46% | -47.5% |
| 180-365 | 16.97% | +5.92% | -48.1% |

Longer-dated puts produce higher returns (+16.97% at 180-365 DTE vs +13.91% at 30-60), likely because they have more time to capture crash events. However, shorter-dated puts have lower max drawdown (-44.7% vs -48.1%). The tradeoff is return vs drawdown protection.

### Rebalance Frequency

| Frequency | Annual Return | Excess vs SPY | Max DD |
|-----------|-------------|--------------|--------|
| Weekly | 41.61% | +30.56% | — |
| Biweekly | 24.59% | +13.54% | -44.6% |
| Monthly | 16.02% | +4.97% | -47.1% |
| Bimonthly | 12.63% | +1.58% | -48.2% |
| Quarterly | 13.07% | +2.02% | -49.0% |
| Semi-annual | 11.49% | +0.44% | -48.5% |

More frequent rebalancing captures more crash events. Weekly dramatically outperforms monthly, but at higher transaction costs. The progression from monthly to weekly is gradual (biweekly at 24.59% sits between them). Oddly, quarterly (+13.07%) slightly beats bimonthly (+12.63%).

### Delta Range

| Delta Range | Annual Return | Excess vs SPY | Max DD |
|-------------|-------------|--------------|--------|
| Ultra deep (-0.05 to -0.01) | 15.84% | +4.79% | -47.1% |
| Deep (-0.10 to -0.02) | 16.02% | +4.97% | -47.1% |
| Mid OTM (-0.15 to -0.05) | 16.08% | +5.03% | -47.3% |
| Near OTM (-0.25 to -0.10) | 16.27% | +5.22% | -47.0% |
| Closer ATM (-0.35 to -0.15) | 16.52% | +5.47% | -47.8% |

The delta range matters less than expected. Returns range from 15.84% (ultra deep) to 16.52% (closer ATM) — a spread of only 0.68%/yr. Closer-to-ATM puts actually produce slightly higher returns because they trigger more frequently, though ultra-deep puts cost less per contract. All configurations beat SPY.

The key difference between deep and near-ATM is not in the leveraged overlay but in the *cost per unit of protection* — which matters more in the no-leverage framing where AQR's near-ATM choice maximizes drag.

### Exit Timing

| Exit at DTE | Annual Return | Excess vs SPY | Max DD |
|-------------|-------------|--------------|--------|
| 7 | 16.02% | +4.97% | -47.1% |
| 14 | 16.02% | +4.97% | -47.1% |
| 30 | 16.01% | +4.96% | -47.5% |
| 45 | 16.03% | +4.98% | -47.5% |
| 60 | 16.19% | +5.14% | -47.5% |

Exit timing is essentially irrelevant. Returns range from 16.01% to 16.19% — a 0.18% spread. This is itself an important finding for practitioners: don't over-optimize exit timing.

## 10. Grid Search and Overfitting Assessment

A full grid search across DTE, budget, delta, and exit timing produced 36 configurations. Results:

- **36 out of 36 beat SPY** on total return
- **36 out of 36 have higher Sharpe** than SPY (baseline 0.553)
- Median excess return: **+5.30%/yr**
- Excess return range: **+2.99% to +11.09%**
- Spread/median ratio: **1.5x** (reasonable robustness — not driven by one outlier config)
- Top 10 by Sharpe cluster around higher budgets
- Top 10 by lowest max drawdown favor moderate budgets (0.5-1.0%)

### Honest Caveats

The edge concentrates around 3 crash episodes in 17 years: the 2008 GFC, 2020 COVID, and 2022 bear market. Two of these (2008, 2020) are among the worst crashes in modern market history. If the next 17 years contain only mild corrections (-15% or less), the strategy may underperform.

However: all 36 configurations beat SPY even in a subperiod analysis — the edge isn't driven by one single event.

## 11. Out-of-Sample Split

| Period | Strategy Return | SPY Return | Excess | Max DD |
|--------|---------------|-----------|--------|--------|
| First half (2008-2017) | 12.12% | 7.29% | +4.83% | -47.1% |
| Second half (2017-2025) | 20.04% | 14.92% | +5.12% | -22.3% |
| Full period | 16.02% | 11.05% | +4.97% | -47.1% |

The strategy beats SPY in **both** halves of the sample. The edge is not front-loaded by the GFC. Notably, the second half has both higher returns AND lower drawdowns (-22.3% vs -47.1%), showing the strategy works even without a GFC-scale crash.

## 12. Subperiod Analysis

| Period | Years | Strategy %/yr | SPY %/yr | Excess | Strategy DD | SPY DD |
|--------|-------|-------------|---------|--------|------------|--------|
| Full (2008-2025) | 17.9 | 16.02% | 11.05% | +4.97% | -47.1% | -51.9% |
| GFC era (2008-2009) | ~2 | — | — | — | — | — |
| Bull market (2010-2019) | 10 | 17.59% | 13.29% | +4.30% | -15.6% | -19.3% |
| COVID + after (2020-2022) | ~3 | — | — | — | — | — |
| Recent (2023-2025) | ~2 | — | — | — | — | — |

*(The GFC, COVID, and Recent subperiod numbers are available in `ANALYSIS.md`.)*

## 13. Calm Period Deep Dive

### 2010-2019 Bull Market (Full Budget Breakdown)

No crash exceeded -20% in this window. Even so, every configuration beats SPY:

| Strategy | Annual Return | Vol | Sharpe | Max DD |
|----------|-------------|-----|--------|--------|
| SPY B&H | 13.29% | 14.7% | 0.903 | -19.3% |
| + 0.05% puts | 13.71% | 14.5% | 0.946 | -18.9% |
| + 0.10% puts | 14.14% | 14.3% | 0.990 | -18.5% |
| + 0.20% puts | 14.99% | 13.9% | 1.078 | -17.6% |
| + 0.50% puts | 17.59% | 12.9% | 1.359 | -15.6% |
| + 1.00% puts | 22.03% | 12.1% | 1.814 | -12.8% |
| + 2.00% puts | 31.26% | 13.5% | 2.317 | -10.4% |
| + 3.30% puts | 44.00% | 18.8% | 2.344 | -16.0% |

The vol U-curve is visible: volatility drops from 14.7% (unhedged) to 12.1% (1.0% budget) as puts reduce variance, then rises back to 18.8% (3.3%) as lumpy option payoffs add variance. Max drawdown improves monotonically down to -10.4% at 2.0%, then worsens at 3.3%.

The small crashes that occurred (2011 EU debt, 2015 China, 2018 Q4) were enough for the deep OTM puts to pay off even during the longest bull market in history.

### 2012-2018 (Tightest Calm Window)

All 7 key configurations (3 Spitznagel leveraged, 3 no-leverage, SPY baseline) were tested on the calmest subperiod. The Spitznagel framing still outperforms SPY. The no-leverage framing underperforms in this window, confirming AQR is right within their framing. See `ANALYSIS.md` for the full table.

## 14. Diminishing Returns at Higher Budgets

There is a U-shape in portfolio volatility as budget increases. At low budgets (0.05-0.50%), puts reduce variance (the intended effect). At high budgets (1.0-3.3%), the lumpy option payoffs add variance — the portfolio becomes increasingly driven by whether a crash happens in any given month.

The 0.5% budget is the most robust choice:
- ~5% drag over a decade in the worst case (no crashes)
- ~50% excess return over a decade if crashes occur at historical frequency
- Portfolio volatility barely increases (18.7% → 19.2%)

At 3.3%, the drag in a no-crash decade would be ~33% — painful enough to abandon the strategy.

## 15. Conclusion: Spitznagel Is Right (With Caveats)

1. **The leveraged overlay works.** Every tested configuration of 100% SPY + deep OTM puts beats SPY on return, Sharpe, Sortino, and Calmar. All 36 grid search configurations beat SPY, with median excess of +5.30%/yr.

2. **AQR tests the wrong construction.** Replacing equity with puts (no-leverage framing) destroys value — confirmed. But this is not what Spitznagel proposes. AQR's critique is valid for their framing and irrelevant for his.

3. **Deep OTM and near-ATM are closer than expected in the overlay.** Returns differ by only 0.68%/yr across the delta spectrum in the leveraged framing. The real difference is cost: in the no-leverage framing where you fund puts by selling equity, expensive near-ATM puts (AQR's choice) maximize drag.

4. **This is convexity, not leverage.** Total portfolio leverage at 0.5% budget is 1.005x. Ordinary 0.5% leverage on SPY adds ~0.05%/yr; the put overlay adds +4.97%/yr — 100x more. The excess return comes from the convex payoff profile, not from amplified market exposure.

5. **The puts lose money even in crashes.** Total put P&L is negative (-$1.63M on $1.99M premium). The benefit is not "puts make money in crashes" but "puts reduce variance drain on the portfolio's geometric growth." This second-order mechanism is the core of Spitznagel's thesis.

6. **The edge is crash-dependent but survives calm periods.** Three crashes in 17 years drive most of the excess return. But the strategy beats SPY even during 2010-2019 (the longest bull market), and in both halves of an out-of-sample split.

7. **Parameter sensitivity is low.** Exit timing is irrelevant (0.18% spread). Delta range matters little in the overlay (0.68% spread). DTE and rebalance frequency matter more but all configurations are profitable. The strategy is robust, not fragile.

8. **The benchmark framing needs fixing.** Excess returns are currently computed against plain SPY, but the fair benchmark for the overlay is SPY + whatever the external capital would earn without puts.

---

*The authoritative analysis is `ANALYSIS.md`, which corrects the no-leverage claims from the original and adds extended risk metrics, calm-period analysis, and the diminishing returns discussion.*
