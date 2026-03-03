# Cross-Asset Spitznagel Tail-Hedge Research

Systematic evaluation of Mark Spitznagel's tail-hedging thesis across five asset
classes: **S&P 500 equity**, **US Treasuries**, **US-UK bond carry**, **commodity
futures**, and a **combined multi-strategy portfolio**.

The core structure is always the same: leveraged long exposure to a positive-carry
asset, hedged with cheap deep-OTM puts that cost a fraction of a percent per month
but pay 50-100x during tail events.

**Data**: Real CME/ICE futures and options from Databento, 2010-2026.

## How to Run

```bash
# Run all five analyses (charts saved to charts/)
python run.py

# Run a single analysis
python run.py equity
python run.py treasury
python run.py bond_carry
python run.py commodity
python run.py combined

# Run a subset
python run.py equity combined
```

Each script writes PNG charts to `charts/` with a prefix:
`eq_`, `tr_`, `bc_`, `cmd_`, `comb_`.

## 1. S&P 500 Equity (`run_equity.py`)

The canonical Spitznagel application. Long ES futures at multiple leverage levels
(1x through 10x) with monthly ~8% OTM puts at three budget tiers (0.3%, 0.5%, 1.0%).

**Key findings**:
- Equity is the ideal Spitznagel asset: strong positive base return (~10% CAGR from
  the equity risk premium) funds the hedge cost.
- At 1x, the 0.5% put budget improves Sharpe without meaningful drag.
- At 3x, puts are essential -- they cut max drawdown materially during COVID (-34%
  vs -67% unhedged).
- Kelly-optimal leverage with puts is around 3-5x, versus 2-3x without.
- Put win rate is low (~10%) but average winning payoff is massive, consistent with
  Spitznagel's thesis that cheap OTM puts are systematically underpriced.

## 2. US Treasuries (`run_treasury.py`)

Applies the structure to ZN (10-Year T-Note) and ZB (30-Year T-Bond). Treasuries
rally during equity crises (flight-to-quality), so put protection here hedges the
opposite tail: rate hikes.

**Key findings**:
- ZN at moderate leverage (2-3x) produces equity-like returns with a different risk
  profile (anti-correlated with equities).
- ZB has more duration/convexity -- higher potential return but also higher rate-hike
  risk (2022 was devastating).
- OTM puts on ZN/ZB paid off handsomely during the 2022 Fed hiking cycle.
- Treasury + puts is naturally anti-correlated with equity + puts, making it an
  excellent diversifier in the combined portfolio.

## 3. US-UK Bond Carry (`run_bond_carry.py`)

Relative value strategy: long the higher-yielding bond, short the lower-yielding
bond (ZN vs Long Gilt). Uses **directional** OZN options -- puts when long ZN,
calls when short ZN -- switching with the carry signal.

Tests six OTM levels (4% to 30%) to validate Spitznagel's thesis that deeper OTM
is more capital-efficient.

**Key findings**:
- The carry trade itself is modestly positive (US-UK yield differential).
- Directional hedging with puts AND calls protects against both tails.
- The 2022 Liz Truss crisis (Gilt crash) and 2022 Fed hikes both generated
  meaningful put payoffs.
- Deeper OTM (20-30%) is more efficient per dollar spent, confirming the thesis.

## 4. Commodity Carry (`run_commodity.py`)

Tests four commodities: Gold (GC/OG), Crude Oil (CL/LO), Copper (HG/HXE), and
Natural Gas (NG/ON). "Carry" here is roll yield from the term structure.

**Key findings**:
- All four commodities have persistent **negative** roll yield (contango):
  Gold -5.4%/yr, Crude -3.5%/yr, Copper -4.3%/yr, NatGas -22.3%/yr.
- Unlike FX or equity carry, commodity "carry" is a drag, not income.
- Puts improve Sharpe across the board (biggest for Crude: +0.35 at 1x) but cannot
  rescue a negative-carry asset.
- At 3x+ leverage, ALL commodities blow up (max DD > 94%).
- Put economics: Crude is the star (19.6% win rate, 5.6x avg payout, 118x best).
- The Spitznagel structure requires positive base carry to work. Commodity contango
  destroys the base return.

## 5. Combined Portfolio (`run_combined.py`)

Combines three uncorrelated hedged strategies:
1. S&P 500 (ES 3x + 25% OTM puts, 0.3% budget)
2. FX Carry (6 pairs vs JPY, 1x hedged, 8% OTM puts, 0.5% budget)
3. US-UK Bond Carry (Long ZN / Short Gilt 3x + 4% OTM puts, 0.3% budget)

Analyses cross-strategy dependencies with Pearson correlation and mutual information,
then constructs four portfolio types: Equal-Weight, Risk-Parity (60-day vol, monthly
rebalance), Min-Variance (252-day cov), and Max-Sharpe (252-day, quarterly rebalance).

**Key findings**:
- Cross-strategy correlations are near zero, making this ideal for diversification.
- Portfolio Sharpe exceeds any individual strategy (diversification benefit).
- Risk-Parity tends to perform well due to stable vol-targeting.
- The combined portfolio achieves strong risk-adjusted returns with bounded tail risk
  from the OTM puts on each individual leg.

## Cross-Asset Ranking (Spitznagel Structure Effectiveness)

| Rank | Asset | Why |
|------|-------|-----|
| 1st | S&P 500 equity | Strong ERP, deepest option liquidity |
| 2nd | FX carry (AUD+MXN/JPY) | Positive carry, decent put payoffs |
| 3rd | US Treasuries | Flight-to-quality return, rate-hike hedge |
| 4th | US-UK bond carry | Modest carry, directional hedge works |
| 5th | Gold | Marginal return, occasional crisis alpha |
| 6th | Crude oil | Negative carry, but huge put payoffs |
| 7th | Copper / NatGas | Negative carry + illiquid options |

## The Spitznagel Thesis Validated

The structure works best on assets with:
1. **Strong positive base returns** (equity > FX carry >> commodities)
2. **Liquid, well-priced OTM options** (ES > FX > commodities)
3. **Occasional fat-tail events** that make cheap OTM puts pay 50-100x
4. **Positive carry to fund the hedge cost** -- this is the key insight

## File Structure

```
cross_asset_spitznagel/
  run.py              -- orchestrator (runs all or selected analyses)
  run_equity.py       -- S&P 500 + tail hedge
  run_treasury.py     -- US Treasuries + tail hedge
  run_bond_carry.py   -- US-UK bond carry + directional hedge
  run_commodity.py    -- Commodity carry + OTM puts
  run_combined.py     -- Multi-strategy portfolio
  charts/             -- output PNGs (eq_*, tr_*, bc_*, cmd_*, comb_*)
  README.md           -- this file
```
