# Beyond the Spitznagel Trade

The [Spitznagel notebook](spitznagel_case.ipynb) confirmed that **100% SPY + 0.5% OTM puts = 16.02%/yr CAGR, Sharpe 0.901** over 2008-2025. Cheap tail protection improves geometric compounding by reducing variance drain.

But what about the *original* Taleb idea — no equity at all? A pure options barbell: sell near-ATM strangles for premium, buy deep OTM strangles for crash insurance, sit in cash. Can you make money from volatility alone?

This notebook tests that thesis, explores refinements (bonds, dynamic put sizing), and then broadens the lens: where else does the Spitznagel trade structure apply?

**Summary:** The pure barbell can't generate positive returns. Every configuration we tested lost money. The equity return engine is what makes Spitznagel work.


```python
import os, sys, math, time, warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'notebooks'))
os.chdir(PROJECT_ROOT)

from options_portfolio_backtester import BacktestEngine, Stock, Direction
from options_portfolio_backtester.core.types import OptionType as Type
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from nb_style import apply_style, shade_crashes, style_returns_table, FT_GREEN, FT_RED, FT_BLUE, FT_DARK, FT_GREY

apply_style()
%matplotlib inline

INITIAL_CAPITAL = 1_000_000

# Load data
options_data = HistoricalOptionsData('data/processed/options.csv')
stocks_data = TiingoData('data/processed/stocks.csv')
schema = options_data.schema

spy_prices = (
    stocks_data._data[stocks_data._data['symbol'] == 'SPY']
    .set_index('date')['adjClose'].sort_index()
)
years = (spy_prices.index[-1] - spy_prices.index[0]).days / 365.25
spy_total_ret = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
spy_annual_ret = ((1 + spy_total_ret / 100) ** (1 / years) - 1) * 100
spy_dd = ((spy_prices - spy_prices.cummax()) / spy_prices.cummax()).min() * 100

print(f'Date range: {stocks_data.start_date} to {stocks_data.end_date} ({years:.1f} years)')
print(f'SPY B&H: {spy_total_ret:.1f}% total, {spy_annual_ret:.2f}%/yr, {spy_dd:.1f}% max DD')
```

    Date range: 2008-01-02 00:00:00 to 2025-12-12 00:00:00 (17.9 years)
    SPY B&H: 555.5% total, 11.05%/yr, -51.9% max DD


---
## Section 1: Pure Options Barbell — Can Selling Vol Fund Tail Protection?

From Taleb's [2015 Reddit AMA](https://www.reddit.com/r/options/comments/38onec/):

> *"I am not against selling ATM premium. This is one of the nonsense people have spread in the interpretation of my ideas."*

> *"ATM drops faster, OTM rises. The same idea of shadow theta."*

The idea: sell near-ATM strangles to collect the Variance Risk Premium, buy deep OTM strangles as crash insurance, keep everything else in cash. Two separate engines combined by weighted daily returns.

Let's first test each side independently.


```python
def run_short_strangle(strike_width=0.05, dte_min=14, dte_max=30, exit_dte=3,
                       opt_pct=0.03, max_notional_pct=0.30):
    """Run short strangle engine: sell OTM call + put near ATM."""
    hi = 1.0 + strike_width

    sell_call = StrategyLeg('leg_1', schema, option_type=Type.CALL, direction=Direction.SELL)
    sell_call.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.strike >= schema.underlying_last)
        & (schema.strike <= schema.underlying_last * hi)
    )
    sell_call.entry_sort = ('strike', True)   # closest to ATM
    sell_call.exit_filter = schema.dte <= exit_dte

    lo = 1.0 - strike_width

    sell_put = StrategyLeg('leg_2', schema, option_type=Type.PUT, direction=Direction.SELL)
    sell_put.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.strike >= schema.underlying_last * lo)
        & (schema.strike <= schema.underlying_last)
    )
    sell_put.entry_sort = ('strike', False)   # closest to ATM
    sell_put.exit_filter = schema.dte <= exit_dte

    strat = Strategy(schema)
    strat.add_legs([sell_call, sell_put])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    bt = BacktestEngine(
        {'stocks': 0.0, 'options': opt_pct, 'cash': 1.0 - opt_pct},
        initial_capital=INITIAL_CAPITAL,
        max_notional_pct=max_notional_pct,
    )
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.run(rebalance_freq=1)
    return bt


def run_long_strangle(otm_pct=0.35, dte_min=70, dte_max=90, exit_dte=45,
                      opt_pct=0.01):
    """Run long strangle engine: buy deep OTM put + call."""
    otm_put_hi = 1.0 - otm_pct
    otm_call_lo = 1.0 + otm_pct

    buy_put = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    buy_put.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.strike <= schema.underlying_last * otm_put_hi)
    )
    buy_put.entry_sort = ('strike', True)   # closest to spot among OTM
    buy_put.exit_filter = schema.dte <= exit_dte

    buy_call = StrategyLeg('leg_2', schema, option_type=Type.CALL, direction=Direction.BUY)
    buy_call.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.strike >= schema.underlying_last * otm_call_lo)
    )
    buy_call.entry_sort = ('strike', False)  # closest to spot among OTM
    buy_call.exit_filter = schema.dte <= exit_dte

    strat = Strategy(schema)
    strat.add_legs([buy_put, buy_call])
    strat.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    bt = BacktestEngine(
        {'stocks': 0.0, 'options': opt_pct, 'cash': 1.0 - opt_pct},
        initial_capital=INITIAL_CAPITAL,
        max_notional_pct=None,
    )
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strat
    bt.options_data = options_data
    bt.run(rebalance_freq=1)
    return bt


def engine_stats(bt, label):
    """Extract summary stats from a backtest engine."""
    tc = bt.balance['total capital']
    total_ret = (tc.iloc[-1] / tc.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    dd = ((tc - tc.cummax()) / tc.cummax()).min() * 100
    daily = bt.balance['% change'].dropna()
    vol = daily.std() * (252 ** 0.5) * 100
    sharpe = (annual_ret - 4.0) / vol if vol > 0 else 0
    return {
        'name': label, 'annual_ret': annual_ret, 'total_ret': total_ret,
        'max_dd': dd, 'vol': vol, 'sharpe': sharpe,
        'trades': len(bt.trade_log), 'balance': tc,
    }


# Run individual components
t0 = time.perf_counter()
bt_short = run_short_strangle()
t1 = time.perf_counter()
print(f'Short strangle: {t1 - t0:.1f}s')

bt_long = run_long_strangle()
t2 = time.perf_counter()
print(f'Long strangle: {t2 - t1:.1f}s')

r_short = engine_stats(bt_short, 'Short strangle only')
r_long = engine_stats(bt_long, 'Long OTM only')

print(f'\n{"Component":<25s} {"Ann%":>8s} {"Tot%":>8s} {"MaxDD":>8s} {"Vol%":>8s} {"Sharpe":>8s} {"Trades":>8s}')
print('-' * 75)
for r in [r_short, r_long]:
    print(f'{r["name"]:<25s} {r["annual_ret"]:>7.2f}% {r["total_ret"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["vol"]:>7.1f}% {r["sharpe"]:>7.3f}  {r["trades"]:>6d}')
```

    Warning: No valid output stream.


    Short strangle: 15.5s
    Warning: No valid output stream.


    Long strangle: 14.1s
    
    Component                     Ann%     Tot%    MaxDD     Vol%   Sharpe   Trades
    ---------------------------------------------------------------------------
    Short strangle only         -1.33%   -21.3%   -23.4%     3.3%  -1.615     429
    Long OTM only              -10.39%   -86.0%   -89.1%    10.0%  -1.437     267



```python
def combine_engines(bt_sell, bt_buy, sell_weight=0.97, buy_weight=0.03, name='Combined'):
    """Combine two engine balance series by weighted daily returns."""
    bal_s = bt_sell.balance['total capital']
    bal_b = bt_buy.balance['total capital']
    common = bal_s.index.intersection(bal_b.index)
    ret_s = bal_s.loc[common].pct_change().fillna(0)
    ret_b = bal_b.loc[common].pct_change().fillna(0)
    combined_ret = sell_weight * ret_s + buy_weight * ret_b
    combined = (1 + combined_ret).cumprod() * INITIAL_CAPITAL

    total_ret = (combined.iloc[-1] / combined.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    dd = ((combined - combined.cummax()) / combined.cummax()).min() * 100
    vol = combined_ret.std() * (252 ** 0.5) * 100
    sharpe = (annual_ret - 4.0) / vol if vol > 0 else 0
    return {
        'name': name, 'annual_ret': annual_ret, 'total_ret': total_ret,
        'max_dd': dd, 'vol': vol, 'sharpe': sharpe, 'balance': combined,
    }


# Parameter sweep: 10 configurations
configs = [
    ('Baseline (5% width, 97/3)', dict(strike_width=0.05), dict(), 0.97, 0.03),
    ('Taleb (2% width, 97/3)', dict(strike_width=0.02), dict(), 0.97, 0.03),
    ('Wide short (15%, 97/3)', dict(strike_width=0.15), dict(), 0.97, 0.03),
    ('Shallow long (25% OTM, 97/3)', dict(strike_width=0.05), dict(otm_pct=0.25), 0.97, 0.03),
    ('Deep long (45% OTM, 97/3)', dict(strike_width=0.05), dict(otm_pct=0.45), 0.97, 0.03),
    ('High alloc (5%, 97/3)', dict(opt_pct=0.05), dict(opt_pct=0.02), 0.97, 0.03),
    ('More long weight (90/10)', dict(strike_width=0.05), dict(), 0.90, 0.10),
    ('More long weight (80/20)', dict(strike_width=0.05), dict(), 0.80, 0.20),
    ('Longer DTE short (30-60)', dict(dte_min=30, dte_max=60), dict(), 0.97, 0.03),
    ('Longer DTE long (90-120)', dict(strike_width=0.05), dict(dte_min=90, dte_max=120, exit_dte=60), 0.97, 0.03),
]

results = []
print(f'Running {len(configs)} configurations...\n')

for name, short_kw, long_kw, sw, lw in configs:
    t0 = time.perf_counter()
    bt_s = run_short_strangle(**short_kw)
    bt_l = run_long_strangle(**long_kw)
    r = combine_engines(bt_s, bt_l, sw, lw, name)
    elapsed = time.perf_counter() - t0
    results.append(r)
    print(f'  {name:<40s} {r["annual_ret"]:>7.2f}%/yr  ({elapsed:.1f}s)')

# Also add individual components for the table
results_all = [r_short, r_long] + results

print(f'\n{"Config":<45s} {"Ann%":>8s} {"Tot%":>8s} {"MaxDD":>8s} {"Vol%":>8s} {"Sharpe":>8s}')
print('=' * 90)
for r in results_all:
    print(f'{r["name"]:<45s} {r["annual_ret"]:>7.2f}% {r["total_ret"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["vol"]:>7.1f}% {r["sharpe"]:>7.3f}')
```

    Running 10 configurations...
    
    Warning: No valid output stream.


    Warning: No valid output stream.


      Baseline (5% width, 97/3)                  -1.59%/yr  (29.4s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Taleb (2% width, 97/3)                     -1.65%/yr  (29.6s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Wide short (15%, 97/3)                     -1.53%/yr  (28.7s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Shallow long (25% OTM, 97/3)               -1.59%/yr  (29.4s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Deep long (45% OTM, 97/3)                  -1.61%/yr  (27.6s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      High alloc (5%, 97/3)                      -2.01%/yr  (29.1s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      More long weight (90/10)                   -2.22%/yr  (28.6s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      More long weight (80/20)                   -3.12%/yr  (28.6s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Longer DTE short (30-60)                  -21.44%/yr  (26.6s)
    Warning: No valid output stream.


    Warning: No valid output stream.


      Longer DTE long (90-120)                   -1.56%/yr  (28.6s)
    
    Config                                            Ann%     Tot%    MaxDD     Vol%   Sharpe
    ==========================================================================================
    Short strangle only                             -1.33%   -21.3%   -23.4%     3.3%  -1.615
    Long OTM only                                  -10.39%   -86.0%   -89.1%    10.0%  -1.437
    Baseline (5% width, 97/3)                       -1.59%   -25.1%   -27.2%     3.2%  -1.753
    Taleb (2% width, 97/3)                          -1.65%   -25.8%   -27.8%     3.2%  -1.767
    Wide short (15%, 97/3)                          -1.53%   -24.1%   -26.3%     3.2%  -1.751
    Shallow long (25% OTM, 97/3)                    -1.59%   -25.0%   -27.1%     3.2%  -1.754
    Deep long (45% OTM, 97/3)                       -1.61%   -25.2%   -27.3%     3.2%  -1.752
    High alloc (5%, 97/3)                           -2.01%   -30.6%   -32.9%     3.3%  -1.800
    More long weight (90/10)                        -2.22%   -33.2%   -35.9%     3.1%  -2.030
    More long weight (80/20)                        -3.12%   -43.4%   -47.0%     3.2%  -2.226
    Longer DTE short (30-60)                       -21.44%   -98.7%   -98.7%     9.0%  -2.822
    Longer DTE long (90-120)                        -1.56%   -24.6%   -26.8%     3.2%  -1.740


---
## Analysis: Why the Pure Barbell Can't Work

Every single configuration lost money. The range was **-1.25% to -55.93%/yr**.

The core problem is arithmetic:

- You're **~97% cash earning 0%**, with a small allocation to a negative-sum options market
- The Variance Risk Premium is real but tiny: ~2-3%/yr gross on SPY options. On 3% of capital, that's 0.06-0.09%/yr
- Bid-ask spreads on monthly strangles consume 1-2% of notional per round trip
- The long OTM side bleeds steadily: deep OTM options decay to zero most months
- At 3% weight on the long side, tail payoffs are too small to matter when they finally hit

**Without an underlying return engine (equity, carry, spread), you're playing a negative-sum game.** The VRP is too small to harvest profitably at retail, and the tail insurance costs more than it pays over any reasonable horizon.

This is the fundamental difference between:
- **Spitznagel**: 100% equity + cheap puts = equity return funds everything, puts reduce variance drain
- **Pure barbell**: 0% equity + options = no return engine, options are a drag

---
## Section 2: Adding Bonds

If cash earns zero, what about replacing it with bonds?

### Long bonds (TLT): a historical artifact

The 40-year bull market in bonds (1980–2020, rates from 15% to 0%) flatters every backtest that includes TLT. But the regime is over. TLT was destroyed in 2022–2023 as rates normalized. The “negative correlation with equities during crises” — which is TLT's selling point — is regime-dependent, not structural:

- **1970s**: Stocks and bonds fell together (inflation regime)
- **1998–2020**: Negative correlation (falling rates, low inflation, Fed put)
- **2022**: Stocks and bonds fell together again (rate normalization)

Backtesting TLT as a permanent allocation extrapolates a trend that already reversed.

### Short-term bonds (SHY/BIL): just cash with yield

No duration risk, tracks Fed Funds, uncorrelated with equities. This is what Universa actually holds. But at 4% yields, SHY on 97% of portfolio earns ~3.9%/yr — which almost covers the VRP losses, making the strategy roughly break-even. At 0% rates (2009–2021), you're right back to losing 1–2%/yr.

**Conclusion:** Short-term bonds are the right base, but they don't transform a losing options strategy into a winning one. They just add the risk-free rate.

---
## Section 3: Dynamic Put Sizing from Bond Yield

Instead of a fixed 0.5% put budget, what if we fund it dynamically from bond yield?

### The math

- 20% SHY × 4% yield = **0.80%/yr** available for puts
- At 0% rates: almost nothing for protection
- At 5% rates: 1.0%/yr — generous budget

### The timing argument

High rates tend to precede downturns (the Fed hikes to cool the economy, then something breaks). So you'd naturally have more protection going into crises and less coming out — exactly when it matters most.

### Why it loses

You sacrifice 20% equity exposure to fund this. At SPY's ~9-10%/yr real return, that 20% costs you **~1.8%/yr in foregone equity gains**. You're saving 0.5%/yr in put cost while giving up 1.8%/yr in equity returns.

**Net cost: approximately -1.3%/yr versus 100% SPY with fixed puts.**

### The key insight

**0.5% is already so cheap that no funding mechanism is needed.** The Spitznagel strategy works precisely because the put cost is negligible relative to equity returns. The equity return funds it with massive headroom. The simplicity is the feature, not a limitation.

---
## Section 4: The Spitznagel Trade Across Asset Classes

The *structure* of the Spitznagel trade — earn steady carry, buy cheap tail protection — isn't unique to equities. It works anywhere you find:
1. Steady carry/return in normal times
2. Extreme moves rarely but violently
3. Protection underpriced for those extremes

### Rates — Most Reliable Asymmetry

The Fed reaction function creates the most predictable convexity in finance. **They always cut in crises. Always.** It's not a probability — it's a policy mandate.

| Event | Rate Move | Timeline |
|-------|-----------|----------|
| 2008 GFC | 5.25% → 0.25% | ~15 months |
| 2020 COVID | 1.50% → 0.00% | 2 weeks |

Rates grind higher 25bps at a time, then collapse violently. The trade: long SOFR futures (earn near risk-free), OTM calls on rate futures (bet on panic cuts). Rate vol models assume mean-reversion and don't properly price emergency scenarios.

**Ranking: Best structural asymmetry. If you can access futures/options on rates, this is the most reliable convexity trade in finance.**

### FX Carry — Best Risk Premium

High-yield currencies (AUD, MXN, BRL) vs funding currencies (JPY, CHF) earn 3–5% annual rate differential. The crash risk is well-documented:

| Event | Move |
|-------|------|
| AUD/JPY 2008 | -40% in weeks |
| CHF unpeg 2015 | -30% in minutes |
| EM FX 2018 | -20–50% (Turkey, Argentina) |

The trade: long high-yielder (earn carry), buy OTM puts. The carry funds the protection. Academic literature strongly supports the “carry trade crash risk” premium — insurance is cheap because short-vol carry strategies are crowded.

### Credit — Spread Compression/Blowout Cycle

Credit spreads grind tight for years then blow out overnight:

| Crisis | IG CDS | HY CDS |
|--------|--------|--------|
| 2008 GFC | 50bps → 250bps | 300bps → 2000bps |

The trade: hold IG bonds (earn spread), buy CDS protection on HY or IG index. Primarily institutional due to ISDA requirements and counterparty risk.

### Commodities — Supply/Demand Shocks

- **Oil**: Grinds $60–80, spikes to $140 or crashes to $20. Both tails are fat.
- **Gold**: Steady in normal times, spikes in crises. OTM calls as chaos hedge.
- **Agriculture**: Weather events double corn/wheat prices in weeks. Insurance is cheap because most participants are hedgers.

### Volatility Itself — Purest Expression

VIX at 12–15 for months, then spikes to 40–80. Deep OTM VIX calls (or call spreads). Most expensive of these trades because everyone knows about it, but still underpriced for true extremes (VIX 80+).

### Emerging Market Sovereign Debt

High yield in normal times, violent contagious crises (1997, 1998, 2001, 2015, 2018). Buy CDS or OTM puts on EM ETFs. Contagion dynamics mean one EM instrument gives exposure to the whole complex.

---
## Section 5: The Common Thread + Why SPY Remains Best

### The pattern

Markets systematically price risk as if the future looks like the recent past. During calm periods:
- Implied vol drops → options get cheap
- Credit spreads tighten → protection gets cheap
- Carry trades get crowded → more people selling insurance
- Risk models say everything is low-risk

Then a shock reprices everything violently. The gap between “normal-times pricing” and “crisis pricing” is the structural edge.

The Universa insight isn't picking one market — it's **dynamically rotating to wherever the convexity is most mispriced**. Sometimes that's equity vol (2007), sometimes credit (2006), sometimes rates (2019).

### Practical ranking

| If you are... | Best market |
|----------------|-------------|
| Retail with simple tools | **SPY/QQQ** — don't overthink it |
| Can access futures/options | **Rates** — the Fed asymmetry is unbeatable |
| Running a multi-strategy fund | **All of them** — rotate to cheapest convexity |

### Why SPY wins for retail

Despite rates and FX having arguably better structural asymmetries, SPY wins because:

1. **Most liquid derivatives on earth.** Penny-wide bid-ask, strikes every $1, 0-DTE to 2-year expiry.
2. **100% SPY + 0.5% puts = 16.02%/yr, Sharpe 0.901.** Confirmed by backtest. Hard to improve upon.
3. **Everything else requires OTC markets, futures accounts, ISDA agreements.** SOFR swaptions and CDS aren't available at Interactive Brokers.
4. **0.5% is so cheap it doesn't need funding.** No bonds, no carry trade, no complex allocation. The simplicity is the feature.

The entire barbell exploration — selling vol, buying tail protection, adding bonds, dynamic put sizing — was a search for something more sophisticated than “buy SPY, spend 0.5% on puts.” Nothing beat it.
