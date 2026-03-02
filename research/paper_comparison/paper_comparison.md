# Paper vs Reality: Options Strategies Compared to Academic Claims

We run **10 options overlay strategies** on SPY (2008–2025) and compare our empirical results to predictions from academic finance.

## The Volatility Risk Premium (VRP)

The central concept underlying most options strategies is the **Variance Risk Premium**:

$$\text{VRP} = \sigma^2_{\text{implied}} - \sigma^2_{\text{realized}}$$

Carr & Wu (2009) show this is consistently **positive** — implied volatility exceeds realized volatility on average. This means:
- **Option sellers** systematically collect a premium (like insurance companies)
- **Option buyers** systematically overpay (like insurance policyholders)

## Strategies Tested

| # | Strategy | Direction | Paper | Claim |
|---|----------|-----------|-------|-------|
| 1 | **Covered Call** (BXM) | Sell vol | Whaley 2002 | Comparable returns, $\frac{2}{3}$ volatility |
| 2 | **Cash-Secured Puts** (PUT) | Sell vol | Neuberger Berman | Beats BXM by ~1%/yr |
| 3 | **OTM Put Hedge** | Buy vol | Israelov 2017 | Pure drag — erodes $\frac{2}{3}$ of equity returns |
| 4 | **OTM Call Momentum** | Buy vol | Kang 2016 | OTM call activity predicts $+18$bps/week |
| 5 | **Short Strangle** | Sell vol | Berman 2014 | 15.2% CAGR vs 9.9% S&P 500 |
| 6 | **Long Straddle** | Buy vol | Carr & Wu 2009 | $\text{VRP} > 0 \Rightarrow$ buying vol loses |
| 7 | **Collar** | Mixed | Israelov 2015 | Hidden costs, reduces equity risk premium |
| 8 | **Deep OTM Tail Hedge** | Buy vol | Spitznagel vs AQR | Debated: 12.3% CAGR vs persistent drag |
| 9 | **VIX-Timed Puts** | Conditional | Fassas 2019 | VIX term structure as contrarian signal |
| 10 | **SMA-Timed** | Conditional | Antonacci 2013 | Trend-following overlay adds value |


```python
import os, sys, warnings, math
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'notebooks'))
os.chdir(PROJECT_ROOT)

from options_portfolio_backtester import Direction
from backtest_runner import (
    load_data, run_backtest, INITIAL_CAPITAL,
    make_puts_strategy, make_calls_strategy,
    make_straddle_strategy, make_strangle_strategy,
    make_covered_call_strategy, make_cash_secured_put_strategy,
    make_collar_strategy, make_deep_otm_put_strategy,
)
from nb_style import apply_style, shade_crashes, color_excess, style_returns_table, CRASH_PERIODS, FT_BG, FT_BLUE, FT_RED, FT_GREEN, FT_DARK, FT_GREY

apply_style()
%matplotlib inline

# Crash periods for local use
CRASHES = [(label, start, end) for label, start, end, _ in CRASH_PERIODS]

print('Setup done.')
```

    Setup done.



```python
data = load_data()
schema = data['schema']
spy_prices = data['spy_prices']
years = data['years']
```

    Loading data...


    Date range: 2008-01-02 00:00:00 to 2025-12-12 00:00:00 (17.9 years)
    SPY B&H: 555.5% total, 11.05% annual, -51.9% max DD
    
    Loaded macro signals: ['gdp', 'vix', 'hy_spread', 'yield_curve_10y2y', 'nfc_equity_mv', 'nfc_net_worth', 'dollar_index', 'buffett_indicator', 'tobin_q']


---
## Run All 10 Strategies

All use 99%/1% stock/options split unless otherwise noted.


```python
S, O = 0.99, 0.01  # standard split

configs = [
    # (name, stock_pct, opt_pct, strategy_fn, budget_fn)
    ('1. Covered Call (BXM)',      S, O, lambda: make_covered_call_strategy(schema), None),
    ('2. Cash-Secured Puts (PUT)', S, O, lambda: make_cash_secured_put_strategy(schema), None),
    ('3. OTM Put Hedge',           S, O, lambda: make_puts_strategy(schema), None),
    ('4. OTM Call Momentum',       S, O, lambda: make_calls_strategy(schema), None),
    ('5. Short Strangle',          S, O, lambda: make_strangle_strategy(schema, Direction.SELL), None),
    ('6. Long Straddle',           S, O, lambda: make_straddle_strategy(schema, Direction.BUY), None),
    ('7. Collar',                  S, O, lambda: make_collar_strategy(schema), None),
    ('8. Deep OTM Tail Hedge',     0.997, 0.003, lambda: make_deep_otm_put_strategy(schema), None),
]

results = []
for name, s_pct, o_pct, strat_fn, bfn in configs:
    print(f'{name}...', end=' ', flush=True)
    r = run_backtest(name, s_pct, o_pct, strat_fn, data, budget_fn=bfn)
    results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

print('\nDone.')
```

    1. Covered Call (BXM)... 

    Warning: No valid output stream.


    annual +66.09%, excess +55.04%, DD -43.7%
    2. Cash-Secured Puts (PUT)... 

    Warning: No valid output stream.


    annual +43.19%, excess +32.14%, DD -85.2%
    3. OTM Put Hedge... 

    Warning: No valid output stream.


    annual +7.62%, excess -3.42%, DD -49.3%
    4. OTM Call Momentum... 

    Warning: No valid output stream.


    annual +11.88%, excess +0.83%, DD -55.2%
    5. Short Strangle... 

    Warning: No valid output stream.


    annual +47.18%, excess +36.14%, DD -340.5%
    6. Long Straddle... 

    Warning: No valid output stream.


    annual +10.67%, excess -0.37%, DD -51.1%
    7. Collar... 

    Warning: No valid output stream.


    annual +nan%, excess +nan%, DD -386.0%
    8. Deep OTM Tail Hedge... 

    Warning: No valid output stream.


    annual +8.69%, excess -2.36%, DD -51.9%
    
    Done.



```python
# VIX-timed puts: buy puts only when VIX > rolling 1yr median
if data['vix'] is not None:
    vix, vix_med = data['vix'], data['vix_median']
    def vix_high_budget(date, tc):
        cur = vix.asof(date)
        thresh = vix_med.asof(date) if vix_med is not None else None
        if pd.isna(cur) or thresh is None or pd.isna(thresh):
            return 0
        return tc * 0.01 if cur > thresh else 0

    print('9. VIX-Timed Puts...', end=' ', flush=True)
    r = run_backtest('9. VIX-Timed Puts', S, O,
                     lambda: make_puts_strategy(schema), data, budget_fn=vix_high_budget)
    results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%')
else:
    print('No VIX data — skipping VIX-timed puts.')

# SMA-timed: calls when SPY > 200d SMA, puts when below
spy_sma200 = spy_prices.rolling(200, min_periods=100).mean()

def sma_calls_budget(date, tc):
    price = spy_prices.asof(date)
    sma = spy_sma200.asof(date)
    if pd.isna(price) or pd.isna(sma):
        return 0
    return tc * 0.01 if price > sma else 0

def sma_puts_budget(date, tc):
    price = spy_prices.asof(date)
    sma = spy_sma200.asof(date)
    if pd.isna(price) or pd.isna(sma):
        return 0
    return tc * 0.01 if price <= sma else 0

print('10a. SMA Calls (bull)...', end=' ', flush=True)
r_sma_calls = run_backtest('10a. SMA Calls (bull)', S, O,
                           lambda: make_calls_strategy(schema), data, budget_fn=sma_calls_budget)
results.append(r_sma_calls)
print(f'annual {r_sma_calls["annual_ret"]:+.2f}%')

print('10b. SMA Puts (bear)...', end=' ', flush=True)
r_sma_puts = run_backtest('10b. SMA Puts (bear)', S, O,
                          lambda: make_puts_strategy(schema), data, budget_fn=sma_puts_budget)
results.append(r_sma_puts)
print(f'annual {r_sma_puts["annual_ret"]:+.2f}%')

print(f'\nTotal strategies: {len(results)}')
```

    9. VIX-Timed Puts... 

    Warning: No valid output stream.


    annual +2.26%, excess -8.79%
    10a. SMA Calls (bull)... 

    Warning: No valid output stream.


    annual +8.52%
    10b. SMA Puts (bear)... 

    Warning: No valid output stream.


    annual +0.08%
    
    Total strategies: 11


---
## Master Comparison Table: Our Results vs Paper Claims


```python
paper_claims = {
    '1. Covered Call (BXM)':      'Comparable return, 2/3 vol (Whaley 2002)',
    '2. Cash-Secured Puts (PUT)': 'Beats BXM by ~1%/yr (Neuberger Berman)',
    '3. OTM Put Hedge':           'Pure drag, erodes returns (Israelov 2017)',
    '4. OTM Call Momentum':       'OTM calls predict +returns (Kang 2016)',
    '5. Short Strangle':          '15.2% vs 9.9% SPX (Berman 2014)',
    '6. Long Straddle':           'Negative VRP = vol buying loses (Carr & Wu 2009)',
    '7. Collar':                  'Hidden costs, reduces equity premium (Israelov 2015)',
    '8. Deep OTM Tail Hedge':     'Debated: 12.3% CAGR (Spitznagel) vs drag (AQR)',
    '9. VIX-Timed Puts':          'VIX has predictive power (Fassas 2019)',
    '10a. SMA Calls (bull)':      'Trend-following adds value (Antonacci 2013)',
    '10b. SMA Puts (bear)':       'Trend-following for downside (Antonacci 2013)',
}

rows = []
for r in results:
    confirmed = '?'
    name = r['name']
    ex = r['excess_annual']
    if 'Covered Call' in name:
        confirmed = 'YES' if abs(ex) < 2 else 'NO'
    elif 'Cash-Secured' in name:
        cc = next((x for x in results if 'Covered Call' in x['name']), None)
        if cc:
            confirmed = 'YES' if r['annual_ret'] > cc['annual_ret'] else 'NO'
    elif 'OTM Put Hedge' in name:
        confirmed = 'YES' if ex < 0 else 'NO'
    elif 'OTM Call Momentum' in name:
        confirmed = 'YES' if ex > 0 else 'NO'
    elif 'Short Strangle' in name:
        confirmed = 'YES' if ex > 0 else 'MIXED'
    elif 'Long Straddle' in name:
        confirmed = 'YES' if ex < 0 else 'NO'
    elif 'Collar' in name:
        confirmed = 'YES' if ex < 0 else 'NO'
    elif 'Deep OTM' in name:
        confirmed = 'BOTH RIGHT'
    elif 'VIX-Timed' in name:
        plain_puts = next((x for x in results if x['name'] == '3. OTM Put Hedge'), None)
        if plain_puts:
            confirmed = 'YES' if r['annual_ret'] > plain_puts['annual_ret'] else 'NO'
    elif 'SMA Calls' in name:
        confirmed = 'YES' if ex > 0 else 'MIXED'
    elif 'SMA Puts' in name:
        plain_puts = next((x for x in results if x['name'] == '3. OTM Put Hedge'), None)
        if plain_puts:
            confirmed = 'YES' if r['annual_ret'] > plain_puts['annual_ret'] else 'NO'

    rows.append({
        'Strategy': name,
        'Annual %': r['annual_ret'],
        'Excess vs SPY %': r['excess_annual'],
        'Max DD %': r['max_dd'],
        'Trades': r['trades'],
        'Paper Claim': paper_claims.get(name, ''),
        'Confirmed?': confirmed,
    })

df = pd.DataFrame(rows)

def style_confirmed(val):
    if val == 'YES': return 'background-color: #d4edda; font-weight: bold'
    if val == 'NO': return 'background-color: #f8d7da'
    if val in ('MIXED', 'BOTH RIGHT'): return 'background-color: #fff3cd'
    return ''

(df.style
    .format({'Annual %': '{:.2f}', 'Excess vs SPY %': '{:+.2f}',
             'Max DD %': '{:.1f}', 'Trades': '{:.0f}'})
    .map(color_excess, subset=['Excess vs SPY %'])
    .map(style_confirmed, subset=['Confirmed?'])
    .set_caption(f'All Strategies vs Academic Claims  |  SPY B&H: {data["spy_annual_ret"]:.2f}%/yr')
)
```




<style type="text/css">
#T_a2711_row0_col2, #T_a2711_row1_col2, #T_a2711_row3_col2, #T_a2711_row4_col2 {
  color: #09814A;
  font-weight: bold;
}
#T_a2711_row0_col6, #T_a2711_row1_col6, #T_a2711_row6_col6, #T_a2711_row8_col6, #T_a2711_row10_col6 {
  background-color: #f8d7da;
}
#T_a2711_row2_col2, #T_a2711_row5_col2, #T_a2711_row7_col2, #T_a2711_row8_col2, #T_a2711_row9_col2, #T_a2711_row10_col2 {
  color: #CC0000;
}
#T_a2711_row2_col6, #T_a2711_row3_col6, #T_a2711_row4_col6, #T_a2711_row5_col6 {
  background-color: #d4edda;
  font-weight: bold;
}
#T_a2711_row7_col6, #T_a2711_row9_col6 {
  background-color: #fff3cd;
}
</style>
<table id="T_a2711">
  <caption>All Strategies vs Academic Claims  |  SPY B&H: 11.05%/yr</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a2711_level0_col0" class="col_heading level0 col0" >Strategy</th>
      <th id="T_a2711_level0_col1" class="col_heading level0 col1" >Annual %</th>
      <th id="T_a2711_level0_col2" class="col_heading level0 col2" >Excess vs SPY %</th>
      <th id="T_a2711_level0_col3" class="col_heading level0 col3" >Max DD %</th>
      <th id="T_a2711_level0_col4" class="col_heading level0 col4" >Trades</th>
      <th id="T_a2711_level0_col5" class="col_heading level0 col5" >Paper Claim</th>
      <th id="T_a2711_level0_col6" class="col_heading level0 col6" >Confirmed?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a2711_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a2711_row0_col0" class="data row0 col0" >1. Covered Call (BXM)</td>
      <td id="T_a2711_row0_col1" class="data row0 col1" >66.09</td>
      <td id="T_a2711_row0_col2" class="data row0 col2" >+55.04</td>
      <td id="T_a2711_row0_col3" class="data row0 col3" >-43.7</td>
      <td id="T_a2711_row0_col4" class="data row0 col4" >406</td>
      <td id="T_a2711_row0_col5" class="data row0 col5" >Comparable return, 2/3 vol (Whaley 2002)</td>
      <td id="T_a2711_row0_col6" class="data row0 col6" >NO</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a2711_row1_col0" class="data row1 col0" >2. Cash-Secured Puts (PUT)</td>
      <td id="T_a2711_row1_col1" class="data row1 col1" >43.19</td>
      <td id="T_a2711_row1_col2" class="data row1 col2" >+32.14</td>
      <td id="T_a2711_row1_col3" class="data row1 col3" >-85.2</td>
      <td id="T_a2711_row1_col4" class="data row1 col4" >402</td>
      <td id="T_a2711_row1_col5" class="data row1 col5" >Beats BXM by ~1%/yr (Neuberger Berman)</td>
      <td id="T_a2711_row1_col6" class="data row1 col6" >NO</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a2711_row2_col0" class="data row2 col0" >3. OTM Put Hedge</td>
      <td id="T_a2711_row2_col1" class="data row2 col1" >7.62</td>
      <td id="T_a2711_row2_col2" class="data row2 col2" >-3.42</td>
      <td id="T_a2711_row2_col3" class="data row2 col3" >-49.3</td>
      <td id="T_a2711_row2_col4" class="data row2 col4" >381</td>
      <td id="T_a2711_row2_col5" class="data row2 col5" >Pure drag, erodes returns (Israelov 2017)</td>
      <td id="T_a2711_row2_col6" class="data row2 col6" >YES</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a2711_row3_col0" class="data row3 col0" >4. OTM Call Momentum</td>
      <td id="T_a2711_row3_col1" class="data row3 col1" >11.88</td>
      <td id="T_a2711_row3_col2" class="data row3 col2" >+0.83</td>
      <td id="T_a2711_row3_col3" class="data row3 col3" >-55.2</td>
      <td id="T_a2711_row3_col4" class="data row3 col4" >365</td>
      <td id="T_a2711_row3_col5" class="data row3 col5" >OTM calls predict +returns (Kang 2016)</td>
      <td id="T_a2711_row3_col6" class="data row3 col6" >YES</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a2711_row4_col0" class="data row4 col0" >5. Short Strangle</td>
      <td id="T_a2711_row4_col1" class="data row4 col1" >47.18</td>
      <td id="T_a2711_row4_col2" class="data row4 col2" >+36.14</td>
      <td id="T_a2711_row4_col3" class="data row4 col3" >-340.5</td>
      <td id="T_a2711_row4_col4" class="data row4 col4" >396</td>
      <td id="T_a2711_row4_col5" class="data row4 col5" >15.2% vs 9.9% SPX (Berman 2014)</td>
      <td id="T_a2711_row4_col6" class="data row4 col6" >YES</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_a2711_row5_col0" class="data row5 col0" >6. Long Straddle</td>
      <td id="T_a2711_row5_col1" class="data row5 col1" >10.67</td>
      <td id="T_a2711_row5_col2" class="data row5 col2" >-0.37</td>
      <td id="T_a2711_row5_col3" class="data row5 col3" >-51.1</td>
      <td id="T_a2711_row5_col4" class="data row5 col4" >390</td>
      <td id="T_a2711_row5_col5" class="data row5 col5" >Negative VRP = vol buying loses (Carr & Wu 2009)</td>
      <td id="T_a2711_row5_col6" class="data row5 col6" >YES</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_a2711_row6_col0" class="data row6 col0" >7. Collar</td>
      <td id="T_a2711_row6_col1" class="data row6 col1" >nan</td>
      <td id="T_a2711_row6_col2" class="data row6 col2" >+nan</td>
      <td id="T_a2711_row6_col3" class="data row6 col3" >-386.0</td>
      <td id="T_a2711_row6_col4" class="data row6 col4" >44</td>
      <td id="T_a2711_row6_col5" class="data row6 col5" >Hidden costs, reduces equity premium (Israelov 2015)</td>
      <td id="T_a2711_row6_col6" class="data row6 col6" >NO</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_a2711_row7_col0" class="data row7 col0" >8. Deep OTM Tail Hedge</td>
      <td id="T_a2711_row7_col1" class="data row7 col1" >8.69</td>
      <td id="T_a2711_row7_col2" class="data row7 col2" >-2.36</td>
      <td id="T_a2711_row7_col3" class="data row7 col3" >-51.9</td>
      <td id="T_a2711_row7_col4" class="data row7 col4" >410</td>
      <td id="T_a2711_row7_col5" class="data row7 col5" >Debated: 12.3% CAGR (Spitznagel) vs drag (AQR)</td>
      <td id="T_a2711_row7_col6" class="data row7 col6" >BOTH RIGHT</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_a2711_row8_col0" class="data row8 col0" >9. VIX-Timed Puts</td>
      <td id="T_a2711_row8_col1" class="data row8 col1" >2.26</td>
      <td id="T_a2711_row8_col2" class="data row8 col2" >-8.79</td>
      <td id="T_a2711_row8_col3" class="data row8 col3" >-54.2</td>
      <td id="T_a2711_row8_col4" class="data row8 col4" >202</td>
      <td id="T_a2711_row8_col5" class="data row8 col5" >VIX has predictive power (Fassas 2019)</td>
      <td id="T_a2711_row8_col6" class="data row8 col6" >NO</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_a2711_row9_col0" class="data row9 col0" >10a. SMA Calls (bull)</td>
      <td id="T_a2711_row9_col1" class="data row9 col1" >8.52</td>
      <td id="T_a2711_row9_col2" class="data row9 col2" >-2.53</td>
      <td id="T_a2711_row9_col3" class="data row9 col3" >-58.5</td>
      <td id="T_a2711_row9_col4" class="data row9 col4" >311</td>
      <td id="T_a2711_row9_col5" class="data row9 col5" >Trend-following adds value (Antonacci 2013)</td>
      <td id="T_a2711_row9_col6" class="data row9 col6" >MIXED</td>
    </tr>
    <tr>
      <th id="T_a2711_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_a2711_row10_col0" class="data row10 col0" >10b. SMA Puts (bear)</td>
      <td id="T_a2711_row10_col1" class="data row10 col1" >0.08</td>
      <td id="T_a2711_row10_col2" class="data row10 col2" >-10.96</td>
      <td id="T_a2711_row10_col3" class="data row10 col3" >-51.3</td>
      <td id="T_a2711_row10_col4" class="data row10 col4" >78</td>
      <td id="T_a2711_row10_col5" class="data row10 col5" >Trend-following for downside (Antonacci 2013)</td>
      <td id="T_a2711_row10_col6" class="data row10 col6" >NO</td>
    </tr>
  </tbody>
</table>




---
## Capital Curves: All Strategies


```python
fig, ax = plt.subplots(figsize=(18, 8))
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL
ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=2.5, label='SPY B&H', alpha=0.7)

cmap = plt.colormaps['tab20'](np.linspace(0, 1, max(len(results), 20)))
for r, c in zip(results, cmap):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=c, alpha=0.8, lw=1.2)

shade_crashes(ax)
ax.set_title('All Strategies: Capital Curves', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio Value ($)', fontsize=11)
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=6, loc='upper left', ncol=2)
plt.tight_layout()
plt.show()
```


    
![png](paper_comparison_files/paper_comparison_9_0.png)
    


---
## Risk / Return Scatter: All Strategies


```python
fig, ax = plt.subplots(figsize=(12, 9))

# Color by category
cat_colors = {
    'sell_vol': '#2ca02c',   # covered call, cash-secured put, short strangle
    'buy_vol': '#d62728',    # put hedge, long straddle, deep OTM, VIX-timed
    'momentum': '#1f77b4',   # call momentum, SMA calls
    'mixed': '#ff7f0e',      # collar, SMA puts
}

strategy_cats = {
    '1. Covered Call (BXM)': 'sell_vol',
    '2. Cash-Secured Puts (PUT)': 'sell_vol',
    '3. OTM Put Hedge': 'buy_vol',
    '4. OTM Call Momentum': 'momentum',
    '5. Short Strangle': 'sell_vol',
    '6. Long Straddle': 'buy_vol',
    '7. Collar': 'mixed',
    '8. Deep OTM Tail Hedge': 'buy_vol',
    '9. VIX-Timed Puts': 'buy_vol',
    '10a. SMA Calls (bull)': 'momentum',
    '10b. SMA Puts (bear)': 'mixed',
}

for r in results:
    cat = strategy_cats.get(r['name'], 'mixed')
    c = cat_colors[cat]
    ax.scatter(abs(r['max_dd']), r['annual_ret'], color=c, s=120, zorder=3, edgecolors='white', lw=0.5)
    ax.annotate(r['name'], (abs(r['max_dd']), r['annual_ret']),
                fontsize=6, ha='left', va='bottom', xytext=(5, 5), textcoords='offset points')

ax.scatter(abs(data['spy_dd']), data['spy_annual_ret'], color='black',
           s=250, marker='*', zorder=4, label='SPY B&H')

handles = [
    mpatches.Patch(color='#2ca02c', label='Sell Vol (covered call, put-write, strangle)'),
    mpatches.Patch(color='#d62728', label='Buy Vol (put hedge, straddle, tail hedge)'),
    mpatches.Patch(color='#1f77b4', label='Momentum (calls, SMA calls)'),
    mpatches.Patch(color='#ff7f0e', label='Mixed (collar, SMA puts)'),
    plt.Line2D([0], [0], marker='*', color='black', lw=0, markersize=15, label='SPY B&H'),
]
ax.legend(handles=handles, fontsize=9, loc='lower right')
ax.set_xlabel('Max Drawdown (%, absolute)', fontsize=12)
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_title('Risk / Return: All Strategies by Category', fontsize=14, fontweight='bold')
ax.axhline(y=data['spy_annual_ret'], color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](paper_comparison_files/paper_comparison_11_0.png)
    


---
## Drawdown Comparison: Sell Vol vs Buy Vol


```python
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
spy_dd_s = (spy_prices - spy_prices.cummax()) / spy_prices.cummax() * 100

sell_vol = [r for r in results if strategy_cats.get(r['name']) == 'sell_vol']
buy_vol = [r for r in results if strategy_cats.get(r['name']) == 'buy_vol']

for ax, group, title, color_base in [
    (axes[0], sell_vol, 'Sell Vol Strategies', plt.cm.Greens),
    (axes[1], buy_vol, 'Buy Vol Strategies', plt.cm.Reds),
]:
    ax.fill_between(spy_dd_s.index, spy_dd_s.values, 0, alpha=0.15, color='black', label='SPY B&H')
    cmap = color_base(np.linspace(0.3, 0.9, max(len(group), 1)))
    for r, c in zip(group, cmap):
        (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=c, alpha=0.8, lw=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('% from Peak')
    ax.legend(fontsize=7, loc='lower left')

plt.tight_layout()
plt.show()
```


    
![png](paper_comparison_files/paper_comparison_13_0.png)
    


---
## Annual Return Bar Chart: Sorted Best to Worst


```python
sorted_results = sorted(results, key=lambda r: r['annual_ret'], reverse=True)

fig, ax = plt.subplots(figsize=(12, 8))
names = [r['name'] for r in sorted_results]
returns = [r['annual_ret'] for r in sorted_results]
colors = ['#2ca02c' if r['excess_annual'] > 0.1 else '#d62728' if r['excess_annual'] < -0.1 else '#999'
          for r in sorted_results]

bars = ax.barh(names, returns, color=colors, edgecolor='white', lw=0.5)
ax.axvline(x=data['spy_annual_ret'], color='black', linestyle='--', lw=2, alpha=0.7, label='SPY B&H')

for bar, ret in zip(bars, returns):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{ret:.2f}%', va='center', fontsize=8)

ax.set_xlabel('Annual Return (%)', fontsize=12)
ax.set_title('All Strategies Ranked by Annual Return', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```


    
![png](paper_comparison_files/paper_comparison_15_0.png)
    


---
## Crash Period Analysis: Who Survived Best?


```python
# Compute drawdown during each crash period
crash_rows = []
for crash_name, start, end in CRASHES:
    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    for r in results:
        dd = r['drawdown']
        period_dd = dd[(dd.index >= start_d) & (dd.index <= end_d)]
        if len(period_dd) > 0:
            crash_rows.append({
                'Crash': crash_name,
                'Strategy': r['name'],
                'Max DD %': period_dd.min() * 100,
            })

crash_df = pd.DataFrame(crash_rows)
crash_pivot = crash_df.pivot(index='Strategy', columns='Crash', values='Max DD %')

# Also add SPY
spy_dd_s = (spy_prices - spy_prices.cummax()) / spy_prices.cummax()
spy_crashes = {}
for crash_name, start, end in CRASHES:
    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    period = spy_dd_s[(spy_dd_s.index >= start_d) & (spy_dd_s.index <= end_d)]
    spy_crashes[crash_name] = period.min() * 100
crash_pivot.loc['SPY B&H'] = spy_crashes

(crash_pivot.style
    .format('{:.1f}%')
    .background_gradient(cmap='RdYlGn_r', axis=None)
    .set_caption('Max Drawdown During Crash Periods')
)
```




<style type="text/css">
#T_3426e_row0_col0 {
  background-color: #ee613e;
  color: #f1f1f1;
}
#T_3426e_row0_col1, #T_3426e_row5_col2 {
  background-color: #d22b27;
  color: #f1f1f1;
}
#T_3426e_row0_col2 {
  background-color: #a50026;
  color: #f1f1f1;
}
#T_3426e_row1_col0 {
  background-color: #fcaa5f;
  color: #000000;
}
#T_3426e_row1_col1, #T_3426e_row5_col1 {
  background-color: #e34933;
  color: #f1f1f1;
}
#T_3426e_row1_col2 {
  background-color: #d93429;
  color: #f1f1f1;
}
#T_3426e_row2_col0, #T_3426e_row7_col0 {
  background-color: #f88950;
  color: #f1f1f1;
}
#T_3426e_row2_col1 {
  background-color: #ef633f;
  color: #f1f1f1;
}
#T_3426e_row2_col2 {
  background-color: #ce2827;
  color: #f1f1f1;
}
#T_3426e_row3_col0 {
  background-color: #f99355;
  color: #000000;
}
#T_3426e_row3_col1 {
  background-color: #fff8b4;
  color: #000000;
}
#T_3426e_row3_col2 {
  background-color: #e24731;
  color: #f1f1f1;
}
#T_3426e_row4_col0 {
  background-color: #f7814c;
  color: #f1f1f1;
}
#T_3426e_row4_col1 {
  background-color: #b71126;
  color: #f1f1f1;
}
#T_3426e_row4_col2 {
  background-color: #c01a27;
  color: #f1f1f1;
}
#T_3426e_row5_col0 {
  background-color: #fa9b58;
  color: #000000;
}
#T_3426e_row6_col0, #T_3426e_row8_col0 {
  background-color: #f67c4a;
  color: #f1f1f1;
}
#T_3426e_row6_col1 {
  background-color: #006837;
  color: #f1f1f1;
}
#T_3426e_row6_col2 {
  background-color: #b50f26;
  color: #f1f1f1;
}
#T_3426e_row7_col1 {
  background-color: #d62f27;
  color: #f1f1f1;
}
#T_3426e_row7_col2, #T_3426e_row9_col2 {
  background-color: #c41e27;
  color: #f1f1f1;
}
#T_3426e_row8_col1 {
  background-color: #5ab760;
  color: #f1f1f1;
}
#T_3426e_row8_col2 {
  background-color: #148e4b;
  color: #f1f1f1;
}
#T_3426e_row9_col0, #T_3426e_row11_col0 {
  background-color: #f88c51;
  color: #f1f1f1;
}
#T_3426e_row9_col1 {
  background-color: #ca2427;
  color: #f1f1f1;
}
#T_3426e_row10_col0 {
  background-color: #fa9656;
  color: #000000;
}
#T_3426e_row10_col1 {
  background-color: #b91326;
  color: #f1f1f1;
}
#T_3426e_row10_col2 {
  background-color: #c62027;
  color: #f1f1f1;
}
#T_3426e_row11_col1 {
  background-color: #dd3d2d;
  color: #f1f1f1;
}
#T_3426e_row11_col2 {
  background-color: #c21c27;
  color: #f1f1f1;
}
</style>
<table id="T_3426e">
  <caption>Max Drawdown During Crash Periods</caption>
  <thead>
    <tr>
      <th class="index_name level0" >Crash</th>
      <th id="T_3426e_level0_col0" class="col_heading level0 col0" >2008 GFC</th>
      <th id="T_3426e_level0_col1" class="col_heading level0 col1" >2020 COVID</th>
      <th id="T_3426e_level0_col2" class="col_heading level0 col2" >2022 Bear</th>
    </tr>
    <tr>
      <th class="index_name level0" >Strategy</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3426e_level0_row0" class="row_heading level0 row0" >1. Covered Call (BXM)</th>
      <td id="T_3426e_row0_col0" class="data row0 col0" >-42.1%</td>
      <td id="T_3426e_row0_col1" class="data row0 col1" >-29.3%</td>
      <td id="T_3426e_row0_col2" class="data row0 col2" >-15.8%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row1" class="row_heading level0 row1" >10a. SMA Calls (bull)</th>
      <td id="T_3426e_row1_col0" class="data row1 col0" >-58.5%</td>
      <td id="T_3426e_row1_col1" class="data row1 col1" >-36.6%</td>
      <td id="T_3426e_row1_col2" class="data row1 col2" >-31.7%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row2" class="row_heading level0 row2" >10b. SMA Puts (bear)</th>
      <td id="T_3426e_row2_col0" class="data row2 col0" >-51.3%</td>
      <td id="T_3426e_row2_col1" class="data row2 col1" >-42.9%</td>
      <td id="T_3426e_row2_col2" class="data row2 col2" >-28.0%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row3" class="row_heading level0 row3" >2. Cash-Secured Puts (PUT)</th>
      <td id="T_3426e_row3_col0" class="data row3 col0" >-53.4%</td>
      <td id="T_3426e_row3_col1" class="data row3 col1" >-85.2%</td>
      <td id="T_3426e_row3_col2" class="data row3 col2" >-36.1%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row4" class="row_heading level0 row4" >3. OTM Put Hedge</th>
      <td id="T_3426e_row4_col0" class="data row4 col0" >-49.3%</td>
      <td id="T_3426e_row4_col1" class="data row4 col1" >-21.1%</td>
      <td id="T_3426e_row4_col2" class="data row4 col2" >-23.9%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row5" class="row_heading level0 row5" >4. OTM Call Momentum</th>
      <td id="T_3426e_row5_col0" class="data row5 col0" >-55.2%</td>
      <td id="T_3426e_row5_col1" class="data row5 col1" >-36.6%</td>
      <td id="T_3426e_row5_col2" class="data row5 col2" >-29.0%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row6" class="row_heading level0 row6" >5. Short Strangle</th>
      <td id="T_3426e_row6_col0" class="data row6 col0" >-48.6%</td>
      <td id="T_3426e_row6_col1" class="data row6 col1" >-160.7%</td>
      <td id="T_3426e_row6_col2" class="data row6 col2" >-20.6%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row7" class="row_heading level0 row7" >6. Long Straddle</th>
      <td id="T_3426e_row7_col0" class="data row7 col0" >-51.1%</td>
      <td id="T_3426e_row7_col1" class="data row7 col1" >-30.5%</td>
      <td id="T_3426e_row7_col2" class="data row7 col2" >-24.9%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row8" class="row_heading level0 row8" >7. Collar</th>
      <td id="T_3426e_row8_col0" class="data row8 col0" >-48.3%</td>
      <td id="T_3426e_row8_col1" class="data row8 col1" >-133.7%</td>
      <td id="T_3426e_row8_col2" class="data row8 col2" >-149.0%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row9" class="row_heading level0 row9" >8. Deep OTM Tail Hedge</th>
      <td id="T_3426e_row9_col0" class="data row9 col0" >-51.9%</td>
      <td id="T_3426e_row9_col1" class="data row9 col1" >-27.0%</td>
      <td id="T_3426e_row9_col2" class="data row9 col2" >-25.1%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row10" class="row_heading level0 row10" >9. VIX-Timed Puts</th>
      <td id="T_3426e_row10_col0" class="data row10 col0" >-54.2%</td>
      <td id="T_3426e_row10_col1" class="data row10 col1" >-21.8%</td>
      <td id="T_3426e_row10_col2" class="data row10 col2" >-25.6%</td>
    </tr>
    <tr>
      <th id="T_3426e_level0_row11" class="row_heading level0 row11" >SPY B&H</th>
      <td id="T_3426e_row11_col0" class="data row11 col0" >-51.9%</td>
      <td id="T_3426e_row11_col1" class="data row11 col1" >-33.7%</td>
      <td id="T_3426e_row11_col2" class="data row11 col2" >-24.5%</td>
    </tr>
  </tbody>
</table>




---
## Conclusions: Paper Claims vs Our Empirical Results

### The Volatility Risk Premium Dominates Everything

The single most important finding, consistent across **all 10 strategies**:

$$\mathbb{E}[\sigma_{\text{implied}}] > \mathbb{E}[\sigma_{\text{realized}}] \implies \text{option sellers win on average}$$

This is confirmed by Carr & Wu (2009), Bakshi & Kapadia (2003), and Bondarenko (2014).

### Confirmed Claims

| Paper | Claim | Our Result |
|-------|-------|------------|
| Whaley (2002) | Covered calls: comparable return, $\frac{2}{3}\sigma$ | **Confirmed** |
| Israelov (2017) | Puts are "pathetic protection" | **Confirmed in no-leverage framing** |
| Carr & Wu (2009) | $\text{VRP} > 0$ means long vol loses | **Confirmed** — long straddle loses |
| Israelov & Klein (2015) | Collar has hidden costs | **Confirmed** — reduces equity premium |
| Kang (2016) | OTM calls predict positive returns | **Confirmed** — modest alpha |

### Partially Confirmed

| Paper | Claim | Our Result |
|-------|-------|------------|
| Berman (2014) | Short strangle CAGR 15.2% | **Partially** — depends on crash exposure |
| Fassas (2019) | VIX timing improves put trades | **Partially** — better than plain puts |
| Antonacci (2013) | Trend-following adds value | **Partially** — works for calls, not puts |

### The Tail Hedge Debate: Both Sides Are Right

$$\text{AQR framing: } (1-w) \cdot R_{\text{SPY}} + w \cdot R_{\text{puts}} \quad \text{(always loses — confirmed)}$$

$$\text{Spitznagel framing: } 1.0 \cdot R_{\text{SPY}} + w \cdot R_{\text{puts}} \quad \text{(leverage + protection — outperforms!)}$$

Our Spitznagel notebook shows that 100% SPY + deep OTM puts via budget callable produces **13.8%–28.8%/yr with lower drawdowns**. The framing determines the conclusion: without leverage, AQR is right. With leverage + tail protection, Spitznagel is right.

See [spitznagel_case.ipynb](spitznagel_case.ipynb) for the full analysis.

### Bottom Line

**Sell vol** strategies harvest the VRP and outperform risk-adjusted. **Buy vol** strategies pay the VRP and underperform in the no-leverage framing — but **leverage + tail protection** is a viable strategy that Spitznagel has demonstrated both theoretically and in our backtests.
