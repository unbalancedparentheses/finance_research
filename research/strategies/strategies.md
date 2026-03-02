# Strategy Showcase

Four distinct options strategies on SPY, each with capital curve, drawdown, and summary stats.

| # | Strategy | Direction | Delta Range | Legs |
|---|----------|-----------|-------------|------|
| 1 | OTM Put Hedge | BUY puts | -0.25 to -0.10 | 1 |
| 2 | OTM Call Momentum | BUY calls | 0.10 to 0.25 | 1 |
| 3 | Long Straddle | BUY call + put (ATM) | ~0.50 | 2 |
| 4 | Short Strangle | SELL call + put (OTM) | 0.10–0.25 | 2 |


```python
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.realpath(os.path.join(os.getcwd(), '..'))
os.chdir(PROJECT_ROOT)

from options_portfolio_backtester import Direction
from backtest_runner import (
    load_data, make_puts_strategy, make_calls_strategy,
    make_straddle_strategy, make_strangle_strategy,
    run_backtest, INITIAL_CAPITAL,
)

sns.set_theme(style='whitegrid', palette='muted')
%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['figure.dpi'] = 110

CRASHES = [
    ('2008 GFC', '2007-10-01', '2009-03-09'),
    ('2020 COVID', '2020-02-19', '2020-03-23'),
    ('2022 Bear', '2022-01-03', '2022-10-12'),
]

def shade_crashes(ax, alpha=0.12):
    colors = ['#d62728', '#ff7f0e', '#9467bd']
    for (label, start, end), color in zip(CRASHES, colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=color, label=label)

print('Setup done.')
```

    Setup done.



```python
data = load_data()
schema = data['schema']
spy_prices = data['spy_prices']
```

    Loading data...


    Date range: 2008-01-02 00:00:00 to 2025-12-12 00:00:00 (17.9 years)
    SPY B&H: 555.5% total, 11.05% annual, -51.9% max DD
    
    Loaded macro signals: ['gdp', 'vix', 'hy_spread', 'yield_curve_10y2y', 'nfc_equity_mv', 'nfc_net_worth', 'dollar_index', 'buffett_indicator', 'tobin_q']


---
## Run All 4 Strategies

All use 99%/1% stock/options split (no leverage), monthly rebalance.


```python
S_PCT, O_PCT = 0.99, 0.01

configs = [
    ('OTM Put Hedge',     lambda: make_puts_strategy(schema)),
    ('OTM Call Momentum', lambda: make_calls_strategy(schema)),
    ('Long Straddle',     lambda: make_straddle_strategy(schema, direction=Direction.BUY)),
    ('Short Strangle',    lambda: make_strangle_strategy(schema, direction=Direction.SELL)),
]

results = []
for name, strat_fn in configs:
    print(f'Running {name}...', end=' ', flush=True)
    r = run_backtest(name, S_PCT, O_PCT, strat_fn, data)
    results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

print('\nAll strategies complete.')
```

    Running OTM Put Hedge... 

    Warning: No valid output stream.


    annual +7.62%, excess -3.42%, DD -49.3%
    Running OTM Call Momentum... 

    Warning: No valid output stream.


    annual +11.88%, excess +0.83%, DD -55.2%
    Running Long Straddle... 

    Warning: No valid output stream.


    annual +10.67%, excess -0.37%, DD -51.1%
    Running Short Strangle... 

    Warning: No valid output stream.


    annual +47.18%, excess +36.14%, DD -340.5%
    
    All strategies complete.


---
## Comparison Table


```python
rows = []
for r in results:
    rows.append({
        'Strategy': r['name'],
        'Annual Return %': r['annual_ret'],
        'Total Return %': r['total_ret'],
        'Max Drawdown %': r['max_dd'],
        'Trades': r['trades'],
        'Excess vs SPY %/yr': r['excess_annual'],
    })
df = pd.DataFrame(rows)

def color_excess(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'color: green; font-weight: bold'
        if val < -0.5: return 'color: red'
    return ''

(df.style
    .format({'Annual Return %': '{:.2f}', 'Total Return %': '{:.1f}',
             'Max Drawdown %': '{:.1f}', 'Trades': '{:.0f}',
             'Excess vs SPY %/yr': '{:+.2f}'})
    .map(color_excess, subset=['Excess vs SPY %/yr'])
    .set_caption(f'Strategy Comparison: {S_PCT*100:.0f}% SPY + {O_PCT*100:.0f}% Options')
)
```




<style type="text/css">
#T_d3e42_row0_col5 {
  color: red;
}
#T_d3e42_row1_col5, #T_d3e42_row3_col5 {
  color: green;
  font-weight: bold;
}
</style>
<table id="T_d3e42">
  <caption>Strategy Comparison: 99% SPY + 1% Options</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d3e42_level0_col0" class="col_heading level0 col0" >Strategy</th>
      <th id="T_d3e42_level0_col1" class="col_heading level0 col1" >Annual Return %</th>
      <th id="T_d3e42_level0_col2" class="col_heading level0 col2" >Total Return %</th>
      <th id="T_d3e42_level0_col3" class="col_heading level0 col3" >Max Drawdown %</th>
      <th id="T_d3e42_level0_col4" class="col_heading level0 col4" >Trades</th>
      <th id="T_d3e42_level0_col5" class="col_heading level0 col5" >Excess vs SPY %/yr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d3e42_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_d3e42_row0_col0" class="data row0 col0" >OTM Put Hedge</td>
      <td id="T_d3e42_row0_col1" class="data row0 col1" >7.62</td>
      <td id="T_d3e42_row0_col2" class="data row0 col2" >273.8</td>
      <td id="T_d3e42_row0_col3" class="data row0 col3" >-49.3</td>
      <td id="T_d3e42_row0_col4" class="data row0 col4" >381</td>
      <td id="T_d3e42_row0_col5" class="data row0 col5" >-3.42</td>
    </tr>
    <tr>
      <th id="T_d3e42_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_d3e42_row1_col0" class="data row1 col0" >OTM Call Momentum</td>
      <td id="T_d3e42_row1_col1" class="data row1 col1" >11.88</td>
      <td id="T_d3e42_row1_col2" class="data row1 col2" >649.1</td>
      <td id="T_d3e42_row1_col3" class="data row1 col3" >-55.2</td>
      <td id="T_d3e42_row1_col4" class="data row1 col4" >365</td>
      <td id="T_d3e42_row1_col5" class="data row1 col5" >+0.83</td>
    </tr>
    <tr>
      <th id="T_d3e42_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_d3e42_row2_col0" class="data row2 col0" >Long Straddle</td>
      <td id="T_d3e42_row2_col1" class="data row2 col1" >10.67</td>
      <td id="T_d3e42_row2_col2" class="data row2 col2" >517.0</td>
      <td id="T_d3e42_row2_col3" class="data row2 col3" >-51.1</td>
      <td id="T_d3e42_row2_col4" class="data row2 col4" >390</td>
      <td id="T_d3e42_row2_col5" class="data row2 col5" >-0.37</td>
    </tr>
    <tr>
      <th id="T_d3e42_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d3e42_row3_col0" class="data row3 col0" >Short Strangle</td>
      <td id="T_d3e42_row3_col1" class="data row3 col1" >47.18</td>
      <td id="T_d3e42_row3_col2" class="data row3 col2" >102695.2</td>
      <td id="T_d3e42_row3_col3" class="data row3 col3" >-340.5</td>
      <td id="T_d3e42_row3_col4" class="data row3 col4" >396</td>
      <td id="T_d3e42_row3_col5" class="data row3 col5" >+36.14</td>
    </tr>
  </tbody>
</table>




---
## Capital Curves


```python
fig, ax = plt.subplots(figsize=(16, 7))
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=2.5, label='SPY B&H', alpha=0.7)

strategy_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
for r, color in zip(results, strategy_colors):
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=color, alpha=0.85, lw=1.5)

shade_crashes(ax)
ax.set_title('Capital Curves: 4 Strategies vs SPY B&H', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio Value ($)', fontsize=11)
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.show()
```


    
![png](strategies_files/strategies_8_0.png)
    


---
## Drawdown Comparison


```python
fig, ax = plt.subplots(figsize=(16, 6))

spy_cummax = spy_prices.cummax()
spy_dd = (spy_prices - spy_cummax) / spy_cummax * 100
ax.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.2, color='black', label='SPY B&H')

for r, color in zip(results, strategy_colors):
    (r['drawdown'] * 100).plot(ax=ax, label=r['name'], color=color, alpha=0.8, lw=1.2)

ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('% from Peak', fontsize=11)
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.show()
```


    
![png](strategies_files/strategies_10_0.png)
    


---
## Individual Strategy Deep Dives

Each strategy gets its own capital curve with annotations.


```python
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

for ax, r, color in zip(axes.flat, results, strategy_colors):
    ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=1.5, label='SPY B&H', alpha=0.6)
    r['balance']['total capital'].plot(ax=ax, label=r['name'], color=color, alpha=0.9, lw=1.8)
    shade_crashes(ax, alpha=0.1)
    ax.set_title(f"{r['name']}  |  {r['annual_ret']:.2f}%/yr  |  DD {r['max_dd']:.1f}%",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('$')
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.show()
```


    
![png](strategies_files/strategies_12_0.png)
    


---
## Risk / Return Scatter


```python
fig, ax = plt.subplots(figsize=(10, 7))

for r, color in zip(results, strategy_colors):
    ax.scatter(abs(r['max_dd']), r['annual_ret'], color=color, s=150, zorder=3, edgecolors='white', lw=0.5)
    ax.annotate(r['name'], (abs(r['max_dd']), r['annual_ret']),
                fontsize=9, ha='left', va='bottom', xytext=(8, 5), textcoords='offset points')

ax.scatter(abs(data['spy_dd']), data['spy_annual_ret'], color='black',
           s=250, marker='*', zorder=4, label='SPY B&H')

ax.set_xlabel('Max Drawdown (%, absolute)', fontsize=12)
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_title('Risk / Return Tradeoff', fontsize=14, fontweight='bold')
ax.axhline(y=data['spy_annual_ret'], color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```


    
![png](strategies_files/strategies_14_0.png)
    


---
## Takeaways

- **OTM Call Momentum** is the only single-leg strategy that consistently beats SPY B&H
- **OTM Put Hedge** costs premium every month — a drag in bull markets
- **Long Straddle** (buy vol) pays off in crashes but bleeds in calm markets
- **Short Strangle** (sell vol) collects premium but faces blow-up risk in crashes

The choice depends on your market view:
- **Bullish** → OTM calls
- **Expect vol** → long straddle
- **Sell premium** → short strangle (with risk management!)
- **Hedge tail risk** → puts (accept the cost as insurance)
