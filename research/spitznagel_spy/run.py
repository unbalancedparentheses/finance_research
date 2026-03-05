#!/usr/bin/env python3
"""
Spitznagel tail-hedge backtest: testing the leveraged deep OTM put overlay
on SPY options data (2008-2025).

Reproduces all analysis from spitznagel_case.md as a standalone script.
Generates charts/ directory with all figures.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

os.makedirs(os.path.join(os.path.dirname(__file__), 'charts'), exist_ok=True)
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')

from backtest_runner import (
    load_data, run_backtest, INITIAL_CAPITAL,
    make_puts_strategy, make_deep_otm_put_strategy, make_atm_put_strategy,
)
from options_portfolio_backtester import Order
from nb_style import apply_style, shade_crashes, color_excess, style_returns_table, FT_GREEN, FT_RED, FT_BLUE

apply_style()
print('Ready.')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_data()
schema = data['schema']
spy_prices = data['spy_prices']
years = data['years']

# ---------------------------------------------------------------------------
# 1. Variance Drain on Actual SPY
# ---------------------------------------------------------------------------
daily_returns = spy_prices.pct_change().dropna()
arith_annual = daily_returns.mean() * 252
geom_annual = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
vol_annual = daily_returns.std() * np.sqrt(252)
drain = arith_annual - geom_annual

rolling_vol = daily_returns.rolling(252).std() * np.sqrt(252)
rolling_drain = (rolling_vol ** 2) / 2

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes[0]
ax.plot(rolling_vol.index, rolling_vol * 100, color=FT_BLUE, lw=1.2)
ax.set_ylabel('Annualized Vol (%)')
ax.set_title('SPY Rolling 1-Year Volatility', fontweight='bold')
shade_crashes(ax)
ax.legend(loc='upper right', fontsize=7)

ax = axes[1]
ax.fill_between(rolling_drain.index, rolling_drain * 100, 0, color=FT_RED, alpha=0.4)
ax.set_ylabel('Variance Drain (%/yr)')
ax.set_title('Rolling Variance Drain (sigma^2/2) — What Volatility Costs You in Compounding', fontweight='bold')
shade_crashes(ax)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'variance_drain.png'), dpi=150, bbox_inches='tight'); plt.close()

print(f'Arithmetic mean: {arith_annual*100:.2f}%  |  Geometric mean: {geom_annual*100:.2f}%  |  Vol: {vol_annual*100:.1f}%')
print(f'Variance drain: {drain*100:.2f}%/yr  (theoretical sigma^2/2 = {(vol_annual**2/2)*100:.2f}%)')
print(f'Peak drain during GFC: {rolling_drain.max()*100:.1f}%/yr')

# ---------------------------------------------------------------------------
# 2. The AQR Test: No Leverage (Allocation Split)
# ---------------------------------------------------------------------------
no_leverage_configs = [
    ('SPY only',            1.0,    0.0,    lambda: make_deep_otm_put_strategy(schema)),
    ('Deep OTM 0.1%',      0.999,  0.001,  lambda: make_deep_otm_put_strategy(schema)),
    ('Deep OTM 0.5%',      0.995,  0.005,  lambda: make_deep_otm_put_strategy(schema)),
    ('Deep OTM 1.0%',      0.99,   0.01,   lambda: make_deep_otm_put_strategy(schema)),
    ('Deep OTM 3.3%',      0.967,  0.033,  lambda: make_deep_otm_put_strategy(schema)),
    ('Std OTM 1.0%',       0.99,   0.01,   lambda: make_puts_strategy(schema)),
]

no_lev_results = []
for name, s, o, fn in no_leverage_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, s, o, fn, data)
    no_lev_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r in no_lev_results:
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({
        'Strategy': r['name'], 'Annual %': r['annual_ret'],
        'Vol %': vol, 'Max DD %': r['max_dd'],
        'Excess %': r['excess_annual'], 'Trades': r['trades'],
    })
df = pd.DataFrame(rows)
styled = df.style.format({'Annual %': '{:.2f}', 'Vol %': '{:.1f}', 'Max DD %': '{:.1f}',
                           'Excess %': '{:+.2f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('AQR framing: reduce equity to fund puts (NO leverage) — always loses')

# ---------------------------------------------------------------------------
# 2b. DEBUG: No Leverage via Budget (spend X% of portfolio as premium, 99% equity)
# ---------------------------------------------------------------------------
print('\n=== DEBUG: No-leverage via BUDGET approach ===')
print('Same equity reduction but using budget_fn instead of allocation split.\n')

no_lev_budget_configs = [
    ('Budget 0.1%',  0.001),
    ('Budget 0.5%',  0.005),
    ('Budget 1.0%',  0.01),
    ('Budget 3.3%',  0.033),
]

no_lev_budget_results = []
for name, bp in no_lev_budget_configs:
    print(f'  {name}...', end=' ', flush=True)
    stock_pct = 1.0 - bp  # e.g. 0.99 for 1% budget
    r = run_backtest(name, stock_pct, 0.0, lambda: make_deep_otm_put_strategy(schema), data, budget_pct=bp)
    no_lev_budget_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows_debug = []
for r in no_lev_budget_results:
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows_debug.append({
        'Strategy': r['name'], 'Annual %': r['annual_ret'],
        'Vol %': vol, 'Max DD %': r['max_dd'],
        'Excess %': r['excess_annual'], 'Trades': r['trades'],
    })
df_debug = pd.DataFrame(rows_debug)
print('\nNo-leverage via budget:')
print(df_debug.to_string(index=False))

# Also compare: what does leveraged (stocks=1.0) give for same budgets?
print('\n=== DEBUG: Leveraged (stocks=1.0) for same budgets ===')
for name, bp in no_lev_budget_configs:
    print(f'  Leveraged {name}...', end=' ', flush=True)
    r = run_backtest(f'Lev {name}', 1.0, 0.0, lambda: make_deep_otm_put_strategy(schema), data, budget_pct=bp)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

print('\n=== Comparison: Allocation vs Budget vs Blog ===')
print(f'{"Config":<20} {"Alloc annual%":>14} {"Budget annual%":>15} {"Blog annual%":>13}')
print('-' * 65)
blog_numbers = {'0.1%': 11.40, '0.5%': 12.63, '1.0%': 14.11, '3.3%': 20.74}
alloc_map = {'0.1%': 1, '0.5%': 2, '1.0%': 3, '3.3%': 4}  # indices into no_lev_results
for pct_label, blog_val in blog_numbers.items():
    alloc_val = no_lev_results[alloc_map[pct_label]]['annual_ret'] if pct_label in alloc_map else 'N/A'
    budget_idx = list(blog_numbers.keys()).index(pct_label)
    budget_val = no_lev_budget_results[budget_idx]['annual_ret']
    print(f'{pct_label:<20} {alloc_val:>14.2f} {budget_val:>15.2f} {blog_val:>13.2f}')

# Run blog params (DTE 90-180, exit DTE 14, monthly-only exits) for comparison
print('\n=== Blog Params Comparison (DTE 90-180, exit 14, monthly exits) ===')
print(f'{"Config":<20} {"Our config":>12} {"Blog params":>12} {"Blog article":>13}')
print('-' * 60)
blog_param_budgets = [0.001, 0.005, 0.01, 0.033]
blog_param_labels = ['0.1%', '0.5%', '1.0%', '3.3%']
for bp, label in zip(blog_param_budgets, blog_param_labels):
    r_blog = run_backtest(
        f'blog-params {label}', 1.0, 0.0,
        lambda: make_deep_otm_put_strategy(schema, dte_min=90, dte_max=180, exit_dte=14),
        data, budget_pct=bp, check_exits_daily=False)
    our_idx = list(blog_numbers.keys()).index(label)
    our_val = no_lev_budget_results[our_idx]['annual_ret']
    blog_val = blog_numbers[label]
    print(f'{label:<20} {our_val:>12.2f} {r_blog["annual_ret"]:>12.2f} {blog_val:>13.2f}')

# ---------------------------------------------------------------------------
# 3. The Spitznagel Test: With Leverage (100% Equity + Puts on Top)
# ---------------------------------------------------------------------------
leverage_configs = [
    ('100% SPY (baseline)',    None),
    ('+ 0.05% deep OTM puts', 0.0005),
    ('+ 0.1% deep OTM puts',  0.001),
    ('+ 0.2% deep OTM puts',  0.002),
    ('+ 0.5% deep OTM puts',  0.005),
    ('+ 1.0% deep OTM puts',  0.01),
    ('+ 2.0% deep OTM puts',  0.02),
    ('+ 3.3% deep OTM puts',  0.033),
]

lev_results = []
for name, budget_pct in leverage_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, 1.0, 0.0, lambda: make_deep_otm_put_strategy(schema), data, budget_pct=budget_pct)
    lev_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r in lev_results:
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({
        'Strategy': r['name'], 'Annual %': r['annual_ret'],
        'Vol %': vol, 'Max DD %': r['max_dd'],
        'Excess %': r['excess_annual'], 'Trades': r['trades'],
    })
df_lev = pd.DataFrame(rows)
styled = df_lev.style.format({'Annual %': '{:.2f}', 'Vol %': '{:.1f}', 'Max DD %': '{:.1f}',
                               'Excess %': '{:+.2f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('Spitznagel framing: 100% SPY + puts on top (WITH leverage)')

# Leverage breakdown
budget_pcts = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.033]
rows_lev = []
for r, bp in zip(lev_results, budget_pcts):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    leverage = 1.0 + bp
    excess = r['excess_annual']
    ret_per_leverage = excess / (bp * 100) if bp > 0 else 0
    rows_lev.append({
        'Strategy': r['name'],
        'Put Budget %/yr': bp * 100,
        'Total Leverage': f'{leverage:.4f}x',
        'Annual Return %': r['annual_ret'],
        'Excess vs SPY %': excess,
        'Return per 1% Budget': ret_per_leverage,
        'Max DD %': r['max_dd'],
        'Vol %': vol,
    })

df_leverage = pd.DataFrame(rows_lev)

styled_lev = (df_leverage.style
    .format({
        'Put Budget %/yr': '{:.2f}',
        'Annual Return %': '{:.2f}',
        'Excess vs SPY %': '{:+.2f}',
        'Return per 1% Budget': '{:.1f}',
        'Max DD %': '{:.1f}',
        'Vol %': '{:.1f}',
    })
    .map(color_excess, subset=['Excess vs SPY %'])
)
style_returns_table(styled_lev).set_caption(
    'Leverage Breakdown: Tiny Leverage, Massive Convex Payoff'
)

# ---------------------------------------------------------------------------
# 3c. Side-by-Side: Deep OTM vs ATM, Leveraged vs No-Leverage
# ---------------------------------------------------------------------------
print('\n=== Side-by-Side Comparison: Strike Selection x Leverage ===')
budgets = [0.005, 0.01, 0.033]
budget_labels = ['0.5%', '1.0%', '3.3%']

comparison_rows = []
# SPY baseline
comparison_rows.append({
    'Strategy': 'SPY B&H', 'Type': '-', 'Framing': '-', 'Budget': '-',
    'Annual %': lev_results[0]['annual_ret'], 'Excess %': 0.0,
    'Max DD %': lev_results[0]['max_dd'],
})

for bp, label in zip(budgets, budget_labels):
    # Deep OTM leveraged (already computed above)
    lev_idx = {0.005: 4, 0.01: 5, 0.033: 7}[bp]
    r = lev_results[lev_idx]
    comparison_rows.append({
        'Strategy': f'Deep OTM {label} (leveraged)', 'Type': 'Deep OTM', 'Framing': 'Leveraged',
        'Budget': label, 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
        'Max DD %': r['max_dd'],
    })
    # Deep OTM no-leverage (allocation-based, NOT budget_pct)
    r = run_backtest(f'Deep OTM {label} (no-lev)', 1.0 - bp, bp,
                     lambda: make_deep_otm_put_strategy(schema), data)
    comparison_rows.append({
        'Strategy': f'Deep OTM {label} (no-leverage)', 'Type': 'Deep OTM', 'Framing': 'No-leverage',
        'Budget': label, 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
        'Max DD %': r['max_dd'],
    })
    # ATM leveraged (externally funded)
    r = run_backtest(f'ATM {label} (leveraged)', 1.0, 0.0,
                     lambda: make_atm_put_strategy(schema), data, budget_pct=bp)
    comparison_rows.append({
        'Strategy': f'ATM {label} (leveraged)', 'Type': 'ATM', 'Framing': 'Leveraged',
        'Budget': label, 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
        'Max DD %': r['max_dd'],
    })
    # ATM no-leverage (allocation-based, NOT budget_pct)
    r = run_backtest(f'ATM {label} (no-lev)', 1.0 - bp, bp,
                     lambda: make_atm_put_strategy(schema), data)
    comparison_rows.append({
        'Strategy': f'ATM {label} (no-leverage)', 'Type': 'ATM', 'Framing': 'No-leverage',
        'Budget': label, 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
        'Max DD %': r['max_dd'],
    })

df_comp = pd.DataFrame(comparison_rows)
styled_comp = (df_comp[['Strategy', 'Annual %', 'Excess %', 'Max DD %']].style
    .format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled_comp).set_caption('Strike Selection x Leverage: All Combinations')

for row in comparison_rows:
    print(f"  {row['Strategy']:<35} annual {row['Annual %']:+.2f}%  excess {row['Excess %']:+.2f}%  DD {row['Max DD %']:.1f}%")

# ---------------------------------------------------------------------------
# 4. Capital Curves: AQR Framing vs Spitznagel Framing
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
spy_norm = spy_prices / spy_prices.iloc[0] * INITIAL_CAPITAL

# AQR framing (no leverage)
ax = axes[0]
ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=2.5, label='SPY B&H', alpha=0.7)
cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(no_lev_results) - 1))
for r, c in zip(no_lev_results[1:], cmap):
    r['balance']['total capital'].plot(ax=ax, label=f"{r['name']} ({r['excess_annual']:+.2f}%)",
                                       color=c, alpha=0.8, lw=1.2)
shade_crashes(ax)
ax.set_title('AQR framing: Reduce equity to fund puts', fontweight='bold', color=FT_RED)
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=6, loc='upper left')

# Spitznagel framing (leverage)
ax = axes[1]
ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=2.5, label='SPY B&H', alpha=0.7)
cmap = plt.cm.Greens(np.linspace(0.3, 0.9, len(lev_results) - 1))
for r, c in zip(lev_results[1:], cmap):
    r['balance']['total capital'].plot(ax=ax, label=f"{r['name']} ({r['excess_annual']:+.2f}%)",
                                       color=c, alpha=0.8, lw=1.2)
shade_crashes(ax)
ax.set_title('Spitznagel framing: 100% equity + puts on top', fontweight='bold', color=FT_GREEN)
ax.set_ylabel('$')
ax.ticklabel_format(style='plain', axis='y')
ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'capital_curves_aqr_vs_spitznagel.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------------------
# 5. Also Try With Standard OTM Puts (Leveraged)
# ---------------------------------------------------------------------------
std_lev_configs = [
    ('+ 0.1% std OTM puts',  0.001),
    ('+ 0.5% std OTM puts',  0.005),
    ('+ 1.0% std OTM puts',  0.01),
]

std_lev_results = [lev_results[0]]  # baseline
for name, budget_pct in std_lev_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, 1.0, 0.0, lambda: make_puts_strategy(schema), data, budget_pct=budget_pct)
    std_lev_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, group, title, palette in [
    (axes[0], lev_results, 'Deep OTM puts (delta -0.10 to -0.02) + 100% SPY', plt.cm.Purples),
    (axes[1], std_lev_results, 'Standard OTM puts (delta -0.25 to -0.10) + 100% SPY', plt.cm.Blues),
]:
    ax.plot(spy_norm.index, spy_norm.values, 'k--', lw=2.5, label='SPY B&H', alpha=0.7)
    cmap = palette(np.linspace(0.3, 0.9, max(len(group) - 1, 1)))
    for r, c in zip(group[1:], cmap):
        r['balance']['total capital'].plot(ax=ax, label=f"{r['name']} ({r['excess_annual']:+.2f}%)",
                                           color=c, alpha=0.85, lw=1.5)
    shade_crashes(ax)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel('$')
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'deep_vs_standard_otm_leveraged.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------------------
# 6. Crash-Period Trade Analysis
# ---------------------------------------------------------------------------
r_analysis = run_backtest('Deep OTM 0.5% (leveraged)', 1.0, 0.0,
                          lambda: make_deep_otm_put_strategy(schema), data,
                          budget_pct=0.005)
trade_log = r_analysis['trade_log']

if len(trade_log) > 0:
    first_leg = trade_log.columns.levels[0][0]
    entry_mask = trade_log[first_leg]['order'].isin([Order.BTO.value, Order.STO.value])
    entries = trade_log[entry_mask]
    exits = trade_log[~entry_mask]

    trades = []
    for _, entry_row in entries.iterrows():
        contract = entry_row[first_leg]['contract']
        exit_rows = exits[exits[first_leg]['contract'] == contract]
        if exit_rows.empty:
            continue
        exit_row = exit_rows.iloc[0]
        entry_cost = entry_row['totals']['cost'] * entry_row['totals']['qty']
        exit_cost = exit_row['totals']['cost'] * exit_row['totals']['qty']
        pnl = -(entry_cost + exit_cost)
        trades.append({
            'entry_date': entry_row['totals']['date'],
            'exit_date': exit_row['totals']['date'],
            'pnl': pnl,
            'entry_cost': entry_cost,
        })

    trades_df = pd.DataFrame(trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

    CRASHES = [
        ('2008 GFC',   '2007-10-01', '2009-06-01'),
        ('2020 COVID', '2020-01-01', '2020-06-01'),
        ('2022 Bear',  '2022-01-01', '2022-12-31'),
        ('Calm periods', None, None),
    ]

    def classify(d):
        for name, s, e in CRASHES[:-1]:
            if pd.Timestamp(s) <= d <= pd.Timestamp(e):
                return name
        return 'Calm periods'

    trades_df['period'] = trades_df['entry_date'].apply(classify)

    period_stats = trades_df.groupby('period').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        best_trade=('pnl', 'max'),
        total_premium=('entry_cost', 'sum'),
    ).round(0)

    styled = period_stats.style.format({
        'total_pnl': '${:,.0f}', 'avg_pnl': '${:,.0f}',
        'best_trade': '${:,.0f}', 'total_premium': '${:,.0f}', 'trades': '{:.0f}'
    })
    style_returns_table(styled).set_caption('Deep OTM Put Trades: P&L by Market Period')
else:
    print('No trades executed.')

if len(trades_df) > 0:
    # Per-trade P&L bar chart colored by period
    sorted_trades = trades_df.sort_values('entry_date')
    period_colors = {'2008 GFC': FT_RED, '2020 COVID': '#FF8833', '2022 Bear': '#9467bd', 'Calm periods': '#AAAAAA'}
    colors = [period_colors.get(p, '#999') for p in sorted_trades['period']]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Per-trade P&L
    ax = axes[0]
    ax.bar(range(len(sorted_trades)), sorted_trades['pnl'], color=colors, width=1.0, edgecolor='none')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('P&L ($)')
    ax.set_title('Per-Trade P&L: Deep OTM Puts (colored by market period)', fontweight='bold')

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=l) for l, c in period_colors.items()]
    ax.legend(handles=handles, fontsize=8)

    # Cumulative P&L
    ax = axes[1]
    cum_pnl = sorted_trades.set_index('exit_date')['pnl'].cumsum()
    ax.plot(cum_pnl.index, cum_pnl.values, color=FT_BLUE, lw=2)
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=cum_pnl.values >= 0, color=FT_GREEN, alpha=0.15)
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=cum_pnl.values < 0, color=FT_RED, alpha=0.15)
    shade_crashes(ax)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('Cumulative Options P&L Over Time', fontweight='bold')
    ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'crash_trade_analysis.png'), dpi=150, bbox_inches='tight'); plt.close()

    total_premium = trades_df['entry_cost'].sum()
    total_pnl = trades_df['pnl'].sum()
    crash_pnl = trades_df[trades_df['period'] != 'Calm periods']['pnl'].sum()
    calm_pnl = trades_df[trades_df['period'] == 'Calm periods']['pnl'].sum()

    print(f'Total premium spent: ${abs(total_premium):,.0f}')
    print(f'Total P&L: ${total_pnl:,.0f}')
    print(f'Crash period P&L: ${crash_pnl:,.0f}')
    print(f'Calm period P&L: ${calm_pnl:,.0f}')
    print(f'Crash payoff / Total premium: {crash_pnl / abs(total_premium) * 100:.1f}%')

# ---------------------------------------------------------------------------
# 7. Drawdown During Crashes
# ---------------------------------------------------------------------------
spy_dd_s = (spy_prices - spy_prices.cummax()) / spy_prices.cummax()

crash_periods = [
    ('2008 GFC',   '2007-10-01', '2009-03-09'),
    ('2020 COVID', '2020-02-19', '2020-03-23'),
    ('2022 Bear',  '2022-01-03', '2022-10-12'),
]

crash_rows = []
for crash_name, start, end in crash_periods:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    # SPY B&H
    spy_period = spy_dd_s[(spy_dd_s.index >= s) & (spy_dd_s.index <= e)]
    crash_rows.append({'Crash': crash_name, 'Strategy': 'SPY B&H', 'Max DD %': spy_period.min() * 100})

    # Each leveraged config
    for r in lev_results[1:]:
        dd = r['drawdown']
        period = dd[(dd.index >= s) & (dd.index <= e)]
        if len(period) > 0:
            crash_rows.append({'Crash': crash_name, 'Strategy': r['name'], 'Max DD %': period.min() * 100})

crash_compare = pd.DataFrame(crash_rows).pivot(index='Strategy', columns='Crash', values='Max DD %')

styled = crash_compare.style.format('{:.1f}%').background_gradient(cmap='RdYlGn_r', axis=None)
style_returns_table(styled).set_caption('Drawdown During Crashes: SPY vs Leveraged Deep OTM Puts')

# ---------------------------------------------------------------------------
# 8. Sharpe Ratio Comparison
# ---------------------------------------------------------------------------
all_configs = no_lev_results + lev_results[1:] + std_lev_results[1:]

rows = []
for r in all_configs:
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    sharpe = r['annual_ret'] / vol if vol > 0 else 0
    framing = 'No leverage' if r in no_lev_results else 'Leveraged'
    rows.append({
        'Framing': framing, 'Strategy': r['name'],
        'Annual %': r['annual_ret'], 'Vol %': vol,
        'Max DD %': r['max_dd'], 'Sharpe': sharpe,
        'Excess %': r['excess_annual'],
    })

df_all = pd.DataFrame(rows)
styled = (df_all.style
    .format({'Annual %': '{:.2f}', 'Vol %': '{:.1f}', 'Max DD %': '{:.1f}',
             'Sharpe': '{:.3f}', 'Excess %': '{:+.2f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption('Full Comparison: No Leverage (AQR) vs Leverage (Spitznagel)')

# ---------------------------------------------------------------------------
# Extended Risk Metrics
# ---------------------------------------------------------------------------
from scipy import stats as sp_stats

def compute_extended_metrics(r, years):
    """Compute Sortino, Calmar, tail ratio, skew, kurtosis, max DD duration."""
    d = r['balance']['% change'].dropna()
    total_cap = r['balance']['total capital']
    annual_ret = r['annual_ret']
    vol = d.std() * np.sqrt(252) * 100

    # Sortino: use downside deviation only
    downside = d[d < 0]
    downside_vol = downside.std() * np.sqrt(252) * 100 if len(downside) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

    # Calmar: annual return / abs(max drawdown)
    max_dd = r['max_dd']
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    # Tail ratio: 95th percentile / abs(5th percentile)
    p95, p5 = np.percentile(d, 95), np.percentile(d, 5)
    tail_ratio = p95 / abs(p5) if p5 != 0 else 0

    # Distribution shape
    skew = sp_stats.skew(d)
    kurt = sp_stats.kurtosis(d)  # excess kurtosis

    # Max drawdown duration (in trading days)
    cummax = total_cap.cummax()
    in_dd = total_cap < cummax
    if in_dd.any():
        groups = (~in_dd).cumsum()
        dd_lengths = in_dd.groupby(groups).sum()
        max_dd_days = int(dd_lengths.max())
    else:
        max_dd_days = 0

    # Monthly stats
    monthly = d.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    pct_pos_months = (monthly > 0).mean() * 100

    # Worst month / best month
    worst_month = monthly.min() * 100
    best_month = monthly.max() * 100

    return {
        'Strategy': r['name'],
        'Annual %': annual_ret,
        'Vol %': vol,
        'Sharpe': annual_ret / vol if vol > 0 else 0,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max DD %': max_dd,
        'Max DD Days': max_dd_days,
        'Tail Ratio': tail_ratio,
        'Skew': skew,
        'Kurtosis': kurt,
        'Pos Months %': pct_pos_months,
        'Worst Mo %': worst_month,
        'Best Mo %': best_month,
    }

# Compute for key strategies
key_configs = [
    no_lev_results[0],   # SPY only
    lev_results[4],      # + 0.5% deep OTM puts
    lev_results[5],      # + 1.0% deep OTM puts
    lev_results[6],      # + 2.0% deep OTM puts
]

ext_rows = [compute_extended_metrics(r, years) for r in key_configs]
df_ext = pd.DataFrame(ext_rows)

styled = (df_ext.style
    .format({
        'Annual %': '{:.2f}', 'Vol %': '{:.1f}', 'Sharpe': '{:.3f}',
        'Sortino': '{:.3f}', 'Calmar': '{:.3f}', 'Max DD %': '{:.1f}',
        'Max DD Days': '{:.0f}', 'Tail Ratio': '{:.3f}',
        'Skew': '{:.3f}', 'Kurtosis': '{:.2f}',
        'Pos Months %': '{:.1f}', 'Worst Mo %': '{:.1f}', 'Best Mo %': '{:.1f}',
    })
)
style_returns_table(styled).set_caption('Extended Risk Metrics: Key Strategies')

# Return distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

spy_d = no_lev_results[0]['balance']['% change'].dropna() * 100
puts_d = lev_results[4]['balance']['% change'].dropna() * 100

for ax, data_series, name, color in [
    (axes[0], spy_d, 'SPY B&H', FT_RED),
    (axes[1], puts_d, 'SPY + 0.5% Puts', FT_GREEN),
]:
    ax.hist(data_series, bins=100, alpha=0.7, color=color, edgecolor='white', linewidth=0.3)
    ax.axvline(data_series.mean(), color='black', linestyle='--', linewidth=1, label=f'Mean: {data_series.mean():.3f}%')
    ax.axvline(data_series.median(), color='gray', linestyle=':', linewidth=1, label=f'Median: {data_series.median():.3f}%')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} — Daily Return Distribution')
    skew = sp_stats.skew(data_series)
    kurt = sp_stats.kurtosis(data_series)
    ax.legend(title=f'Skew: {skew:.3f}, Kurt: {kurt:.1f}')

fig.suptitle('Distribution Shape: Does Tail Hedging Improve the Left Tail?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'return_distribution.png'), dpi=150, bbox_inches='tight'); plt.close()

# Year-by-year returns
yearly_rows = []
for r in key_configs:
    d = r['balance']['% change'].dropna()
    total_cap = r['balance']['total capital']
    yearly = total_cap.resample('YE').last().pct_change().dropna() * 100
    for date, ret in yearly.items():
        yearly_rows.append({'Year': date.year, 'Strategy': r['name'], 'Return %': ret})

df_yearly = pd.DataFrame(yearly_rows).pivot(index='Year', columns='Strategy', values='Return %')
# Reorder columns
col_order = [r['name'] for r in key_configs if r['name'] in df_yearly.columns]
df_yearly = df_yearly[col_order]

# Add summary rows
avg = df_yearly.mean()
med = df_yearly.median()
df_yearly.loc['Average'] = avg
df_yearly.loc['Median'] = med
df_yearly.loc['% Positive'] = ((df_yearly.iloc[:-2] > 0).mean() * 100)

styled = (df_yearly.style
    .format('{:.1f}')
    .map(lambda v: f'color: {FT_GREEN}; font-weight: bold' if v > 0 else f'color: {FT_RED}')
)
style_returns_table(styled).set_caption('Calendar Year Returns (%)')

# Rolling 1-year Sharpe comparison
fig, ax = plt.subplots(figsize=(18, 6))
window = 252

for r, color, ls in [
    (no_lev_results[0], FT_RED, '-'),
    (lev_results[4], FT_GREEN, '-'),
    (lev_results[5], FT_BLUE, '--'),
]:
    d = r['balance']['% change'].dropna()
    rolling_mean = d.rolling(window).mean() * 252
    rolling_std = d.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    ax.plot(rolling_sharpe.index, rolling_sharpe, label=r['name'],
            color=color, linestyle=ls, linewidth=1.2, alpha=0.85)

ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
shade_crashes(ax)
ax.set_title('Rolling 1-Year Sharpe Ratio (rf=0%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Sharpe Ratio')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'rolling_sharpe.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------------------
# The Vol U-Curve and Diminishing Returns
# ---------------------------------------------------------------------------
budgets = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.3]
all_lev = [no_lev_results[0]] + lev_results[1:]
vols = []
sharpes = []
for r in all_lev:
    d = r['balance']['% change'].dropna()
    v = d.std() * np.sqrt(252) * 100
    vols.append(v)
    sharpes.append(r['annual_ret'] / v if v > 0 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Left: Vol vs budget
ax1.plot(budgets, vols, 'o-', color=FT_BLUE, linewidth=2, markersize=8)
ax1.axhline(vols[0], color='gray', linestyle=':', linewidth=0.8, label=f'SPY vol: {vols[0]:.1f}%')
min_vol_idx = vols.index(min(vols))
ax1.annotate(f'Min vol: {vols[min_vol_idx]:.1f}%\n({budgets[min_vol_idx]}% budget)',
             xy=(budgets[min_vol_idx], vols[min_vol_idx]),
             xytext=(budgets[min_vol_idx]+0.5, vols[min_vol_idx]-1.5),
             arrowprops=dict(arrowstyle='->', color=FT_GREEN), fontsize=11, color=FT_GREEN)
ax1.set_xlabel('Annual Put Budget (%)')
ax1.set_ylabel('Annualized Volatility (%)')
ax1.set_title('The Vol U-Curve: Puts Reduce Then Add Variance', fontweight='bold')
ax1.legend()

# Right: Sharpe vs budget
ax2.plot(budgets, sharpes, 'o-', color=FT_GREEN, linewidth=2, markersize=8)
ax2.axhline(sharpes[0], color='gray', linestyle=':', linewidth=0.8, label=f'SPY Sharpe: {sharpes[0]:.3f}')
ax2.set_xlabel('Annual Put Budget (%)')
ax2.set_ylabel('Sharpe Ratio (rf=0%)')
ax2.set_title('Sharpe Keeps Rising — But Only Because We Had 3 Crashes', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'vol_ucurve_and_sharpe.png'), dpi=150, bbox_inches='tight'); plt.close()

# Bull market subperiod: 2010-2019 (no crash > -20%)
print('\n=== Bull Market Subperiod: 2010-2019 (no major crash) ===\n')
print(f'{"Strategy":<30} {"Annual %":>10} {"Vol %":>8} {"Sharpe":>8} {"Max DD %":>10}')
print('-' * 70)
for r in all_lev:
    cap = r['balance']['total capital']
    # Filter to 2010-2019
    sub_cap = cap[(cap.index >= '2010-01-01') & (cap.index < '2020-01-01')]
    if len(sub_cap) < 100:
        continue
    sub_d = sub_cap.pct_change().dropna()
    total_ret = sub_cap.iloc[-1] / sub_cap.iloc[0]
    yrs = len(sub_d) / 252
    ann = (total_ret ** (1/yrs) - 1) * 100
    vol = sub_d.std() * np.sqrt(252) * 100
    sh = ann / vol if vol > 0 else 0
    cummax = sub_cap.cummax()
    dd = ((sub_cap - cummax) / cummax).min() * 100
    print(f'{r["name"]:<30} {ann:>10.2f} {vol:>8.1f} {sh:>8.3f} {dd:>10.1f}')

# ---------------------------------------------------------------------------
# 10a. DTE Range Sweep
# ---------------------------------------------------------------------------
dte_configs = [
    ('DTE 30-60',   30,  60,  14),
    ('DTE 60-90',   60,  90,  30),
    ('DTE 90-120',  90,  120, 45),
    ('DTE 120-180', 120, 180, 60),
    ('DTE 180-365', 180, 365, 90),
]

dte_results = []
for name, dte_min, dte_max, exit_dte in dte_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda dmin=dte_min, dmax=dte_max, edte=exit_dte: make_deep_otm_put_strategy(
            schema, dte_min=dmin, dte_max=dmax, exit_dte=edte),
        data,
        budget_pct=0.005,
    )
    dte_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r, (name, dmin, dmax, edte) in zip(dte_results, dte_configs):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({'DTE Range': name, 'Entry DTE': f'{dmin}-{dmax}', 'Exit DTE': edte,
                 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
                 'Max DD %': r['max_dd'], 'Vol %': vol, 'Trades': r['trades']})
df_dte = pd.DataFrame(rows)
styled = df_dte.style.format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}',
                               'Vol %': '{:.1f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('DTE Sweep: 0.5% budget, leveraged, deep OTM puts')

# ---------------------------------------------------------------------------
# 10b. Rebalance Frequency Sweep
# ---------------------------------------------------------------------------
rebal_configs = [
    ('Monthly (1)',    1),
    ('Bimonthly (2)',  2),
    ('Quarterly (3)',  3),
    ('Semi-annual (6)', 6),
]

rebal_results = []
for name, freq in rebal_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda: make_deep_otm_put_strategy(schema),
        data,
        budget_pct=0.005,
        rebal_months=freq,
    )
    rebal_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r, (name, freq) in zip(rebal_results, rebal_configs):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({'Rebalance': name, 'Freq (months)': freq,
                 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
                 'Max DD %': r['max_dd'], 'Vol %': vol, 'Trades': r['trades']})
df_rebal = pd.DataFrame(rows)
styled = df_rebal.style.format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}',
                                 'Vol %': '{:.1f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('Rebalance Frequency Sweep: 0.5% budget, leveraged')

# ---------------------------------------------------------------------------
# 10c. Delta Range Sweep
# ---------------------------------------------------------------------------
delta_configs = [
    ('Ultra deep: delta -0.05 to -0.01', -0.05, -0.01),
    ('Deep: delta -0.10 to -0.02',       -0.10, -0.02),
    ('Mid OTM: delta -0.15 to -0.05',    -0.15, -0.05),
    ('Near OTM: delta -0.25 to -0.10',   -0.25, -0.10),
    ('Closer ATM: delta -0.35 to -0.15', -0.35, -0.15),
]

delta_results = []
for name, dmin, dmax in delta_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda d1=dmin, d2=dmax: make_deep_otm_put_strategy(
            schema, delta_min=d1, delta_max=d2),
        data,
        budget_pct=0.005,
    )
    delta_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r, (name, dmin, dmax) in zip(delta_results, delta_configs):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({'Delta Range': name, 'delta min': dmin, 'delta max': dmax,
                 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
                 'Max DD %': r['max_dd'], 'Vol %': vol, 'Trades': r['trades']})
df_delta = pd.DataFrame(rows)
styled = df_delta.style.format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}',
                                 'Vol %': '{:.1f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('Delta Sweep: How deep OTM? (0.5% budget, leveraged)')

# ---------------------------------------------------------------------------
# 10d. Exit Timing Sweep
# ---------------------------------------------------------------------------
exit_configs = [
    ('Exit at DTE 7 (near expiry)',  7),
    ('Exit at DTE 14',               14),
    ('Exit at DTE 30',               30),
    ('Exit at DTE 45',               45),
    ('Exit at DTE 60',               60),
]

exit_results = []
for name, exit_dte in exit_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda edte=exit_dte: make_deep_otm_put_strategy(schema, exit_dte=edte),
        data,
        budget_pct=0.005,
    )
    exit_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r, (name, edte) in zip(exit_results, exit_configs):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    rows.append({'Exit Rule': name, 'Exit DTE': edte,
                 'Annual %': r['annual_ret'], 'Excess %': r['excess_annual'],
                 'Max DD %': r['max_dd'], 'Vol %': vol, 'Trades': r['trades']})
df_exit = pd.DataFrame(rows)
styled = df_exit.style.format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}',
                                'Vol %': '{:.1f}', 'Trades': '{:.0f}'}).map(color_excess, subset=['Excess %'])
style_returns_table(styled).set_caption('Exit Timing Sweep: When to sell? (0.5% budget, leveraged)')

# ---------------------------------------------------------------------------
# 10e. Multi-dimensional grid search
# ---------------------------------------------------------------------------
from itertools import product

grid_dte = [(60, 90, 30), (90, 120, 45)]
grid_delta = [(-0.10, -0.02), (-0.15, -0.05)]
grid_exit = [21, 30, 45]
grid_budget = [0.003, 0.005, 0.01]

combos = list(product(grid_dte, grid_delta, grid_exit, grid_budget))
print(f'Running {len(combos)} combinations via Rust parallel sweep...\n')

import time as _time
import polars as pl
import pyarrow as pa
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData

_opts_data = data['options_data']
_stocks_data = data['stocks_data']
_opts_schema = _opts_data.schema
_stocks_schema = _stocks_data.schema

# Build rebalance dates as nanosecond timestamps
_dates_df = (
    pd.DataFrame(_opts_data._data[["quotedate", "volume"]])
    .drop_duplicates("quotedate")
    .set_index("quotedate")
)
_rb_days = pd.to_datetime(
    _dates_df.groupby(pd.Grouper(freq="1BMS"))
    .apply(lambda x: x.index.min())
    .values
)
_rb_date_ns = [int(d.value) for d in _rb_days if not pd.isna(d)]

# Build base config using default deep OTM put strategy
_base_strat = make_deep_otm_put_strategy(schema)
_base_leg = _base_strat.legs[0]
_base_config = {
    "allocation": {"stocks": 1.0, "options": 0.0, "cash": 0.0},
    "initial_capital": float(INITIAL_CAPITAL),
    "shares_per_contract": 100,
    "rebalance_dates": _rb_date_ns,
    "legs": [{
        "name": _base_leg.name,
        "entry_filter": _base_leg.entry_filter.query,
        "exit_filter": _base_leg.exit_filter.query,
        "direction": _base_leg.direction.price_column,
        "type": _base_leg.type.value,
        "entry_sort_col": _base_leg.entry_sort[0] if _base_leg.entry_sort else None,
        "entry_sort_asc": _base_leg.entry_sort[1] if _base_leg.entry_sort else True,
    }],
    "profit_pct": None,
    "loss_pct": None,
    "stocks": [("SPY", 1.0)],
    "options_budget_pct": 0.005,
    "check_exits_daily": True,
}

_schema_mapping = {
    "contract": _opts_schema["contract"],
    "date": _opts_schema["date"],
    "stocks_date": _stocks_schema["date"],
    "stocks_symbol": _stocks_schema["symbol"],
    "stocks_price": _stocks_schema["adjClose"],
    "underlying": _opts_schema["underlying"],
    "expiration": _opts_schema["expiration"],
    "type": _opts_schema["type"],
    "strike": _opts_schema["strike"],
}

# Convert to Polars
_opts_pl = pl.from_arrow(pa.Table.from_pandas(_opts_data._data, preserve_index=False))
_stocks_pl = pl.from_arrow(pa.Table.from_pandas(_stocks_data._data, preserve_index=False))

# Build param overrides
_param_grid = []
for (dte_min, dte_max, _), (d_min, d_max), exit_dte, budget in combos:
    label = f'DTE{dte_min}-{dte_max} delta({d_min},{d_max}) exit{exit_dte} b{budget*100:.1f}%'
    entry_q = (f"((type == 'put') & (ask > 0)) & (((((underlying == 'SPY')"
               f" & (dte >= {dte_min})) & (dte <= {dte_max}))"
               f" & (delta >= {d_min})) & (delta <= {d_max}))")
    exit_q = f"(type == 'put') & (dte <= {exit_dte})"
    _param_grid.append({
        "label": label,
        "leg_entry_filters": [entry_q],
        "leg_exit_filters": [exit_q],
        "options_budget_pct": budget,
    })

_t0 = _time.perf_counter()
from options_portfolio_backtester._ob_rust import parallel_sweep
_sweep_results = parallel_sweep(
    _opts_pl, _stocks_pl, _base_config, _schema_mapping, _param_grid, None,
)
_elapsed = _time.perf_counter() - _t0
print(f'Parallel sweep done in {_elapsed:.1f}s')

# Convert sweep results to the format the rest of the script expects
spy_annual = data['spy_annual_ret']
grid_results = []
for res, ((dte_min, dte_max, _), (d_min, d_max), exit_dte, budget) in zip(_sweep_results, combos):
    ann_ret = res['annualized_return'] * 100
    max_dd = res['max_drawdown'] * 100
    sharpe = res.get('sharpe_ratio', 0.0)
    excess = ann_ret - spy_annual
    grid_results.append({
        'DTE': f'{dte_min}-{dte_max}', 'Delta': f'({d_min},{d_max})',
        'Exit DTE': exit_dte, 'Budget %': budget * 100,
        'Annual %': ann_ret, 'Excess %': excess,
        'Max DD %': max_dd, 'Vol %': 0.0,  # not available from sweep stats
        'Sharpe': sharpe,
        'Trades': res.get('total_trades', 0),
    })

df_grid = pd.DataFrame(grid_results)
print(f'\nDone. {len(df_grid)} configs tested.')

# Top 10 by Sharpe
top_sharpe = df_grid.sort_values('Sharpe', ascending=False).head(10)
styled = (top_sharpe.style
    .format({'Budget %': '{:.1f}', 'Annual %': '{:.2f}', 'Excess %': '{:+.2f}',
             'Max DD %': '{:.1f}', 'Vol %': '{:.1f}', 'Sharpe': '{:.3f}', 'Trades': '{:.0f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption('Top 10 Configs by Sharpe Ratio')

# Top 10 by lowest max drawdown
top_dd = df_grid.sort_values('Max DD %', ascending=False).head(10)
styled = (top_dd.style
    .format({'Budget %': '{:.1f}', 'Annual %': '{:.2f}', 'Excess %': '{:+.2f}',
             'Max DD %': '{:.1f}', 'Vol %': '{:.1f}', 'Sharpe': '{:.3f}', 'Trades': '{:.0f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption('Top 10 Configs by Lowest Max Drawdown')

# Heatmap
import seaborn as sns_hm

pivot = df_grid.pivot_table(index='Budget %', columns='DTE', values='Annual %', aggfunc='mean')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, metric, cmap, title in [
    (axes[0], 'Annual %', 'YlGn', 'Annual Return (%) by DTE x Budget'),
    (axes[1], 'Max DD %', 'RdYlGn', 'Max Drawdown (%) by DTE x Budget'),
    (axes[2], 'Sharpe', 'YlGn', 'Sharpe Ratio by DTE x Budget'),
]:
    pv = df_grid.pivot_table(index='Budget %', columns='DTE', values=metric, aggfunc='mean')
    sns_hm.heatmap(pv, annot=True, fmt='.1f' if 'DD' in metric else '.2f',
                   cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(title, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'grid_search_heatmaps.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------------------
# 10f. Overfitting check
# ---------------------------------------------------------------------------
spy_annual = data['spy_annual_ret']
all_beat = (df_grid['Annual %'] > spy_annual).all()
min_excess = df_grid['Excess %'].min()
max_excess = df_grid['Excess %'].max()
median_excess = df_grid['Excess %'].median()

print('=== Overfitting Check ===\n')
print(f'Total configs tested: {len(df_grid)}')
print(f'All beat SPY? {all_beat}')
print(f'Excess return range: {min_excess:+.2f}% to {max_excess:+.2f}%')
print(f'Median excess: {median_excess:+.2f}%')
print(f'Worst config still beats SPY by: {min_excess:+.2f}%/yr')
print()

spread = max_excess - min_excess
print(f'Spread (best - worst): {spread:.2f}%')
print(f'If spread is small relative to median, the strategy is robust to parameter choice.')
print(f'Ratio spread/median: {spread/median_excess:.1f}x')
print()

spy_vol = daily_returns.std() * np.sqrt(252) * 100
spy_sharpe = spy_annual / spy_vol
configs_better_sharpe = (df_grid['Sharpe'] > spy_sharpe).sum()
print(f'SPY Sharpe: {spy_sharpe:.3f}')
print(f'Configs with higher Sharpe: {configs_better_sharpe}/{len(df_grid)} ({configs_better_sharpe/len(df_grid)*100:.0f}%)')

# ---------------------------------------------------------------------------
# 10g. Out-of-sample test
# ---------------------------------------------------------------------------
spy_mid = spy_prices.index[0] + (spy_prices.index[-1] - spy_prices.index[0]) / 2
print(f'Split date: {spy_mid.strftime("%Y-%m-%d")}\n')

first_half_prices = spy_prices[spy_prices.index <= spy_mid]
first_years = (first_half_prices.index[-1] - first_half_prices.index[0]).days / 365.25

second_half_prices = spy_prices[spy_prices.index > spy_mid]
second_years = (second_half_prices.index[-1] - second_half_prices.index[0]).days / 365.25

r_full = run_backtest(
    'Full period', 1.0, 0.0,
    lambda: make_deep_otm_put_strategy(schema),
    data,
    budget_pct=0.005,
)

bal = r_full['balance']
total_cap = bal['total capital']

# First half performance
first_cap = total_cap[total_cap.index <= spy_mid]
if len(first_cap) > 1:
    first_ret = (first_cap.iloc[-1] / first_cap.iloc[0] - 1) * 100
    first_annual = ((1 + first_ret / 100) ** (1 / first_years) - 1) * 100
    first_dd = ((first_cap - first_cap.cummax()) / first_cap.cummax()).min() * 100
    spy_first = first_half_prices
    spy_first_ret = ((spy_first.iloc[-1] / spy_first.iloc[0] - 1) * 100)
    spy_first_annual = ((1 + spy_first_ret / 100) ** (1 / first_years) - 1) * 100

# Second half performance
second_cap = total_cap[total_cap.index > spy_mid]
if len(second_cap) > 1:
    second_ret = (second_cap.iloc[-1] / second_cap.iloc[0] - 1) * 100
    second_annual = ((1 + second_ret / 100) ** (1 / second_years) - 1) * 100
    second_dd = ((second_cap - second_cap.cummax()) / second_cap.cummax()).min() * 100
    spy_second = second_half_prices
    spy_second_ret = ((spy_second.iloc[-1] / spy_second.iloc[0] - 1) * 100)
    spy_second_annual = ((1 + spy_second_ret / 100) ** (1 / second_years) - 1) * 100

print('Out-of-Sample Check: same params (0.5% budget, default deep OTM), split in half\n')
print(f'{"Period":<25} {"Strategy":>12} {"SPY B&H":>12} {"Excess":>10} {"Max DD":>10}')
print('-' * 70)
print(f'{"First half (2008-~2017)":<25} {first_annual:>11.2f}% {spy_first_annual:>11.2f}% {first_annual-spy_first_annual:>+9.2f}% {first_dd:>9.1f}%')
print(f'{"Second half (~2017-2025)":<25} {second_annual:>11.2f}% {spy_second_annual:>11.2f}% {second_annual-spy_second_annual:>+9.2f}% {second_dd:>9.1f}%')
print(f'{"Full period":<25} {r_full["annual_ret"]:>11.2f}% {data["spy_annual_ret"]:>11.2f}% {r_full["excess_annual"]:>+9.2f}% {r_full["max_dd"]:>9.1f}%')
print()
both_positive = (first_annual > spy_first_annual) and (second_annual > spy_second_annual)
print(f'Beats SPY in BOTH halves? {"YES" if both_positive else "NO"}')

# ---------------------------------------------------------------------------
# 10i. Subperiod analysis
# ---------------------------------------------------------------------------
subperiods = [
    ('Full (2008-2025)',  None, None),
    ('GFC era (2008-2009)', '2008-01-01', '2010-01-01'),
    ('Bull market (2010-2019)', '2010-01-01', '2020-01-01'),
    ('COVID + after (2020-2022)', '2020-01-01', '2023-01-01'),
    ('Recent (2023-2025)', '2023-01-01', None),
]

r_sub = run_backtest(
    'subperiod test', 1.0, 0.0,
    lambda: make_deep_otm_put_strategy(schema),
    data,
    budget_pct=0.005,
)

bal = r_sub['balance']
total_cap = bal['total capital']

sub_rows = []
for name, start, end in subperiods:
    s = pd.Timestamp(start) if start else total_cap.index[0]
    e = pd.Timestamp(end) if end else total_cap.index[-1]

    # Strategy
    period_cap = total_cap[(total_cap.index >= s) & (total_cap.index <= e)]
    if len(period_cap) < 20:
        continue
    period_years = (period_cap.index[-1] - period_cap.index[0]).days / 365.25
    if period_years < 0.5:
        continue
    period_ret = (period_cap.iloc[-1] / period_cap.iloc[0] - 1) * 100
    period_annual = ((1 + period_ret / 100) ** (1 / period_years) - 1) * 100
    period_dd = ((period_cap - period_cap.cummax()) / period_cap.cummax()).min() * 100

    # SPY
    spy_period = spy_prices[(spy_prices.index >= s) & (spy_prices.index <= e)]
    spy_ret = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
    spy_annual = ((1 + spy_ret / 100) ** (1 / period_years) - 1) * 100
    spy_dd_p = ((spy_period - spy_period.cummax()) / spy_period.cummax()).min() * 100

    sub_rows.append({
        'Period': name,
        'Years': period_years,
        'Strategy %/yr': period_annual,
        'SPY %/yr': spy_annual,
        'Excess %': period_annual - spy_annual,
        'Strategy DD %': period_dd,
        'SPY DD %': spy_dd_p,
    })

df_sub = pd.DataFrame(sub_rows)
styled = (df_sub.style
    .format({'Years': '{:.1f}', 'Strategy %/yr': '{:.2f}', 'SPY %/yr': '{:.2f}',
             'Excess %': '{:+.2f}', 'Strategy DD %': '{:.1f}', 'SPY DD %': '{:.1f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption(
    'Subperiod Analysis: Does the strategy work in calm markets? (0.5% budget, leveraged)'
)

# ---------------------------------------------------------------------------
# 10i-b. Calm-Period Deep Dive: 2012-2018 (All Configurations)
# ---------------------------------------------------------------------------
CALM_START = pd.Timestamp('2012-01-01')
CALM_END   = pd.Timestamp('2019-01-01')

calm_configs = [
    ('SPY only',           'baseline',    1.0,   0.0,   None),
    ('Spitznagel 0.5%',   'spitznagel',  1.0,   0.0,   0.005),
    ('Spitznagel 1.0%',   'spitznagel',  1.0,   0.0,   0.01),
    ('Spitznagel 3.3%',   'spitznagel',  1.0,   0.0,   0.033),
    ('No-leverage 0.5%',  'no-leverage',  0.995, 0.005, None),
    ('No-leverage 1.0%',  'no-leverage',  0.99,  0.01,  None),
    ('No-leverage 3.3%',  'no-leverage',  0.967, 0.033, None),
]

calm_results = []
for name, framing, spct, opct, budget_pct in calm_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, spct, opct, lambda: make_deep_otm_put_strategy(schema), data, budget_pct=budget_pct)
    r['framing'] = framing
    calm_results.append(r)
    print(f'done')

# Extract 2012-2018 subperiod stats with vol and Sharpe
def calm_subperiod_stats(r, spy_prices, start, end):
    total_cap = r['balance']['total capital']
    daily_ret = r['balance']['% change']
    mask = (total_cap.index >= start) & (total_cap.index < end)
    cap = total_cap[mask]
    d = daily_ret[mask].dropna()
    yrs = (cap.index[-1] - cap.index[0]).days / 365.25
    annual = ((cap.iloc[-1] / cap.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = ((cap - cap.cummax()) / cap.cummax()).min() * 100
    vol = d.std() * np.sqrt(252) * 100
    sharpe = annual / vol if vol > 0 else 0
    spy_m = (spy_prices.index >= start) & (spy_prices.index < end)
    spy = spy_prices[spy_m]
    spy_annual = ((spy.iloc[-1] / spy.iloc[0]) ** (1 / yrs) - 1) * 100
    return {'Strategy': r['name'], 'Annual %': annual, 'vs SPY %': annual - spy_annual,
            'Max DD %': dd, 'Vol %': vol, 'Sharpe': sharpe}

calm_rows = [calm_subperiod_stats(r, spy_prices, CALM_START, CALM_END) for r in calm_results]
df_calm = pd.DataFrame(calm_rows)

styled = (df_calm.style
    .format({'Annual %': '{:+.2f}', 'vs SPY %': '{:+.2f}', 'Max DD %': '{:.1f}',
             'Vol %': '{:.1f}', 'Sharpe': '{:.3f}'})
    .map(color_excess, subset=['vs SPY %'])
)
style_returns_table(styled).set_caption('Calm-Period Comparison: 2012-2018 (no correction > -19.3%)')

# Year-by-year for Spitznagel 0.5% during 2012-2018
r_spitz_calm = calm_results[1]  # Spitznagel 0.5%
cap = r_spitz_calm['balance']['total capital']
cap_calm = cap[(cap.index >= CALM_START) & (cap.index < CALM_END)]

spy_calm = spy_prices[(spy_prices.index >= CALM_START) & (spy_prices.index < CALM_END)]

yby_rows = []
for yr in range(2012, 2019):
    yr_start = pd.Timestamp(f'{yr}-01-01')
    yr_end = pd.Timestamp(f'{yr+1}-01-01')
    yr_cap = cap_calm[(cap_calm.index >= yr_start) & (cap_calm.index < yr_end)]
    yr_spy = spy_calm[(spy_calm.index >= yr_start) & (spy_calm.index < yr_end)]
    if len(yr_cap) < 10 or len(yr_spy) < 10:
        continue
    st = (yr_cap.iloc[-1] / yr_cap.iloc[0] - 1) * 100
    sr = (yr_spy.iloc[-1] / yr_spy.iloc[0] - 1) * 100
    yby_rows.append({'Year': yr, 'SPY %': sr, 'Strategy %': st, 'Excess %': st - sr})

df_yby = pd.DataFrame(yby_rows)
styled = (df_yby.style
    .format({'SPY %': '{:+.2f}', 'Strategy %': '{:+.2f}', 'Excess %': '{:+.2f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption('Year-by-Year: Spitznagel 0.5% vs SPY (2012-2018)')

# ---------------------------------------------------------------------------
# 10k. Weekly vs Monthly
# ---------------------------------------------------------------------------
freq_configs = [
    ('Monthly',   1, 'BMS'),
    ('Biweekly',  2, 'W-MON'),
    ('Weekly',    1, 'W-MON'),
]

freq_results = []
for name, freq, unit in freq_configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda: make_deep_otm_put_strategy(schema),
        data,
        budget_pct=0.005,
        rebal_months=freq,
        rebal_unit=unit,
    )
    freq_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

rows = []
for r, (name, freq, unit) in zip(freq_results, freq_configs):
    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    sharpe = r['annual_ret'] / vol if vol > 0 else 0
    rows.append({
        'Frequency': name,
        'Annual %': r['annual_ret'],
        'Excess %': r['excess_annual'],
        'Max DD %': r['max_dd'],
        'Vol %': vol,
        'Sharpe': sharpe,
        'Trades': r['trades'],
    })
df_freq = pd.DataFrame(rows)
styled = (df_freq.style
    .format({'Annual %': '{:.2f}', 'Excess %': '{:+.2f}', 'Max DD %': '{:.1f}',
             'Vol %': '{:.1f}', 'Sharpe': '{:.3f}', 'Trades': '{:.0f}'})
    .map(color_excess, subset=['Excess %'])
)
style_returns_table(styled).set_caption(
    'Rebalance Frequency: Monthly vs Biweekly vs Weekly (0.5% budget, leveraged)'
)

print('\n=== All charts saved to charts/ directory ===')
