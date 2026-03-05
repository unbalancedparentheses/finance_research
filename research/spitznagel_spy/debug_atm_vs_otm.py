#!/usr/bin/env python3
"""Debug: Why does ATM appear to outperform deep OTM?
Compare trade-level details for both strategies."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import (
    load_data, run_backtest, INITIAL_CAPITAL,
    make_deep_otm_put_strategy, make_atm_put_strategy,
)

data = load_data()
schema = data['schema']

# Run both at 0.5% budget, leveraged
print('\n=== Deep OTM (delta -0.10 to -0.02) leveraged 0.5% ===')
r_otm = run_backtest('Deep OTM 0.5%', 1.0, 0.0,
    lambda: make_deep_otm_put_strategy(schema), data, budget_pct=0.005)

print(f'Annual: {r_otm["annual_ret"]:.2f}%, Excess: {r_otm["excess_annual"]:+.2f}%, DD: {r_otm["max_dd"]:.1f}%')
print(f'Trades: {r_otm["trades"]}')

print('\n=== ATM (delta -0.55 to -0.40) leveraged 0.5% ===')
r_atm = run_backtest('ATM 0.5%', 1.0, 0.0,
    lambda: make_atm_put_strategy(schema), data, budget_pct=0.005)

print(f'Annual: {r_atm["annual_ret"]:.2f}%, Excess: {r_atm["excess_annual"]:+.2f}%, DD: {r_atm["max_dd"]:.1f}%')
print(f'Trades: {r_atm["trades"]}')

# Compare trade logs
def analyze_trades(trade_log, name):
    print(f'\n--- {name}: Trade Log Analysis ---')
    if trade_log.empty:
        print('No trades!')
        return

    # Get column structure
    print(f'Columns: {trade_log.columns.tolist()[:10]}...')
    print(f'Total rows: {len(trade_log)}')

    # Show first few trades
    print(f'\nFirst 5 entries:')
    cols_to_show = []
    for col in trade_log.columns:
        if isinstance(col, tuple):
            if col[1] in ('date', 'qty', 'cost', 'strike', 'delta', 'dte', 'type', 'action'):
                cols_to_show.append(col)
        else:
            if col in ('date', 'qty', 'cost', 'strike', 'delta', 'dte', 'type', 'action'):
                cols_to_show.append(col)

    if cols_to_show:
        print(trade_log[cols_to_show].head(10).to_string())
    else:
        print(trade_log.head(5).to_string())

    # Entry trades only
    if ('totals', 'qty') in trade_log.columns:
        entries = trade_log[trade_log[('totals', 'qty')] > 0]
        exits = trade_log[trade_log[('totals', 'qty')] < 0]

        print(f'\nEntry trades: {len(entries)}')
        print(f'Exit trades: {len(exits)}')

        if len(entries) > 0:
            entry_costs = entries[('totals', 'cost')].abs()
            entry_qty = entries[('totals', 'qty')]

            print(f'\nEntry costs (total $ per trade):')
            print(f'  Mean: ${entry_costs.mean():,.2f}')
            print(f'  Median: ${entry_costs.median():,.2f}')
            print(f'  Min: ${entry_costs.min():,.2f}')
            print(f'  Max: ${entry_costs.max():,.2f}')

            print(f'\nContracts per entry:')
            print(f'  Mean: {entry_qty.mean():.1f}')
            print(f'  Median: {entry_qty.median():.1f}')
            print(f'  Min: {entry_qty.min():.0f}')
            print(f'  Max: {entry_qty.max():.0f}')

        # Compute per-trade P&L by matching entries/exits
        if ('totals', 'date') in trade_log.columns:
            entry_dates = entries[('totals', 'date')].values
            exit_dates = exits[('totals', 'date')].values

            print(f'\nFirst 10 entry dates: {entry_dates[:10]}')
            print(f'First 10 exit dates: {exit_dates[:10]}')

    # Try leg_1 columns for strike/delta info
    for prefix in ['leg_1', 'totals']:
        strike_col = (prefix, 'strike') if (prefix, 'strike') in trade_log.columns else None
        delta_col = (prefix, 'delta') if (prefix, 'delta') in trade_log.columns else None

        if strike_col:
            entries_mask = trade_log[('totals', 'qty')] > 0
            entry_strikes = trade_log.loc[entries_mask, strike_col]
            print(f'\n{prefix} Entry strikes:')
            print(f'  Mean: ${entry_strikes.mean():,.2f}')
            print(f'  Min: ${entry_strikes.min():,.2f}')
            print(f'  Max: ${entry_strikes.max():,.2f}')

        if delta_col:
            entry_deltas = trade_log.loc[entries_mask, delta_col]
            print(f'{prefix} Entry deltas:')
            print(f'  Mean: {entry_deltas.mean():.4f}')
            print(f'  Min: {entry_deltas.min():.4f}')
            print(f'  Max: {entry_deltas.max():.4f}')

analyze_trades(r_otm['trade_log'], 'Deep OTM')
analyze_trades(r_atm['trade_log'], 'ATM')

# Compare crash period P&L
print('\n\n=== Crash Period Comparison ===')
CRASHES = [
    ('2008 GFC',   '2007-10-01', '2009-06-01'),
    ('2020 COVID', '2020-01-01', '2020-06-01'),
    ('2022 Bear',  '2022-01-01', '2022-12-31'),
]

for strategy_name, r in [('Deep OTM', r_otm), ('ATM', r_atm)]:
    tl = r['trade_log']
    if tl.empty:
        continue

    entries = tl[tl[('totals', 'qty')] > 0].copy()
    exits = tl[tl[('totals', 'qty')] < 0].copy()

    print(f'\n--- {strategy_name} ---')

    # Match entries to exits and compute P&L
    entry_list = []
    for idx, row in entries.iterrows():
        entry_date = row[('totals', 'date')]
        entry_cost = row[('totals', 'cost')]
        entry_qty = row[('totals', 'qty')]
        # Find corresponding exit
        exit_rows = exits[exits.index > idx]
        if not exit_rows.empty:
            exit_row = exit_rows.iloc[0]
            exit_cost = exit_row[('totals', 'cost')]
            pnl = -(entry_cost + exit_cost)
            entry_list.append({
                'entry_date': entry_date,
                'exit_date': exit_row[('totals', 'date')],
                'qty': entry_qty,
                'entry_cost': entry_cost,
                'exit_cost': exit_cost,
                'pnl': pnl,
            })

    trades_df = pd.DataFrame(entry_list)
    if trades_df.empty:
        continue

    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])

    total_pnl = trades_df['pnl'].sum()
    total_premium = trades_df['entry_cost'].sum()
    print(f'Total trades: {len(trades_df)}')
    print(f'Total premium: ${abs(total_premium):,.0f}')
    print(f'Total P&L: ${total_pnl:,.0f}')

    for crash_name, start, end in CRASHES:
        mask = (trades_df['entry_date'] >= pd.Timestamp(start)) & (trades_df['entry_date'] <= pd.Timestamp(end))
        crash_trades = trades_df[mask]
        if len(crash_trades) > 0:
            print(f'  {crash_name}: {len(crash_trades)} trades, P&L ${crash_trades["pnl"].sum():,.0f}, '
                  f'avg qty {crash_trades["qty"].mean():.0f}')
        else:
            print(f'  {crash_name}: no trades in period')

# Capital comparison during crashes
print('\n\n=== Capital During Crashes ===')
for crash_name, start, end in CRASHES:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for strategy_name, r in [('Deep OTM', r_otm), ('ATM', r_atm)]:
        cap = r['balance']['total capital']
        period = cap[(cap.index >= s) & (cap.index <= e)]
        if len(period) > 1:
            peak = period.iloc[0]
            trough = period.min()
            dd = (trough - peak) / peak * 100
            print(f'  {crash_name} - {strategy_name}: peak ${peak:,.0f} -> trough ${trough:,.0f} (DD {dd:.1f}%)')

# Year-by-year comparison
print('\n\n=== Year-by-Year Returns ===')
print(f'{"Year":<6} {"Deep OTM":>10} {"ATM":>10} {"SPY":>10} {"OTM excess":>12} {"ATM excess":>12}')
print('-' * 65)
for r_name, r in [('Deep OTM', r_otm), ('ATM', r_atm)]:
    cap = r['balance']['total capital']

spy = data['spy_prices']
otm_cap = r_otm['balance']['total capital']
atm_cap = r_atm['balance']['total capital']

for yr in range(2008, 2026):
    yr_start = pd.Timestamp(f'{yr}-01-01')
    yr_end = pd.Timestamp(f'{yr+1}-01-01')

    for cap_series, vals in [(otm_cap, []), (atm_cap, []), (spy, [])]:
        period = cap_series[(cap_series.index >= yr_start) & (cap_series.index < yr_end)]
        if len(period) > 10:
            ret = (period.iloc[-1] / period.iloc[0] - 1) * 100
            vals.append(ret)

    otm_period = otm_cap[(otm_cap.index >= yr_start) & (otm_cap.index < yr_end)]
    atm_period = atm_cap[(atm_cap.index >= yr_start) & (atm_cap.index < yr_end)]
    spy_period = spy[(spy.index >= yr_start) & (spy.index < yr_end)]

    if len(otm_period) > 10 and len(atm_period) > 10 and len(spy_period) > 10:
        otm_ret = (otm_period.iloc[-1] / otm_period.iloc[0] - 1) * 100
        atm_ret = (atm_period.iloc[-1] / atm_period.iloc[0] - 1) * 100
        spy_ret = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
        print(f'{yr:<6} {otm_ret:>10.2f} {atm_ret:>10.2f} {spy_ret:>10.2f} {otm_ret-spy_ret:>+12.2f} {atm_ret-spy_ret:>+12.2f}')
