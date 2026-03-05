#!/usr/bin/env python3
"""Debug: Detailed trade-by-trade comparison during GFC for ATM vs deep OTM."""

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
spy_prices = data['spy_prices']

# Run both at 0.5% budget, leveraged
r_otm = run_backtest('Deep OTM 0.5%', 1.0, 0.0,
    lambda: make_deep_otm_put_strategy(schema), data, budget_pct=0.005)
r_atm = run_backtest('ATM 0.5%', 1.0, 0.0,
    lambda: make_atm_put_strategy(schema), data, budget_pct=0.005)

# Show ALL columns available
print('=== Trade Log Column Structure ===')
tl = r_otm['trade_log']
print(f'All columns: {tl.columns.tolist()}')
print()

# Show entries/exits properly using cost sign
def show_trade_details(tl, name, start_date, end_date):
    """Show trade-by-trade details for a period."""
    print(f'\n{"="*80}')
    print(f'{name}: Trades from {start_date} to {end_date}')
    print(f'{"="*80}')

    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)

    # Filter by date
    dates = pd.to_datetime(tl[('totals', 'date')])
    mask = (dates >= s) & (dates <= e)
    period = tl[mask].copy()

    if period.empty:
        print('No trades in period!')
        return

    print(f'Total rows: {len(period)}')
    print()

    # Determine entry vs exit by cost sign
    # Positive cost = entry (paid premium), Negative cost = exit (received money)
    entries = period[period[('totals', 'cost')] > 0]
    exits = period[period[('totals', 'cost')] <= 0]

    print(f'Entries: {len(entries)}, Exits: {len(exits)}')
    print()

    # Show each trade
    print(f'{"Date":<12} {"Type":<6} {"Strike":>8} {"Qty":>5} {"Cost/ct":>10} {"Total $":>12} {"SPY price":>10}')
    print('-' * 75)

    for idx, row in period.iterrows():
        date = str(row[('totals', 'date')])[:10]
        cost_per = row[('totals', 'cost')]
        qty = row[('totals', 'qty')]
        total_cost = cost_per * qty
        strike = row[('leg_1', 'strike')]
        action = 'ENTRY' if cost_per > 0 else 'EXIT'

        # Get SPY price on that date
        trade_date = pd.Timestamp(row[('totals', 'date')])
        spy_on_date = spy_prices[spy_prices.index <= trade_date].iloc[-1] if len(spy_prices[spy_prices.index <= trade_date]) > 0 else 0

        print(f'{date:<12} {action:<6} {strike:>8.0f} {qty:>5.0f} {cost_per:>10,.0f} {total_cost:>12,.0f} {spy_on_date:>10.2f}')

    # Compute P&L by pairing entries with next exit
    print(f'\nTrade P&L:')
    print(f'{"Entry date":<12} {"Exit date":<12} {"Strike":>8} {"Qty":>5} {"Entry $":>10} {"Exit $":>10} {"P&L":>10}')
    print('-' * 75)

    total_pnl = 0
    i = 0
    rows_list = list(period.iterrows())
    while i < len(rows_list) - 1:
        idx1, row1 = rows_list[i]
        idx2, row2 = rows_list[i + 1]

        c1 = row1[('totals', 'cost')]
        c2 = row2[('totals', 'cost')]

        if c1 > 0 and c2 <= 0:  # entry followed by exit
            entry_total = c1 * row1[('totals', 'qty')]
            exit_total = c2 * row2[('totals', 'qty')]
            pnl = -(entry_total + exit_total)
            total_pnl += pnl

            entry_date = str(row1[('totals', 'date')])[:10]
            exit_date = str(row2[('totals', 'date')])[:10]
            strike = row1[('leg_1', 'strike')]
            qty = row1[('totals', 'qty')]

            print(f'{entry_date:<12} {exit_date:<12} {strike:>8.0f} {qty:>5.0f} {entry_total:>10,.0f} {exit_total:>10,.0f} {pnl:>+10,.0f}')
            i += 2
        else:
            i += 1

    print(f'\nTotal P&L in period: ${total_pnl:>+,.0f}')

# GFC period
show_trade_details(r_otm['trade_log'], 'DEEP OTM', '2007-10-01', '2009-06-01')
show_trade_details(r_atm['trade_log'], 'ATM', '2007-10-01', '2009-06-01')

# COVID period
show_trade_details(r_otm['trade_log'], 'DEEP OTM', '2020-01-01', '2020-06-01')
show_trade_details(r_atm['trade_log'], 'ATM', '2020-01-01', '2020-06-01')

# Calm period 2013
show_trade_details(r_otm['trade_log'], 'DEEP OTM', '2013-01-01', '2013-12-31')
show_trade_details(r_atm['trade_log'], 'ATM', '2013-01-01', '2013-12-31')

# Total options P&L over full period
print('\n\n=== FULL PERIOD OPTIONS P&L ===')
for name, r in [('Deep OTM', r_otm), ('ATM', r_atm)]:
    tl = r['trade_log']
    entries = tl[tl[('totals', 'cost')] > 0]
    exits = tl[tl[('totals', 'cost')] <= 0]

    entry_total = (entries[('totals', 'cost')] * entries[('totals', 'qty')]).sum()
    exit_total = (exits[('totals', 'cost')] * exits[('totals', 'qty')]).sum()
    net_pnl = -(entry_total + exit_total)

    print(f'\n{name}:')
    print(f'  Total premium paid: ${entry_total:,.0f}')
    print(f'  Total received at exit: ${abs(exit_total):,.0f}')
    print(f'  Net options P&L: ${net_pnl:+,.0f}')
    print(f'  P&L / premium: {net_pnl / entry_total * 100:.1f}%')
    print(f'  Number of round-trips: {len(entries)}')
    print(f'  Avg premium per trade: ${entry_total / len(entries):,.0f}')

# Balance comparison at key dates
print('\n\n=== PORTFOLIO VALUE AT KEY DATES ===')
key_dates = [
    '2008-01-02', '2008-09-15', '2009-03-09', '2009-12-31',
    '2020-02-19', '2020-03-23', '2020-12-31',
    '2022-01-03', '2022-10-12', '2025-12-12',
]

print(f'{"Date":<12} {"SPY":>12} {"Deep OTM":>12} {"ATM":>12} {"OTM vs SPY":>12} {"ATM vs SPY":>12}')
print('-' * 75)

for d in key_dates:
    dt = pd.Timestamp(d)
    # Find nearest date
    for series_name, series in [('SPY', spy_prices)]:
        nearby = series[series.index <= dt + pd.Timedelta(days=5)]
        if len(nearby) > 0:
            spy_val = nearby.iloc[-1] / spy_prices.iloc[0] * INITIAL_CAPITAL
        else:
            spy_val = 0

    otm_cap = r_otm['balance']['total capital']
    nearby = otm_cap[otm_cap.index <= dt + pd.Timedelta(days=5)]
    otm_val = nearby.iloc[-1] if len(nearby) > 0 else 0

    atm_cap = r_atm['balance']['total capital']
    nearby = atm_cap[atm_cap.index <= dt + pd.Timedelta(days=5)]
    atm_val = nearby.iloc[-1] if len(nearby) > 0 else 0

    otm_vs_spy = (otm_val - spy_val)
    atm_vs_spy = (atm_val - spy_val)

    print(f'{d:<12} {spy_val:>12,.0f} {otm_val:>12,.0f} {atm_val:>12,.0f} {otm_vs_spy:>+12,.0f} {atm_vs_spy:>+12,.0f}')
