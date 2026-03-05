#!/usr/bin/env python3
"""Test different OTM configurations to find realistic Spitznagel settings."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, run_backtest, INITIAL_CAPITAL
from options_portfolio_backtester import OptionType as Type, Direction
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

data = load_data()
schema = data['schema']

def make_put(schema, delta_min, delta_max, dte_min=60, dte_max=90, exit_dte=30, sort_asc=True):
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', sort_asc)  # True=ascending picks most negative (closest to ATM in range)
    leg.exit_filter = (schema.dte <= exit_dte)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s

# First: check what the actual entry strikes/deltas look like
print('=== What options does each config actually pick? ===\n')

# Quick helper to show what gets picked
def show_picks(name, delta_min, delta_max, sort_asc):
    r = run_backtest(name, 1.0, 0.0,
        lambda d1=delta_min, d2=delta_max, sa=sort_asc: make_put(schema, d1, d2, sort_asc=sa),
        data, budget_pct=0.005)
    tl = r['trade_log']
    entries = tl[tl[('totals', 'cost')] > 0]

    strikes = entries[('leg_1', 'strike')]
    # Compute OTM percentage using the spy_prices on entry dates
    spy = data['spy_prices']
    otm_pcts = []
    for idx, row in entries.iterrows():
        entry_date = pd.Timestamp(row[('totals', 'date')])
        spy_nearby = spy[spy.index <= entry_date]
        if len(spy_nearby) > 0:
            spy_price = spy_nearby.iloc[-1]
            strike = row[('leg_1', 'strike')]
            # But spy_prices is adjClose, strike is unadjusted...
            # Just use strike/underlying_last from the raw data if available
            otm_pcts.append(0)  # placeholder

    costs = entries[('totals', 'cost')]
    qtys = entries[('totals', 'qty')]
    total_costs = costs * qtys

    print(f'{name}:')
    print(f'  Trades: {len(entries)}')
    print(f'  Strike: mean ${strikes.mean():.0f}, min ${strikes.min():.0f}, max ${strikes.max():.0f}')
    print(f'  Cost/contract: mean ${costs.mean():.0f}, median ${costs.median():.0f}')
    print(f'  Qty/entry: mean {qtys.mean():.0f}, median {qtys.median():.0f}')
    print(f'  Total cost/entry: mean ${total_costs.mean():,.0f}')
    print(f'  Annual: {r["annual_ret"]:+.2f}%, Excess: {r["excess_annual"]:+.2f}%, DD: {r["max_dd"]:.1f}%')
    print()
    return r

# Current deep OTM: picks delta -0.02 (deepest)
r1 = show_picks('Deep OTM sort=desc (picks -0.02)', -0.10, -0.02, sort_asc=False)

# Deep OTM but pick the LEAST deep (delta -0.10)
r2 = show_picks('Deep OTM sort=asc (picks -0.10)', -0.10, -0.02, sort_asc=True)

# Spitznagel-realistic: delta -0.15 to -0.05, pick -0.15
r3 = show_picks('Spitznagel-like sort=asc (picks -0.15)', -0.15, -0.05, sort_asc=True)

# Slightly wider: delta -0.20 to -0.05, pick -0.20
r4 = show_picks('Wider OTM sort=asc (picks -0.20)', -0.20, -0.05, sort_asc=True)

# ATM for comparison
r5 = show_picks('ATM sort=asc (picks -0.55)', -0.55, -0.45, sort_asc=True)

# Now the big comparison: vary delta with ASCENDING sort (picks least deep OTM in range)
print('\n' + '='*120)
print('FULL COMPARISON: All ascending sort (picks closest-to-ATM in each range)')
print('='*120 + '\n')

configs = [
    # (name, delta_min, delta_max, sort_asc)
    ('d -0.05 to -0.01, pick -0.05',  -0.05, -0.01, True),
    ('d -0.10 to -0.02, pick -0.10',  -0.10, -0.02, True),
    ('d -0.10 to -0.02, pick -0.02',  -0.10, -0.02, False),  # current default
    ('d -0.15 to -0.05, pick -0.15',  -0.15, -0.05, True),
    ('d -0.15 to -0.05, pick -0.05',  -0.15, -0.05, False),
    ('d -0.20 to -0.10, pick -0.20',  -0.20, -0.10, True),
    ('d -0.20 to -0.10, pick -0.10',  -0.20, -0.10, False),
    ('d -0.30 to -0.15, pick -0.30',  -0.30, -0.15, True),
    ('d -0.30 to -0.15, pick -0.15',  -0.30, -0.15, False),
    ('d -0.40 to -0.20, pick -0.40',  -0.40, -0.20, True),
    ('d -0.50 to -0.30, pick -0.50',  -0.50, -0.30, True),
    ('d -0.55 to -0.45, pick -0.55',  -0.55, -0.45, True),  # true ATM
]

spy_annual = data['spy_annual_ret']

print(f'{"Config":<38} {"Annual%":>8} {"Excess%":>8} {"MaxDD%":>8} {"Trades":>7} {"Avg $/trade":>12}')
print('-' * 90)
print(f'{"SPY B&H":<38} {spy_annual:>8.2f} {"--":>8} {data["spy_dd"]:>8.1f}')

for name, dmin, dmax, sa in configs:
    r = run_backtest(name, 1.0, 0.0,
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, sort_asc=s),
        data, budget_pct=0.005)

    tl = r['trade_log']
    entries = tl[tl[('totals', 'cost')] > 0]
    avg_cost = (entries[('totals', 'cost')] * entries[('totals', 'qty')]).mean() if len(entries) > 0 else 0

    print(f'{name:<38} {r["annual_ret"]:>8.2f} {r["excess_annual"]:>+8.2f} {r["max_dd"]:>8.1f} {r["trades"]:>7} {avg_cost:>12,.0f}')

# Also test with different DTE settings
print(f'\n\n{"="*120}')
print('DTE VARIATIONS for deep OTM (d -0.15 to -0.05, picks -0.15)')
print('='*120 + '\n')

dte_configs = [
    ('DTE 30-60, exit 14',   30,  60,  14),
    ('DTE 60-90, exit 30',   60,  90,  30),
    ('DTE 90-120, exit 30',  90,  120, 30),
    ('DTE 90-180, exit 14',  90,  180, 14),
    ('DTE 120-180, exit 30', 120, 180, 30),
    ('DTE 180-365, exit 30', 180, 365, 30),
    ('DTE 60-90, exit 7',    60,  90,  7),
    ('DTE 60-90, exit 14',   60,  90,  14),
    ('DTE 60-90, exit 45',   60,  90,  45),
]

print(f'{"DTE Config":<28} {"Annual%":>8} {"Excess%":>8} {"MaxDD%":>8} {"Trades":>7}')
print('-' * 65)

for name, dte_min, dte_max, exit_dte in dte_configs:
    r = run_backtest(name, 1.0, 0.0,
        lambda dmin=dte_min, dmax=dte_max, edte=exit_dte: make_put(
            schema, -0.15, -0.05, dte_min=dmin, dte_max=dmax, exit_dte=edte, sort_asc=True),
        data, budget_pct=0.005)
    print(f'{name:<28} {r["annual_ret"]:>8.2f} {r["excess_annual"]:>+8.2f} {r["max_dd"]:>8.1f} {r["trades"]:>7}')
