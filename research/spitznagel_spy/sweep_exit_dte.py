#!/usr/bin/env python3
"""Sweep exit DTE for top OTM levels with DTE 30-60 entry."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, run_backtest
from options_portfolio_backtester import OptionType as Type, Direction
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

data = load_data()
schema = data['schema']
spy_annual = data['spy_annual_ret']

def make_put(schema, delta_min, delta_max, dte_min, dte_max, exit_dte, sort_asc=True):
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= dte_min) & (schema.dte <= dte_max)
        & (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', sort_asc)
    leg.exit_filter = (schema.dte <= exit_dte)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s

# Top OTM levels from Phase 1
otm_levels = [
    ('10%OTM',  -0.25, -0.15, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('15%OTM',  -0.15, -0.08, True),
    ('5%OTM',   -0.40, -0.30, True),
    ('ATM',     -0.55, -0.45, True),
    ('35%OTM',  -0.025, -0.005, True),
]

# Exit DTEs to test (for 30-60 entry)
exit_dtes = [1, 3, 5, 7, 10, 14, 18, 21, 25]

# Also test entry windows
entry_windows = [
    ('30-60', 30, 60),
    ('30-45', 30, 45),
    ('45-60', 45, 60),
]

# =====================================================================
# Sweep 1: Exit DTE x OTM for 30-60 entry, 0.5% budget
# =====================================================================
print('='*120)
print('EXIT DTE SWEEP: OTM x exit DTE (entry 30-60, 0.5% leveraged)')
print('='*120)

otm_names = [o[0] for o in otm_levels]

print(f'\n--- Excess Annual % ---')
print(f'{"Exit DTE":<10}', end='')
for n in otm_names:
    print(f' {n:>10}', end='')
print(f' {"BEST":>10}')
print('-' * (10 + 11 * len(otm_names) + 11))

results = {}
for exit_dte in exit_dtes:
    print(f'DTE {exit_dte:<6}', end='', flush=True)
    row_best = None
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} exit{exit_dte}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, edte=exit_dte:
                make_put(schema, d1, d2, 30, 60, edte, sort_asc=s),
            data, budget_pct=0.005)
        results[(otm_name, exit_dte)] = r
        exc = r['excess_annual']
        print(f' {exc:>+10.2f}', end='', flush=True)
        if row_best is None or exc > row_best[1]:
            row_best = (otm_name, exc)
    print(f' {row_best[0]:>10}')

print(f'\n--- Max Drawdown % ---')
print(f'{"Exit DTE":<10}', end='')
for n in otm_names:
    print(f' {n:>10}', end='')
print()
print('-' * (10 + 11 * len(otm_names)))

for exit_dte in exit_dtes:
    print(f'DTE {exit_dte:<6}', end='')
    for otm_name, _, _, _ in otm_levels:
        r = results[(otm_name, exit_dte)]
        print(f' {r["max_dd"]:>10.1f}', end='')
    print()

print(f'\n--- Trades ---')
print(f'{"Exit DTE":<10}', end='')
for n in otm_names:
    print(f' {n:>10}', end='')
print()
print('-' * (10 + 11 * len(otm_names)))

for exit_dte in exit_dtes:
    print(f'DTE {exit_dte:<6}', end='')
    for otm_name, _, _, _ in otm_levels:
        r = results[(otm_name, exit_dte)]
        print(f' {r["trades"]:>10}', end='')
    print()

# =====================================================================
# Sweep 2: Narrower entry windows for top combo
# =====================================================================
print('\n\n' + '='*120)
print('ENTRY WINDOW SWEEP: 10%OTM with different entry/exit combos')
print('='*120 + '\n')

print(f'{"Entry":<10} {"Exit DTE":<10} {"Excess%":>10} {"MaxDD%":>10} {"Trades":>8}')
print('-' * 52)

for ew_name, dte_min, dte_max in entry_windows:
    for exit_dte in [1, 3, 5, 7, 10, 14]:
        if exit_dte >= dte_min:
            continue
        r = run_backtest(
            f'10%OTM {ew_name}/exit{exit_dte}', 1.0, 0.0,
            lambda dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, -0.25, -0.15, dmin_, dmax_, edte, sort_asc=True),
            data, budget_pct=0.005)
        print(f'{ew_name:<10} {exit_dte:<10} {r["excess_annual"]:>+10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Sweep 3: Budget sweep for best exit DTE combos
# =====================================================================
print('\n\n' + '='*120)
print('BUDGET SWEEP: Top exit DTE combos at higher budgets')
print('='*120 + '\n')

# Find top 5 from sweep 1
ranked = sorted(results.items(), key=lambda x: x[1]['excess_annual'], reverse=True)[:5]

budgets = [0.005, 0.01, 0.02, 0.033]
blabels = ['0.5%', '1.0%', '2.0%', '3.3%']

print(f'{"OTM":<10} {"ExitDTE":<8}', end='')
for bl in blabels:
    print(f' {bl+" exc":>10} {bl+" DD":>8}', end='')
print()
print('-' * (18 + len(budgets) * 19))

for (otm_name, exit_dte), base_r in ranked:
    otm_params = next(o for o in otm_levels if o[0] == otm_name)
    _, dmin, dmax, sa = otm_params
    print(f'{otm_name:<10} {exit_dte:<8}', end='', flush=True)

    for bp in budgets:
        if bp == 0.005:
            r = base_r
        else:
            r = run_backtest(
                f'{otm_name} exit{exit_dte} {bp}', 1.0, 0.0,
                lambda d1=dmin, d2=dmax, s=sa, edte=exit_dte:
                    make_put(schema, d1, d2, 30, 60, edte, sort_asc=s),
                data, budget_pct=bp)
        print(f' {r["excess_annual"]:>+10.2f} {r["max_dd"]:>8.1f}', end='', flush=True)
    print()

print('\nDone.')
