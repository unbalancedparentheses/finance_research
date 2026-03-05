#!/usr/bin/env python3
"""Precise DTE sweep: narrow entry windows, hold near-expiry."""

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

otm_levels = [
    ('ATM',     -0.55, -0.45, True),
    ('5%OTM',   -0.40, -0.30, True),
    ('10%OTM',  -0.25, -0.15, True),
    ('15%OTM',  -0.15, -0.08, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('35%OTM',  -0.025, -0.005, True),
]

# =====================================================================
# Sweep 1: Narrow entry windows (5-day wide), exit 1 DTE before min
# "Buy at X DTE, hold to near expiry"
# =====================================================================
print('='*120)
print('NARROW ENTRY DTE: 5-day windows, exit at entry_min - 1 (hold to near-expiry)')
print('='*120 + '\n')

narrow_entries = [
    # (name, dte_min, dte_max, exit_dte)
    ('28-33/1',  28, 33, 1),
    ('28-33/3',  28, 33, 3),
    ('28-33/5',  28, 33, 5),
    ('28-33/7',  28, 33, 7),
    ('28-33/14', 28, 33, 14),
    ('35-40/1',  35, 40, 1),
    ('35-40/3',  35, 40, 3),
    ('35-40/7',  35, 40, 7),
    ('35-40/14', 35, 40, 14),
    ('42-47/1',  42, 47, 1),
    ('42-47/7',  42, 47, 7),
    ('42-47/14', 42, 47, 14),
    ('56-61/1',  56, 61, 1),
    ('56-61/7',  56, 61, 7),
    ('56-61/14', 56, 61, 14),
    ('56-61/30', 56, 61, 30),
    ('88-93/1',  88, 93, 1),
    ('88-93/7',  88, 93, 7),
    ('88-93/30', 88, 93, 30),
    ('88-93/60', 88, 93, 60),
]

otm_names = [o[0] for o in otm_levels]
print(f'{"DTE config":<14}', end='')
for n in otm_names:
    print(f' {n:>8}', end='')
print(f' {"BEST":>10} {"DD(best)":>10}')
print('-' * (14 + 9 * len(otm_names) + 22))

for entry_name, dte_min, dte_max, exit_dte in narrow_entries:
    print(f'{entry_name:<14}', end='', flush=True)
    row_best = None
    row_results = {}
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} {entry_name}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=0.005)
        row_results[otm_name] = r
        exc = r['excess_annual']
        print(f' {exc:>+8.2f}', end='', flush=True)
        if row_best is None or exc > row_best[1]:
            row_best = (otm_name, exc, r['max_dd'])
    print(f' {row_best[0]:>10} {row_best[2]:>10.1f}')

# =====================================================================
# Sweep 2: For the optimal "hold to expiry" pattern, test exact DTEs
# Buy at DTE X, sell at DTE Y (Y close to 0)
# =====================================================================
print('\n\n' + '='*120)
print('EXACT TARGET DTE: Buy at ~X, sell at 1 DTE (hold to near-expiry)')
print('='*120 + '\n')

# Target specific DTEs with tight 3-day windows
target_dtes = [21, 28, 30, 35, 42, 45, 60, 90]

print(f'{"Target DTE":<12}', end='')
for n in otm_names:
    print(f' {n:>8}', end='')
print(f' {"BEST":>10} {"DD(best)":>10} {"Trades":>8}')
print('-' * (12 + 9 * len(otm_names) + 30))

for target in target_dtes:
    dte_min = target - 1
    dte_max = target + 2
    exit_dte = 1
    print(f'DTE ~{target:<7}', end='', flush=True)
    row_best = None
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} ~{target}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=0.005)
        exc = r['excess_annual']
        print(f' {exc:>+8.2f}', end='', flush=True)
        if row_best is None or exc > row_best[1]:
            row_best = (otm_name, exc, r['max_dd'], r['trades'])
    print(f' {row_best[0]:>10} {row_best[2]:>10.1f} {row_best[3]:>8}')

# =====================================================================
# Sweep 3: Same but exit at 3 DTE instead of 1
# =====================================================================
print(f'\n{"Target DTE":<12}', end='')
for n in otm_names:
    print(f' {n:>8}', end='')
print(f'  (exit at 3 DTE)')
print('-' * (12 + 9 * len(otm_names) + 20))

for target in target_dtes:
    dte_min = target - 1
    dte_max = target + 2
    exit_dte = 3
    print(f'DTE ~{target:<7}', end='', flush=True)
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} ~{target}/3', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=0.005)
        print(f' {r["excess_annual"]:>+8.2f}', end='', flush=True)
    print()

# =====================================================================
# Sweep 4: Budget sweep for best precise configs
# =====================================================================
print('\n\n' + '='*120)
print('BUDGET SWEEP: Best precise DTE configs')
print('='*120 + '\n')

best_configs = [
    ('5%OTM',  -0.40, -0.30, True, 29, 33, 1, '~30/1'),
    ('5%OTM',  -0.40, -0.30, True, 34, 38, 1, '~35/1'),
    ('5%OTM',  -0.40, -0.30, True, 41, 46, 1, '~42/1'),
    ('10%OTM', -0.25, -0.15, True, 29, 33, 1, '~30/1'),
    ('25%OTM', -0.06, -0.02, True, 29, 33, 1, '~30/1'),
    ('ATM',    -0.55, -0.45, True, 29, 33, 1, '~30/1'),
    ('35%OTM', -0.025, -0.005, True, 29, 33, 1, '~30/1'),
]

budgets = [0.005, 0.01, 0.02, 0.033]
blabels = ['0.5%', '1.0%', '2.0%', '3.3%']

print(f'{"OTM":<10} {"DTE":<10}', end='')
for bl in blabels:
    print(f' {bl+" exc":>10} {bl+" DD":>8}', end='')
print()
print('-' * (20 + len(budgets) * 19))

for otm_name, dmin, dmax, sa, dte_min, dte_max, exit_dte, dte_label in best_configs:
    print(f'{otm_name:<10} {dte_label:<10}', end='', flush=True)
    for bp in budgets:
        r = run_backtest(
            f'{otm_name} {dte_label} {bp}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=bp)
        print(f' {r["excess_annual"]:>+10.2f} {r["max_dd"]:>8.1f}', end='', flush=True)
    print()

print('\nDone.')
