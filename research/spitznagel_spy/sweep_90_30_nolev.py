#!/usr/bin/env python3
"""Test: 40% OTM, ~90 DTE entry, exit ~30 DTE, no leverage.
Sweep exit DTE and OTM level around this region."""

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
spy_prices = data['spy_prices']

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
    ('20%OTM',  -0.10, -0.04, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('30%OTM',  -0.04, -0.01, True),
    ('35%OTM',  -0.025, -0.005, True),
    ('40%OTM',  -0.015, -0.002, True),
]

budgets = [0.005, 0.01, 0.02, 0.033]
budget_labels = ['0.5%', '1%', '2%', '3.3%']

# =====================================================================
# Test 1: 90 DTE / exit 30, all OTM levels, no leverage, budget sweep
# =====================================================================
print('='*130)
print('NO LEVERAGE: ~90 DTE entry, exit 30 DTE (hold ~60 days)')
print('='*130)

for bp, bl in zip(budgets, budget_labels):
    stock_pct = 1.0 - bp
    print(f'\n--- Budget {bl} (stocks {stock_pct*100:.1f}%) ---')
    print(f'{"OTM":<10} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
    print('-' * 52)
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} 90/30 nolev {bl}', stock_pct, bp,
            lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 88, 93, 30, sort_asc=s),
            data)
        print(f'{otm_name:<10} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Test 2: Vary exit DTE for 40% OTM at ~90 DTE, no leverage, 1% budget
# =====================================================================
print('\n\n' + '='*130)
print('40% OTM at ~90 DTE, NO LEVERAGE 1%: sweep exit DTE')
print('='*130 + '\n')

print(f'{"Exit DTE":<12} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
print('-' * 55)

for exit_dte in [1, 5, 10, 14, 21, 30, 40, 45, 50, 55, 60, 70, 80, 85]:
    r = run_backtest(
        f'40%OTM 90/exit{exit_dte} nolev', 0.99, 0.01,
        lambda edte=exit_dte: make_put(schema, -0.015, -0.002, 88, 93, edte, sort_asc=True),
        data)
    print(f'DTE {exit_dte:<8} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Test 3: Also try wider entry window (60-120) with exit 30
# =====================================================================
print('\n\n' + '='*130)
print('NO LEVERAGE 1%: Wider entry windows, exit 30')
print('='*130 + '\n')

entry_windows = [
    ('60-90',   60,  90),
    ('60-120',  60, 120),
    ('88-93',   88,  93),
    ('90-120',  90, 120),
    ('90-180',  90, 180),
    ('120-180', 120, 180),
]

otm_names = [o[0] for o in otm_levels]
print(f'{"Entry":<12}', end='')
for n in otm_names:
    print(f' {n:>8}', end='')
print(f' {"BEST":>10}')
print('-' * (12 + 9 * len(otm_names) + 11))

for ew_name, dte_min, dte_max in entry_windows:
    print(f'{ew_name:<12}', end='', flush=True)
    row_best = None
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} {ew_name}/30 nolev', 0.99, 0.01,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max:
                make_put(schema, d1, d2, dmin_, dmax_, 30, sort_asc=s),
            data)
        exc = r['excess_annual']
        print(f' {exc:>+8.2f}', end='', flush=True)
        if row_best is None or exc > row_best[1]:
            row_best = (otm_name, exc)
    print(f' {row_best[0]:>10}')

# =====================================================================
# Test 4: Year-by-year for 40% OTM 90/30 no leverage at 1%
# =====================================================================
print('\n\n' + '='*130)
print('YEAR-BY-YEAR: No leverage 1%, ~90 DTE / exit 30')
print('='*130 + '\n')

yoy_otms = [
    ('ATM',     -0.55, -0.45, True),
    ('10%OTM',  -0.25, -0.15, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('40%OTM',  -0.015, -0.002, True),
]

yoy_results = {}
for otm_name, dmin, dmax, sa in yoy_otms:
    r = run_backtest(
        f'{otm_name} 90/30 nolev', 0.99, 0.01,
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 88, 93, 30, sort_asc=s),
        data)
    yoy_results[otm_name] = r

yoy_names = [o[0] for o in yoy_otms]
print(f'{"Year":<6} {"SPY":>8}', end='')
for n in yoy_names:
    print(f' {n:>10}', end='')
print(f' {"Winner":>10}')
print('-' * (6 + 9 + 11 * len(yoy_names) + 11))

for yr in range(2008, 2026):
    s = pd.Timestamp(f'{yr}-01-01')
    e = pd.Timestamp(f'{yr+1}-01-01')
    pspy = spy_prices[(spy_prices.index >= s) & (spy_prices.index < e)]
    if len(pspy) < 10:
        continue
    retspy = (pspy.iloc[-1] / pspy.iloc[0] - 1) * 100
    print(f'{yr:<6} {retspy:>8.1f}', end='')
    best_exc = -999
    best_name = ''
    for otm_name in yoy_names:
        cap = yoy_results[otm_name]['balance']['total capital']
        p = cap[(cap.index >= s) & (cap.index < e)]
        if len(p) > 10:
            ret = (p.iloc[-1] / p.iloc[0] - 1) * 100
            exc = ret - retspy
            print(f' {exc:>+10.2f}', end='')
            if exc > best_exc:
                best_exc = exc
                best_name = otm_name
        else:
            print(f' {"n/a":>10}', end='')
    print(f' {best_name:>10}')

print('\nDone.')
