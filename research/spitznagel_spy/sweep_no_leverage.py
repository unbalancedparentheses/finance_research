#!/usr/bin/env python3
"""No-leverage sweep: puts funded by reducing equity (stock + opt = 100%).

This is the fair comparison: money in puts = money NOT in stocks.
Deep OTM costs less per contract → more money stays in stocks.
"""

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

dte_configs = [
    ('30-60/14',  30,  60, 14),
    ('30-60/25',  30,  60, 25),
    ('60-90/30',  60,  90, 30),
    ('88-93/10',  88,  93, 10),
    ('88-93/60',  88,  93, 60),
]

budgets = [0.005, 0.01, 0.02, 0.033, 0.05]
budget_labels = ['0.5%', '1%', '2%', '3.3%', '5%']

# =====================================================================
# Phase 1: OTM x DTE grid, NO LEVERAGE, 0.5% allocation
# stock_pct = 0.995, opt_pct = 0.005
# =====================================================================
print('='*130)
print('NO LEVERAGE: stock_pct + opt_pct = 100% (puts funded from equity)')
print('='*130)

for bp, bl in zip(budgets, budget_labels):
    stock_pct = 1.0 - bp
    opt_pct = bp

    print(f'\n--- Budget: {bl} (stocks {stock_pct*100:.1f}%, puts {opt_pct*100:.1f}%) ---\n')

    otm_names = [o[0] for o in otm_levels]
    print(f'{"DTE":<14}', end='')
    for n in otm_names:
        print(f' {n:>8}', end='')
    print(f' {"BEST":>10}')
    print('-' * (14 + 9 * len(otm_names) + 11))

    for dte_name, dte_min, dte_max, exit_dte in dte_configs:
        print(f'{dte_name:<14}', end='', flush=True)
        row_best = None
        for otm_name, dmin, dmax, sa in otm_levels:
            r = run_backtest(
                f'{otm_name} {dte_name} nolev {bl}',
                stock_pct, opt_pct,
                lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                    make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
                data)  # NO budget_pct — pure allocation
            exc = r['excess_annual']
            print(f' {exc:>+8.2f}', end='', flush=True)
            if row_best is None or exc > row_best[1]:
                row_best = (otm_name, exc)
        print(f' {row_best[0]:>10}')

# =====================================================================
# Phase 2: Year-by-year, no leverage, 3.3% budget, 30-60/14
# =====================================================================
print('\n\n' + '='*130)
print('YEAR-BY-YEAR: No leverage, 3.3% budget, DTE 30-60/14')
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
        f'{otm_name} nolev 3.3%', 0.967, 0.033,
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 30, 60, 14, sort_asc=s),
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

# =====================================================================
# Phase 3: Summary — best OTM at each budget level (no leverage)
# =====================================================================
print('\n\n' + '='*130)
print('SUMMARY: Best OTM level at each budget (no leverage, DTE 30-60/14)')
print('='*130 + '\n')

print(f'{"Budget":<8} {"Best OTM":<10} {"Excess%":>10} {"MaxDD%":>10} {"Worst OTM":<10} {"Excess%":>10}')
print('-' * 65)

for bp, bl in zip(budgets, budget_labels):
    stock_pct = 1.0 - bp
    best_r = None
    worst_r = None
    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_backtest(
            f'{otm_name} nolev {bl}', stock_pct, bp,
            lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 30, 60, 14, sort_asc=s),
            data)
        if best_r is None or r['excess_annual'] > best_r['excess_annual']:
            best_r = r
            best_name = otm_name
        if worst_r is None or r['excess_annual'] < worst_r['excess_annual']:
            worst_r = r
            worst_name = otm_name
    print(f'{bl:<8} {best_name:<10} {best_r["excess_annual"]:>+10.2f} {best_r["max_dd"]:>10.1f} '
          f'{worst_name:<10} {worst_r["excess_annual"]:>+10.2f}')

print('\nDone.')
