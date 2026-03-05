#!/usr/bin/env python3
"""Full grid sweep: OTM level x DTE window x exit DTE x budget.

Finds the best put hedge configuration across all dimensions.
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

# =====================================================================
# Dimension definitions
# =====================================================================

# OTM levels (name, delta_min, delta_max, sort_ascending)
# sort_asc=True picks most negative delta (closest to ATM within range)
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

# DTE windows (name, dte_min, dte_max, exit_dte)
dte_configs = [
    ('30-60/14',   30,  60,  14),
    ('30-60/7',    30,  60,   7),
    ('60-90/30',   60,  90,  30),
    ('60-90/14',   60,  90,  14),
    ('60-90/7',    60,  90,   7),
    ('90-120/30',  90, 120,  30),
    ('90-180/30',  90, 180,  30),
    ('120-180/60', 120, 180, 60),
    ('180-365/90', 180, 365, 90),
]

budgets = [0.005, 0.01, 0.02, 0.033]
budget_labels = ['0.5%', '1.0%', '2.0%', '3.3%']

# =====================================================================
# Phase 1: OTM x DTE grid at 0.5% budget (find best combos)
# =====================================================================
print('='*130)
print('PHASE 1: OTM level x DTE window (0.5% leveraged)')
print('='*130)

# Header
otm_names = [o[0] for o in otm_levels]
print(f'\n{"DTE config":<14}', end='')
for n in otm_names:
    print(f' {n:>10}', end='')
print(f' {"BEST OTM":>12}')
print('-' * (14 + 11 * len(otm_names) + 13))

all_results = {}  # (otm_name, dte_name) -> result
best_overall = None

for dte_name, dte_min, dte_max, exit_dte in dte_configs:
    print(f'{dte_name:<14}', end='', flush=True)
    row_best = None

    for otm_name, dmin, dmax, sa in otm_levels:
        key = (otm_name, dte_name)
        r = run_backtest(
            f'{otm_name} {dte_name}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=0.005)
        all_results[key] = r
        exc = r['excess_annual']
        print(f' {exc:>+10.2f}', end='', flush=True)

        if row_best is None or exc > row_best[1]:
            row_best = (otm_name, exc)
        if best_overall is None or exc > best_overall[2]:
            best_overall = (otm_name, dte_name, exc)

    print(f' {row_best[0]:>12}')

print(f'\nBest combo at 0.5%: {best_overall[0]} + {best_overall[1]} = {best_overall[2]:+.2f}% excess')

# =====================================================================
# Phase 1b: Same grid but show MaxDD
# =====================================================================
print(f'\n{"DTE config":<14}', end='')
for n in otm_names:
    print(f' {n:>10}', end='')
print(f'  (MaxDD%)')
print('-' * (14 + 11 * len(otm_names) + 12))

for dte_name, _, _, _ in dte_configs:
    print(f'{dte_name:<14}', end='')
    for otm_name, _, _, _ in otm_levels:
        r = all_results[(otm_name, dte_name)]
        print(f' {r["max_dd"]:>10.1f}', end='')
    print()

# =====================================================================
# Phase 2: Top 10 combos with budget sweep
# =====================================================================
print('\n\n' + '='*130)
print('PHASE 2: Top 10 combos — budget sweep (leveraged)')
print('='*130 + '\n')

# Sort all combos by excess
ranked = sorted(all_results.items(), key=lambda x: x[1]['excess_annual'], reverse=True)
top10 = ranked[:10]

print(f'{"Rank":<5} {"OTM":<8} {"DTE":<14}', end='')
for bl in budget_labels:
    print(f' {bl+" exc":>10} {bl+" DD":>8}', end='')
print()
print('-' * (5 + 8 + 14 + len(budgets) * 19))

for rank, ((otm_name, dte_name), base_r) in enumerate(top10, 1):
    # Find the OTM and DTE params
    otm_params = next(o for o in otm_levels if o[0] == otm_name)
    dte_params = next(d for d in dte_configs if d[0] == dte_name)
    _, dmin, dmax, sa = otm_params
    _, dte_min, dte_max, exit_dte = dte_params

    print(f'{rank:<5} {otm_name:<8} {dte_name:<14}', end='', flush=True)

    for bp in budgets:
        if bp == 0.005:
            r = base_r  # already computed
        else:
            r = run_backtest(
                f'{otm_name} {dte_name} {bp}', 1.0, 0.0,
                lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                    make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
                data, budget_pct=bp)
        print(f' {r["excess_annual"]:>+10.2f} {r["max_dd"]:>8.1f}', end='', flush=True)
    print()

# =====================================================================
# Phase 3: Top 5 combos year-by-year
# =====================================================================
print('\n\n' + '='*130)
print('PHASE 3: Top 5 combos — year-by-year excess return (0.5% leveraged)')
print('='*130 + '\n')

top5 = ranked[:5]
top5_names = [f'{otm}+{dte}' for (otm, dte), _ in top5]

print(f'{"Year":<6} {"SPY":>8}', end='')
for n in top5_names:
    print(f' {n:>14}', end='')
print()
print('-' * (6 + 9 + 15 * len(top5_names)))

spy_prices = data['spy_prices']
for yr in range(2008, 2026):
    s = pd.Timestamp(f'{yr}-01-01')
    e = pd.Timestamp(f'{yr+1}-01-01')
    pspy = spy_prices[(spy_prices.index >= s) & (spy_prices.index < e)]
    if len(pspy) < 10:
        continue
    retspy = (pspy.iloc[-1] / pspy.iloc[0] - 1) * 100
    print(f'{yr:<6} {retspy:>8.1f}', end='')

    for (otm_name, dte_name), r in top5:
        cap = r['balance']['total capital']
        p = cap[(cap.index >= s) & (cap.index < e)]
        if len(p) > 10:
            ret = (p.iloc[-1] / p.iloc[0] - 1) * 100
            exc = ret - retspy
            print(f' {exc:>+14.2f}', end='')
        else:
            print(f' {"n/a":>14}', end='')
    print()

# =====================================================================
# Phase 4: No-leverage version of top 5
# =====================================================================
print('\n\n' + '='*130)
print('PHASE 4: Top 5 combos — NO LEVERAGE (reduce equity to fund puts, 0.5%)')
print('='*130 + '\n')

print(f'{"Rank":<5} {"OTM":<8} {"DTE":<14} {"Lev Exc%":>10} {"NoLev Exc%":>12} {"NoLev DD%":>10}')
print('-' * 65)

for rank, ((otm_name, dte_name), lev_r) in enumerate(top5, 1):
    otm_params = next(o for o in otm_levels if o[0] == otm_name)
    dte_params = next(d for d in dte_configs if d[0] == dte_name)
    _, dmin, dmax, sa = otm_params
    _, dte_min, dte_max, exit_dte = dte_params

    nolev_r = run_backtest(
        f'{otm_name} {dte_name} nolev', 0.995, 0.005,
        lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
            make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
        data)
    print(f'{rank:<5} {otm_name:<8} {dte_name:<14} {lev_r["excess_annual"]:>+10.2f} '
          f'{nolev_r["excess_annual"]:>+12.2f} {nolev_r["max_dd"]:>10.1f}')

print('\nDone.')
