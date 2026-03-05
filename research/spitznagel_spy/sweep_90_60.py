#!/usr/bin/env python3
"""Test user hypothesis: 40% OTM, buy at 90 DTE, sell at 60 DTE.
Also sweep nearby configs to find the optimum in that region."""

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
    ('20%OTM',  -0.10, -0.04, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('30%OTM',  -0.04, -0.01, True),
    ('35%OTM',  -0.025, -0.005, True),
    ('40%OTM',  -0.015, -0.002, True),
]

# =====================================================================
# Test 1: Exact user config — 90 DTE entry, 60 DTE exit, all OTM levels
# =====================================================================
print('='*120)
print('USER HYPOTHESIS: Buy at ~90 DTE, sell at ~60 DTE (hold 30 days)')
print('='*120 + '\n')

print(f'{"OTM level":<12} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
print('-' * 55)
print(f'{"SPY B&H":<12} {"--":>10} {spy_annual:>10.2f} {data["spy_dd"]:>10.1f}')

for otm_name, dmin, dmax, sa in otm_levels:
    r = run_backtest(
        f'{otm_name} 90/60', 1.0, 0.0,
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 88, 93, 60, sort_asc=s),
        data, budget_pct=0.005)
    print(f'{otm_name:<12} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Test 2: Vary exit DTE for 40% OTM at 90 DTE entry
# =====================================================================
print('\n\n' + '='*120)
print('40% OTM at ~90 DTE: vary exit DTE')
print('='*120 + '\n')

print(f'{"Exit DTE":<12} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
print('-' * 55)

for exit_dte in [1, 3, 5, 7, 10, 14, 21, 30, 45, 55, 60, 65, 70, 75, 80, 85]:
    r = run_backtest(
        f'40%OTM 90/exit{exit_dte}', 1.0, 0.0,
        lambda edte=exit_dte: make_put(schema, -0.015, -0.002, 88, 93, edte, sort_asc=True),
        data, budget_pct=0.005)
    print(f'DTE {exit_dte:<8} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Test 3: Vary entry DTE for 40% OTM, exit at 60 DTE
# =====================================================================
print('\n\n' + '='*120)
print('40% OTM, exit at 60 DTE: vary entry DTE')
print('='*120 + '\n')

print(f'{"Entry DTE":<14} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
print('-' * 57)

for dte_min, dte_max, label in [
    (68, 73, '~70'),
    (78, 83, '~80'),
    (88, 93, '~90'),
    (98, 103, '~100'),
    (108, 113, '~110'),
    (118, 123, '~120'),
    (148, 153, '~150'),
    (178, 183, '~180'),
    (88, 120, '90-120'),
    (88, 180, '90-180'),
    (60, 120, '60-120'),
]:
    r = run_backtest(
        f'40%OTM {label}/60', 1.0, 0.0,
        lambda dmin=dte_min, dmax=dte_max: make_put(schema, -0.015, -0.002, dmin, dmax, 60, sort_asc=True),
        data, budget_pct=0.005)
    print(f'DTE {label:<10} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} {r["max_dd"]:>10.1f} {r["trades"]:>8}')

# =====================================================================
# Test 4: Budget sweep for 40% OTM 90/60 vs previous winners
# =====================================================================
print('\n\n' + '='*120)
print('BUDGET SWEEP: 40% OTM 90/60 vs previous winners')
print('='*120 + '\n')

configs = [
    ('40%OTM 90/60',   -0.015, -0.002, True, 88, 93, 60),
    ('10%OTM 30-60/14', -0.25, -0.15, True, 30, 60, 14),
    ('5%OTM 30-60/25',  -0.40, -0.30, True, 30, 60, 25),
    ('25%OTM 30-60/14', -0.06, -0.02, True, 30, 60, 14),
    ('ATM 30-60/14',    -0.55, -0.45, True, 30, 60, 14),
]

budgets = [0.005, 0.01, 0.02, 0.033, 0.05]
blabels = ['0.5%', '1.0%', '2.0%', '3.3%', '5.0%']

print(f'{"Config":<22}', end='')
for bl in blabels:
    print(f' {bl+" exc":>10} {bl+" DD":>8}', end='')
print()
print('-' * (22 + len(budgets) * 19))

for name, dmin, dmax, sa, dte_min, dte_max, exit_dte in configs:
    print(f'{name:<22}', end='', flush=True)
    for bp in budgets:
        r = run_backtest(
            f'{name} {bp}', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            data, budget_pct=bp)
        print(f' {r["excess_annual"]:>+10.2f} {r["max_dd"]:>8.1f}', end='', flush=True)
    print()

# =====================================================================
# Test 5: Year-by-year for 40% OTM 90/60 vs 10%OTM 30-60/14
# =====================================================================
print('\n\n' + '='*120)
print('YEAR-BY-YEAR: 40% OTM 90/60 vs 10%OTM 30-60/14 (0.5% leveraged)')
print('='*120 + '\n')

r40 = run_backtest('40%OTM 90/60', 1.0, 0.0,
    lambda: make_put(schema, -0.015, -0.002, 88, 93, 60, sort_asc=True),
    data, budget_pct=0.005)
r10 = run_backtest('10%OTM 30-60/14', 1.0, 0.0,
    lambda: make_put(schema, -0.25, -0.15, 30, 60, 14, sort_asc=True),
    data, budget_pct=0.005)

spy_prices = data['spy_prices']
cap40 = r40['balance']['total capital']
cap10 = r10['balance']['total capital']

print(f'{"Year":<6} {"SPY":>8} {"40%OTM 90/60":>14} {"10%OTM 30-60":>14} {"Winner":>10}')
print('-' * 58)

for yr in range(2008, 2026):
    s = pd.Timestamp(f'{yr}-01-01')
    e = pd.Timestamp(f'{yr+1}-01-01')
    pspy = spy_prices[(spy_prices.index >= s) & (spy_prices.index < e)]
    p40 = cap40[(cap40.index >= s) & (cap40.index < e)]
    p10 = cap10[(cap10.index >= s) & (cap10.index < e)]
    if len(pspy) > 10 and len(p40) > 10 and len(p10) > 10:
        retspy = (pspy.iloc[-1] / pspy.iloc[0] - 1) * 100
        ret40 = (p40.iloc[-1] / p40.iloc[0] - 1) * 100
        ret10 = (p10.iloc[-1] / p10.iloc[0] - 1) * 100
        exc40 = ret40 - retspy
        exc10 = ret10 - retspy
        w = '40%OTM' if exc40 > exc10 else '10%OTM'
        print(f'{yr:<6} {retspy:>8.1f} {exc40:>+14.2f} {exc10:>+14.2f} {w:>10}')

print('\nDone.')
