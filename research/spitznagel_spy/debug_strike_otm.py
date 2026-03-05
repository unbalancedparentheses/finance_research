#!/usr/bin/env python3
"""Map OTM% to delta from actual data, then sweep using delta filters."""

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
spy_prices = data['spy_prices']
spy_annual = data['spy_annual_ret']

# Step 1: Map OTM% to delta from actual data
raw = data['options_data']._data
puts = raw[(raw['underlying'] == 'SPY') & (raw['type'] == 'put') &
           (raw['dte'] >= 60) & (raw['dte'] <= 90)].copy()

puts['otm_pct'] = (puts['underlying_last'] - puts['strike']) / puts['underlying_last'] * 100

print('=== OTM% to Delta Mapping (SPY puts, DTE 60-90) ===\n')
print(f'{"OTM% range":<20} {"Mean delta":>12} {"Median delta":>14} {"Count":>8}')
print('-' * 58)

otm_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 50)]
delta_map = {}

for lo, hi in otm_bins:
    mask = (puts['otm_pct'] >= lo) & (puts['otm_pct'] < hi)
    subset = puts[mask]
    if len(subset) > 0:
        mean_d = subset['delta'].mean()
        med_d = subset['delta'].median()
        delta_map[(lo, hi)] = (mean_d, med_d)
        print(f'{lo}-{hi}% OTM{"":<10} {mean_d:>12.4f} {med_d:>14.4f} {len(subset):>8,}')
    else:
        print(f'{lo}-{hi}% OTM{"":<10} {"no data":>12}')

# Step 2: Build delta-based configs that approximate each OTM%
print('\n\n=== SWEEP: OTM by strike percentage (delta-mapped, 0.5% leveraged) ===\n')

def make_put(schema, delta_min, delta_max, sort_asc=True):
    leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == 'SPY')
        & (schema.dte >= 60) & (schema.dte <= 90)
        & (schema.delta >= delta_min) & (schema.delta <= delta_max)
    )
    leg.entry_sort = ('delta', sort_asc)
    leg.exit_filter = (schema.dte <= 30)
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s

# Use delta ranges that correspond to OTM percentages
# Sort ascending = picks most negative delta = closest to ATM within range
configs = [
    ('ATM (~0-2% OTM)',     -0.55, -0.45, True),    # picks -0.55
    ('~5% OTM',             -0.40, -0.30, True),    # picks -0.40
    ('~10% OTM',            -0.25, -0.15, True),    # picks -0.25
    ('~15% OTM',            -0.15, -0.08, True),    # picks -0.15
    ('~20% OTM',            -0.10, -0.04, True),    # picks -0.10
    ('~25% OTM',            -0.06, -0.02, True),    # picks -0.06
    ('~30% OTM',            -0.04, -0.01, True),    # picks -0.04
    ('~35% OTM',            -0.025, -0.005, True),  # picks -0.025
    ('~40% OTM',            -0.015, -0.002, True),  # picks -0.015
]

print(f'{"Config":<20} {"Delta range":<20} {"Annual%":>8} {"Excess%":>8} {"MaxDD%":>8} {"Trades":>7}')
print('-' * 75)
print(f'{"SPY B&H":<20} {"":<20} {spy_annual:>8.2f} {"--":>8} {data["spy_dd"]:>8.1f}')

results = {}
for name, dmin, dmax, sa in configs:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, 1.0, 0.0,
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, sort_asc=s),
        data, budget_pct=0.005)
    results[name] = r
    print(f'{name:<20} {f"({dmin}, {dmax})":<20} {r["annual_ret"]:>8.2f} {r["excess_annual"]:>+8.2f} {r["max_dd"]:>8.1f} {r["trades"]:>7}')

# Year-by-year: ~35% OTM vs ATM
print(f'\n\n=== Year-by-Year: ~35% OTM vs ATM (0.5% leveraged) ===\n')
r35 = results['~35% OTM']
ratm = results['ATM (~0-2% OTM)']

cap_35 = r35['balance']['total capital']
cap_atm = ratm['balance']['total capital']

print(f'{"Year":<6} {"~35%OTM":>10} {"ATM":>10} {"SPY":>10} {"35%exc":>10} {"ATMexc":>10} {"Winner":>8}')
print('-' * 70)

for yr in range(2008, 2026):
    s = pd.Timestamp(f'{yr}-01-01')
    e = pd.Timestamp(f'{yr+1}-01-01')
    p35 = cap_35[(cap_35.index >= s) & (cap_35.index < e)]
    patm = cap_atm[(cap_atm.index >= s) & (cap_atm.index < e)]
    pspy = spy_prices[(spy_prices.index >= s) & (spy_prices.index < e)]

    if len(p35) > 10 and len(patm) > 10 and len(pspy) > 10:
        ret35 = (p35.iloc[-1] / p35.iloc[0] - 1) * 100
        retatm = (patm.iloc[-1] / patm.iloc[0] - 1) * 100
        retspy = (pspy.iloc[-1] / pspy.iloc[0] - 1) * 100
        w = '35%OTM' if (ret35 - retspy) > (retatm - retspy) else 'ATM'
        print(f'{yr:<6} {ret35:>10.2f} {retatm:>10.2f} {retspy:>10.2f} {ret35-retspy:>+10.2f} {retatm-retspy:>+10.2f} {w:>8}')

# Budget sweep for ~35% OTM vs ATM
print(f'\n\n=== Budget Sweep: ~35% OTM vs ATM (leveraged) ===\n')
print(f'{"Budget":<10} {"35%OTM Ann%":>12} {"35%OTM Exc%":>12} {"ATM Ann%":>10} {"ATM Exc%":>10}')
print('-' * 60)

for bp, label in [(0.005, '0.5%'), (0.01, '1.0%'), (0.02, '2.0%'), (0.033, '3.3%')]:
    r35 = run_backtest(f'35%OTM {label}', 1.0, 0.0,
        lambda: make_put(schema, -0.025, -0.005, sort_asc=True),
        data, budget_pct=bp)
    ra = run_backtest(f'ATM {label}', 1.0, 0.0,
        lambda: make_put(schema, -0.55, -0.45, sort_asc=True),
        data, budget_pct=bp)
    print(f'{label:<10} {r35["annual_ret"]:>12.2f} {r35["excess_annual"]:>+12.2f} {ra["annual_ret"]:>10.2f} {ra["excess_annual"]:>+10.2f}')
