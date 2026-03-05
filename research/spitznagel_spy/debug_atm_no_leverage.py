#!/usr/bin/env python3
"""Compare delta range performance: leveraged vs no-leverage."""

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

def make_custom_put(schema, delta_min, delta_max, sort_asc=False):
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

deltas = [
    ('Deep OTM (d -0.10 to -0.02)',      -0.10, -0.02, False),
    ('Std OTM (d -0.25 to -0.10)',       -0.25, -0.10, False),
    ('Near OTM (d -0.35 to -0.20)',      -0.35, -0.20, False),
    ('Moderate (d -0.45 to -0.35)',      -0.45, -0.35, False),
    ('Near ATM (d -0.55 to -0.40)',      -0.55, -0.40, False),
    ('True ATM (d -0.55 to -0.45)',      -0.55, -0.45, False),
    ('ATM (d -0.60 to -0.50)',           -0.60, -0.50, False),
]

budget = 0.005  # 0.5%

# ---- LEVERAGED ----
print('=== LEVERAGED (100% equity + puts on top) ===\n')
lev_results = []
for name, dmin, dmax, sa in deltas:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, 1.0, 0.0,
        lambda d1=dmin, d2=dmax, s=sa: make_custom_put(schema, d1, d2, sort_asc=s),
        data, budget_pct=budget)
    lev_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

# ---- NO LEVERAGE ----
print('\n=== NO LEVERAGE (reduce equity to fund puts) ===\n')
nolev_results = []
for name, dmin, dmax, sa in deltas:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, 1.0 - budget, budget,
        lambda d1=dmin, d2=dmax, s=sa: make_custom_put(schema, d1, d2, sort_asc=s),
        data)  # NO budget_pct → allocation-based
    nolev_results.append(r)
    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, DD {r["max_dd"]:.1f}%')

# ---- SUMMARY ----
spy_annual = data['spy_annual_ret']

print(f'\n\n{"="*120}')
print(f'COMPARISON: Leveraged vs No-Leverage across delta range (0.5% budget, DTE 60-90, exit DTE 30)')
print(f'{"="*120}\n')

print(f'{"Strategy":<35} {"LEV Annual%":>12} {"LEV Excess%":>12} {"LEV MaxDD%":>11} {"NOLEV Annual%":>14} {"NOLEV Excess%":>14} {"NOLEV MaxDD%":>13}')
print('-' * 120)
print(f'{"SPY B&H":<35} {spy_annual:>12.2f} {"--":>12} {data["spy_dd"]:>11.1f} {spy_annual:>14.2f} {"--":>14} {data["spy_dd"]:>13.1f}')

for (name, _, _, _), lr, nr in zip(deltas, lev_results, nolev_results):
    print(f'{name:<35} {lr["annual_ret"]:>12.2f} {lr["excess_annual"]:>+12.2f} {lr["max_dd"]:>11.1f} '
          f'{nr["annual_ret"]:>14.2f} {nr["excess_annual"]:>+14.2f} {nr["max_dd"]:>13.1f}')

# Also try higher budgets
print(f'\n\n{"="*120}')
print(f'HIGHER BUDGETS: Deep OTM vs True ATM')
print(f'{"="*120}\n')

for bp, bp_label in [(0.005, '0.5%'), (0.01, '1.0%'), (0.033, '3.3%')]:
    print(f'\n--- Budget: {bp_label} ---')
    print(f'{"Strategy":<35} {"LEV Annual%":>12} {"LEV Excess%":>12} {"NOLEV Annual%":>14} {"NOLEV Excess%":>14}')
    print('-' * 90)

    for name, dmin, dmax, sa in [
        ('Deep OTM (d -0.10 to -0.02)', -0.10, -0.02, False),
        ('True ATM (d -0.60 to -0.50)',  -0.60, -0.50, False),
    ]:
        # Leveraged
        rl = run_backtest(f'{name} lev', 1.0, 0.0,
            lambda d1=dmin, d2=dmax, s=sa: make_custom_put(schema, d1, d2, sort_asc=s),
            data, budget_pct=bp)

        # No leverage
        rn = run_backtest(f'{name} nolev', 1.0 - bp, bp,
            lambda d1=dmin, d2=dmax, s=sa: make_custom_put(schema, d1, d2, sort_asc=s),
            data)

        print(f'{name:<35} {rl["annual_ret"]:>12.2f} {rl["excess_annual"]:>+12.2f} '
              f'{rn["annual_ret"]:>14.2f} {rn["excess_annual"]:>+14.2f}')
