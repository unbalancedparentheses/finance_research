#!/usr/bin/env python3
"""Sweep: what holding period is optimal? Test entry DTE x exit DTE more finely."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, run_backtest
from options_portfolio_backtester import OptionType as Type, Direction
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

data = load_data()
schema = data['schema']

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

# Use 5%OTM (winner) and 10%OTM (runner-up)
otm_levels = [
    ('5%OTM',   -0.40, -0.30, True),
    ('10%OTM',  -0.25, -0.15, True),
    ('25%OTM',  -0.06, -0.02, True),
]

# Entry windows x exit DTEs — focus on the hold period
combos = [
    # (entry_name, dte_min, dte_max, exit_dtes)
    ('30-60',  30,  60,  [1, 3, 5, 7, 10, 14, 18, 21, 25, 27, 28, 29]),
    ('45-75',  45,  75,  [1, 5, 10, 14, 21, 25, 30, 35, 40, 42, 44]),
    ('60-90',  60,  90,  [1, 5, 10, 14, 21, 30, 40, 45, 50, 55, 58]),
    ('90-120', 90, 120,  [1, 5, 10, 14, 21, 30, 45, 60, 75, 85, 89]),
]

for otm_name, dmin, dmax, sa in otm_levels:
    print(f'\n{"="*100}')
    print(f'{otm_name}: Excess% by entry window x exit DTE (0.5% leveraged)')
    print(f'{"="*100}\n')

    for entry_name, dte_min, dte_max, exit_dtes in combos:
        valid_exits = [e for e in exit_dtes if e < dte_min]
        if not valid_exits:
            continue
        # Compute hold period
        print(f'  Entry {entry_name}:  ', end='')
        for edte in valid_exits:
            hold_min = dte_min - dte_max  # not meaningful, use entry-exit
            print(f' exit{edte:>3}', end='')
        print()
        print(f'  {"":14}', end='')

        best_exc = -999
        best_edte = None
        for edte in valid_exits:
            r = run_backtest(
                f'{otm_name} {entry_name}/e{edte}', 1.0, 0.0,
                lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, e=edte:
                    make_put(schema, d1, d2, dmin_, dmax_, e, sort_asc=s),
                data, budget_pct=0.005)
            exc = r['excess_annual']
            print(f' {exc:>+6.2f}', end='', flush=True)
            if exc > best_exc:
                best_exc = exc
                best_edte = edte
        print(f'  -> best: exit {best_edte} ({best_exc:+.2f}%)')

        # MaxDD row
        print(f'  {"DD%":14}', end='')
        for edte in valid_exits:
            r = run_backtest(
                f'{otm_name} {entry_name}/e{edte}', 1.0, 0.0,
                lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, e=edte:
                    make_put(schema, d1, d2, dmin_, dmax_, e, sort_asc=s),
                data, budget_pct=0.005)
            print(f' {r["max_dd"]:>6.1f}', end='', flush=True)
        print('\n')

print('\nDone.')
