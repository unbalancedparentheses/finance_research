#!/usr/bin/env python3
"""Compare deep OTM vs various ATM definitions at 0.5% leveraged."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import (
    load_data, run_backtest, INITIAL_CAPITAL,
    make_deep_otm_put_strategy, make_atm_put_strategy,
)
from options_portfolio_backtester import OptionType as Type, Direction
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

data = load_data()
schema = data['schema']

def make_custom_put_strategy(schema, delta_min, delta_max, dte_min=60, dte_max=90, exit_dte=30, sort_asc=False):
    """Custom put strategy with configurable delta range."""
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

configs = [
    # (name, delta_min, delta_max, sort_ascending, description)
    ('Deep OTM (d -0.10 to -0.02)',       -0.10, -0.02, False, 'picks -0.02 (deepest)'),
    ('Std OTM (d -0.25 to -0.10)',        -0.25, -0.10, False, 'picks -0.10'),
    ('Near OTM (d -0.35 to -0.20)',       -0.35, -0.20, False, 'picks -0.20'),
    ('Moderate OTM (d -0.45 to -0.35)',   -0.45, -0.35, False, 'picks -0.35'),
    ('Old "ATM" (d -0.55 to -0.40)',      -0.55, -0.40, False, 'picks -0.40'),
    ('True ATM (d -0.55 to -0.45)',       -0.55, -0.45, False, 'picks -0.45'),
    ('True ATM asc (d -0.55 to -0.45)',   -0.55, -0.45, True,  'picks -0.55 (most ATM)'),
    ('Deep ATM (d -0.60 to -0.50)',       -0.60, -0.50, False, 'picks -0.50'),
    ('Deep ATM asc (d -0.60 to -0.50)',   -0.60, -0.50, True,  'picks -0.60 (ITM)'),
]

results = []
for name, dmin, dmax, sort_asc, desc in configs:
    print(f'  {name} ({desc})...', end=' ', flush=True)
    r = run_backtest(
        name, 1.0, 0.0,
        lambda d1=dmin, d2=dmax, sa=sort_asc: make_custom_put_strategy(schema, d1, d2, sort_asc=sa),
        data, budget_pct=0.005,
    )
    results.append(r)

    # Get options P&L
    tl = r['trade_log']
    entries = tl[tl[('totals', 'cost')] > 0]
    exits = tl[tl[('totals', 'cost')] <= 0]
    premium_paid = (entries[('totals', 'cost')] * entries[('totals', 'qty')]).sum()
    received = abs((exits[('totals', 'cost')] * exits[('totals', 'qty')]).sum())
    net_pnl = received - premium_paid
    loss_pct = net_pnl / premium_paid * 100 if premium_paid > 0 else 0

    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    sharpe = r['annual_ret'] / vol if vol > 0 else 0

    print(f'annual {r["annual_ret"]:+.2f}%, excess {r["excess_annual"]:+.2f}%, '
          f'DD {r["max_dd"]:.1f}%, Sharpe {sharpe:.3f}, '
          f'options P&L ${net_pnl:+,.0f} ({loss_pct:+.1f}% of premium)')

# Summary table
print('\n\n=== SUMMARY: Delta Range vs Performance (0.5% leveraged, DTE 60-90, exit DTE 30) ===\n')
print(f'{"Strategy":<38} {"Annual%":>8} {"Excess%":>8} {"MaxDD%":>8} {"Sharpe":>8} {"Trades":>7} {"Opt P&L%":>10}')
print('-' * 100)

spy_annual = data['spy_annual_ret']
print(f'{"SPY B&H":<38} {spy_annual:>8.2f} {"--":>8} {data["spy_dd"]:>8.1f} {"":>8} {"":>7} {"":>10}')

for r, (name, dmin, dmax, sort_asc, desc) in zip(results, configs):
    tl = r['trade_log']
    entries = tl[tl[('totals', 'cost')] > 0]
    exits = tl[tl[('totals', 'cost')] <= 0]
    premium_paid = (entries[('totals', 'cost')] * entries[('totals', 'qty')]).sum()
    received = abs((exits[('totals', 'cost')] * exits[('totals', 'qty')]).sum())
    net_pnl = received - premium_paid
    loss_pct = net_pnl / premium_paid * 100 if premium_paid > 0 else 0

    d = r['balance']['% change'].dropna()
    vol = d.std() * np.sqrt(252) * 100
    sharpe = r['annual_ret'] / vol if vol > 0 else 0

    print(f'{name:<38} {r["annual_ret"]:>8.2f} {r["excess_annual"]:>+8.2f} {r["max_dd"]:>8.1f} {sharpe:>8.3f} {r["trades"]:>7} {loss_pct:>+10.1f}%')

# Year-by-year for deep OTM vs true ATM (d -0.55 to -0.45, ascending = picks -0.55)
print('\n\n=== Year-by-Year: Deep OTM vs True ATM (picks delta ~-0.55) ===\n')
r_deep = results[0]  # Deep OTM
r_true_atm = results[6]  # True ATM ascending (picks -0.55)

spy = data['spy_prices']
deep_cap = r_deep['balance']['total capital']
atm_cap = r_true_atm['balance']['total capital']

print(f'{"Year":<6} {"Deep OTM":>10} {"True ATM":>10} {"SPY":>10} {"OTM exc":>10} {"ATM exc":>10} {"Winner":>10}')
print('-' * 70)

for yr in range(2008, 2026):
    yr_start = pd.Timestamp(f'{yr}-01-01')
    yr_end = pd.Timestamp(f'{yr+1}-01-01')

    otm_period = deep_cap[(deep_cap.index >= yr_start) & (deep_cap.index < yr_end)]
    atm_period = atm_cap[(atm_cap.index >= yr_start) & (atm_cap.index < yr_end)]
    spy_period = spy[(spy.index >= yr_start) & (spy.index < yr_end)]

    if len(otm_period) > 10 and len(atm_period) > 10 and len(spy_period) > 10:
        otm_ret = (otm_period.iloc[-1] / otm_period.iloc[0] - 1) * 100
        atm_ret = (atm_period.iloc[-1] / atm_period.iloc[0] - 1) * 100
        spy_ret = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
        otm_exc = otm_ret - spy_ret
        atm_exc = atm_ret - spy_ret
        winner = 'OTM' if otm_exc > atm_exc else 'ATM'
        print(f'{yr:<6} {otm_ret:>10.2f} {atm_ret:>10.2f} {spy_ret:>10.2f} {otm_exc:>+10.2f} {atm_exc:>+10.2f} {winner:>10}')
