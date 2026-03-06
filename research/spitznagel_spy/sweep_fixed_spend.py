#!/usr/bin/env python3
"""Fair comparison: FIXED monthly premium spend (same $ for ATM and OTM).

The budget_pct model spends less on ATM (positions retain value).
This test converts fixed dollar amounts to pct of initial capital so both
strategies target the same premium fraction. Equivalent at t=0 and close
enough thereafter for the ATM-vs-OTM comparison purpose.

Also: deep dive into what happens to 40% OTM during crashes.
"""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from options_portfolio_backtester import BacktestEngine as Backtest, Stock, OptionType as Type, Direction
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg

from backtest_runner import load_data, INITIAL_CAPITAL

data = load_data()
schema = data['schema']
spy_annual = data['spy_annual_ret']
spy_prices = data['spy_prices']
years = data['years']

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

def run_fixed_budget(name, strategy_fn, fixed_budget):
    bt = Backtest(
        {'stocks': 1.0, 'options': 0.0, 'cash': 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_pct = fixed_budget / INITIAL_CAPITAL
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = data['stocks_data']
    bt.options_strategy = strategy_fn()
    bt.options_data = data['options_data']
    bt.run(rebalance_freq=1, rebalance_unit='BMS', check_exits_daily=True)

    balance = bt.balance
    total_cap = balance['total capital']
    total_ret = (balance['accumulated return'].iloc[-1] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = total_cap.cummax()
    drawdown = (total_cap - cummax) / cummax
    max_dd = drawdown.min() * 100

    return {
        'name': name,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - spy_annual,
        'balance': balance,
        'trade_log': bt.trade_log,
    }

def run_pct_budget(name, strategy_fn, budget_pct):
    bt = Backtest(
        {'stocks': 1.0, 'options': 0.0, 'cash': 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.options_budget_pct = budget_pct
    bt.stocks = [Stock('SPY', 1.0)]
    bt.stocks_data = data['stocks_data']
    bt.options_strategy = strategy_fn()
    bt.options_data = data['options_data']
    bt.run(rebalance_freq=1, rebalance_unit='BMS', check_exits_daily=True)

    balance = bt.balance
    total_cap = balance['total capital']
    total_ret = (balance['accumulated return'].iloc[-1] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = total_cap.cummax()
    drawdown = (total_cap - cummax) / cummax
    max_dd = drawdown.min() * 100

    return {
        'name': name,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'trades': len(bt.trade_log),
        'excess_annual': annual_ret - spy_annual,
        'balance': balance,
        'trade_log': bt.trade_log,
    }

otm_levels = [
    ('ATM',     -0.55, -0.45, True),
    ('5%OTM',   -0.40, -0.30, True),
    ('10%OTM',  -0.25, -0.15, True),
    ('15%OTM',  -0.15, -0.08, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('35%OTM',  -0.025, -0.005, True),
    ('40%OTM',  -0.015, -0.002, True),
]

# DTE configs to test
dte_configs = [
    ('30-60/14',  30,  60, 14),
    ('30-60/25',  30,  60, 25),
    ('60-90/30',  60,  90, 30),
    ('88-93/10',  88,  93, 10),  # best exit for 40% OTM at 90 DTE
    ('88-93/60',  88,  93, 60),  # user's proposed config
]

# =====================================================================
# Test 1: FIXED $5,000/month budget — fair comparison
# =====================================================================
print('='*120)
print('FIXED BUDGET: $5,000/month (same spend for all OTM levels)')
print('='*120 + '\n')

for dte_name, dte_min, dte_max, exit_dte in dte_configs:
    print(f'--- DTE {dte_name} ---')
    print(f'{"OTM":<10} {"Excess%":>10} {"Annual%":>10} {"MaxDD%":>10} {"Trades":>8}')
    print('-' * 52)

    for otm_name, dmin, dmax, sa in otm_levels:
        r = run_fixed_budget(
            f'{otm_name} {dte_name} fixed',
            lambda d1=dmin, d2=dmax, s=sa, dmin_=dte_min, dmax_=dte_max, edte=exit_dte:
                make_put(schema, d1, d2, dmin_, dmax_, edte, sort_asc=s),
            5000)
        print(f'{otm_name:<10} {r["excess_annual"]:>+10.2f} {r["annual_ret"]:>10.2f} '
              f'{r["max_dd"]:>10.1f} {r["trades"]:>8}')
    print()

# =====================================================================
# Test 2: Fixed budget sweep ($2k, $5k, $10k, $20k, $33k)
# =====================================================================
print('\n' + '='*120)
print('FIXED BUDGET SWEEP: 40%OTM vs ATM vs 10%OTM (DTE 30-60/14)')
print('='*120 + '\n')

fixed_budgets = [2000, 5000, 10000, 20000, 33000]
test_otms = [
    ('ATM',     -0.55, -0.45, True),
    ('5%OTM',   -0.40, -0.30, True),
    ('10%OTM',  -0.25, -0.15, True),
    ('25%OTM',  -0.06, -0.02, True),
    ('40%OTM',  -0.015, -0.002, True),
]

print(f'{"OTM":<10}', end='')
for fb in fixed_budgets:
    print(f'  ${fb/1000:.0f}k exc  ${fb/1000:.0f}k DD', end='')
print()
print('-' * (10 + len(fixed_budgets) * 18))

for otm_name, dmin, dmax, sa in test_otms:
    print(f'{otm_name:<10}', end='', flush=True)
    for fb in fixed_budgets:
        r = run_fixed_budget(
            f'{otm_name} ${fb}',
            lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 30, 60, 14, sort_asc=s),
            fb)
        print(f'  {r["excess_annual"]:>+6.2f} {r["max_dd"]:>6.1f}', end='', flush=True)
    print()

# Same for 90 DTE configs
print(f'\n--- DTE 88-93/10 (best for 40% OTM) ---')
print(f'{"OTM":<10}', end='')
for fb in fixed_budgets:
    print(f'  ${fb/1000:.0f}k exc  ${fb/1000:.0f}k DD', end='')
print()
print('-' * (10 + len(fixed_budgets) * 18))

for otm_name, dmin, dmax, sa in test_otms:
    print(f'{otm_name:<10}', end='', flush=True)
    for fb in fixed_budgets:
        r = run_fixed_budget(
            f'{otm_name} 90/10 ${fb}',
            lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 88, 93, 10, sort_asc=s),
            fb)
        print(f'  {r["excess_annual"]:>+6.2f} {r["max_dd"]:>6.1f}', end='', flush=True)
    print()

# =====================================================================
# Test 3: Compare pct-budget vs fixed-budget for same configs
# =====================================================================
print('\n\n' + '='*120)
print('PCT vs FIXED BUDGET: Does the budget model explain ATM advantage?')
print('='*120 + '\n')

print(f'{"Config":<25} {"Pct 0.5% exc":>14} {"Fixed $5k exc":>14} {"Diff":>8}')
print('-' * 65)

for otm_name, dmin, dmax, sa in test_otms:
    rp = run_pct_budget(
        f'{otm_name} pct',
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 30, 60, 14, sort_asc=s),
        0.005)
    rf = run_fixed_budget(
        f'{otm_name} fixed',
        lambda d1=dmin, d2=dmax, s=sa: make_put(schema, d1, d2, 30, 60, 14, sort_asc=s),
        5000)
    diff = rf['excess_annual'] - rp['excess_annual']
    print(f'{otm_name + " 30-60/14":<25} {rp["excess_annual"]:>+14.2f} {rf["excess_annual"]:>+14.2f} {diff:>+8.2f}')

# =====================================================================
# Test 4: Year-by-year with fixed $5k budget
# =====================================================================
print('\n\n' + '='*120)
print('YEAR-BY-YEAR: Fixed $5k/month, 30-60/14')
print('='*120 + '\n')

r40f = run_fixed_budget('40%OTM fixed',
    lambda: make_put(schema, -0.015, -0.002, 30, 60, 14, sort_asc=True), 5000)
r10f = run_fixed_budget('10%OTM fixed',
    lambda: make_put(schema, -0.25, -0.15, 30, 60, 14, sort_asc=True), 5000)
ratmf = run_fixed_budget('ATM fixed',
    lambda: make_put(schema, -0.55, -0.45, 30, 60, 14, sort_asc=True), 5000)

cap40 = r40f['balance']['total capital']
cap10 = r10f['balance']['total capital']
capatm = ratmf['balance']['total capital']

print(f'{"Year":<6} {"SPY":>8} {"40%OTM":>10} {"10%OTM":>10} {"ATM":>10} {"Winner":>10}')
print('-' * 60)

for yr in range(2008, 2026):
    s = pd.Timestamp(f'{yr}-01-01')
    e = pd.Timestamp(f'{yr+1}-01-01')
    pspy = spy_prices[(spy_prices.index >= s) & (spy_prices.index < e)]
    p40 = cap40[(cap40.index >= s) & (cap40.index < e)]
    p10 = cap10[(cap10.index >= s) & (cap10.index < e)]
    patm = capatm[(capatm.index >= s) & (capatm.index < e)]
    if len(pspy) > 10 and len(p40) > 10:
        retspy = (pspy.iloc[-1] / pspy.iloc[0] - 1) * 100
        ret40 = (p40.iloc[-1] / p40.iloc[0] - 1) * 100
        ret10 = (p10.iloc[-1] / p10.iloc[0] - 1) * 100
        retatm = (patm.iloc[-1] / patm.iloc[0] - 1) * 100
        exc40 = ret40 - retspy
        exc10 = ret10 - retspy
        excatm = retatm - retspy
        best_exc = max(exc40, exc10, excatm)
        w = '40%OTM' if exc40 == best_exc else ('10%OTM' if exc10 == best_exc else 'ATM')
        print(f'{yr:<6} {retspy:>8.1f} {exc40:>+10.2f} {exc10:>+10.2f} {excatm:>+10.2f} {w:>10}')

print('\nDone.')
