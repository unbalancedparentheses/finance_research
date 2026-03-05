#!/usr/bin/env python3
"""Debug: Compare actual insurance costs and check raw option data quality."""

import os, sys, warnings, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, INITIAL_CAPITAL
from options_portfolio_backtester.data.providers import HistoricalOptionsData

data = load_data()
schema = data['schema']
options_data = data['options_data']

# Raw options data
raw = options_data._data

# Check what delta ranges are available in the data
print('=== Raw Options Data Structure ===')
print(f'Columns: {list(raw.columns)}')
print(f'Total rows: {len(raw):,}')
print()

# Filter to SPY puts only
spy_puts = raw[(raw['underlying'] == 'SPY') & (raw['type'] == 'put')].copy()
print(f'SPY puts: {len(spy_puts):,}')

# Check delta distribution
print(f'\nDelta stats:')
print(spy_puts['delta'].describe())
print(f'NaN deltas: {spy_puts["delta"].isna().sum()}')
print(f'Zero deltas: {(spy_puts["delta"] == 0).sum()}')
print(f'Positive deltas (wrong sign?): {(spy_puts["delta"] > 0).sum()}')

# Sample of deep OTM puts (delta -0.10 to -0.02)
deep_otm = spy_puts[(spy_puts['delta'] >= -0.10) & (spy_puts['delta'] <= -0.02)]
print(f'\nDeep OTM (delta -0.10 to -0.02): {len(deep_otm):,} rows')
if len(deep_otm) > 0:
    print(f'  Ask price: mean ${deep_otm["ask"].mean():.2f}, median ${deep_otm["ask"].median():.2f}')
    print(f'  Bid price: mean ${deep_otm["bid"].mean():.2f}, median ${deep_otm["bid"].median():.2f}')
    print(f'  Bid-Ask spread: mean ${(deep_otm["ask"] - deep_otm["bid"]).mean():.2f}')
    print(f'  Spread as % of ask: {((deep_otm["ask"] - deep_otm["bid"]) / deep_otm["ask"].replace(0, np.nan)).mean() * 100:.1f}%')
    # OTM percentage
    if 'underlying_last' in raw.columns:
        deep_otm_pct = (deep_otm['underlying_last'] - deep_otm['strike']) / deep_otm['underlying_last'] * 100
        print(f'  OTM %: mean {deep_otm_pct.mean():.1f}%, median {deep_otm_pct.median():.1f}%')

# Sample of ATM puts (delta -0.55 to -0.40)
atm = spy_puts[(spy_puts['delta'] >= -0.55) & (spy_puts['delta'] <= -0.40)]
print(f'\nATM (delta -0.55 to -0.40): {len(atm):,} rows')
if len(atm) > 0:
    print(f'  Ask price: mean ${atm["ask"].mean():.2f}, median ${atm["ask"].median():.2f}')
    print(f'  Bid price: mean ${atm["bid"].mean():.2f}, median ${atm["bid"].median():.2f}')
    print(f'  Bid-Ask spread: mean ${(atm["ask"] - atm["bid"]).mean():.2f}')
    print(f'  Spread as % of ask: {((atm["ask"] - atm["bid"]) / atm["ask"].replace(0, np.nan)).mean() * 100:.1f}%')
    if 'underlying_last' in raw.columns:
        atm_pct = (atm['underlying_last'] - atm['strike']) / atm['underlying_last'] * 100
        print(f'  OTM %: mean {atm_pct.mean():.1f}%, median {atm_pct.median():.1f}%')

# Check: which column name is the underlying price?
price_cols = [c for c in raw.columns if 'price' in c.lower() or 'last' in c.lower() or 'close' in c.lower() or 'underlying' in c.lower()]
print(f'\nPrice-related columns: {price_cols}')

# Show a few deep OTM rows
print(f'\nSample deep OTM rows (2020-02-03):')
sample = deep_otm[deep_otm['quotedate'] == '2020-02-03']
if len(sample) == 0:
    # Try finding the date
    dates = deep_otm['quotedate'].unique()
    target_dates = [d for d in dates if '2020-02' in str(d)]
    if target_dates:
        sample = deep_otm[deep_otm['quotedate'] == target_dates[0]]
if len(sample) > 0:
    cols = ['quotedate', 'strike', 'delta', 'bid', 'ask', 'dte', 'underlying_last']
    cols = [c for c in cols if c in sample.columns]
    print(sample[cols].head(10).to_string())

print(f'\nSample ATM rows (2020-02-03):')
sample = atm[atm['quotedate'] == '2020-02-03']
if len(sample) == 0:
    dates = atm['quotedate'].unique()
    target_dates = [d for d in dates if '2020-02' in str(d)]
    if target_dates:
        sample = atm[atm['quotedate'] == target_dates[0]]
if len(sample) > 0:
    cols = ['quotedate', 'strike', 'delta', 'bid', 'ask', 'dte', 'underlying_last']
    cols = [c for c in cols if c in sample.columns]
    print(sample[cols].head(10).to_string())

# Check entry_sort behavior: what option gets picked?
# For deep OTM: sort by delta descending → picks delta closest to 0 (deepest OTM)
# For ATM: sort by delta descending → picks delta closest to 0 (most OTM in ATM range = -0.40)
print('\n\n=== Entry Sort Behavior ===')
print('Deep OTM: picks delta closest to 0 (deepest OTM)')
print('ATM: picks delta closest to -0.40 (most OTM in "ATM" range)')
print()
print('ISSUE: "ATM" picks delta -0.40 which is actually moderately OTM, not truly ATM')
print('True ATM = delta ~-0.50. Our "ATM" is picking the boundary (-0.40).')
print()

# What about the budget spending?
# The key question: does the budget-target model (remaining_budget = target - current_options)
# cause ATM to spend less because old positions retain value?
# Let's check with actual balance data
from backtest_runner import run_backtest, make_deep_otm_put_strategy, make_atm_put_strategy

print('\n=== Checking Balance Cash Column ===')
r_otm = run_backtest('Deep OTM 0.5%', 1.0, 0.0,
    lambda: make_deep_otm_put_strategy(schema), data, budget_pct=0.005)
r_atm = run_backtest('ATM 0.5%', 1.0, 0.0,
    lambda: make_atm_put_strategy(schema), data, budget_pct=0.005)

print(f'\nBalance columns: {list(r_otm["balance"].columns)}')

# Check cash levels over time
for name, r in [('Deep OTM', r_otm), ('ATM', r_atm)]:
    bal = r['balance']
    cash = bal['cash'] if 'cash' in bal.columns else None
    total_cap = bal['total capital']
    if cash is not None:
        print(f'\n{name}:')
        print(f'  Cash: min ${cash.min():,.0f}, max ${cash.max():,.0f}, mean ${cash.mean():,.0f}')
        print(f'  Negative cash days: {(cash < 0).sum()}')
        print(f'  Cash at start: ${cash.iloc[0]:,.0f}')
        print(f'  Cash at end: ${cash.iloc[-1]:,.0f}')

    # Check if there's an options_value column
    for col in bal.columns:
        if 'option' in col.lower() or 'cap' in col.lower():
            print(f'  {col}: mean ${bal[col].mean():,.0f}')

# Compare total capital trajectories
print('\n=== Key Capital Checkpoints ===')
print(f'{"Date":<12} {"OTM total":>12} {"ATM total":>12} {"Diff":>12} {"Diff%":>8}')
print('-' * 60)
for d in ['2009-01-02', '2012-01-03', '2015-01-02', '2018-01-02', '2020-01-02', '2023-01-03', '2025-12-12']:
    dt = pd.Timestamp(d)
    otm = r_otm['balance']['total capital']
    atm = r_atm['balance']['total capital']
    o = otm[otm.index <= dt + pd.Timedelta(days=5)].iloc[-1] if len(otm[otm.index <= dt + pd.Timedelta(days=5)]) > 0 else 0
    a = atm[atm.index <= dt + pd.Timedelta(days=5)].iloc[-1] if len(atm[atm.index <= dt + pd.Timedelta(days=5)]) > 0 else 0
    diff = a - o
    pct = diff / o * 100 if o > 0 else 0
    print(f'{d:<12} {o:>12,.0f} {a:>12,.0f} {diff:>+12,.0f} {pct:>+7.1f}%')
