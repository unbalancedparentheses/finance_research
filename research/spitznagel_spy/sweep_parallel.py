#!/usr/bin/env python3
"""Parallel grid sweep using Rust's parallel_sweep (Rayon).

Tests OTM level x DTE x exit DTE x budget in parallel.
Both leveraged and no-leverage modes.
"""

import os, sys, warnings, math, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, INITIAL_CAPITAL
from options_portfolio_backtester import OptionType as Type, Direction
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester._ob_rust import parallel_sweep
from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData

data = load_data()
schema = data['schema']
spy_annual = data['spy_annual_ret']
spy_dd = data['spy_dd']

# =====================================================================
# Prepare data for Rust engine
# =====================================================================
opts_data = data['options_data']
stocks_data = data['stocks_data']
opts_schema = opts_data.schema
stocks_schema = stocks_data.schema

# Rebalance dates (business month start)
dates_df = (
    pd.DataFrame(opts_data._data[["quotedate", "volume"]])
    .drop_duplicates("quotedate")
    .set_index("quotedate")
)
rb_days = pd.to_datetime(
    dates_df.groupby(pd.Grouper(freq="1BMS"))
    .apply(lambda x: x.index.min())
    .values
)
rb_date_ns = [int(d.value) for d in rb_days if not pd.isna(d)]

# Schema mapping
schema_mapping = {
    "contract": opts_schema["contract"],
    "date": opts_schema["date"],
    "stocks_date": stocks_schema["date"],
    "stocks_symbol": stocks_schema["symbol"],
    "stocks_price": stocks_schema["adjClose"],
    "underlying": opts_schema["underlying"],
    "expiration": opts_schema["expiration"],
    "type": opts_schema["type"],
    "strike": opts_schema["strike"],
}

# Convert to Polars (once)
opts_pl = pl.from_arrow(pa.Table.from_pandas(opts_data._data, preserve_index=False))
stocks_pl = pl.from_arrow(pa.Table.from_pandas(stocks_data._data, preserve_index=False))

# Build a base config template
def make_base_config(stock_pct, opt_pct, budget_pct=None):
    cfg = {
        "allocation": {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        "initial_capital": float(INITIAL_CAPITAL),
        "shares_per_contract": 100,
        "rebalance_dates": rb_date_ns,
        "legs": [{
            "name": "leg_1",
            "entry_filter": "((type == 'put') & (ask > 0)) & (((((underlying == 'SPY') & (dte >= 30)) & (dte <= 60)) & (delta >= -0.25)) & (delta <= -0.15))",
            "exit_filter": "(type == 'put') & (dte <= 14)",
            "direction": "ask",
            "type": "put",
            "entry_sort_col": "delta",
            "entry_sort_asc": True,
        }],
        "profit_pct": None,
        "loss_pct": None,
        "stocks": [("SPY", 1.0)],
        "options_budget_pct": budget_pct,
        "check_exits_daily": True,
    }
    return cfg

def make_entry_filter(delta_min, delta_max, dte_min, dte_max):
    return (f"((type == 'put') & (ask > 0)) & (((((underlying == 'SPY')"
            f" & (dte >= {dte_min})) & (dte <= {dte_max}))"
            f" & (delta >= {delta_min})) & (delta <= {delta_max}))")

def make_exit_filter(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

# =====================================================================
# Grid dimensions
# =====================================================================
otm_levels = [
    ('ATM',     -0.55, -0.45),
    ('5%OTM',   -0.40, -0.30),
    ('10%OTM',  -0.25, -0.15),
    ('15%OTM',  -0.15, -0.08),
    ('20%OTM',  -0.10, -0.04),
    ('25%OTM',  -0.06, -0.02),
    ('30%OTM',  -0.04, -0.01),
    ('35%OTM',  -0.025, -0.005),
    ('40%OTM',  -0.015, -0.002),
]

dte_configs = [
    ('30-60/7',    30,  60,  7),
    ('30-60/10',   30,  60, 10),
    ('30-60/14',   30,  60, 14),
    ('30-60/21',   30,  60, 21),
    ('30-60/25',   30,  60, 25),
    ('60-90/14',   60,  90, 14),
    ('60-90/30',   60,  90, 30),
    ('88-93/10',   88,  93, 10),
    ('88-93/30',   88,  93, 30),
    ('90-120/30',  90, 120, 30),
]

budgets_lev = [0.005, 0.01, 0.02, 0.033]
budgets_nolev = [0.005, 0.01, 0.02, 0.033]

# =====================================================================
# LEVERAGED SWEEP
# =====================================================================
print('='*130)
print('LEVERAGED SWEEP (100% stocks + puts on top)')
print('='*130 + '\n')

param_grid = []
combo_keys = []
for otm_name, dmin, dmax in otm_levels:
    for dte_name, dte_min, dte_max, exit_dte in dte_configs:
        for bp in budgets_lev:
            label = f'{otm_name} {dte_name} b{bp*100:.1f}%'
            param_grid.append({
                "label": label,
                "leg_entry_filters": [make_entry_filter(dmin, dmax, dte_min, dte_max)],
                "leg_exit_filters": [make_exit_filter(exit_dte)],
                "options_budget_pct": bp,
            })
            combo_keys.append((otm_name, dte_name, bp))

base_config = make_base_config(1.0, 0.0, budget_pct=0.005)

print(f'Running {len(param_grid)} configs in parallel...', flush=True)
t0 = time.perf_counter()
results = parallel_sweep(opts_pl, stocks_pl, base_config, schema_mapping, param_grid, None)
elapsed = time.perf_counter() - t0
print(f'Done in {elapsed:.1f}s ({len(param_grid)/elapsed:.0f} configs/sec)\n')

# Parse results into a DataFrame
rows = []
for res, (otm_name, dte_name, bp) in zip(results, combo_keys):
    ann_ret = res['annualized_return'] * 100
    max_dd = res['max_drawdown'] * 100
    excess = ann_ret - spy_annual
    rows.append({
        'OTM': otm_name, 'DTE': dte_name, 'Budget': bp * 100,
        'Annual%': ann_ret, 'Excess%': excess, 'MaxDD%': max_dd,
        'Trades': res.get('total_trades', 0),
    })
df_lev = pd.DataFrame(rows)

# Print pivot: OTM x DTE at each budget
for bp in budgets_lev:
    bl = f'{bp*100:.1f}%'
    sub = df_lev[df_lev['Budget'] == bp * 100]
    pivot = sub.pivot_table(index='DTE', columns='OTM', values='Excess%')
    # Reorder columns
    otm_order = [o[0] for o in otm_levels]
    pivot = pivot[[c for c in otm_order if c in pivot.columns]]
    print(f'\n--- Leveraged, Budget {bl}: Excess Annual % ---')
    print(pivot.to_string(float_format='{:+.2f}'.format))
    # Find best
    best_idx = sub['Excess%'].idxmax()
    best = sub.loc[best_idx]
    print(f'Best: {best["OTM"]} {best["DTE"]} = {best["Excess%"]:+.2f}%')

# =====================================================================
# NO-LEVERAGE SWEEP
# =====================================================================
print('\n\n' + '='*130)
print('NO-LEVERAGE SWEEP (stock + opt = 100%)')
print('='*130 + '\n')

nolev_grid = []
nolev_keys = []
for otm_name, dmin, dmax in otm_levels:
    for dte_name, dte_min, dte_max, exit_dte in dte_configs:
        for bp in budgets_nolev:
            label = f'{otm_name} {dte_name} nolev b{bp*100:.1f}%'
            nolev_grid.append({
                "label": label,
                "leg_entry_filters": [make_entry_filter(dmin, dmax, dte_min, dte_max)],
                "leg_exit_filters": [make_exit_filter(exit_dte)],
                "options_budget_pct": None,  # Use allocation-based, not external budget
            })
            nolev_keys.append((otm_name, dte_name, bp))

# For no-leverage we need different base configs per budget level.
# parallel_sweep uses a single base config, so we run one sweep per budget.
for bp in budgets_nolev:
    bl = f'{bp*100:.1f}%'
    stock_pct = 1.0 - bp
    base_nolev = make_base_config(stock_pct, bp, budget_pct=None)

    sub_grid = []
    sub_keys = []
    for otm_name, dmin, dmax in otm_levels:
        for dte_name, dte_min, dte_max, exit_dte in dte_configs:
            sub_grid.append({
                "label": f'{otm_name} {dte_name} nolev {bl}',
                "leg_entry_filters": [make_entry_filter(dmin, dmax, dte_min, dte_max)],
                "leg_exit_filters": [make_exit_filter(exit_dte)],
            })
            sub_keys.append((otm_name, dte_name, bp))

    print(f'No-leverage {bl}: {len(sub_grid)} configs...', end=' ', flush=True)
    t0 = time.perf_counter()
    nolev_results = parallel_sweep(opts_pl, stocks_pl, base_nolev, schema_mapping, sub_grid, None)
    elapsed = time.perf_counter() - t0
    print(f'{elapsed:.1f}s')

    rows = []
    for res, (otm_name, dte_name, _) in zip(nolev_results, sub_keys):
        ann_ret = res['annualized_return'] * 100
        max_dd = res['max_drawdown'] * 100
        excess = ann_ret - spy_annual
        rows.append({
            'OTM': otm_name, 'DTE': dte_name, 'Budget': bp * 100,
            'Annual%': ann_ret, 'Excess%': excess, 'MaxDD%': max_dd,
            'Trades': res.get('total_trades', 0),
        })
    df_sub = pd.DataFrame(rows)

    pivot = df_sub.pivot_table(index='DTE', columns='OTM', values='Excess%')
    otm_order = [o[0] for o in otm_levels]
    pivot = pivot[[c for c in otm_order if c in pivot.columns]]
    print(f'\n--- No-Leverage {bl}: Excess Annual % ---')
    print(pivot.to_string(float_format='{:+.2f}'.format))
    best_idx = df_sub['Excess%'].idxmax()
    best = df_sub.loc[best_idx]
    print(f'Best: {best["OTM"]} {best["DTE"]} = {best["Excess%"]:+.2f}%\n')

print('\nDone.')
