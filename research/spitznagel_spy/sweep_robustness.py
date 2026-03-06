#!/usr/bin/env python3
"""Robustness check: run top configs from 5D sweep across sub-periods.

Tests for overfitting by checking if results hold across:
- Full period (2008-2025)
- First half (2008-2016)
- Second half (2016-2025)
- Calm period (2012-2018, no major crash)
- Post-GFC (2010-2019)
- Recent (2019-2025, includes COVID + 2022)
"""

import os, sys, warnings, time
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import polars as pl
import pyarrow as pa

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, INITIAL_CAPITAL
from options_portfolio_backtester._ob_rust import parallel_sweep
from sweep_parallel import init_engine

# =====================================================================
# Top configs to test (from 5D sweep results)
# =====================================================================
# Format: (name, strike_lo, strike_hi, dte_min, dte_max, exit_dte, budget)
TOP_CONFIGS = [
    # Best per budget (leveraged)
    ('40% 90-120/x30 b0.25%',  0.58, 0.62, 90, 120, 30, 0.0025),
    ('40% 90-120/x30 b0.5%',   0.58, 0.62, 90, 120, 30, 0.005),
    ('40% 90-120/x30 b1%',     0.58, 0.62, 90, 120, 30, 0.01),
    ('18% 30-60/x25 b1%',      0.80, 0.84, 30, 60,  25, 0.01),
    ('18% 30-60/x25 b2%',      0.80, 0.84, 30, 60,  25, 0.02),
    ('18% 30-60/x25 b3.3%',    0.80, 0.84, 30, 60,  25, 0.033),
    # Best Sharpe
    ('5% 30-60/x21 b3.3%',     0.93, 0.97, 30, 60,  21, 0.033),
    # Best Calmar
    ('5% 45-75/x30 b3.3%',     0.93, 0.97, 45, 75,  30, 0.033),
    # Deep OTM alternatives
    ('28% 45-75/x30 b3.3%',    0.70, 0.74, 45, 75,  30, 0.033),
    ('35% 90-120/x30 b0.5%',   0.63, 0.67, 90, 120, 30, 0.005),
    ('35% 90-120/x30 b1%',     0.63, 0.67, 90, 120, 30, 0.01),
    # Medium OTM sweet spots
    ('15% 45-75/x30 b2%',      0.83, 0.87, 45, 75,  30, 0.02),
    ('10% 45-75/x30 b2%',      0.88, 0.92, 45, 75,  30, 0.02),
    ('20% 45-75/x30 b1%',      0.78, 0.82, 45, 75,  30, 0.01),
    # Simple default: recommended config
    ('20% 60-90/x30 b0.5%',    0.78, 0.82, 60, 90,  30, 0.005),
    ('28% 60-90/x30 b0.5%',    0.70, 0.74, 60, 90,  30, 0.005),
]

# Time periods to test
PERIODS = [
    ('Full 2008-2025', None, None),
    ('First half 2008-2016', '2008-01-01', '2016-12-31'),
    ('Second half 2016-2025', '2016-01-01', '2025-12-31'),
    ('Calm 2012-2018', '2012-01-01', '2018-12-31'),
    ('Post-GFC 2010-2019', '2010-01-01', '2019-12-31'),
    ('Recent 2019-2025', '2019-01-01', '2025-12-31'),
]

def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def filter_data_by_period(opts_pl, stocks_pl, start, end):
    """Filter Polars DataFrames to a date range."""
    if start is None and end is None:
        return opts_pl, stocks_pl

    # Options: filter on quotedate
    o = opts_pl
    s = stocks_pl
    if start:
        start_dt = pd.Timestamp(start)
        o = o.filter(pl.col('quotedate') >= start_dt)
        s = s.filter(pl.col('date') >= start_dt)
    if end:
        end_dt = pd.Timestamp(end)
        o = o.filter(pl.col('quotedate') <= end_dt)
        s = s.filter(pl.col('date') <= end_dt)
    return o, s

def compute_spy_annual(stocks_pl, start, end):
    """Compute SPY buy-and-hold annual return for a sub-period."""
    s = stocks_pl.filter(pl.col('symbol') == 'SPY').sort('date')
    if start:
        s = s.filter(pl.col('date') >= pd.Timestamp(start))
    if end:
        s = s.filter(pl.col('date') <= pd.Timestamp(end))
    prices = s['adjClose'].to_list()
    if len(prices) < 2:
        return 0.0
    total_ret = prices[-1] / prices[0]
    dates = s['date'].to_list()
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0:
        return 0.0
    annual = (total_ret ** (1/years) - 1) * 100
    return annual

def get_rebalance_dates(opts_pl, start=None, end=None):
    """Get business month start rebalance dates for a period."""
    dates_series = opts_pl['quotedate'].unique().sort()
    dates_pd = pd.DatetimeIndex(dates_series.to_list())
    if start:
        dates_pd = dates_pd[dates_pd >= pd.Timestamp(start)]
    if end:
        dates_pd = dates_pd[dates_pd <= pd.Timestamp(end)]

    dates_df = pd.DataFrame({'quotedate': dates_pd, 'volume': 1}).set_index('quotedate')
    rb_days = pd.to_datetime(
        dates_df.groupby(pd.Grouper(freq="1BMS"))
        .apply(lambda x: x.index.min())
        .values
    )
    return [int(d.value) for d in rb_days if not pd.isna(d)]

def run_period(opts_pl_full, stocks_pl_full, schema_mapping, period_name, start, end):
    """Run all top configs for a single period. Returns DataFrame."""
    print(f'\n--- {period_name} ---')

    # Filter data
    opts_pl, stocks_pl = filter_data_by_period(opts_pl_full, stocks_pl_full, start, end)
    spy_annual = compute_spy_annual(stocks_pl_full, start, end)
    rb_ns = get_rebalance_dates(opts_pl, start, end)

    print(f'  SPY annual: {spy_annual:.2f}%, rebalance dates: {len(rb_ns)}')

    # Build grid
    grid, labels, budgets = [], [], []
    for name, lo, hi, dte_min, dte_max, exit_dte, bp in TOP_CONFIGS:
        grid.append({
            "label": name,
            "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
            "leg_exit_filters": [exit_q(exit_dte)],
            "options_budget_pct": bp,
        })
        labels.append(name)
        budgets.append(bp)

    # Base config (leveraged: 100% stocks + puts on top)
    base = {
        "allocation": {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        "initial_capital": float(INITIAL_CAPITAL),
        "shares_per_contract": 100,
        "rebalance_dates": rb_ns,
        "legs": [{
            "name": "leg_1",
            "entry_filter": "(type == 'put') & (ask > 0) & (underlying == 'SPY')",
            "exit_filter": "(type == 'put') & (dte <= 14)",
            "direction": "ask", "type": "put",
            "entry_sort_col": "strike", "entry_sort_asc": True,
        }],
        "profit_pct": None, "loss_pct": None,
        "stocks": [("SPY", 1.0)],
        "options_budget_pct": 0.005,
        "check_exits_daily": True,
    }

    t0 = time.perf_counter()
    raw = parallel_sweep(opts_pl, stocks_pl, base, schema_mapping, grid, None)
    elapsed = time.perf_counter() - t0
    print(f'  {len(grid)} configs in {elapsed:.1f}s')

    rows = []
    for res, name, bp in zip(raw, labels, budgets):
        ann = res['annualized_return'] * 100
        rows.append({
            'Config': name,
            'Annual%': ann,
            'Excess%': ann - spy_annual,
            'MaxDD%': res['max_drawdown'] * 100,
            'Sharpe': res.get('sharpe_ratio', 0),
            'Calmar': res.get('calmar_ratio', 0),
            'Trades': res.get('total_trades', 0),
        })

    return pd.DataFrame(rows), spy_annual

def main():
    print('Loading data...')
    engine = init_engine()
    opts_pl = engine['opts_pl']
    stocks_pl = engine['stocks_pl']
    schema_mapping = engine['schema_mapping']

    all_results = {}

    for period_name, start, end in PERIODS:
        df, spy_ann = run_period(opts_pl, stocks_pl, schema_mapping, period_name, start, end)
        all_results[period_name] = (df, spy_ann)

    # =====================================================================
    # Display: config x period matrix
    # =====================================================================
    print('\n' + '='*120)
    print('ROBUSTNESS: Excess% across periods (leveraged)')
    print('='*120)

    # Build pivot: rows=config, cols=period
    config_names = [c[0] for c in TOP_CONFIGS]
    period_names = [p[0] for p in PERIODS]

    matrix = pd.DataFrame(index=config_names, columns=period_names, dtype=float)
    for period_name in period_names:
        df, _ = all_results[period_name]
        for _, row in df.iterrows():
            matrix.loc[row['Config'], period_name] = row['Excess%']

    # Add mean and min columns
    matrix['Mean'] = matrix[period_names].mean(axis=1)
    matrix['Min'] = matrix[period_names].min(axis=1)
    matrix['StdDev'] = matrix[period_names].std(axis=1)

    print(matrix.to_string(float_format='{:+.2f}'.format))

    # =====================================================================
    # Same for MaxDD
    # =====================================================================
    print('\n' + '='*120)
    print('ROBUSTNESS: MaxDD% across periods (leveraged)')
    print('='*120)

    dd_matrix = pd.DataFrame(index=config_names, columns=period_names, dtype=float)
    for period_name in period_names:
        df, _ = all_results[period_name]
        for _, row in df.iterrows():
            dd_matrix.loc[row['Config'], period_name] = row['MaxDD%']

    dd_matrix['Worst'] = dd_matrix[period_names].max(axis=1)  # max because DD is positive here
    print(dd_matrix.to_string(float_format='{:.1f}'.format))

    # =====================================================================
    # Same for Sharpe
    # =====================================================================
    print('\n' + '='*120)
    print('ROBUSTNESS: Sharpe across periods (leveraged)')
    print('='*120)

    sh_matrix = pd.DataFrame(index=config_names, columns=period_names, dtype=float)
    for period_name in period_names:
        df, _ = all_results[period_name]
        for _, row in df.iterrows():
            sh_matrix.loc[row['Config'], period_name] = row['Sharpe']

    sh_matrix['Mean'] = sh_matrix[period_names].mean(axis=1)
    sh_matrix['Min'] = sh_matrix[period_names].min(axis=1)
    print(sh_matrix.to_string(float_format='{:.2f}'.format))

    # =====================================================================
    # Summary: which configs beat SPY in ALL periods?
    # =====================================================================
    print('\n' + '='*120)
    print('SUMMARY: Configs that beat SPY (Excess% > 0) in ALL periods')
    print('='*120)
    for config in config_names:
        excess_vals = [matrix.loc[config, p] for p in period_names]
        all_positive = all(v > 0 for v in excess_vals)
        min_excess = min(excess_vals)
        mean_excess = np.mean(excess_vals)
        if all_positive:
            print(f'  [PASS] {config:35s}  min={min_excess:+.2f}%  mean={mean_excess:+.2f}%')
        else:
            neg_periods = [p for p, v in zip(period_names, excess_vals) if v <= 0]
            print(f'  [FAIL] {config:35s}  min={min_excess:+.2f}%  fails: {", ".join(neg_periods)}')

    # SPY baselines per period
    print('\n' + '='*120)
    print('SPY Buy & Hold annual returns per period')
    print('='*120)
    for period_name in period_names:
        _, spy_ann = all_results[period_name]
        print(f'  {period_name:30s}: {spy_ann:+.2f}%')

if __name__ == '__main__':
    main()
