#!/usr/bin/env python3
"""Head-to-head: same budget, different OTM levels, across all periods."""

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

BUDGETS = [0.005, 0.01, 0.02, 0.033]

OTM_LEVELS = [
    ('5%',   0.93, 0.97),
    ('10%',  0.88, 0.92),
    ('15%',  0.83, 0.87),
    ('18%',  0.80, 0.84),
    ('20%',  0.78, 0.82),
    ('25%',  0.73, 0.77),
    ('28%',  0.70, 0.74),
    ('30%',  0.68, 0.72),
    ('35%',  0.63, 0.67),
    ('40%',  0.58, 0.62),
]

# Best DTE per OTM from 5D sweep: use 45-75/x30 as a good general choice,
# and 90-120/x30 for deep OTM
DTE_CONFIGS = [
    ('30-60/x25',  30,  60, 25),
    ('45-75/x30',  45,  75, 30),
    ('60-90/x30',  60,  90, 30),
    ('90-120/x30', 90, 120, 30),
]

PERIODS = [
    ('Full',       None, None),
    ('2008-2016',  '2008-01-01', '2016-12-31'),
    ('2016-2025',  '2016-01-01', '2025-12-31'),
    ('Calm 12-18', '2012-01-01', '2018-12-31'),
    ('2010-2019',  '2010-01-01', '2019-12-31'),
    ('2019-2025',  '2019-01-01', '2025-12-31'),
]

def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def filter_data(opts_pl, stocks_pl, start, end):
    if start is None and end is None:
        return opts_pl, stocks_pl
    o, s = opts_pl, stocks_pl
    if start:
        dt = pd.Timestamp(start)
        o = o.filter(pl.col('quotedate') >= dt)
        s = s.filter(pl.col('date') >= dt)
    if end:
        dt = pd.Timestamp(end)
        o = o.filter(pl.col('quotedate') <= dt)
        s = s.filter(pl.col('date') <= dt)
    return o, s

def spy_annual(stocks_pl, start, end):
    s = stocks_pl.filter(pl.col('symbol') == 'SPY').sort('date')
    if start: s = s.filter(pl.col('date') >= pd.Timestamp(start))
    if end: s = s.filter(pl.col('date') <= pd.Timestamp(end))
    prices = s['adjClose'].to_list()
    if len(prices) < 2: return 0.0
    dates = s['date'].to_list()
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0: return 0.0
    return ((prices[-1] / prices[0]) ** (1/years) - 1) * 100

def get_rb_dates(opts_pl, start=None, end=None):
    dates = pd.DatetimeIndex(opts_pl['quotedate'].unique().sort().to_list())
    if start: dates = dates[dates >= pd.Timestamp(start)]
    if end: dates = dates[dates <= pd.Timestamp(end)]
    df = pd.DataFrame({'q': dates, 'v': 1}).set_index('q')
    rb = pd.to_datetime(df.groupby(pd.Grouper(freq="1BMS")).apply(lambda x: x.index.min()).values)
    return [int(d.value) for d in rb if not pd.isna(d)]

def main():
    print('Loading data...')
    engine = init_engine()
    opts_pl = engine['opts_pl']
    stocks_pl = engine['stocks_pl']
    schema = engine['schema_mapping']

    # Build full grid: OTM x DTE x Budget (leveraged)
    grid_info = []  # (otm_name, dte_name, budget)
    grid_dicts = []
    for oname, lo, hi in OTM_LEVELS:
        for dname, dte_min, dte_max, exit_dte in DTE_CONFIGS:
            for bp in BUDGETS:
                grid_dicts.append({
                    "label": f'{oname} {dname} b{bp*100:.1f}%',
                    "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
                    "leg_exit_filters": [exit_q(exit_dte)],
                    "options_budget_pct": bp,
                })
                grid_info.append((oname, dname, bp))

    print(f'Grid: {len(grid_dicts)} configs x {len(PERIODS)} periods')

    # Run each period
    period_results = {}
    for pname, start, end in PERIODS:
        print(f'\n  {pname}...', end=' ', flush=True)
        o, s = filter_data(opts_pl, stocks_pl, start, end)
        spy_ann = spy_annual(stocks_pl, start, end)
        rb_ns = get_rb_dates(o, start, end)

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
        raw = parallel_sweep(o, s, base, schema, grid_dicts, None)
        print(f'{time.perf_counter()-t0:.1f}s')

        rows = []
        for res, (oname, dname, bp) in zip(raw, grid_info):
            ann = res['annualized_return'] * 100
            rows.append({
                'OTM': oname, 'DTE': dname, 'Budget': bp * 100,
                'Annual%': ann, 'Excess%': ann - spy_ann,
                'MaxDD%': res['max_drawdown'] * 100,
                'Sharpe': res.get('sharpe_ratio', 0),
            })
        period_results[pname] = pd.DataFrame(rows)

    # =====================================================================
    # Analysis: For each budget, show OTM x Period pivot (best DTE per OTM)
    # =====================================================================
    period_names = [p[0] for p in PERIODS]
    otm_names = [o[0] for o in OTM_LEVELS]

    for bp in BUDGETS:
        print(f'\n{"="*100}')
        print(f'Budget {bp*100:.1f}% - Best DTE per OTM - Excess% across periods')
        print(f'{"="*100}')

        matrix = pd.DataFrame(index=otm_names, columns=period_names, dtype=float)
        dte_chosen = pd.DataFrame(index=otm_names, columns=period_names, dtype=object)

        for pname in period_names:
            df = period_results[pname]
            sub = df[df['Budget'] == bp * 100]
            for oname in otm_names:
                osub = sub[sub['OTM'] == oname]
                if len(osub) == 0:
                    continue
                # Best DTE for this OTM in full period (to avoid per-period optimization)
                best_idx = osub['Excess%'].idxmax()
                matrix.loc[oname, pname] = osub.loc[best_idx, 'Excess%']
                dte_chosen.loc[oname, pname] = osub.loc[best_idx, 'DTE']

        matrix['Mean'] = matrix[period_names].mean(axis=1)
        matrix['Min'] = matrix[period_names].min(axis=1)
        matrix['StdDev'] = matrix[period_names].std(axis=1)
        print(matrix.to_string(float_format='{:+.2f}'.format))

    # =====================================================================
    # Same but fix DTE to avoid per-period cheating: use full-period best DTE
    # =====================================================================
    print(f'\n\n{"="*100}')
    print(f'FIXED DTE (chosen from full period only) - Excess% across periods')
    print(f'{"="*100}')

    full_df = period_results['Full']

    for bp in BUDGETS:
        print(f'\n--- Budget {bp*100:.1f}% ---')
        sub_full = full_df[full_df['Budget'] == bp * 100]

        # Find best DTE per OTM on full period
        best_dte = {}
        for oname in otm_names:
            osub = sub_full[sub_full['OTM'] == oname]
            if len(osub) > 0:
                best_idx = osub['Excess%'].idxmax()
                best_dte[oname] = osub.loc[best_idx, 'DTE']

        matrix = pd.DataFrame(index=otm_names, columns=period_names + ['Mean', 'Min', 'StdDev', 'BestDTE'], dtype=object)

        for oname in otm_names:
            if oname not in best_dte:
                continue
            dte = best_dte[oname]
            matrix.loc[oname, 'BestDTE'] = dte
            vals = []
            for pname in period_names:
                df = period_results[pname]
                sub = df[(df['Budget'] == bp * 100) & (df['OTM'] == oname) & (df['DTE'] == dte)]
                if len(sub) > 0:
                    v = sub.iloc[0]['Excess%']
                    matrix.loc[oname, pname] = f'{v:+.2f}'
                    vals.append(v)
            if vals:
                matrix.loc[oname, 'Mean'] = f'{np.mean(vals):+.2f}'
                matrix.loc[oname, 'Min'] = f'{min(vals):+.2f}'
                matrix.loc[oname, 'StdDev'] = f'{np.std(vals):.2f}'

        print(matrix.to_string())

    # =====================================================================
    # MaxDD comparison (fixed DTE from full period)
    # =====================================================================
    print(f'\n\n{"="*100}')
    print(f'FIXED DTE - MaxDD% across periods')
    print(f'{"="*100}')

    for bp in BUDGETS:
        print(f'\n--- Budget {bp*100:.1f}% ---')
        sub_full = full_df[full_df['Budget'] == bp * 100]
        best_dte = {}
        for oname in otm_names:
            osub = sub_full[sub_full['OTM'] == oname]
            if len(osub) > 0:
                best_idx = osub['Excess%'].idxmax()
                best_dte[oname] = osub.loc[best_idx, 'DTE']

        matrix = pd.DataFrame(index=otm_names, columns=period_names + ['Worst', 'BestDTE'], dtype=object)
        for oname in otm_names:
            if oname not in best_dte:
                continue
            dte = best_dte[oname]
            matrix.loc[oname, 'BestDTE'] = dte
            vals = []
            for pname in period_names:
                df = period_results[pname]
                sub = df[(df['Budget'] == bp * 100) & (df['OTM'] == oname) & (df['DTE'] == dte)]
                if len(sub) > 0:
                    v = sub.iloc[0]['MaxDD%']
                    matrix.loc[oname, pname] = f'{v:.1f}'
                    vals.append(v)
            if vals:
                matrix.loc[oname, 'Worst'] = f'{max(vals):.1f}'
        print(matrix.to_string())

if __name__ == '__main__':
    main()
