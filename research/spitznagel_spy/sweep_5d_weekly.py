#!/usr/bin/env python3
"""Full 5D sweep with weekly rebalancing (faster put-profit reinvestment)."""

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
sys.path.insert(0, 'research/spitznagel_spy')
from sweep_parallel import init_engine

BATCH_SIZE = 300  # smaller batches since weekly = more work per config

OTM_LEVELS = [
    ('5%',   0.93, 0.97), ('10%',  0.88, 0.92), ('15%',  0.83, 0.87),
    ('18%',  0.80, 0.84), ('20%',  0.78, 0.82), ('23%',  0.75, 0.79),
    ('25%',  0.73, 0.77), ('28%',  0.70, 0.74), ('30%',  0.68, 0.72),
    ('35%',  0.63, 0.67), ('40%',  0.58, 0.62),
]
ENTRY_DTE = [
    ('30-60',30,60), ('45-75',45,75), ('60-90',60,90), ('75-105',75,105),
    ('90-120',90,120), ('120-180',120,180), ('150-210',150,210), ('180-270',180,270),
]
EXIT_DTES = [1, 7, 10, 14, 21, 25, 30, 45]
BUDGETS = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.033]

def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def build_grid():
    grid, keys = [], []
    for oname, lo, hi in OTM_LEVELS:
        for ename, dte_min, dte_max in ENTRY_DTE:
            for exit_dte in EXIT_DTES:
                if exit_dte >= dte_min:
                    continue
                for bp in BUDGETS:
                    grid.append({
                        "label": f'{oname} {ename}/x{exit_dte} b{bp*100:.1f}%',
                        "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
                        "leg_exit_filters": [exit_q(exit_dte)],
                        "options_budget_pct": bp,
                    })
                    keys.append((oname, f'{ename}/x{exit_dte}', bp))
    return grid, keys

def run_batched(engine, grid, keys, base, spy_annual, label=""):
    all_raw = []
    total = len(grid)
    t0 = time.perf_counter()
    for i in range(0, total, BATCH_SIZE):
        batch = grid[i:i+BATCH_SIZE]
        end = min(i + BATCH_SIZE, total)
        print(f'  {label} batch {i+1}-{end}/{total}...', end=' ', flush=True)
        bt0 = time.perf_counter()
        raw = parallel_sweep(engine['opts_pl'], engine['stocks_pl'], base,
                             engine['schema_mapping'], batch, None)
        all_raw.extend(raw)
        print(f'{time.perf_counter()-bt0:.1f}s')
    elapsed = time.perf_counter() - t0
    print(f'  {label} total: {total} configs in {elapsed:.1f}s ({total/elapsed:.1f} configs/sec)')

    rows = []
    for res, (otm, dte, bp) in zip(all_raw, keys):
        ann = res['annualized_return'] * 100
        rows.append({
            'OTM': otm, 'DTE': dte, 'Budget': bp * 100,
            'Annual%': ann, 'Excess%': ann - spy_annual,
            'MaxDD%': res['max_drawdown'] * 100,
            'Sharpe': res.get('sharpe_ratio', 0),
            'Calmar': res.get('calmar_ratio', 0),
            'Sortino': res.get('sortino_ratio', 0),
            'Trades': res.get('total_trades', 0),
        })
    return pd.DataFrame(rows)

def show_top(df, n=20, by='Calmar', title=''):
    top = df.sort_values(by, ascending=False).head(n)
    print(f'\n{"="*100}')
    print(f'Top {n} by {by} {title}')
    print(f'{"="*100}')
    print(top.to_string(index=False, float_format='{:.2f}'.format))

def show_best_per_budget(df, by='Calmar'):
    print(f'\n{"="*100}')
    print(f'Best config per budget (by {by})')
    print(f'{"="*100}')
    for budget in sorted(df['Budget'].unique()):
        sub = df[df['Budget'] == budget]
        best = sub.loc[sub[by].idxmax()]
        print(f'  Budget {budget:.2f}%: {best["OTM"]} {best["DTE"]} '
              f'Excess={best["Excess%"]:+.2f}% MaxDD={best["MaxDD%"]:.1f}% '
              f'Calmar={best["Calmar"]:.2f} Sharpe={best["Sharpe"]:.2f} Sortino={best["Sortino"]:.2f}')

def show_pivot(df, rows='OTM', cols='DTE', value='Calmar', title='', budget=None):
    sub = df if budget is None else df[abs(df['Budget'] - budget) < 0.01]
    piv = sub.pivot_table(index=rows, columns=cols, values=value, aggfunc='mean')
    otm_order = [o[0] for o in OTM_LEVELS]
    if rows == 'OTM':
        piv = piv.reindex([o for o in otm_order if o in piv.index])
    print(f'\n--- {title}: mean {value} ---')
    print(piv.to_string(float_format='{:.2f}'.format))

def analyze(df, title=''):
    show_top(df, 20, 'Calmar', title)
    show_top(df, 20, 'Sharpe', title)
    show_top(df, 20, 'Excess%', title)
    show_best_per_budget(df, 'Calmar')
    show_best_per_budget(df, 'Excess%')
    df2 = df.copy()
    df2['EntryDTE'] = df2['DTE'].str.split('/').str[0]
    df2['ExitDTE'] = df2['DTE'].str.split('/x').str[1]
    show_pivot(df2, 'OTM', 'EntryDTE', 'Calmar', f'{title} OTM x EntryDTE (all budgets avg)')
    show_pivot(df2, 'OTM', 'ExitDTE', 'Calmar', f'{title} OTM x ExitDTE (all budgets avg)')
    for bp in [0.5, 1.0, 2.0]:
        show_pivot(df2[abs(df2['Budget']-bp)<0.01], 'OTM', 'EntryDTE', 'Calmar',
                   f'{title} OTM x EntryDTE @ {bp}% budget')

if __name__ == '__main__':
    print('Loading data...')
    engine = init_engine()
    spy_annual = engine['spy_annual']

    # Weekly rebalance dates
    daily_dates = sorted(engine['opts_pl']['quotedate'].unique().to_list())
    weekly_ns = [int(d.value) for d in pd.DatetimeIndex(daily_dates[::5])]
    monthly_ns = engine['rb_date_ns']

    grid, keys = build_grid()
    print(f'\nTotal: {len(grid)} leveraged configs')

    # --- Monthly (baseline) ---
    print(f'\n{"="*100}')
    print(f'MONTHLY REBALANCING ({len(monthly_ns)} dates)')
    print(f'{"="*100}')
    base_m = {
        "allocation": {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        "initial_capital": float(INITIAL_CAPITAL),
        "shares_per_contract": 100,
        "rebalance_dates": monthly_ns,
        "legs": [{"name": "leg_1",
                  "entry_filter": "(type == 'put') & (ask > 0) & (underlying == 'SPY')",
                  "exit_filter": "(type == 'put') & (dte <= 14)",
                  "direction": "ask", "type": "put",
                  "entry_sort_col": "strike", "entry_sort_asc": True}],
        "profit_pct": None, "loss_pct": None,
        "stocks": [("SPY", 1.0)],
        "options_budget_pct": 0.005,
        "check_exits_daily": True,
    }
    df_monthly = run_batched(engine, grid, keys, base_m, spy_annual, label='MONTHLY')
    analyze(df_monthly, 'Monthly')

    # --- Weekly ---
    print(f'\n\n{"="*100}')
    print(f'WEEKLY REBALANCING ({len(weekly_ns)} dates)')
    print(f'{"="*100}')
    base_w = dict(base_m)
    base_w["rebalance_dates"] = weekly_ns
    df_weekly = run_batched(engine, grid, keys, base_w, spy_annual, label='WEEKLY')
    analyze(df_weekly, 'Weekly')

    # --- Comparison: top 10 monthly vs same configs weekly ---
    print(f'\n\n{"="*100}')
    print('MONTHLY vs WEEKLY: Top 20 monthly configs compared')
    print(f'{"="*100}')
    top_m = df_monthly.sort_values('Calmar', ascending=False).head(20)
    for _, row in top_m.iterrows():
        w = df_weekly[(df_weekly['OTM']==row['OTM']) & (df_weekly['DTE']==row['DTE']) & (abs(df_weekly['Budget']-row['Budget'])<0.01)]
        if len(w) > 0:
            wr = w.iloc[0]
            print(f'{row["OTM"]:>4s} {row["DTE"]:<12s} b{row["Budget"]:.1f}%  '
                  f'Monthly: Exc={row["Excess%"]:+.1f}% DD={row["MaxDD%"]:.0f}% Cal={row["Calmar"]:.2f}  |  '
                  f'Weekly: Exc={wr["Excess%"]:+.1f}% DD={wr["MaxDD%"]:.0f}% Cal={wr["Calmar"]:.2f}')

    # Save
    df_monthly.to_csv('research/spitznagel_spy/sweep_5d_monthly.csv', index=False)
    df_weekly.to_csv('research/spitznagel_spy/sweep_5d_weekly.csv', index=False)
    print('\nSaved: sweep_5d_monthly.csv, sweep_5d_weekly.csv')
