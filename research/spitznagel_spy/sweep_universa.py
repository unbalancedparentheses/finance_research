#!/usr/bin/env python3
"""Universa-style sweep: annual budget split across rebalances.

Models the actual Universa approach:
- 3.33% of portfolio per YEAR allocated to tail puts
- Spent in equal installments at each rebalance
- 100% stocks + puts on top (leveraged overlay)
- Put profits reinvested into stocks at next rebalance

Tests monthly (12x/yr) and bimonthly (6x/yr) rebalancing.
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
sys.path.insert(0, 'research/spitznagel_spy')
from sweep_parallel import init_engine

BATCH_SIZE = 500

# Annual budgets to test (% of portfolio per year)
ANNUAL_BUDGETS = [0.02, 0.0333, 0.05]

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

def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def build_grid(per_rebalance_budgets):
    """Build grid with per-rebalance budgets (already divided by frequency)."""
    grid, keys = [], []
    for oname, lo, hi in OTM_LEVELS:
        for ename, dte_min, dte_max in ENTRY_DTE:
            for exit_dte in EXIT_DTES:
                if exit_dte >= dte_min:
                    continue
                for bp in per_rebalance_budgets:
                    grid.append({
                        "label": f'{oname} {ename}/x{exit_dte} b{bp*100:.3f}%',
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
    return all_raw

def show_top(df, n=20, by='Calmar', title=''):
    top = df.sort_values(by, ascending=False).head(n)
    print(f'\n{"="*110}')
    print(f'Top {n} by {by} {title}')
    print(f'{"="*110}')
    print(top.to_string(index=False, float_format='{:.2f}'.format))

def show_best_per_budget(df, by='Calmar'):
    print(f'\n{"="*110}')
    print(f'Best config per annual budget (by {by})')
    print(f'{"="*110}')
    for budget in sorted(df['AnnualBudget%'].unique()):
        sub = df[df['AnnualBudget%'] == budget]
        best = sub.loc[sub[by].idxmax()]
        print(f'  Annual {budget:.2f}%: {best["OTM"]} {best["DTE"]} '
              f'Excess={best["Excess%"]:+.2f}% MaxDD={best["MaxDD%"]:.1f}% '
              f'Calmar={best["Calmar"]:.2f} Sharpe={best["Sharpe"]:.2f} Sortino={best["Sortino"]:.2f}')

def show_pivot(df, rows='OTM', cols='DTE', value='Calmar', title='', ann_budget=None):
    sub = df if ann_budget is None else df[abs(df['AnnualBudget%'] - ann_budget) < 0.01]
    if len(sub) == 0:
        return
    piv = sub.pivot_table(index=rows, columns=cols, values=value, aggfunc='mean')
    otm_order = [o[0] for o in OTM_LEVELS]
    if rows == 'OTM':
        piv = piv.reindex([o for o in otm_order if o in piv.index])
    print(f'\n--- {title}: mean {value} ---')
    print(piv.to_string(float_format='{:.2f}'.format))

def analyze(df, title=''):
    show_top(df, 20, 'Calmar', title)
    show_top(df, 20, 'Excess%', title)
    show_best_per_budget(df, 'Calmar')
    show_best_per_budget(df, 'Excess%')
    df2 = df.copy()
    df2['EntryDTE'] = df2['DTE'].str.split('/').str[0]
    df2['ExitDTE'] = df2['DTE'].str.split('/x').str[1]
    for ab in [2.0, 3.33, 5.0]:
        show_pivot(df2[abs(df2['AnnualBudget%']-ab)<0.1], 'OTM', 'EntryDTE', 'Calmar',
                   f'{title} OTM x EntryDTE @ {ab}% annual budget')

def get_bimonthly_dates(daily_dates):
    """Every ~2 months: pick every other monthly rebalance date."""
    df = pd.DataFrame({'q': daily_dates, 'v': 1}).set_index('q')
    monthly = pd.to_datetime(
        df.groupby(pd.Grouper(freq="1BMS")).apply(lambda x: x.index.min()).values
    )
    monthly = [d for d in monthly if not pd.isna(d)]
    return monthly[::2]  # every other month

if __name__ == '__main__':
    print('Loading data...')
    engine = init_engine()
    spy_annual = engine['spy_annual']

    daily_dates = sorted(engine['opts_pl']['quotedate'].unique().to_list())
    monthly_dates = pd.DatetimeIndex(engine['rb_date_ns']).tolist()
    # Reconstruct monthly dates from ns
    monthly_rb = [pd.Timestamp(ns, unit='ns') for ns in engine['rb_date_ns']]
    bimonthly_rb = monthly_rb[::2]

    monthly_ns = engine['rb_date_ns']
    bimonthly_ns = [int(d.value) for d in bimonthly_rb]

    freqs = [
        ('Monthly (12x/yr)', monthly_ns, 12),
        ('Bimonthly (6x/yr)', bimonthly_ns, 6),
    ]

    for freq_name, rb_ns, rebal_per_year in freqs:
        print(f'\n\n{"="*110}')
        print(f'{freq_name} REBALANCING')
        print(f'{"="*110}')

        # Convert annual budgets to per-rebalance
        per_rebal = [ab / rebal_per_year for ab in ANNUAL_BUDGETS]
        grid, keys = build_grid(per_rebal)
        print(f'Grid: {len(grid)} configs (annual budgets: {[f"{ab*100:.2f}%" for ab in ANNUAL_BUDGETS]})')
        print(f'Per-rebalance budgets: {[f"{p*100:.3f}%" for p in per_rebal]}')

        base = {
            "allocation": {"stocks": 1.0, "options": 0.0, "cash": 0.0},
            "initial_capital": float(INITIAL_CAPITAL),
            "shares_per_contract": 100,
            "rebalance_dates": rb_ns,
            "legs": [{"name": "leg_1",
                      "entry_filter": "(type == 'put') & (ask > 0) & (underlying == 'SPY')",
                      "exit_filter": "(type == 'put') & (dte <= 14)",
                      "direction": "ask", "type": "put",
                      "entry_sort_col": "strike", "entry_sort_asc": True}],
            "profit_pct": None, "loss_pct": None,
            "stocks": [("SPY", 1.0)],
            "options_budget_pct": per_rebal[0],
            "check_exits_daily": True,
        }

        all_raw = run_batched(engine, grid, keys, base, spy_annual, label=freq_name)

        # Map per-rebalance budget back to annual
        rows = []
        for res, (otm, dte, bp_per_rebal) in zip(all_raw, keys):
            ann_budget = bp_per_rebal * rebal_per_year
            ann = res['annualized_return'] * 100
            rows.append({
                'OTM': otm, 'DTE': dte,
                'AnnualBudget%': round(ann_budget * 100, 2),
                'PerRebal%': round(bp_per_rebal * 100, 3),
                'Annual%': ann, 'Excess%': ann - spy_annual,
                'MaxDD%': res['max_drawdown'] * 100,
                'Sharpe': res.get('sharpe_ratio', 0),
                'Calmar': res.get('calmar_ratio', 0),
                'Sortino': res.get('sortino_ratio', 0),
                'Trades': res.get('total_trades', 0),
            })
        df = pd.DataFrame(rows)
        analyze(df, freq_name)

        # Save
        tag = freq_name.split('(')[0].strip().lower()
        df.to_csv(f'research/spitznagel_spy/sweep_universa_{tag}.csv', index=False)
        print(f'\nSaved: sweep_universa_{tag}.csv')

    # --- Cross-frequency comparison ---
    print(f'\n\n{"="*110}')
    print('MONTHLY vs BIMONTHLY: Top 20 monthly configs (by Calmar)')
    print(f'{"="*110}')

    df_m = pd.read_csv('research/spitznagel_spy/sweep_universa_monthly.csv')
    df_b = pd.read_csv('research/spitznagel_spy/sweep_universa_bimonthly.csv')

    top_m = df_m.sort_values('Calmar', ascending=False).head(20)
    for _, row in top_m.iterrows():
        w = df_b[(df_b['OTM']==row['OTM']) & (df_b['DTE']==row['DTE'])
                 & (abs(df_b['AnnualBudget%']-row['AnnualBudget%'])<0.01)]
        if len(w) > 0:
            wr = w.iloc[0]
            print(f'{row["OTM"]:>4s} {row["DTE"]:<12s} {row["AnnualBudget%"]:.1f}%/yr  '
                  f'Mo: Exc={row["Excess%"]:+.1f}% DD={row["MaxDD%"]:.0f}% Cal={row["Calmar"]:.2f}  |  '
                  f'Bi: Exc={wr["Excess%"]:+.1f}% DD={wr["MaxDD%"]:.0f}% Cal={wr["Calmar"]:.2f}')
