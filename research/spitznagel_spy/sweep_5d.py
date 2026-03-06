#!/usr/bin/env python3
"""Full 5-dimension strike-based sweep, batched to avoid OOM.

Dimensions: OTM level, entry DTE window, exit DTE, budget, leverage mode.
Batches configs in groups of ~500 per parallel_sweep call.
"""

import os, sys, warnings, time
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
import pyarrow as pa

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, INITIAL_CAPITAL
from options_portfolio_backtester._ob_rust import parallel_sweep

BATCH_SIZE = 500

# =====================================================================
# Grid dimensions (strike-based)
# =====================================================================
OTM_LEVELS = [
    ('5%',   0.93, 0.97),
    ('10%',  0.88, 0.92),
    ('15%',  0.83, 0.87),
    ('18%',  0.80, 0.84),
    ('20%',  0.78, 0.82),
    ('23%',  0.75, 0.79),
    ('25%',  0.73, 0.77),
    ('28%',  0.70, 0.74),
    ('30%',  0.68, 0.72),
    ('35%',  0.63, 0.67),
    ('40%',  0.58, 0.62),
]

ENTRY_DTE = [
    ('30-60',   30,  60),
    ('45-75',   45,  75),
    ('60-90',   60,  90),
    ('75-105',  75, 105),
    ('90-120',  90, 120),
    ('120-180', 120, 180),
    ('150-210', 150, 210),
    ('180-270', 180, 270),
]

EXIT_DTES = [1, 7, 10, 14, 21, 25, 30, 45]

BUDGETS = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.033]

# =====================================================================
# Helpers
# =====================================================================
def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def base_cfg(rb_ns, stock_pct, opt_pct, budget_pct):
    return {
        "allocation": {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
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
        "options_budget_pct": budget_pct,
        "check_exits_daily": True,
    }

def init_engine():
    data = load_data()
    opts = data['options_data']
    stocks = data['stocks_data']
    oschema = opts.schema
    sschema = stocks.schema

    dates_df = (
        pd.DataFrame(opts._data[["quotedate", "volume"]])
        .drop_duplicates("quotedate")
        .set_index("quotedate")
    )
    rb_days = pd.to_datetime(
        dates_df.groupby(pd.Grouper(freq="1BMS"))
        .apply(lambda x: x.index.min())
        .values
    )

    return {
        'opts_pl': pl.from_arrow(pa.Table.from_pandas(opts._data, preserve_index=False)),
        'stocks_pl': pl.from_arrow(pa.Table.from_pandas(stocks._data, preserve_index=False)),
        'schema_mapping': {
            "contract": oschema["contract"], "date": oschema["date"],
            "stocks_date": sschema["date"], "stocks_symbol": sschema["symbol"],
            "stocks_price": sschema["adjClose"], "underlying": oschema["underlying"],
            "expiration": oschema["expiration"], "type": oschema["type"],
            "strike": oschema["strike"],
        },
        'rb_date_ns': [int(d.value) for d in rb_days if not pd.isna(d)],
        'spy_annual': data['spy_annual_ret'],
        'spy_dd': data['spy_dd'],
    }

def build_grid():
    """Build full grid + keys, skipping impossible combos (exit >= entry_min)."""
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

def run_batched(engine, grid, keys, base, label=""):
    """Run grid in batches of BATCH_SIZE, return parsed DataFrame."""
    spy_annual = engine['spy_annual']
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
            'Trades': res.get('total_trades', 0),
        })
    return pd.DataFrame(rows)

# =====================================================================
# Analysis & display
# =====================================================================
def show_top(df, n=20, by='Excess%', title=''):
    top = df.sort_values(by, ascending=False).head(n)
    print(f'\n{"="*80}')
    print(f'Top {n} by {by} {title}')
    print(f'{"="*80}')
    print(top.to_string(index=False, float_format='{:.2f}'.format))

def show_pivot(df, rows='OTM', cols='DTE', value='Excess%', title='', budget=None):
    sub = df if budget is None else df[df['Budget'] == budget]
    # Average across other dimensions
    piv = sub.pivot_table(index=rows, columns=cols, values=value, aggfunc='mean')
    # Sort OTM by numeric value
    otm_order = [o[0] for o in OTM_LEVELS]
    if rows == 'OTM':
        piv = piv.reindex([o for o in otm_order if o in piv.index])
    print(f'\n--- {title}: mean {value} ---')
    print(piv.to_string(float_format='{:+.2f}'.format))

def show_best_per_budget(df):
    print(f'\n{"="*80}')
    print('Best config per budget level')
    print(f'{"="*80}')
    for budget in sorted(df['Budget'].unique()):
        sub = df[df['Budget'] == budget]
        best = sub.loc[sub['Excess%'].idxmax()]
        print(f'  Budget {budget:.2f}%: {best["OTM"]} {best["DTE"]} '
              f'Excess={best["Excess%"]:+.2f}% MaxDD={best["MaxDD%"]:.2f}% '
              f'Sharpe={best["Sharpe"]:.2f}')

def analyze(df, title=''):
    show_top(df, 20, 'Excess%', title)
    show_top(df, 20, 'Sharpe', title)
    show_top(df, 20, 'Calmar', title)
    show_best_per_budget(df)
    # Pivot: OTM x Entry DTE (averaged across exit DTE and budget)
    # Split DTE into entry and exit for pivots
    df2 = df.copy()
    df2['EntryDTE'] = df2['DTE'].str.split('/').str[0]
    df2['ExitDTE'] = df2['DTE'].str.split('/x').str[1]
    show_pivot(df2, 'OTM', 'EntryDTE', 'Excess%', f'{title} OTM x EntryDTE (all budgets avg)')
    show_pivot(df2, 'OTM', 'ExitDTE', 'Excess%', f'{title} OTM x ExitDTE (all budgets avg)')
    # Per-budget pivots for 1% and 3.3%
    for bp in [1.0, 3.3]:
        if bp in df['Budget'].unique():
            show_pivot(df2[df2['Budget']==bp], 'OTM', 'EntryDTE', 'Excess%',
                       f'{title} OTM x EntryDTE @ {bp}% budget')

# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print('Loading data...')
    engine = init_engine()

    grid, keys = build_grid()
    print(f'\nTotal configs: {len(grid)} leveraged + {len(grid)} no-leverage = {len(grid)*2}')

    # --- Leveraged ---
    print(f'\n{"="*80}')
    print('LEVERAGED (100% stocks + puts on top)')
    print(f'{"="*80}')
    base_lev = base_cfg(engine['rb_date_ns'], 1.0, 0.0, BUDGETS[0])
    df_lev = run_batched(engine, grid, keys, base_lev, label='LEV')
    analyze(df_lev, 'Leveraged')

    # --- No-leverage: batch per budget ---
    print(f'\n\n{"="*80}')
    print('NO-LEVERAGE (stock + opt = 100%)')
    print(f'{"="*80}')
    nolev_frames = []
    # Build grid without budget dimension (budget set in base)
    for bp in BUDGETS:
        nl_grid, nl_keys = [], []
        for oname, lo, hi in OTM_LEVELS:
            for ename, dte_min, dte_max in ENTRY_DTE:
                for exit_dte in EXIT_DTES:
                    if exit_dte >= dte_min:
                        continue
                    nl_grid.append({
                        "label": f'{oname} {ename}/x{exit_dte} nolev {bp*100:.1f}%',
                        "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
                        "leg_exit_filters": [exit_q(exit_dte)],
                    })
                    nl_keys.append((oname, f'{ename}/x{exit_dte}', bp))

        base_nl = base_cfg(engine['rb_date_ns'], 1.0 - bp, bp, None)
        df_nl = run_batched(engine, nl_grid, nl_keys, base_nl,
                            label=f'NOLEV {bp*100:.1f}%')
        nolev_frames.append(df_nl)

    df_nolev = pd.concat(nolev_frames, ignore_index=True)
    analyze(df_nolev, 'No-leverage')

    # Save to CSV
    df_lev.to_csv('research/spitznagel_spy/sweep_5d_leveraged.csv', index=False)
    df_nolev.to_csv('research/spitznagel_spy/sweep_5d_noleveraged.csv', index=False)
    print('\nSaved: sweep_5d_leveraged.csv, sweep_5d_noleveraged.csv')
