#!/usr/bin/env python3
"""Parallel put-hedge grid sweep using Rust Rayon.

Importable module + CLI. ~36 configs/sec on Apple Silicon.

Usage as library:
    from sweep_parallel import init_engine, run_sweep, print_pivot
    engine = init_engine()
    df = run_sweep(engine, leveraged=True)
    print_pivot(df, engine)

Usage as CLI:
    python sweep_parallel.py                # full default grid
    python sweep_parallel.py --lev          # leveraged only
    python sweep_parallel.py --nolev        # no-leverage only
"""

import os, sys, warnings, time, argparse
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
import pyarrow as pa

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
os.chdir(PROJECT_ROOT)

from backtest_runner import load_data, INITIAL_CAPITAL
from options_portfolio_backtester._ob_rust import parallel_sweep

# =====================================================================
# Default grid dimensions
# =====================================================================
DEFAULT_OTM = [
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

DEFAULT_DTE = [
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

DEFAULT_BUDGETS = [0.005, 0.01, 0.02, 0.033]

# =====================================================================
# Engine init (expensive — call once, reuse)
# =====================================================================
def init_engine(data=None):
    """Load data, build Polars frames & schema mapping. Returns engine dict."""
    if data is None:
        data = load_data()

    opts = data['options_data']
    stocks = data['stocks_data']
    oschema = opts.schema
    sschema = stocks.schema

    # Rebalance dates (business month start)
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
        'data': data,
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

# =====================================================================
# Internal helpers
# =====================================================================
def _entry_q(dmin, dmax, dte_min, dte_max):
    return (f"((type == 'put') & (ask > 0)) & (((((underlying == 'SPY')"
            f" & (dte >= {dte_min})) & (dte <= {dte_max}))"
            f" & (delta >= {dmin})) & (delta <= {dmax}))")

def _exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def _base_cfg(rb_ns, stock_pct, opt_pct, budget_pct):
    return {
        "allocation": {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        "initial_capital": float(INITIAL_CAPITAL),
        "shares_per_contract": 100,
        "rebalance_dates": rb_ns,
        "legs": [{
            "name": "leg_1",
            "entry_filter": "((type == 'put') & (ask > 0)) & (underlying == 'SPY')",
            "exit_filter": "(type == 'put') & (dte <= 14)",
            "direction": "ask", "type": "put",
            "entry_sort_col": "delta", "entry_sort_asc": True,
        }],
        "profit_pct": None, "loss_pct": None,
        "stocks": [("SPY", 1.0)],
        "options_budget_pct": budget_pct,
        "check_exits_daily": True,
    }

def _parse_results(raw, keys, spy_annual):
    rows = []
    for res, (otm, dte, bp) in zip(raw, keys):
        ann = res['annualized_return'] * 100
        rows.append({
            'OTM': otm, 'DTE': dte, 'Budget': bp * 100,
            'Annual%': ann, 'Excess%': ann - spy_annual,
            'MaxDD%': res['max_drawdown'] * 100,
            'Trades': res.get('total_trades', 0),
        })
    return pd.DataFrame(rows)

# =====================================================================
# Core sweep
# =====================================================================
def run_sweep(engine, otm=None, dte=None, budgets=None, leveraged=False, verbose=True):
    """Run parallel grid sweep. Returns DataFrame with OTM, DTE, Budget, Annual%, Excess%, MaxDD%, Trades."""
    otm = otm or DEFAULT_OTM
    dte = dte or DEFAULT_DTE
    budgets = budgets or DEFAULT_BUDGETS
    spy_annual = engine['spy_annual']
    rb_ns = engine['rb_date_ns']

    if leveraged:
        grid, keys = [], []
        for oname, dmin, dmax in otm:
            for dname, dmin_dte, dmax_dte, exit_dte in dte:
                for bp in budgets:
                    grid.append({
                        "label": f'{oname} {dname} b{bp*100:.1f}%',
                        "leg_entry_filters": [_entry_q(dmin, dmax, dmin_dte, dmax_dte)],
                        "leg_exit_filters": [_exit_q(exit_dte)],
                        "options_budget_pct": bp,
                    })
                    keys.append((oname, dname, bp))

        base = _base_cfg(rb_ns, 1.0, 0.0, budgets[0])
        if verbose:
            print(f'Leveraged: {len(grid)} configs...', end=' ', flush=True)
        t0 = time.perf_counter()
        raw = parallel_sweep(engine['opts_pl'], engine['stocks_pl'], base,
                             engine['schema_mapping'], grid, None)
        if verbose:
            print(f'{time.perf_counter()-t0:.1f}s')
        return _parse_results(raw, keys, spy_annual)

    # No-leverage: one sweep per budget (different base allocation)
    frames = []
    for bp in budgets:
        base = _base_cfg(rb_ns, 1.0 - bp, bp, None)
        grid, keys = [], []
        for oname, dmin, dmax in otm:
            for dname, dmin_dte, dmax_dte, exit_dte in dte:
                grid.append({
                    "label": f'{oname} {dname} nolev {bp*100:.1f}%',
                    "leg_entry_filters": [_entry_q(dmin, dmax, dmin_dte, dmax_dte)],
                    "leg_exit_filters": [_exit_q(exit_dte)],
                })
                keys.append((oname, dname, bp))

        if verbose:
            print(f'No-leverage {bp*100:.1f}%: {len(grid)} configs...', end=' ', flush=True)
        t0 = time.perf_counter()
        raw = parallel_sweep(engine['opts_pl'], engine['stocks_pl'], base,
                             engine['schema_mapping'], grid, None)
        if verbose:
            print(f'{time.perf_counter()-t0:.1f}s')
        frames.append(_parse_results(raw, keys, spy_annual))

    return pd.concat(frames, ignore_index=True)

# =====================================================================
# Display
# =====================================================================
def print_pivot(df, engine=None, otm=None, value='Excess%', title=''):
    """Print OTM x DTE pivot for each budget level."""
    otm = otm or DEFAULT_OTM
    order = [o[0] for o in otm]
    for budget in sorted(df['Budget'].unique()):
        sub = df[df['Budget'] == budget]
        piv = sub.pivot_table(index='DTE', columns='OTM', values=value)
        piv = piv[[c for c in order if c in piv.columns]]
        lbl = f'{title} {budget:.1f}%' if title else f'Budget {budget:.1f}%'
        print(f'\n--- {lbl}: {value} ---')
        print(piv.to_string(float_format='{:+.2f}'.format))
        best = sub.loc[sub[value].idxmax()]
        print(f'Best: {best["OTM"]} {best["DTE"]} = {best[value]:+.2f}%')

def print_top(df, n=10, by='Excess%'):
    """Print top N configs."""
    top = df.sort_values(by, ascending=False).head(n)
    print(f'\nTop {n} by {by}:')
    print(top.to_string(index=False, float_format='{:.2f}'.format))

# =====================================================================
# CLI
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel put-hedge sweep')
    parser.add_argument('--lev', action='store_true', help='Leveraged only')
    parser.add_argument('--nolev', action='store_true', help='No-leverage only')
    args = parser.parse_args()

    do_lev = args.lev
    do_nolev = args.nolev or not args.lev

    engine = init_engine()

    if do_lev:
        print('='*100)
        print('LEVERAGED SWEEP (100% stocks + puts on top)')
        print('='*100)
        df_lev = run_sweep(engine, leveraged=True)
        print_pivot(df_lev, title='Leveraged')
        print_top(df_lev)

    if do_nolev:
        print('\n' + '='*100)
        print('NO-LEVERAGE SWEEP (stock + opt = 100%)')
        print('='*100)
        df_nolev = run_sweep(engine, leveraged=False)
        print_pivot(df_nolev, title='No-leverage')
        print_top(df_nolev)
