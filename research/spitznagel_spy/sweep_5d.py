#!/usr/bin/env python3
"""Full 6-dimension strike-based sweep, batched to avoid OOM.

Dimensions: OTM level, entry DTE window, exit DTE, annual budget,
rebalance frequency, leverage mode.
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

ANNUAL_BUDGETS = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.033, 0.04, 0.05, 0.065, 0.08, 0.10, 0.15]

REBAL_FREQS = [
    ('monthly',   1, 12),   # (name, slice_step, rebalances_per_year)
    ('bimonthly', 2, 6),
    ('quarterly', 3, 4),
]

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

def build_grid(per_rebalance_budgets):
    """Build full grid + keys, skipping impossible combos (exit >= entry_min)."""
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

def run_batched(engine, grid, keys, base, freq_name, rebal_per_year, label=""):
    """Run grid in batches, return DataFrame with Freq and AnnualBudget% columns."""
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
    for res, (otm, dte, bp_per_rebal) in zip(all_raw, keys):
        ann = res['annualized_return'] * 100
        ann_budget = bp_per_rebal * rebal_per_year
        rows.append({
            'Freq': freq_name, 'OTM': otm, 'DTE': dte,
            'AnnualBudget%': round(ann_budget * 100, 2),
            'PerRebal%': round(bp_per_rebal * 100, 3),
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
    print(f'\n{"="*110}')
    print(f'Top {n} by {by} {title}')
    print(f'{"="*110}')
    print(top.to_string(index=False, float_format='{:.2f}'.format))

def show_pivot(df, rows='OTM', cols='DTE', value='Excess%', title='', ann_budget=None):
    sub = df if ann_budget is None else df[abs(df['AnnualBudget%'] - ann_budget) < 0.01]
    if len(sub) == 0:
        return
    piv = sub.pivot_table(index=rows, columns=cols, values=value, aggfunc='mean')
    otm_order = [o[0] for o in OTM_LEVELS]
    if rows == 'OTM':
        piv = piv.reindex([o for o in otm_order if o in piv.index])
    print(f'\n--- {title}: mean {value} ---')
    print(piv.to_string(float_format='{:+.2f}'.format))

def show_best_per_budget(df, by='Excess%'):
    print(f'\n{"="*110}')
    print(f'Best config per annual budget (by {by})')
    print(f'{"="*110}')
    for budget in sorted(df['AnnualBudget%'].unique()):
        sub = df[df['AnnualBudget%'] == budget]
        best = sub.loc[sub[by].idxmax()]
        freq = best.get('Freq', '')
        print(f'  Annual {budget:.2f}%: {freq} {best["OTM"]} {best["DTE"]} '
              f'Excess={best["Excess%"]:+.2f}% MaxDD={best["MaxDD%"]:.1f}% '
              f'Calmar={best["Calmar"]:.2f} Sharpe={best["Sharpe"]:.2f}')

def analyze(df, title=''):
    show_top(df, 20, 'Excess%', title)
    show_top(df, 20, 'Sharpe', title)
    show_top(df, 20, 'Calmar', title)
    show_best_per_budget(df, 'Excess%')
    show_best_per_budget(df, 'Calmar')
    df2 = df.copy()
    df2['EntryDTE'] = df2['DTE'].str.split('/').str[0]
    df2['ExitDTE'] = df2['DTE'].str.split('/x').str[1]
    show_pivot(df2, 'OTM', 'EntryDTE', 'Excess%', f'{title} OTM x EntryDTE (all budgets avg)')
    show_pivot(df2, 'OTM', 'ExitDTE', 'Excess%', f'{title} OTM x ExitDTE (all budgets avg)')
    # Per-budget pivots for representative annual budgets
    for ab in [1.0, 2.0, 3.3]:
        show_pivot(df2[abs(df2['AnnualBudget%']-ab)<0.1], 'OTM', 'EntryDTE', 'Excess%',
                   f'{title} OTM x EntryDTE @ ~{ab}% annual budget')

def show_cross_freq(df, by='Calmar'):
    """Compare same config across rebalance frequencies."""
    freqs = sorted(df['Freq'].unique())
    if len(freqs) < 2:
        return
    print(f'\n{"="*110}')
    print(f'CROSS-FREQUENCY COMPARISON: best {freqs[0]} configs (by {by})')
    print(f'{"="*110}')
    base_freq = freqs[0]
    base = df[df['Freq'] == base_freq]
    top = base.sort_values(by, ascending=False).head(20)
    for _, row in top.iterrows():
        parts = [f'{base_freq}: Exc={row["Excess%"]:+.1f}% DD={row["MaxDD%"]:.0f}% '
                 f'Cal={row["Calmar"]:.2f}']
        for f in freqs[1:]:
            w = df[(df['Freq'] == f) & (df['OTM'] == row['OTM']) & (df['DTE'] == row['DTE'])
                   & (abs(df['AnnualBudget%'] - row['AnnualBudget%']) < 0.01)]
            if len(w) > 0:
                wr = w.iloc[0]
                parts.append(f'{f}: Exc={wr["Excess%"]:+.1f}% DD={wr["MaxDD%"]:.0f}% '
                             f'Cal={wr["Calmar"]:.2f}')
        print(f'  {row["OTM"]:>4s} {row["DTE"]:<12s} {row["AnnualBudget%"]:.2f}%/yr  '
              f'{"  |  ".join(parts)}')

# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print('Loading data...')
    engine = init_engine()

    # Derive rebalance date lists per frequency
    monthly_rb = [pd.Timestamp(ns, unit='ns') for ns in engine['rb_date_ns']]
    freq_rb_ns = {}
    for fname, step, _ in REBAL_FREQS:
        dates = monthly_rb[::step]
        freq_rb_ns[fname] = [int(d.value) for d in dates]

    # Count configs per frequency
    sample_per_rebal = [ab / REBAL_FREQS[0][2] for ab in ANNUAL_BUDGETS]
    sample_grid, _ = build_grid(sample_per_rebal)
    n_per_freq = len(sample_grid)
    n_freqs = len(REBAL_FREQS)
    print(f'\nConfigs per frequency: {n_per_freq}')
    print(f'Total: {n_per_freq} x {n_freqs} freqs x 2 leverage = {n_per_freq * n_freqs * 2}')

    # --- Leveraged ---
    print(f'\n{"="*110}')
    print('LEVERAGED (100% stocks + puts on top)')
    print(f'{"="*110}')
    lev_frames = []
    for fname, step, rebal_per_year in REBAL_FREQS:
        per_rebal = [ab / rebal_per_year for ab in ANNUAL_BUDGETS]
        grid, keys = build_grid(per_rebal)
        rb_ns = freq_rb_ns[fname]
        print(f'\n  --- {fname} ({rebal_per_year}x/yr, {len(grid)} configs) ---')
        base_lev = base_cfg(rb_ns, 1.0, 0.0, per_rebal[0])
        df = run_batched(engine, grid, keys, base_lev, fname, rebal_per_year,
                         label=f'LEV {fname}')
        lev_frames.append(df)

    df_lev = pd.concat(lev_frames, ignore_index=True)
    analyze(df_lev, 'Leveraged')
    show_cross_freq(df_lev, 'Calmar')
    show_cross_freq(df_lev, 'Excess%')

    # --- No-leverage: loop freq x budget ---
    print(f'\n\n{"="*110}')
    print('NO-LEVERAGE (stock + opt = 100%)')
    print(f'{"="*110}')
    nolev_frames = []
    for fname, step, rebal_per_year in REBAL_FREQS:
        rb_ns = freq_rb_ns[fname]
        for ab in ANNUAL_BUDGETS:
            bp = ab / rebal_per_year  # per-rebalance budget
            nl_grid, nl_keys = [], []
            for oname, lo, hi in OTM_LEVELS:
                for ename, dte_min, dte_max in ENTRY_DTE:
                    for exit_dte in EXIT_DTES:
                        if exit_dte >= dte_min:
                            continue
                        nl_grid.append({
                            "label": f'{oname} {ename}/x{exit_dte} nolev {fname} {ab*100:.1f}%',
                            "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
                            "leg_exit_filters": [exit_q(exit_dte)],
                        })
                        nl_keys.append((oname, f'{ename}/x{exit_dte}', bp))

            print(f'\n  --- {fname} {ab*100:.1f}% annual ---')
            base_nl = base_cfg(rb_ns, 1.0 - ab, ab, None)
            df_nl = run_batched(engine, nl_grid, nl_keys, base_nl, fname, rebal_per_year,
                                label=f'NOLEV {fname} {ab*100:.1f}%')
            nolev_frames.append(df_nl)

    df_nolev = pd.concat(nolev_frames, ignore_index=True)
    analyze(df_nolev, 'No-leverage')
    show_cross_freq(df_nolev, 'Calmar')
    show_cross_freq(df_nolev, 'Excess%')

    # Save to CSV
    df_lev.to_csv('research/spitznagel_spy/sweep_6d_leveraged.csv', index=False)
    df_nolev.to_csv('research/spitznagel_spy/sweep_6d_noleveraged.csv', index=False)
    print('\nSaved: sweep_6d_leveraged.csv, sweep_6d_noleveraged.csv')
