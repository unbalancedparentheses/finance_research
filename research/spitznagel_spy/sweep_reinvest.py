#!/usr/bin/env python3
"""Test immediate reinvestment of put profits via rebalance_stocks_on_exit.

Compares:
- Standard: put profits sit in cash until next bimonthly rebalance
- Reinvest: put profits immediately buy stocks (new Rust flag)

This models Universa's actual behavior: when puts pay off during a crash,
immediately reinvest into discounted stocks.
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

# Top configs from Universa sweep
CONFIGS = [
    ('40% 90-120/x30',  0.58, 0.62, 90, 120, 30),
    ('35% 90-120/x30',  0.63, 0.67, 90, 120, 30),
    ('30% 90-120/x30',  0.68, 0.72, 90, 120, 30),
    ('28% 90-120/x30',  0.70, 0.74, 90, 120, 30),
    ('25% 75-105/x1',   0.73, 0.77, 75, 105,  1),
    ('23% 60-90/x1',    0.75, 0.79, 60,  90,  1),
    ('20% 60-90/x1',    0.78, 0.82, 60,  90,  1),
    ('15% 45-75/x30',   0.83, 0.87, 45,  75, 30),
    ('5% 30-60/x25',    0.93, 0.97, 30,  60, 25),
]

ANNUAL_BUDGET = 0.0333

def strike_entry_q(lo, hi, dte_min, dte_max):
    return (f"(type == 'put') & (ask > 0) & (underlying == 'SPY')"
            f" & (dte >= {dte_min}) & (dte <= {dte_max})"
            f" & (strike >= underlying_last * {lo})"
            f" & (strike <= underlying_last * {hi})")

def exit_q(exit_dte):
    return f"(type == 'put') & (dte <= {exit_dte})"

def run_configs(engine, rb_ns, budget_per_rebal, spy_annual, reinvest=False, label=""):
    grid = []
    for name, lo, hi, dte_min, dte_max, exit_dte in CONFIGS:
        d = {
            "label": name,
            "leg_entry_filters": [strike_entry_q(lo, hi, dte_min, dte_max)],
            "leg_exit_filters": [exit_q(exit_dte)],
            "options_budget_pct": budget_per_rebal,
        }
        if reinvest:
            d["rebalance_stocks_on_exit"] = True
        grid.append(d)

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
        "options_budget_pct": budget_per_rebal,
        "check_exits_daily": True,
        "rebalance_stocks_on_exit": reinvest,
    }

    print(f'  {label}...', end=' ', flush=True)
    t0 = time.perf_counter()
    raw = parallel_sweep(engine['opts_pl'], engine['stocks_pl'], base,
                         engine['schema_mapping'], grid, None)
    print(f'{time.perf_counter()-t0:.1f}s')

    rows = []
    for res, (name, *_) in zip(raw, CONFIGS):
        ann = res['annualized_return'] * 100
        rows.append({
            'Config': name,
            'Annual%': ann,
            'Excess%': ann - spy_annual,
            'MaxDD%': res['max_drawdown'] * 100,
            'Sharpe': res.get('sharpe_ratio', 0),
            'Calmar': res.get('calmar_ratio', 0),
            'Sortino': res.get('sortino_ratio', 0),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Loading data...')
    engine = init_engine()
    spy_annual = engine['spy_annual']

    monthly_rb = [pd.Timestamp(ns, unit='ns') for ns in engine['rb_date_ns']]
    bimonthly_rb = monthly_rb[::2]
    bimonthly_ns = [int(d.value) for d in bimonthly_rb]
    monthly_ns = engine['rb_date_ns']

    per_rebal_bi = ANNUAL_BUDGET / 6   # bimonthly
    per_rebal_mo = ANNUAL_BUDGET / 12  # monthly

    modes = [
        ('Bimonthly, no reinvest',   bimonthly_ns, per_rebal_bi, False),
        ('Bimonthly, reinvest',      bimonthly_ns, per_rebal_bi, True),
        ('Monthly, no reinvest',     monthly_ns,   per_rebal_mo, False),
        ('Monthly, reinvest',        monthly_ns,   per_rebal_mo, True),
    ]

    results = {}
    for name, rb_ns, per_rebal, reinvest in modes:
        df = run_configs(engine, rb_ns, per_rebal, spy_annual, reinvest=reinvest, label=name)
        results[name] = df

    # Print comparison
    mode_names = [m[0] for m in modes]
    config_names = [c[0] for c in CONFIGS]

    for metric in ['Excess%', 'MaxDD%', 'Calmar', 'Sortino']:
        print(f'\n{"="*110}')
        print(f'{metric}')
        print(f'{"="*110}')
        header = f'{"Config":<22s}'
        for mn in mode_names:
            short = mn[:20]
            header += f'  {short:>20s}'
        print(header)

        for cfg in config_names:
            line = f'{cfg:<22s}'
            for mn in mode_names:
                df = results[mn]
                row = df[df['Config'] == cfg]
                if len(row) > 0:
                    v = row.iloc[0][metric]
                    if metric in ['Excess%', 'MaxDD%']:
                        line += f'  {v:>+19.2f}%'
                    else:
                        line += f'  {v:>20.2f}'
                line += ''
            print(line)

    # Summary: reinvest improvement
    print(f'\n{"="*110}')
    print('IMPACT OF IMMEDIATE REINVESTMENT (bimonthly)')
    print(f'{"="*110}')
    df_no = results['Bimonthly, no reinvest']
    df_yes = results['Bimonthly, reinvest']
    print(f'{"Config":<22s}  {"Excess (no)":>12s}  {"Excess (yes)":>12s}  {"Diff":>8s}  '
          f'{"Calmar (no)":>12s}  {"Calmar (yes)":>12s}  {"Diff":>8s}')
    for cfg in config_names:
        no = df_no[df_no['Config'] == cfg].iloc[0]
        yes = df_yes[df_yes['Config'] == cfg].iloc[0]
        print(f'{cfg:<22s}  {no["Excess%"]:>+11.2f}%  {yes["Excess%"]:>+11.2f}%  '
              f'{yes["Excess%"]-no["Excess%"]:>+7.2f}%  '
              f'{no["Calmar"]:>12.2f}  {yes["Calmar"]:>12.2f}  '
              f'{yes["Calmar"]-no["Calmar"]:>+7.2f}')
