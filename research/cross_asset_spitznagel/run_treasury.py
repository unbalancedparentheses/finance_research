#!/usr/bin/env python3
"""Leveraged Treasuries + Tail Hedge (Spitznagel Structure).

Analyzes leveraged US Treasuries (ZN 10yr, ZB 30yr) + OTM put protection.
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from databento_helpers import (
    compute_stats, load_front_month, parse_treasury_option,
    MONTH_CODES, build_settlement_lookup, get_settlement,
    precompute_settlements,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'databento')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)


def _load_treasury_front_month(filename, prefix='ZN'):
    """Load treasury futures with prefix filtering (ZN or ZB)."""
    fut = pd.read_parquet(os.path.join(DATA_DIR, filename))
    outrights = fut[
        (~fut['symbol'].str.contains('-', na=False)) &
        (~fut['symbol'].str.startswith('UD:', na=False)) &
        (fut['symbol'].str.startswith(prefix, na=False))
    ].copy()
    outrights = outrights.sort_index()

    grouped = outrights.groupby(outrights.index)
    dates = sorted(grouped.groups.keys())

    front_records = []
    for ts in dates:
        day = grouped.get_group(ts)
        max_vol_pos = day['volume'].values.argmax()
        front = day.iloc[max_vol_pos]
        date_norm = ts.normalize().tz_localize(None) if ts.tz is not None else ts.normalize()

        front_records.append({
            'date': date_norm,
            'symbol': front['symbol'],
            'close': float(front['close']),
            'volume': int(front['volume']),
        })

    df = pd.DataFrame(front_records).set_index('date')
    df = df[~df.index.duplicated(keep='first')]
    df['return'] = df['close'].pct_change()
    return df


def _load_treasury_options(filename, label):
    """Load raw treasury option parquet."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f'{label} NOT FOUND: {path}')
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if 'ts_event' not in df.columns and df.index.name == 'ts_event':
        df = df.reset_index()
    elif 'ts_event' not in df.columns:
        df['ts_event'] = df.index
    df = df[df['symbol'].notna()].copy()
    print(f'{label}: {len(df):,} rows, {df["symbol"].nunique()} symbols')
    print(f'  Date range: {df["ts_event"].min()} to {df["ts_event"].max()}')
    return df


def _select_monthly_treasury_puts(opts_df, front_prices, otm_target=0.96, min_vol=3):
    """Select one OTM put per month for treasury options."""
    puts = opts_df[opts_df['opt_type'] == 'P'].copy()
    if len(puts) == 0:
        return pd.DataFrame()

    puts['ym'] = puts['date'].dt.to_period('M')
    selections = []

    for ym, group in puts.groupby('ym'):
        entry_date = group['date'].min()
        near_idx = front_prices.index.get_indexer([entry_date], method='nearest')
        if near_idx[0] < 0:
            continue
        underlying = float(front_prices.iloc[near_idx[0]]['close'])
        if underlying <= 0:
            continue

        first_day = group[group['date'] == entry_date].copy()
        if len(first_day) == 0:
            continue

        first_day['moneyness'] = first_day['strike'] / underlying
        otm = first_day[
            (first_day['moneyness'] < 1.0) &
            (first_day['moneyness'] > 0.85) &
            (first_day['close'] > 0) &
            (first_day['volume'] >= min_vol)
        ]
        if len(otm) == 0:
            continue

        otm = otm.copy()
        otm['dist'] = abs(otm['moneyness'] - otm_target)
        best = otm.nsmallest(3, 'dist')
        selected = best.loc[best['volume'].idxmax()]

        selections.append({
            'entry_date': entry_date,
            'symbol': selected['symbol'],
            'strike': selected['strike'],
            'entry_price': selected['close'],
            'expiry': selected['expiry'],
            'underlying': underlying,
            'moneyness': selected['moneyness'],
            'volume': selected['volume'],
        })

    return pd.DataFrame(selections)


def main():
    # ── 1. Load Treasury Front-Month Futures ──
    zn_fut = _load_treasury_front_month('ZN_FUT_ohlcv1d.parquet', prefix='ZN')
    zb_fut = _load_treasury_front_month('ZB_FUT_ohlcv1d.parquet', prefix='ZB')

    for name, fut in [('ZN (10yr)', zn_fut), ('ZB (30yr)', zb_fut)]:
        print(f'{name}:')
        print(f'  Days: {len(fut):,}')
        print(f'  Range: {fut.index.min().date()} to {fut.index.max().date()}')
        print(f'  Price: {fut["close"].min():.3f} to {fut["close"].max():.3f}')
        print(f'  Avg volume: {fut["volume"].mean():,.0f}')
        print()

    # ── 2. Treasury Return Profile ──
    for name, fut in [('ZN (10yr)', zn_fut), ('ZB (30yr)', zb_fut)]:
        daily = fut[['close', 'return']].dropna()
        annual = daily['close'].resample('YE').last().pct_change().dropna()

        print(f'{name} Annual Returns')
        print('=' * 40)
        for date, ret in annual.items():
            print(f'  {date.year}: {ret:>7.1%}')
        avg = annual.mean()
        med = annual.median()
        print(f'  {"Avg":>4}: {avg:>7.1%}')
        print(f'  {"Med":>4}: {med:>7.1%}')

        total = daily['close'].iloc[-1] / daily['close'].iloc[0]
        years = (daily.index[-1] - daily.index[0]).days / 365.25
        cagr = total ** (1/years) - 1
        vol = daily['return'].std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        print(f'  Cumulative: {total:.3f}x in {years:.1f} years')
        print(f'  CAGR: {cagr:.1%}, Vol: {vol:.1%}, Sharpe: {sharpe:.3f}')
        print()

    # ── 3. Load Treasury Options ──
    ozn_opts_raw = _load_treasury_options('OZN_OPT_ohlcv1d.parquet', 'OZN (10yr opts)')
    ozb_opts_raw = _load_treasury_options('OZB_OPT_ohlcv1d.parquet', 'OZB (30yr opts)')

    # ── 4. Parse Options ──
    ozn_opts = pd.DataFrame()
    ozb_opts = pd.DataFrame()

    for label, opts_raw, fut_name in [('OZN', ozn_opts_raw, 'ZN'),
                                        ('OZB', ozb_opts_raw, 'ZB')]:
        if len(opts_raw) == 0:
            print(f'{label}: No data')
            continue

        parsed = []
        for _, row in opts_raw.iterrows():
            trade_date = row['ts_event']
            p = parse_treasury_option(row['symbol'],
                                       prefix=label,
                                       strike_divisor=10.0 if label == 'OZN' else 1.0)
            if p is not None:
                parsed.append({
                    'date': pd.Timestamp(trade_date).tz_localize(None) if pd.Timestamp(trade_date).tz is not None else pd.Timestamp(trade_date),
                    'symbol': row['symbol'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'opt_type': p['opt_type'],
                    'strike': p['strike'],
                    'month': p['month'],
                    'year': p['year'],
                    'expiry': p['expiry'],
                    'root': p.get('root', label),
                })

        if len(parsed) == 0:
            print(f'{label}: No options parsed')
            continue

        df = pd.DataFrame(parsed)
        puts = df[df['opt_type'] == 'P']
        calls = df[df['opt_type'] == 'C']
        print()
        print(f'{label}: Parsed {len(df):,} options ({len(puts):,} puts, {len(calls):,} calls)')
        print(f'  Strike range: {df["strike"].min():.1f} to {df["strike"].max():.1f}')
        print(f'  Date range: {df["date"].min().date()} to {df["date"].max().date()}')

        if label == 'OZN':
            ozn_opts = df
        else:
            ozb_opts = df

    # ── 5. Monthly Put Selection ──
    zn_put_sels = pd.DataFrame()
    zb_put_sels = pd.DataFrame()

    for label, opts, fut in [('ZN', ozn_opts, zn_fut), ('ZB', ozb_opts, zb_fut)]:
        if len(opts) == 0:
            print(f'{label}: No options data')
            continue
        sels = _select_monthly_treasury_puts(opts, fut, otm_target=0.96)
        print(f'{label} put selections: {len(sels)} months')
        if len(sels) > 0:
            print(f'  Avg moneyness: {sels["moneyness"].mean():.3f}')
            print(f'  Avg entry price: {sels["entry_price"].mean():.4f}')
            print(f'  Avg volume: {sels["volume"].mean():.0f}')
            print('  First 5:')
            for _, r in sels.head(5).iterrows():
                print(f'    {r["entry_date"].strftime("%Y-%m") if hasattr(r["entry_date"], "strftime") else r["entry_date"]}  {r["symbol"]:25s}  K={r["strike"]:.1f}  S={r["underlying"]:.3f}  m={r["moneyness"]:.3f}  px={r["entry_price"]:.4f}')
        print()

        if label == 'ZN':
            zn_put_sels = sels
        else:
            zb_put_sels = sels

    # ── 6. Run All Backtests ──
    leverage_levels = [1, 2, 3, 5, 7, 10]
    put_budgets = [0.003, 0.005, 0.010]

    all_results = {}

    for instr, fut, put_sels, opts in [('ZN', zn_fut, zn_put_sels, ozn_opts),
                                          ('ZB', zb_fut, zb_put_sels, ozb_opts)]:
        daily_rets = fut[['close', 'return']].dropna()['return']
        has_opts = len(opts) > 0 and len(put_sels) > 0

        if has_opts:
            print(f'Building {instr} settlement lookup...')
            settlement_lookup = build_settlement_lookup(opts)
            put_map = precompute_settlements(put_sels, settlement_lookup, fut)
            put_map_naive = {}
            for k, v in put_map.items():
                k_naive = k.tz_localize(None) if hasattr(k, 'tz') and k.tz is not None else k
                put_map_naive[k_naive] = v
            put_map = put_map_naive
            print(f'  {len(put_map)} months with put data')
        else:
            put_map = {}
            print(f'{instr}: No options data - running unhedged only')

        for lev in leverage_levels:
            print(f'  {instr} {lev}x unhedged...')
            cap = 100.0
            records = []
            for date, ret in daily_rets.items():
                if cap <= 0:
                    records.append({'date': date, 'capital': 0})
                    continue
                cap += cap * lev * ret
                records.append({'date': date, 'capital': cap})
            df = pd.DataFrame(records).set_index('date')
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            all_results[(instr, lev, 0)] = df

            if has_opts:
                for budget in put_budgets:
                    cap = 100.0
                    records = []
                    current_month = None
                    for date, ret in daily_rets.items():
                        if cap <= 0:
                            records.append({'date': date, 'capital': 0, 'put_pnl': 0})
                            continue
                        notional = cap * lev
                        spot_pnl = notional * ret

                        p_pnl = 0
                        date_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz is not None else date
                        ym = pd.Timestamp(date_naive).to_period('M')
                        if ym != current_month:
                            current_month = ym
                            if date_naive in put_map:
                                cost = budget * notional
                                p_pnl = cost * put_map[date_naive]['pnl_ratio']

                        cap += spot_pnl + p_pnl
                        records.append({'date': date, 'capital': cap, 'put_pnl': p_pnl})
                    df = pd.DataFrame(records).set_index('date')
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    all_results[(instr, lev, budget)] = df

    print()
    print(f'Total backtests: {len(all_results)}')

    # ── 7. Results Summary ──
    print('=' * 130)
    print('US TREASURY FUTURES + PUT HEDGE -- FULL RESULTS')
    print('=' * 130)

    header = f'{"Strategy":>35s} {"CAGR":>8s} {"Vol":>8s} {"Sharpe":>8s} {"Sortino":>8s} {"Calmar":>8s} {"MaxDD":>8s} {"Skew":>7s} {"Kurt":>7s} {"Total":>8s}'
    print(header)
    print('-' * 130)

    for instr in ['ZN', 'ZB']:
        label_10_30 = '(10yr)' if instr == 'ZN' else '(30yr)'
        print()
        print(f'  --- {instr} {label_10_30} ---')
        for lev in leverage_levels:
            for budget in [0] + put_budgets:
                key = (instr, lev, budget)
                if key not in all_results:
                    continue
                cap = all_results[key]['capital']
                stats = compute_stats(cap)
                if stats is None:
                    continue
                if budget == 0:
                    label = f'{instr} {lev}x unhedged'
                else:
                    label = f'{instr} {lev}x + {budget*100:.1f}% puts'
                print(f'{label:>35s} {stats["CAGR"]:>7.2%} {stats["Vol"]:>7.1%} {stats["Sharpe"]:>8.3f} {stats["Sortino"]:>8.3f} {stats["Calmar"]:>8.3f} {stats["MaxDD"]:>7.1%} {stats["Skew"]:>7.2f} {stats["Kurt"]:>7.1f} {stats["Total"]:>7.1f}x')
            print()

    # ── 8. Equity Curves ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    for budget in [0] + put_budgets:
        key = ('ZN', 1, budget)
        if key in all_results:
            cap = all_results[key]['capital'] / 100
            label = 'Unhedged' if budget == 0 else f'{budget*100:.1f}% puts'
            style = '--' if budget == 0 else '-'
            ax.plot(cap.index, cap, linestyle=style, linewidth=1.5, label=label)
    ax.set_title('ZN (10yr) 1x Leverage')
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[0, 1]
    for budget in [0] + put_budgets:
        key = ('ZB', 1, budget)
        if key in all_results:
            cap = all_results[key]['capital'] / 100
            label = 'Unhedged' if budget == 0 else f'{budget*100:.1f}% puts'
            style = '--' if budget == 0 else '-'
            ax.plot(cap.index, cap, linestyle=style, linewidth=1.5, label=label)
    ax.set_title('ZB (30yr) 1x Leverage')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 0]
    for lev in leverage_levels:
        key = ('ZN', lev, 0.005)
        if key not in all_results:
            key = ('ZN', lev, 0)
        if key in all_results:
            cap = all_results[key]['capital'] / 100
            if cap.iloc[-1] > 0:
                ax.plot(cap.index, cap, linewidth=1.5, label=f'ZN {lev}x')
    ax.set_title('ZN (10yr) -- Leverage Comparison')
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 1]
    for lev in leverage_levels:
        key = ('ZB', lev, 0.005)
        if key not in all_results:
            key = ('ZB', lev, 0)
        if key in all_results:
            cap = all_results[key]['capital'] / 100
            if cap.iloc[-1] > 0:
                ax.plot(cap.index, cap, linewidth=1.5, label=f'ZB {lev}x')
    ax.set_title('ZB (30yr) -- Leverage Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('US Treasury Futures -- Spitznagel Tail Hedge', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'tr_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 9. Year-by-Year Returns ──
    print('=' * 140)
    print('YEAR-BY-YEAR RETURNS -- TREASURY FUTURES')
    print('=' * 140)

    configs = [
        (('ZN', 1, 0), 'ZN 1x unh'),
        (('ZN', 1, 0.005), 'ZN 1x 0.5%'),
        (('ZN', 3, 0), 'ZN 3x unh'),
        (('ZN', 3, 0.005), 'ZN 3x 0.5%'),
        (('ZB', 1, 0), 'ZB 1x unh'),
        (('ZB', 1, 0.005), 'ZB 1x 0.5%'),
        (('ZB', 3, 0), 'ZB 3x unh'),
        (('ZB', 3, 0.005), 'ZB 3x 0.5%'),
    ]

    yearly_data = {}
    cols = []
    for key, label in configs:
        if key not in all_results:
            continue
        cap = all_results[key]['capital']
        cap = cap[cap > 0]
        if len(cap) == 0:
            continue
        yearly = cap.resample('YE').last().pct_change().dropna()
        yearly_data[label] = yearly
        cols.append(label)

    all_years = sorted(set(y.year for ys in yearly_data.values() for y in ys.index))

    header = f'{"Year":>6}'
    for c in cols:
        header += f' {c:>12}'
    print(header)
    print('-' * 140)

    for y in all_years:
        row = f'{y:>6}'
        for c in cols:
            if c in yearly_data:
                ys = yearly_data[c]
                match = ys[ys.index.year == y]
                if len(match) > 0:
                    row += f' {match.iloc[0]:>11.1%}'
                else:
                    row += f' {"":>12}'
            else:
                row += f' {"":>12}'
        print(row)

    # ── 10. Crisis Performance ──
    crises = [
        ('2013 Taper Tantrum',  '2013-05-01', '2013-09-30'),
        ('2015 China Deval',    '2015-08-01', '2015-09-30'),
        ('2018 Q4 Selloff',     '2018-10-01', '2018-12-31'),
        ('2020 COVID',          '2020-02-19', '2020-03-23'),
        ('2022 Rate Hikes',     '2022-01-01', '2022-10-14'),
        ('2023 SVB Crisis',     '2023-03-01', '2023-03-31'),
        ('2025 Tariff Shock',   '2025-01-20', '2025-04-30'),
    ]

    print('=' * 130)
    print('CRISIS PERFORMANCE -- TREASURY FUTURES')
    print('=' * 130)

    crisis_configs = [
        (('ZN', 1, 0), 'ZN 1x unh'),
        (('ZN', 1, 0.005), 'ZN 1x 0.5%'),
        (('ZN', 3, 0), 'ZN 3x unh'),
        (('ZB', 1, 0), 'ZB 1x unh'),
        (('ZB', 1, 0.005), 'ZB 1x 0.5%'),
        (('ZB', 3, 0), 'ZB 3x unh'),
    ]

    header = f'{"Crisis":>25} {"Dates":>25}'
    for _, label in crisis_configs:
        header += f' {label:>12}'
    print(header)
    print('-' * 130)

    for name, start, end in crises:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        row = f'{name:>25} {start} to {end}'
        for key, label in crisis_configs:
            if key not in all_results:
                row += f' {"N/A":>12}'
                continue
            cap = all_results[key]['capital']
            window = cap[(cap.index >= s) & (cap.index <= e)]
            if len(window) >= 2:
                ret = window.iloc[-1] / window.iloc[0] - 1
                row += f' {ret:>11.1%}'
            else:
                row += f' {"N/A":>12}'
        print(row)

    # ── 11. Leverage Analysis ──
    print('=' * 110)
    print('LEVERAGE ANALYSIS -- SHARPE AND CAGR BY LEVERAGE')
    print('=' * 110)

    for instr in ['ZN', 'ZB']:
        label_10_30 = '(10yr)' if instr == 'ZN' else '(30yr)'
        print()
        print(f'  --- {instr} {label_10_30} ---')
        print(f'{"Lever":>8}', end='')
        for budget in [0] + put_budgets:
            label = 'Unhedged' if budget == 0 else f'{budget*100:.1f}% puts'
            print(f' {label:>15} {"":>8}', end='')
        print()

        print(f'{"":>8}', end='')
        for _ in [0] + put_budgets:
            print(f' {"Sharpe":>8} {"CAGR":>8} {"MaxDD":>7}', end='')
        print()
        print('-' * 110)

        for lev in leverage_levels:
            print(f'{lev:>6}x  ', end='')
            for budget in [0] + put_budgets:
                key = (instr, lev, budget)
                if key in all_results:
                    cap = all_results[key]['capital']
                    s = compute_stats(cap)
                    if s:
                        print(f' {s["Sharpe"]:>8.3f} {s["CAGR"]:>7.1%} {s["MaxDD"]:>6.1%}', end='')
                    else:
                        print(f' {"blown":>8} {"":>8} {"":>7}', end='')
                else:
                    print(f' {"N/A":>8} {"":>8} {"":>7}', end='')
            print()

    print()
    print('KELLY-OPTIMAL LEVERAGE:')
    for instr in ['ZN', 'ZB']:
        for budget in [0] + put_budgets:
            best_lev = None
            best_cagr = -999
            for lev in leverage_levels:
                key = (instr, lev, budget)
                if key in all_results:
                    cap = all_results[key]['capital']
                    s = compute_stats(cap)
                    if s and s['CAGR'] > best_cagr:
                        best_cagr = s['CAGR']
                        best_lev = lev
            label = 'Unhedged' if budget == 0 else f'{budget*100:.1f}% puts'
            if best_lev:
                key = (instr, best_lev, budget)
                s = compute_stats(all_results[key]['capital'])
                print(f'  {instr} {label:>12}: {best_lev}x -> CAGR {best_cagr:.1%}, Sharpe {s["Sharpe"]:.3f}, MaxDD {s["MaxDD"]:.1%}')

    # ── 12. Drawdown Chart ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    configs_plot = {
        'ZN': [
            (('ZN', 1, 0), 'ZN 1x unhedged', 'gray', '--'),
            (('ZN', 1, 0.005), 'ZN 1x + 0.5%', 'blue', '-'),
            (('ZN', 3, 0), 'ZN 3x unhedged', 'lightblue', '--'),
            (('ZN', 3, 0.005), 'ZN 3x + 0.5%', 'darkblue', '-'),
        ],
        'ZB': [
            (('ZB', 1, 0), 'ZB 1x unhedged', 'gray', '--'),
            (('ZB', 1, 0.005), 'ZB 1x + 0.5%', 'red', '-'),
            (('ZB', 3, 0), 'ZB 3x unhedged', 'lightsalmon', '--'),
            (('ZB', 3, 0.005), 'ZB 3x + 0.5%', 'darkred', '-'),
        ]
    }

    for col_idx, instr in enumerate(['ZN', 'ZB']):
        ax = axes[0, col_idx]
        for key, label, color, style in configs_plot[instr]:
            if key in all_results:
                cap = all_results[key]['capital'] / 100
                ax.plot(cap.index, cap, color=color, linestyle=style, linewidth=1.5, label=label)
        ax.set_title(f'{instr} -- Equity Curves')
        ax.set_ylabel('Portfolio Value ($1 start)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax = axes[1, col_idx]
        for key, label, color, style in configs_plot[instr]:
            if key in all_results:
                cap = all_results[key]['capital']
                dd = cap / cap.cummax() - 1
                ax.plot(dd.index, dd * 100, color=color, linestyle=style, linewidth=1, label=label, alpha=0.8)
        ax.set_title(f'{instr} -- Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)

    plt.suptitle('US Treasury Futures -- Equity Curves & Drawdowns', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'tr_drawdowns.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 13. Cross-Asset Comparison ──
    print('=' * 120)
    print('CROSS-ASSET COMPARISON -- SPITZNAGEL STRUCTURE')
    print('=' * 120)

    header = f'{"Strategy":>35s} {"CAGR":>8s} {"Vol":>8s} {"Sharpe":>8s} {"Sortino":>8s} {"MaxDD":>8s} {"Total":>8s}'
    print(header)
    print('-' * 120)

    for instr in ['ZN', 'ZB']:
        for lev in [1, 3]:
            for budget in [0, 0.005]:
                key = (instr, lev, budget)
                if key not in all_results:
                    continue
                cap = all_results[key]['capital']
                stats = compute_stats(cap)
                if stats is None:
                    continue
                if budget == 0:
                    label = f'{instr} {lev}x unhedged'
                else:
                    label = f'{instr} {lev}x + 0.5% puts'
                print(f'{label:>35s} {stats["CAGR"]:>7.2%} {stats["Vol"]:>7.1%} {stats["Sharpe"]:>8.3f} {stats["Sortino"]:>8.3f} {stats["MaxDD"]:>7.1%} {stats["Total"]:>7.1f}x')

    print()
    print('Reference benchmarks from other notebooks:')
    print('  ES 1x unhedged:                  Sharpe  0.818, CAGR 12.7%, MaxDD -35.4%')
    print('  ES 1x + 0.5% puts:              Sharpe  0.702, CAGR 11.7%')
    print('  FX Carry High-Carry 1x hedged:   Sharpe ~1.03,  CAGR ~14.3%')
    print('  FX Carry EW All-6 1x hedged:     Sharpe ~0.93,  CAGR ~10.4%')
    print('  Gold 1x hedged:                  Sharpe  0.17,  CAGR  3.6%')

    # ── 14. Conclusions ──
    lines = []
    lines.append('US TREASURY FUTURES + TAIL HEDGE -- KEY FINDINGS')
    lines.append('=' * 60)
    lines.append('')

    for instr in ['ZN', 'ZB']:
        best_1x = None
        best_1x_sharpe = -999
        best_3x = None
        best_3x_sharpe = -999
        best_overall = None
        best_overall_cagr = -999

        for (i, lev, budget), df in all_results.items():
            if i != instr:
                continue
            s = compute_stats(df['capital'])
            if s is None:
                continue
            if lev == 1 and s['Sharpe'] > best_1x_sharpe:
                best_1x_sharpe = s['Sharpe']
                best_1x = (lev, budget, s)
            if lev == 3 and s['Sharpe'] > best_3x_sharpe:
                best_3x_sharpe = s['Sharpe']
                best_3x = (lev, budget, s)
            if s['CAGR'] > best_overall_cagr and s['MaxDD'] > -0.99:
                best_overall_cagr = s['CAGR']
                best_overall = (lev, budget, s)

        label = '10-Year Note' if instr == 'ZN' else '30-Year Bond'
        lines.append(f'{instr} ({label}):')
        if best_1x:
            lev, budget, s = best_1x
            b = f'{budget*100:.1f}% puts' if budget > 0 else 'unhedged'
            lines.append(f'  Best 1x: {b} -> Sharpe {s["Sharpe"]:.3f}, CAGR {s["CAGR"]:.1%}, MaxDD {s["MaxDD"]:.1%}')
        if best_3x:
            lev, budget, s = best_3x
            b = f'{budget*100:.1f}% puts' if budget > 0 else 'unhedged'
            lines.append(f'  Best 3x: {b} -> Sharpe {s["Sharpe"]:.3f}, CAGR {s["CAGR"]:.1%}, MaxDD {s["MaxDD"]:.1%}')
        if best_overall:
            lev, budget, s = best_overall
            b = f'{budget*100:.1f}% puts' if budget > 0 else 'unhedged'
            lines.append(f'  Kelly-optimal: {lev}x {b} -> CAGR {s["CAGR"]:.1%}, Sharpe {s["Sharpe"]:.3f}, MaxDD {s["MaxDD"]:.1%}')
        lines.append('')

    lines.append('KEY INSIGHTS:')
    lines.append('')
    lines.append('1. TREASURIES AS DIVERSIFIER:')
    lines.append('   Bonds are negatively correlated with equities during crises.')
    lines.append('   A 60/40 equity/bond portfolio naturally hedges both tails.')
    lines.append('')
    lines.append('2. THE 2022 PROBLEM:')
    lines.append('   The worst bond selloff in 50 years (-30% for 30yr) destroyed')
    lines.append('   any leveraged bond strategy. Puts on bonds would have protected.')
    lines.append('')
    lines.append('3. LEVERAGE IN TREASURIES:')
    lines.append('   ZN vol is ~6-8% vs ES vol of ~15-18%, so 2-3x levered ZN')
    lines.append('   produces equity-like returns with different risk characteristics.')
    lines.append('')
    lines.append('4. SPITZNAGEL STRUCTURE ON BONDS:')
    lines.append('   Unlike equities where puts protect against market crashes,')
    lines.append('   bond puts protect against rate spikes. This is the OPPOSITE tail.')
    lines.append('   A combined portfolio with equity puts + bond puts covers both scenarios.')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
