#!/usr/bin/env python3
"""US-UK Bond Carry Trade with Directional Tail Hedge.

Strategy: Long the higher-yielding bond, short the lower-yielding bond,
with OZN options for tail protection. Tests multiple OTM levels (4-30%)
to validate Spitznagel's thesis that deeper OTM is more efficient.
"""
import sys
import os
import re

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from databento_helpers import compute_stats, MONTH_CODES

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'databento')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)


def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['figure.dpi'] = 100

    # ── Load & clean futures ──
    zn_raw = pd.read_parquet(os.path.join(DATA_DIR, 'ZN_FUT_ohlcv1d.parquet'))
    gilt_raw = pd.read_parquet(os.path.join(DATA_DIR, 'R_FUT_ohlcv1d.parquet'))

    zn_raw.index = zn_raw.index.tz_localize(None) if zn_raw.index.tz else zn_raw.index
    gilt_raw.index = gilt_raw.index.tz_localize(None) if gilt_raw.index.tz else gilt_raw.index

    zn_all = zn_raw[~zn_raw['symbol'].str.contains('-', na=False)]
    zn_all = zn_all[~zn_all['symbol'].str.startswith('UD:', na=False)]
    zn_all = zn_all.dropna(subset=['close'])
    zn_all = zn_all[zn_all['close'] > 50]
    print(f"ZN: {len(zn_all):,} rows, {zn_all['symbol'].nunique()} contracts")

    gilt_all = gilt_raw[~gilt_raw['symbol'].str.contains('-', na=False)]
    gilt_all = gilt_all[~gilt_all['symbol'].str.contains('_Z', na=False)]
    gilt_all = gilt_all.dropna(subset=['close'])
    gilt_all = gilt_all[gilt_all['close'] > 50]
    print(f"Gilt: {len(gilt_all):,} rows, {gilt_all['symbol'].nunique()} contracts")

    # ── Build front-month continuous series ──
    zn_front = zn_all.loc[zn_all.groupby(zn_all.index)['volume'].idxmax()]
    zn_front = zn_front[['close', 'volume', 'symbol']].copy()
    zn_front = zn_front[~zn_front.index.duplicated(keep='first')]
    zn_front.columns = ['zn_close', 'zn_vol', 'zn_sym']

    gilt_front = gilt_all.loc[gilt_all.groupby(gilt_all.index)['volume'].idxmax()]
    gilt_front = gilt_front[['close', 'volume', 'symbol']].copy()
    gilt_front = gilt_front[~gilt_front.index.duplicated(keep='first')]
    gilt_front.columns = ['gilt_close', 'gilt_vol', 'gilt_sym']

    df = zn_front.join(gilt_front, how='inner').sort_index()
    df['zn_ret'] = df['zn_close'].pct_change()
    df['gilt_ret'] = df['gilt_close'].pct_change()

    df.loc[df['zn_ret'].abs() > 0.03, 'zn_ret'] = 0.0
    df.loc[df['gilt_ret'].abs() > 0.03, 'gilt_ret'] = 0.0
    df = df.dropna(subset=['zn_ret', 'gilt_ret'])

    print(f"Overlapping dates: {len(df):,}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    # ── Price chart ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(df.index, df['zn_close'], label='ZN (US 10yr)', color='tab:blue')
    ax1.plot(df.index, df['gilt_close'], label='Gilt (UK 10yr)', color='tab:red')
    ax1.set_ylabel('Futures Price')
    ax1.legend()
    ax1.set_title('US 10yr vs UK Gilt Futures Prices')
    ratio = df['zn_close'] / df['gilt_close']
    ax2.plot(df.index, ratio, color='tab:green')
    ax2.axhline(ratio.mean(), color='gray', ls='--', alpha=0.5)
    ax2.set_ylabel('ZN / Gilt Ratio')
    ax2.set_title(f'Price Ratio (mean={ratio.mean():.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'bc_prices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation: {df['zn_ret'].corr(df['gilt_ret']):.3f}")

    # ── Carry signal and strategy returns ──
    monthly = df[['zn_ret', 'gilt_ret']].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly['zn_3m'] = (1 + monthly['zn_ret']).rolling(3).apply(lambda x: x.prod()) - 1
    monthly['gilt_3m'] = (1 + monthly['gilt_ret']).rolling(3).apply(lambda x: x.prod()) - 1
    monthly['carry_signal'] = np.where(monthly['zn_3m'].shift(1) < monthly['gilt_3m'].shift(1), 1, -1)
    monthly['long_zn'] = monthly['zn_ret'] - monthly['gilt_ret']
    monthly['long_gilt'] = monthly['gilt_ret'] - monthly['zn_ret']
    monthly['carry_trade'] = monthly['carry_signal'] * (monthly['zn_ret'] - monthly['gilt_ret'])
    monthly = monthly.dropna()

    print("Strategy Monthly Returns Summary")
    print("=" * 70)
    for name, col in [('Long ZN / Short Gilt', 'long_zn'),
                       ('Long Gilt / Short ZN', 'long_gilt'),
                       ('Dynamic Carry', 'carry_trade')]:
        r = monthly[col]
        ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + r).prod()
        print(f"  {name:30s}  CAGR={ann_ret:+.1%}  Vol={ann_vol:.1%}  Sharpe={sharpe:.3f}  Total={cum:.2f}x")

    # ── Load OZN options (puts AND calls) ──
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

    ozn_raw = pd.read_parquet(os.path.join(DATA_DIR, 'OZN_OPT_ohlcv1d.parquet'))
    if ozn_raw.index.tz is not None:
        ozn_raw.index = ozn_raw.index.tz_localize(None)
    print(f"OZN options: {len(ozn_raw):,} rows")

    pat = re.compile(r'^OZN\s*([FGHJKMNQUVXZ])(\d)\s+([PC])(\d+)')
    records = []
    for sym in ozn_raw['symbol'].unique():
        m = pat.match(str(sym))
        if not m:
            continue
        month_code, year_digit, pc, strike_str = m.groups()
        records.append({
            'symbol': sym,
            'opt_month': month_map[month_code],
            'opt_year_digit': int(year_digit),
            'opt_type': pc,
            'strike_raw': int(strike_str),
            'strike': int(strike_str) / 10.0,
        })

    opt_meta = pd.DataFrame(records)
    print(f"Parsed: {len(opt_meta):,} option symbols ({(opt_meta['opt_type']=='P').sum():,} puts, {(opt_meta['opt_type']=='C').sum():,} calls)")

    puts_meta = opt_meta[opt_meta['opt_type'] == 'P'].copy()
    ozn_puts = ozn_raw.reset_index().merge(puts_meta, on='symbol', how='inner').set_index('ts_event')
    ozn_puts = ozn_puts[ozn_puts.index >= df.index.min()]
    print(f"OZN puts in overlap: {len(ozn_puts):,} rows")

    calls_meta = opt_meta[opt_meta['opt_type'] == 'C'].copy()
    ozn_calls = ozn_raw.reset_index().merge(calls_meta, on='symbol', how='inner').set_index('ts_event')
    ozn_calls = ozn_calls[ozn_calls.index >= df.index.min()]
    print(f"OZN calls in overlap: {len(ozn_calls):,} rows")

    # ── Option selection at multiple OTM levels ──
    def select_options(ozn_data, monthly_index, df_prices, opt_type='P', target_moneyness=0.96):
        lo = target_moneyness - 0.04
        hi = target_moneyness + 0.04
        selections = []
        for month_start in monthly_index:
            yr = month_start.year
            mo = month_start.month
            mask_zn = (df_prices.index.year == yr) & (df_prices.index.month == mo)
            zn_prices = df_prices.loc[mask_zn, 'zn_close']
            if len(zn_prices) == 0:
                continue
            spot = zn_prices.iloc[0]
            month_opts = ozn_data[(ozn_data.index.year == yr) & (ozn_data.index.month == mo)]
            if len(month_opts) == 0:
                continue
            first_day = month_opts.index.min()
            day_opts = month_opts[month_opts.index == first_day].copy()
            if len(day_opts) == 0:
                continue
            day_opts['moneyness'] = day_opts['strike'] / spot
            candidates = day_opts[(day_opts['moneyness'].values > lo) & (day_opts['moneyness'].values < hi)]
            candidates = candidates[candidates['close'].values > 0]
            if abs(target_moneyness - 1.0) <= 0.08:
                candidates = candidates[candidates['volume'].values > 0]
            if len(candidates) == 0:
                continue
            candidates = candidates.copy()
            candidates['dist'] = (candidates['moneyness'].values - target_moneyness).__abs__()
            best_idx = candidates['dist'].values.argmin()
            best = candidates.iloc[best_idx]
            best_sym = best['symbol']
            sym_arr = ozn_data['symbol'].values
            yr_arr = ozn_data.index.year
            mo_arr = ozn_data.index.month
            mo_next = mo + 1 if mo < 12 else 1
            yr_next = yr + (1 if mo == 12 else 0)
            mask = (sym_arr == best_sym) & (
                ((yr_arr == yr) & (mo_arr == mo)) |
                ((yr_arr == yr_next) & (mo_arr == mo_next))
            )
            end_opts = ozn_data[mask]
            settle = end_opts['close'].iloc[-1] if len(end_opts) > 0 else 0.0
            settle = settle if pd.notna(settle) else 0.0
            entry_px = float(best['close'])
            selections.append({
                'date': month_start,
                'symbol': best_sym,
                'strike': float(best['strike']),
                'spot': spot,
                'moneyness': float(best['moneyness']),
                'entry_price': entry_px,
                'settle_price': settle,
                'pnl_ratio': (settle / entry_px - 1) if entry_px > 0 else 0.0,
            })
        return pd.DataFrame(selections).set_index('date') if selections else pd.DataFrame()

    otm_levels = {
        '4% OTM':  (0.96, 1.04),
        '10% OTM': (0.90, 1.10),
        '15% OTM': (0.85, 1.15),
        '20% OTM': (0.80, 1.20),
        '25% OTM': (0.75, 1.25),
        '30% OTM': (0.70, 1.30),
    }

    all_puts = {}
    all_calls = {}

    print("OTM LEVEL COMPARISON")
    print("=" * 100)
    for level_name, (put_target, call_target) in otm_levels.items():
        for opt_type, target, ozn_data, store in [
            ('Put', put_target, ozn_puts, all_puts),
            ('Call', call_target, ozn_calls, all_calls),
        ]:
            sel = select_options(ozn_data, monthly.index, df,
                                opt_type=opt_type[0], target_moneyness=target)
            store[level_name] = sel
            if len(sel) > 0:
                wr = (sel['pnl_ratio'] > 0).mean()
                print(f"  {level_name:>10}  {opt_type:>5}  {len(sel):>8}  {sel['moneyness'].mean():>9.3f}  "
                      f"{sel['entry_price'].mean():>9.4f}  {wr:>7.1%}  {sel['pnl_ratio'].mean():>+7.2f}x  "
                      f"{sel['pnl_ratio'].max():>+8.1f}x")

    # ── Backtest engine ──
    def backtest_carry(monthly_rets, puts_df, calls_df, leverage=1.0, opt_budget=0.0,
                       hedge_mode='puts'):
        capital = [1.0]
        for date, row in monthly_rets.items():
            spread_ret = row * leverage
            opt_pnl = 0.0
            if opt_budget > 0:
                if hedge_mode == 'puts':
                    opt_df = puts_df
                elif hedge_mode == 'calls':
                    opt_df = calls_df
                else:
                    sig = monthly.loc[date, 'carry_signal'] if date in monthly.index else 1
                    opt_df = puts_df if sig == 1 else calls_df
                if len(opt_df) > 0 and date in opt_df.index:
                    opt = opt_df.loc[date]
                    if isinstance(opt, pd.DataFrame):
                        opt = opt.iloc[0]
                    opt_pnl = opt_budget * opt['pnl_ratio']
            total_ret = spread_ret + opt_pnl
            capital.append(capital[-1] * (1 + total_ret))
        dates = list(monthly_rets.index)
        cap = pd.Series(capital[1:], index=dates)
        return cap

    def _compute_stats_monthly(cap):
        rets = cap.pct_change().dropna()
        n = len(rets)
        if n < 2:
            return {}
        ann_ret = cap.iloc[-1] ** (12 / n) - 1
        ann_vol = rets.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = cap / cap.cummax() - 1
        max_dd = dd.min()
        downside = rets[rets < 0].std() * np.sqrt(12)
        sortino = ann_ret / downside if downside > 0 else 0
        return {
            'CAGR': ann_ret, 'Vol': ann_vol, 'Sharpe': sharpe,
            'Sortino': sortino, 'MaxDD': max_dd, 'Total': cap.iloc[-1],
        }

    strat_rets = monthly['long_zn']
    leverages = [1, 3, 5]
    opt_budgets = [0.0, 0.003, 0.005, 0.01]

    results = []
    all_caps = {}

    for otm_name in otm_levels.keys():
        puts_df_level = all_puts[otm_name]
        calls_df_level = all_calls[otm_name]
        for lev in leverages:
            for ob in opt_budgets:
                cap = backtest_carry(strat_rets, puts_df_level, calls_df_level,
                                    leverage=lev, opt_budget=ob, hedge_mode='puts')
                stats = _compute_stats_monthly(cap)
                label = f"LongZN {lev}x {otm_name}"
                if ob > 0:
                    label += f" {ob*100:.1f}%"
                else:
                    label += " unhgd"
                stats['Strategy'] = 'Long ZN / Short Gilt'
                stats['Leverage'] = lev
                stats['Opt Budget'] = ob
                stats['OTM Level'] = otm_name
                stats['Label'] = label
                results.append(stats)
                all_caps[label] = cap

    results_df = pd.DataFrame(results)
    print(f"Ran {len(results)} backtest combinations")

    # ── OTM level comparison table ──
    print("=" * 120)
    print("LONG ZN / SHORT GILT -- OTM LEVEL x LEVERAGE x BUDGET")
    print("=" * 120)

    for lev in leverages:
        print(f"\n  -- {lev}x LEVERAGE --")
        print(f"  {'OTM Level':>10}  {'Budget':>8}  {'CAGR':>7}  {'Vol':>7}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>7}  {'Total':>7}")
        print("-" * 90)
        base = results_df[(results_df['Leverage'] == lev) & (results_df['Opt Budget'] == 0.0)]
        if len(base) > 0:
            b = base.iloc[0]
            print(f"  {'baseline':>10}  {'unhgd':>8}  {b['CAGR']:>+6.1%}  {b['Vol']:>6.1%}  {b['Sharpe']:>+6.3f}  {b['Sortino']:>+7.3f}  {b['MaxDD']:>6.1%}  {b['Total']:>6.2f}x")
            print()

        for otm_name in otm_levels.keys():
            for ob in [0.003, 0.005, 0.01]:
                sub = results_df[(results_df['Leverage'] == lev) &
                                 (results_df['OTM Level'] == otm_name) &
                                 (results_df['Opt Budget'] == ob)]
                if len(sub) == 1:
                    r = sub.iloc[0]
                    better = '  <--' if len(base) > 0 and r['Sharpe'] > b['Sharpe'] else ''
                    print(f"  {otm_name:>10}  {ob*100:.1f}%put  {r['CAGR']:>+6.1%}  {r['Vol']:>6.1%}  "
                          f"{r['Sharpe']:>+6.3f}  {r['Sortino']:>+7.3f}  {r['MaxDD']:>6.1%}  {r['Total']:>6.2f}x{better}")
            print()

    # ── Equity curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    base_label = 'LongZN 5x 4% OTM unhgd'
    if base_label in all_caps:
        ax.plot(all_caps[base_label].index, all_caps[base_label].values,
                label='5x unhedged', lw=2, color='black')
    for otm_name in otm_levels.keys():
        label = f'LongZN 5x {otm_name} 0.5%'
        if label in all_caps:
            ax.plot(all_caps[label].index, all_caps[label].values,
                    label=f'{otm_name}', alpha=0.7)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.set_title('5x + 0.5% budget: OTM comparison', fontsize=11)
    ax.set_ylabel('Capital')
    ax.legend(fontsize=7)

    ax = axes[1]
    for ob_str, ob in [('unhgd', 0.0), ('0.3%', 0.003), ('0.5%', 0.005), ('1.0%', 0.01)]:
        for otm_name in ['25% OTM', '30% OTM']:
            if ob == 0.0:
                label = f'LongZN 5x {otm_name} unhgd'
            else:
                label = f'LongZN 5x {otm_name} {ob*100:.1f}%'
            if label in all_caps:
                ax.plot(all_caps[label].index, all_caps[label].values,
                        label=f'{otm_name} {ob_str}', alpha=0.7)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.set_title('5x: Deep OTM budget comparison', fontsize=11)
    ax.set_ylabel('Capital')
    ax.legend(fontsize=7)

    ax = axes[2]
    for lev in [1, 3, 5]:
        label = f'LongZN {lev}x 25% OTM unhgd'
        if label in all_caps:
            ax.plot(all_caps[label].index, all_caps[label].values,
                    label=f'{lev}x unhgd', alpha=0.7, ls='-')
        label = f'LongZN {lev}x 25% OTM 0.5%'
        if label in all_caps:
            ax.plot(all_caps[label].index, all_caps[label].values,
                    label=f'{lev}x +0.5% puts', alpha=0.7, ls='--')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.set_title('25% OTM: leverage comparison', fontsize=11)
    ax.set_ylabel('Capital')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'bc_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Drawdown analysis ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    compare = [
        ('LongZN 5x 4% OTM unhgd', '5x unhedged', '-', 'black'),
        ('LongZN 5x 4% OTM 0.5%', '5x +4% OTM puts', '--', 'tab:red'),
        ('LongZN 5x 25% OTM 0.5%', '5x +25% OTM puts', '--', 'tab:blue'),
        ('LongZN 5x 30% OTM 0.5%', '5x +30% OTM puts', '--', 'tab:green'),
    ]
    for label, short_label, ls, color in compare:
        if label not in all_caps:
            continue
        cap = all_caps[label]
        axes[0].plot(cap.index, cap.values, ls=ls, color=color, label=short_label)
        dd = cap / cap.cummax() - 1
        axes[1].fill_between(dd.index, dd.values, 0, alpha=0.2, color=color, label=short_label)
        axes[1].plot(dd.index, dd.values, ls=ls, color=color, alpha=0.5, lw=0.8)
    axes[0].set_ylabel('Capital')
    axes[0].legend(fontsize=9)
    axes[0].set_title('Long ZN / Short Gilt @ 5x: Near vs Far OTM Puts')
    axes[1].set_ylabel('Drawdown')
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'bc_drawdowns.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Cross-asset comparison ──
    print("\n" + "=" * 110)
    print("CROSS-ASSET COMPARISON")
    print("=" * 110)

    for lev in leverages:
        sub = results_df[results_df['Leverage'] == lev]
        base = sub[sub['Opt Budget'] == 0.0]
        hedged = sub[sub['Opt Budget'] > 0]
        best_h = hedged.loc[hedged['Sharpe'].idxmax()] if len(hedged) > 0 else None
        print(f"\n  Long ZN / Short Gilt @ {lev}x:")
        if len(base) > 0:
            b = base.iloc[0]
            print(f"    Unhedged:    Sharpe={b['Sharpe']:+.3f}  CAGR={b['CAGR']:+.1%}  MaxDD={b['MaxDD']:.1%}")
        if best_h is not None:
            print(f"    Best hedged: Sharpe={best_h['Sharpe']:+.3f}  CAGR={best_h['CAGR']:+.1%}  MaxDD={best_h['MaxDD']:.1%}  ({best_h['OTM Level']} + {best_h['Opt Budget']*100:.1f}%)")

    print()
    print("  Reference benchmarks:")
    print("    ES (S&P 500) 1x unhedged:        Sharpe  0.818, CAGR 12.7%, MaxDD -35.4%")
    print("    ZN (10yr) 1x unhedged:            Sharpe -0.063, CAGR -0.3%, MaxDD -24.8%")
    print("    FX Carry EW All-6 1x hedged:      Sharpe ~0.93,  CAGR ~10.4%")


if __name__ == '__main__':
    main()
