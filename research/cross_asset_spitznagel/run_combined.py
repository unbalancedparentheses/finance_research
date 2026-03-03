#!/usr/bin/env python3
"""Combined Multi-Strategy Portfolio.

Combines three uncorrelated strategies with Spitznagel-style hedging:
  1. S&P 500 (ES 3x + 25% OTM puts, 0.3% budget)
  2. FX Carry (6 pairs vs JPY, 1x hedged, 8% OTM puts, 0.5% budget)
  3. US-UK Bond Carry (Long ZN / Short Gilt 3x + 4% OTM puts, 0.3% budget)

Analyses dependencies with Pearson correlation and mutual information,
then constructs EW / Risk-Parity / Min-Variance / Max-Sharpe portfolios.
"""
import sys
import os
import re

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from databento_helpers import (
    compute_stats, load_front_month, parse_es_option,
    parse_option_generic, load_fx_options, select_monthly_puts,
    build_settlement_lookup, get_settlement, precompute_settlements,
    mutual_info, MONTH_CODES, STRIKE_DIVISORS, OPT_CONFIGS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'databento')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)


def _run_carry_backtest(cross_df, front_prices, put_sels, all_opts,
                        leverage=1, put_budget=0.005):
    """Run leveraged carry + puts backtest for a single FX pair."""
    if put_budget > 0 and len(put_sels) > 0:
        lookup = build_settlement_lookup(all_opts)
        put_map = precompute_settlements(put_sels, lookup, front_prices)
    else:
        put_map = {}

    capital = 100.0
    records = []
    current_month = None

    for date in cross_df.index:
        if capital <= 0:
            records.append({'date': date, 'capital': 0})
            continue

        notional = capital * leverage
        carry_income = notional * cross_df.loc[date, 'daily_carry']
        spot_pnl = notional * cross_df.loc[date, 'cross_ret']

        put_pnl = 0
        ym = pd.Timestamp(date).to_period('M')
        if ym != current_month:
            current_month = ym
            date_tz = pd.Timestamp(date, tz='UTC')
            if put_budget > 0 and date_tz in put_map:
                cost = put_budget * notional
                put_pnl = cost * put_map[date_tz]['pnl_ratio']

        capital += carry_income + spot_pnl + put_pnl
        records.append({'date': date, 'capital': capital})

    return pd.DataFrame(records).set_index('date')['capital']


def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['figure.dpi'] = 100

    # ═══════════════════════════════════════════════════════════════════
    # Strategy 1: S&P 500 (ES 3x + 25% OTM Puts)
    # ═══════════════════════════════════════════════════════════════════
    print('=== Strategy 1: S&P 500 ===')
    es_fut = load_front_month(DATA_DIR, 'ES_FUT_ohlcv1d.parquet')
    es_daily = es_fut[['close', 'return']].copy().dropna()

    print(f'ES futures: {len(es_daily):,} trading days')
    print(f'Date range: {es_daily.index.min().date()} to {es_daily.index.max().date()}')

    opt_file = os.path.join(DATA_DIR, 'ES_OPT_ohlcv1d.parquet')
    has_es_opts = os.path.exists(opt_file)

    if has_es_opts:
        print('Loading ES options...')
        es_opts_raw = pd.read_parquet(opt_file)
        es_opts_raw.index = es_opts_raw.index.tz_localize(None) if es_opts_raw.index.tz else es_opts_raw.index

        es_puts = es_opts_raw[es_opts_raw['symbol'].str.contains(' P', na=False)].copy()
        es_puts = es_puts[es_puts['close'] > 0]
        es_puts['strike'] = es_puts['symbol'].str.extract(r' P(\d+)').astype(float)
        es_puts = es_puts.dropna(subset=['strike'])
        print(f'ES puts: {len(es_puts):,} rows')

        es_puts['ym'] = es_puts.index.to_period('M')
        puts_by_ym = {ym: grp for ym, grp in es_puts.groupby('ym')}

        OTM_TARGET = 0.75
        es_put_map = {}
        monthly_starts = es_daily.resample('MS').first().index

        for ms in monthly_starts:
            ym = ms.to_period('M')
            if ym not in puts_by_ym:
                continue
            month_mask = (es_daily.index.year == ms.year) & (es_daily.index.month == ms.month)
            if not month_mask.any():
                continue
            spot = es_daily.loc[month_mask, 'close'].iloc[0]

            month_puts = puts_by_ym[ym]
            first_day = month_puts.index.min()
            day_puts = month_puts[month_puts.index == first_day].copy()
            if len(day_puts) == 0:
                continue

            day_puts['moneyness'] = day_puts['strike'] / spot
            candidates = day_puts[(day_puts['moneyness'] > 0.65) & (day_puts['moneyness'] < 0.85)]
            if len(candidates) == 0:
                continue

            candidates = candidates.copy()
            candidates['dist'] = (candidates['moneyness'] - OTM_TARGET).abs()
            best = candidates.iloc[candidates['dist'].values.argmin()]

            sym = str(best['symbol'])
            next_mo = ms.month + 1 if ms.month < 12 else 1
            next_yr = ms.year + (1 if ms.month == 12 else 0)
            settle_mask = (es_puts['symbol'] == sym) & (
                ((es_puts.index.year == ms.year) & (es_puts.index.month == ms.month)) |
                ((es_puts.index.year == next_yr) & (es_puts.index.month == next_mo))
            )
            settle_data = es_puts[settle_mask]
            settle = settle_data['close'].iloc[-1] if len(settle_data) > 0 else 0.0

            entry_px = best['close']
            pnl = (settle / entry_px - 1) if entry_px > 0 else 0.0
            es_put_map[ms] = {'pnl_ratio': pnl, 'symbol': sym, 'entry': entry_px,
                              'settle': settle, 'moneyness': best['moneyness']}

        print(f'Selected {len(es_put_map)} monthly puts (25% OTM target)')
    else:
        es_put_map = {}
        print('ES options not found -- will run unhedged')

    # Run ES backtest
    ES_LEVERAGE = 3
    ES_PUT_BUDGET = 0.003

    capital = 100.0
    es_records = []
    current_month = None

    for date, row in es_daily.iterrows():
        if capital <= 0:
            es_records.append({'date': date, 'capital': 0.0})
            continue
        daily_ret = row['return'] * ES_LEVERAGE
        opt_pnl = 0.0
        ym = date.to_period('M')
        if ym != current_month:
            current_month = ym
            ms = pd.Timestamp(date.year, date.month, 1)
            if ms in es_put_map and ES_PUT_BUDGET > 0:
                opt_pnl = ES_PUT_BUDGET * es_put_map[ms]['pnl_ratio']
        capital *= (1 + daily_ret + opt_pnl)
        es_records.append({'date': date, 'capital': capital})

    es_cap = pd.DataFrame(es_records).set_index('date')['capital']
    es_stats = compute_stats(es_cap)
    print(f'ES {ES_LEVERAGE}x + {ES_PUT_BUDGET*100:.1f}% puts (25% OTM)')
    if es_stats:
        print(f'  CAGR: {es_stats["CAGR"]:.1%}, Vol: {es_stats["Vol"]:.1%}, Sharpe: {es_stats["Sharpe"]:.3f}')
        print(f'  MaxDD: {es_stats["MaxDD"]:.1%}, Total: {es_stats["Total"]:.1f}x')

    # ═══════════════════════════════════════════════════════════════════
    # Strategy 2: FX Carry (6 Pairs vs JPY, Hedged)
    # ═══════════════════════════════════════════════════════════════════
    print('\n=== Strategy 2: FX Carry ===')
    FX_PAIRS = {'AUD': '6A', 'GBP': '6B', 'CAD': '6C', 'EUR': '6E', 'MXN': '6M', 'CHF': '6S'}

    boj_rates = {y: 0.0 for y in range(2010, 2027)}
    boj_rates[2024] = 0.25
    boj_rates[2025] = 0.50
    boj_rates[2026] = 0.50

    policy_rates = {
        'AUD': {2010: 4.25, 2011: 4.50, 2012: 3.50, 2013: 2.75, 2014: 2.50,
                2015: 2.00, 2016: 1.75, 2017: 1.50, 2018: 1.50, 2019: 1.00,
                2020: 0.25, 2021: 0.10, 2022: 1.85, 2023: 4.10, 2024: 4.35,
                2025: 4.35, 2026: 4.10},
        'GBP': {2010: 0.50, 2011: 0.50, 2012: 0.50, 2013: 0.50, 2014: 0.50,
                2015: 0.50, 2016: 0.25, 2017: 0.35, 2018: 0.65, 2019: 0.75,
                2020: 0.25, 2021: 0.15, 2022: 2.00, 2023: 4.75, 2024: 5.00,
                2025: 4.50, 2026: 4.25},
        'CAD': {2010: 0.50, 2011: 1.00, 2012: 1.00, 2013: 1.00, 2014: 1.00,
                2015: 0.75, 2016: 0.50, 2017: 0.75, 2018: 1.50, 2019: 1.75,
                2020: 0.50, 2021: 0.25, 2022: 2.50, 2023: 4.75, 2024: 4.50,
                2025: 3.25, 2026: 3.00},
        'EUR': {2010: 1.00, 2011: 1.25, 2012: 0.75, 2013: 0.50, 2014: 0.15,
                2015: 0.05, 2016: 0.00, 2017: 0.00, 2018: 0.00, 2019: 0.00,
                2020: 0.00, 2021: 0.00, 2022: 1.25, 2023: 4.00, 2024: 4.25,
                2025: 3.15, 2026: 2.65},
        'CHF': {2010: 0.25, 2011: 0.00, 2012: 0.00, 2013: 0.00, 2014: 0.00,
                2015: -0.75, 2016: -0.75, 2017: -0.75, 2018: -0.75, 2019: -0.75,
                2020: -0.75, 2021: -0.75, 2022: 0.25, 2023: 1.50, 2024: 1.50,
                2025: 0.50, 2026: 0.25},
        'MXN': {2010: 4.50, 2011: 4.50, 2012: 4.50, 2013: 4.00, 2014: 3.50,
                2015: 3.25, 2016: 5.00, 2017: 7.00, 2018: 8.00, 2019: 8.00,
                2020: 5.00, 2021: 4.50, 2022: 8.50, 2023: 11.25, 2024: 10.75,
                2025: 9.50, 2026: 8.50},
    }

    fx_data = {}
    for ccy, code_str in FX_PAIRS.items():
        fx_data[ccy] = load_front_month(DATA_DIR, f'{code_str}_FUT_ohlcv1d.parquet')
        print(f'  {ccy}: {len(fx_data[ccy]):,} days')

    jpy = load_front_month(DATA_DIR, '6J_FUT_ohlcv1d.parquet')
    print(f'  JPY: {len(jpy):,} days')

    cross_rates = {}
    for ccy in FX_PAIRS:
        f = fx_data[ccy]
        common = f.index.intersection(jpy.index)
        cr = pd.DataFrame({
            'ccy_ret': f.loc[common, 'return'],
            'jpy_ret': jpy.loc[common, 'return'],
            'cross_ret': f.loc[common, 'return'] - jpy.loc[common, 'return'],
        }, index=common)
        ccy_rate = pd.Series(
            [policy_rates[ccy].get(y, 0) / 100 for y in cr.index.year], index=cr.index)
        jpn_rate = pd.Series(
            [boj_rates.get(y, 0) / 100 for y in cr.index.year], index=cr.index)
        cr['daily_carry'] = (ccy_rate - jpn_rate) / 365
        cross_rates[ccy] = cr

    FX_LEVERAGE = 1
    FX_PUT_BUDGET = 0.005
    FX_OTM = 0.92

    fx_pair_caps = {}
    for ccy in FX_PAIRS:
        cr = cross_rates[ccy]
        front = fx_data[ccy]
        try:
            all_opts = load_fx_options(DATA_DIR, ccy)
            if len(all_opts) > 0:
                put_sels = select_monthly_puts(all_opts, front, otm_target=FX_OTM)
            else:
                put_sels = pd.DataFrame()
        except Exception as e:
            print(f'  {ccy}: option loading failed ({e}), running unhedged')
            put_sels = pd.DataFrame()

        n_puts = len(put_sels) if len(put_sels) > 0 else 0
        budget = FX_PUT_BUDGET if n_puts > 0 else 0

        cap = _run_carry_backtest(cr, front, put_sels,
                                   all_opts if n_puts > 0 else pd.DataFrame(),
                                   leverage=FX_LEVERAGE, put_budget=budget)
        fx_pair_caps[ccy] = cap
        s = compute_stats(cap)
        hedged_str = f'hedged ({n_puts} puts)' if n_puts > 0 else 'unhedged'
        if s:
            print(f'  {ccy}/JPY {FX_LEVERAGE}x {hedged_str}: '
                  f'Sharpe={s["Sharpe"]:.3f}, CAGR={s["CAGR"]:.1%}, MaxDD={s["MaxDD"]:.1%}')

    # Build FX EW portfolio
    daily_rets = {}
    for ccy, cap in fx_pair_caps.items():
        cap = cap[cap > 0]
        if len(cap) > 60:
            daily_rets[ccy] = cap.pct_change().fillna(0)

    ret_df_fx = pd.DataFrame(daily_rets).dropna()
    fx_ew_ret = ret_df_fx.mean(axis=1)
    fx_cap = (1 + fx_ew_ret).cumprod() * 100

    fx_stats = compute_stats(fx_cap)
    print(f'FX Carry EW 6-pair {FX_LEVERAGE}x hedged')
    if fx_stats:
        print(f'  CAGR: {fx_stats["CAGR"]:.1%}, Vol: {fx_stats["Vol"]:.1%}, Sharpe: {fx_stats["Sharpe"]:.3f}')

    # ═══════════════════════════════════════════════════════════════════
    # Strategy 3: US-UK Bond Carry (3x + 4% OTM Puts)
    # ═══════════════════════════════════════════════════════════════════
    print('\n=== Strategy 3: Bond Carry ===')
    zn_raw = pd.read_parquet(os.path.join(DATA_DIR, 'ZN_FUT_ohlcv1d.parquet'))
    gilt_raw = pd.read_parquet(os.path.join(DATA_DIR, 'R_FUT_ohlcv1d.parquet'))
    zn_raw.index = zn_raw.index.tz_localize(None) if zn_raw.index.tz else zn_raw.index
    gilt_raw.index = gilt_raw.index.tz_localize(None) if gilt_raw.index.tz else gilt_raw.index

    zn_all = zn_raw[~zn_raw['symbol'].str.contains('-', na=False)]
    zn_all = zn_all[~zn_all['symbol'].str.startswith('UD:', na=False)]
    zn_all = zn_all.dropna(subset=['close'])
    zn_all = zn_all[zn_all['close'] > 50]

    gilt_all = gilt_raw[~gilt_raw['symbol'].str.contains('-', na=False)]
    gilt_all = gilt_all[~gilt_all['symbol'].str.contains('_Z', na=False)]
    gilt_all = gilt_all.dropna(subset=['close'])
    gilt_all = gilt_all[gilt_all['close'] > 50]

    zn_front = zn_all.loc[zn_all.groupby(zn_all.index)['volume'].idxmax()]
    zn_front = zn_front[['close', 'volume', 'symbol']].copy()
    zn_front = zn_front[~zn_front.index.duplicated(keep='first')]
    zn_front.columns = ['zn_close', 'zn_vol', 'zn_sym']

    gilt_front = gilt_all.loc[gilt_all.groupby(gilt_all.index)['volume'].idxmax()]
    gilt_front = gilt_front[['close', 'volume', 'symbol']].copy()
    gilt_front = gilt_front[~gilt_front.index.duplicated(keep='first')]
    gilt_front.columns = ['gilt_close', 'gilt_vol', 'gilt_sym']

    bond_df = zn_front.join(gilt_front, how='inner').sort_index()
    bond_df['zn_ret'] = bond_df['zn_close'].pct_change()
    bond_df['gilt_ret'] = bond_df['gilt_close'].pct_change()
    bond_df = bond_df.dropna()

    monthly = bond_df[['zn_ret', 'gilt_ret']].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly['long_zn'] = monthly['zn_ret'] - monthly['gilt_ret']
    monthly = monthly.dropna()

    print(f'Bond carry data: {len(bond_df):,} daily, {len(monthly)} monthly')

    # Load OZN options
    ozn_raw = pd.read_parquet(os.path.join(DATA_DIR, 'OZN_OPT_ohlcv1d.parquet'))
    ozn_raw.index = ozn_raw.index.tz_localize(None) if ozn_raw.index.tz else ozn_raw.index

    pat = re.compile(r'^OZN\s*([FGHJKMNQUVXZ])(\d)\s+([PC])(\d+)')
    records = []
    for sym in ozn_raw['symbol'].unique():
        m = pat.match(str(sym))
        if not m:
            continue
        month_code, year_digit, pc, strike_str = m.groups()
        records.append({'symbol': sym, 'opt_type': pc, 'strike': int(strike_str) / 10.0})

    opt_meta = pd.DataFrame(records)
    puts_meta = opt_meta[opt_meta['opt_type'] == 'P'].copy()
    ozn_puts = ozn_raw.reset_index().merge(puts_meta, on='symbol', how='inner').set_index('ts_event')
    ozn_puts = ozn_puts[ozn_puts.index >= bond_df.index.min()]
    ozn_puts = ozn_puts[ozn_puts['close'] > 0]

    BOND_OTM = 0.96
    ozn_put_map = {}
    for month_start in monthly.index:
        yr, mo = month_start.year, month_start.month
        mask_zn = (bond_df.index.year == yr) & (bond_df.index.month == mo)
        zn_prices = bond_df.loc[mask_zn, 'zn_close']
        if len(zn_prices) == 0:
            continue
        spot = zn_prices.iloc[0]
        month_opts = ozn_puts[(ozn_puts.index.year == yr) & (ozn_puts.index.month == mo)]
        if len(month_opts) == 0:
            continue
        first_day = month_opts.index.min()
        day_opts = month_opts[month_opts.index == first_day].copy()
        if len(day_opts) == 0:
            continue
        day_opts['moneyness'] = day_opts['strike'] / spot
        lo, hi = BOND_OTM - 0.04, BOND_OTM + 0.04
        candidates = day_opts[(day_opts['moneyness'].values > lo) & (day_opts['moneyness'].values < hi)]
        candidates = candidates[candidates['close'].values > 0]
        candidates = candidates[candidates['volume'].values > 0]
        if len(candidates) == 0:
            continue
        candidates = candidates.copy()
        candidates['dist'] = (candidates['moneyness'].values - BOND_OTM).__abs__()
        best = candidates.iloc[candidates['dist'].values.argmin()]
        best_sym = best['symbol']
        mo_next = mo + 1 if mo < 12 else 1
        yr_next = yr + (1 if mo == 12 else 0)
        mask = (ozn_puts['symbol'].values == best_sym) & (
            ((ozn_puts.index.year == yr) & (ozn_puts.index.month == mo)) |
            ((ozn_puts.index.year == yr_next) & (ozn_puts.index.month == mo_next))
        )
        end_opts = ozn_puts[mask]
        settle = end_opts['close'].iloc[-1] if len(end_opts) > 0 else 0.0
        settle = settle if pd.notna(settle) else 0.0
        entry_px = float(best['close'])
        ozn_put_map[month_start] = {
            'pnl_ratio': (settle / entry_px - 1) if entry_px > 0 else 0.0,
            'entry': entry_px, 'settle': settle,
        }

    print(f'Selected {len(ozn_put_map)} monthly OZN puts (4% OTM)')

    BOND_LEVERAGE = 3
    BOND_PUT_BUDGET = 0.003

    capital_list = [1.0]
    for date, row in monthly['long_zn'].items():
        spread_ret = row * BOND_LEVERAGE
        opt_pnl = 0.0
        if BOND_PUT_BUDGET > 0 and date in ozn_put_map:
            opt = ozn_put_map[date]
            opt_pnl = BOND_PUT_BUDGET * opt['pnl_ratio']
        total_ret = spread_ret + opt_pnl
        capital_list.append(capital_list[-1] * (1 + total_ret))

    bond_cap_monthly = pd.Series(capital_list[1:], index=monthly.index) * 100
    bond_cap = bond_cap_monthly.reindex(bond_df.index, method='ffill').dropna()
    if len(bond_cap) > 0:
        bond_cap = bond_cap / bond_cap.iloc[0] * 100

    bond_stats = compute_stats(bond_cap)
    print(f'Bond Carry {BOND_LEVERAGE}x + {BOND_PUT_BUDGET*100:.1f}% OZN puts')
    if bond_stats:
        print(f'  CAGR: {bond_stats["CAGR"]:.1%}, Vol: {bond_stats["Vol"]:.1%}, Sharpe: {bond_stats["Sharpe"]:.3f}')

    # ═══════════════════════════════════════════════════════════════════
    # Individual Performance
    # ═══════════════════════════════════════════════════════════════════
    strat_caps = {
        'ES 3x hedged': es_cap,
        'FX Carry EW': fx_cap,
        'Bond Carry': bond_cap,
    }

    print('\n' + '=' * 110)
    print('INDIVIDUAL STRATEGY PERFORMANCE')
    print('=' * 110)
    header = f'  {"Strategy":>20s}  {"CAGR":>7s}  {"Vol":>7s}  {"Sharpe":>7s}  {"Sortino":>8s}  {"Calmar":>7s}  {"MaxDD":>7s}  {"Total":>7s}'
    print(header)
    print('-' * 110)
    for name, cap in strat_caps.items():
        s = compute_stats(cap)
        if s:
            print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
                  f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Total"]:>6.1f}x')

    # ── Individual equity curves ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    colors = {'ES 3x hedged': 'royalblue', 'FX Carry EW': 'forestgreen', 'Bond Carry': 'firebrick'}
    for name, cap in strat_caps.items():
        norm = cap / cap.iloc[0]
        ax1.plot(norm.index, norm.values, label=name, color=colors[name], lw=1.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Return (log scale)')
    ax1.set_title('Individual Strategy Equity Curves')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    for name, cap in strat_caps.items():
        dd = cap / cap.cummax() - 1
        ax2.fill_between(dd.index, dd.values * 100, 0, alpha=0.2, color=colors[name])
        ax2.plot(dd.index, dd.values * 100, color=colors[name], lw=0.8, alpha=0.7, label=name)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'comb_individual_equity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # Dependency Analysis
    # ═══════════════════════════════════════════════════════════════════
    aligned = pd.DataFrame({name: cap for name, cap in strat_caps.items()}).dropna()
    print(f'\nCommon date range: {aligned.index.min().date()} to {aligned.index.max().date()} ({len(aligned):,} days)')

    ret_df = aligned.pct_change().dropna()
    strat_names = list(strat_caps.keys())
    n = len(strat_names)

    pearson = ret_df.corr()
    print('\nPEARSON CORRELATION MATRIX:')
    print(pearson.round(4).to_string())

    mi_matrix = pd.DataFrame(np.zeros((n, n)), index=strat_names, columns=strat_names)
    for i in range(n):
        for j in range(n):
            mi_matrix.iloc[i, j] = mutual_info(ret_df[strat_names[i]].values,
                                                 ret_df[strat_names[j]].values, bins=50)

    print('\nMUTUAL INFORMATION MATRIX:')
    print(mi_matrix.round(4).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    short_names = ['ES', 'FX', 'Bond']
    p_plot = pearson.copy()
    p_plot.index = short_names
    p_plot.columns = short_names
    sns.heatmap(p_plot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax1, square=True, linewidths=0.5)
    ax1.set_title('Pearson Correlation', fontsize=13)
    m_plot = mi_matrix.copy()
    m_plot.index = short_names
    m_plot.columns = short_names
    sns.heatmap(m_plot, annot=True, fmt='.4f', cmap='YlOrRd', vmin=0, ax=ax2, square=True, linewidths=0.5)
    ax2.set_title('Mutual Information (nats)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'comb_dependency.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # Portfolio Construction
    # ═══════════════════════════════════════════════════════════════════
    def solve_min_variance(cov_matrix, max_iter=1000):
        n_assets = cov_matrix.shape[0]
        if n_assets == 0:
            return np.array([])
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            return np.ones(n_assets) / n_assets
        w = np.ones(n_assets) / n_assets
        lr = 0.5
        for _ in range(max_iter):
            grad = 2 * cov_matrix @ w
            w_new = w - lr * grad
            w_new = np.maximum(w_new, 0)
            s = w_new.sum()
            w_new = w_new / s if s > 0 else np.ones(n_assets) / n_assets
            if np.max(np.abs(w_new - w)) < 1e-8:
                break
            w = w_new
        return w

    def solve_max_sharpe(mean_ret, cov_matrix, rf=0.0, max_iter=2000):
        n_assets = len(mean_ret)
        if n_assets == 0:
            return np.array([])
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            return np.ones(n_assets) / n_assets
        excess = mean_ret - rf
        w = np.ones(n_assets) / n_assets
        lr = 0.3
        for _ in range(max_iter):
            port_ret = w @ excess
            port_var = w @ cov_matrix @ w
            port_vol = np.sqrt(max(port_var, 1e-12))
            d_ret = excess
            d_var = 2 * cov_matrix @ w
            d_vol = d_var / (2 * port_vol)
            grad = (port_vol * d_ret - port_ret * d_vol) / (port_var + 1e-12)
            w_new = w + lr * grad
            w_new = np.maximum(w_new, 0)
            s = w_new.sum()
            w_new = w_new / s if s > 0 else np.ones(n_assets) / n_assets
            if np.max(np.abs(w_new - w)) < 1e-8:
                break
            w = w_new
        return w

    port_results = {}
    port_weights = {}

    # 1. Equal-Weight
    ew_ret = ret_df.mean(axis=1)
    ew_cap = (1 + ew_ret).cumprod() * 100
    port_results['Equal-Weight'] = ew_cap

    # 2. Risk-Parity
    VOL_WINDOW = 60
    trailing_vol = ret_df.rolling(VOL_WINDOW).std() * np.sqrt(252)
    rebal_monthly = ret_df.resample('ME').last().index
    rp_rets = []
    current_w = None
    for date in ret_df.index:
        if date in rebal_monthly or current_w is None:
            vol_row = trailing_vol.loc[:date].iloc[-1] if date in trailing_vol.index else None
            if vol_row is not None and vol_row.notna().sum() >= 2:
                inv_vol = 1.0 / vol_row.replace(0, np.nan).dropna()
                if len(inv_vol) > 0:
                    current_w = inv_vol / inv_vol.sum()
        if current_w is not None:
            day_ret = ret_df.loc[date]
            cc = current_w.index.intersection(day_ret.dropna().index)
            if len(cc) > 0:
                ww = current_w[cc] / current_w[cc].sum()
                rp_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})
    rp_df = pd.DataFrame(rp_rets).set_index('date')
    rp_cap = (1 + rp_df['return']).cumprod() * 100
    port_results['Risk-Parity'] = rp_cap

    # 3. Min-Variance
    COV_WINDOW = 252
    mv_rets = []
    current_w = None
    for date in ret_df.index:
        if date in rebal_monthly or current_w is None:
            lookback = ret_df.loc[:date].tail(COV_WINDOW)
            avail = lookback.dropna(axis=1, how='all')
            avail = avail.loc[:, avail.notna().sum() >= COV_WINDOW // 2]
            if avail.shape[1] >= 2:
                cov = avail.cov().values
                cols = list(avail.columns)
                w = solve_min_variance(cov)
                current_w = pd.Series(w, index=cols)
        if current_w is not None:
            day_ret = ret_df.loc[date]
            cc = current_w.index.intersection(day_ret.dropna().index)
            if len(cc) > 0:
                ww = current_w[cc] / current_w[cc].sum()
                mv_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})
    mv_df = pd.DataFrame(mv_rets).set_index('date')
    mv_cap = (1 + mv_df['return']).cumprod() * 100
    port_results['Min-Variance'] = mv_cap

    # 4. Max-Sharpe
    rebal_quarterly = ret_df.resample('QE').last().index
    ms_rets = []
    current_w = None
    for date in ret_df.index:
        if date in rebal_quarterly or current_w is None:
            lookback = ret_df.loc[:date].tail(COV_WINDOW)
            avail = lookback.dropna(axis=1, how='all')
            avail = avail.loc[:, avail.notna().sum() >= COV_WINDOW // 2]
            if avail.shape[1] >= 2:
                mean_r = avail.mean().values * 252
                cov = avail.cov().values * 252
                cols = list(avail.columns)
                w = solve_max_sharpe(mean_r, cov)
                current_w = pd.Series(w, index=cols)
        if current_w is not None:
            day_ret = ret_df.loc[date]
            cc = current_w.index.intersection(day_ret.dropna().index)
            if len(cc) > 0:
                ww = current_w[cc] / current_w[cc].sum()
                ms_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})
    ms_df = pd.DataFrame(ms_rets).set_index('date')
    ms_cap = (1 + ms_df['return']).cumprod() * 100
    port_results['Max-Sharpe'] = ms_cap

    # ── Portfolio stats ──
    print('\n' + '=' * 115)
    print('PORTFOLIO vs INDIVIDUAL STRATEGY PERFORMANCE')
    print('=' * 115)
    header = f'  {"Name":>20s}  {"CAGR":>7s}  {"Vol":>7s}  {"Sharpe":>7s}  {"Sortino":>8s}  {"Calmar":>7s}  {"MaxDD":>7s}  {"Total":>7s}'
    print(header)
    print('-' * 115)

    print('  -- Individual Strategies (common period) --')
    for name in strat_names:
        s = compute_stats(aligned[name])
        if s:
            print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
                  f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Total"]:>6.1f}x')

    print('  -- Portfolios --')
    best_port_sharpe = -999
    best_port_name = ''
    for name, cap in port_results.items():
        s = compute_stats(cap)
        if s:
            print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
                  f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Total"]:>6.1f}x')
            if s['Sharpe'] > best_port_sharpe:
                best_port_sharpe = s['Sharpe']
                best_port_name = name

    best_ind_sharpe = max(compute_stats(aligned[name])['Sharpe'] for name in strat_names
                          if compute_stats(aligned[name]) is not None)
    print()
    print(f'Best portfolio: {best_port_name} (Sharpe {best_port_sharpe:.3f})')
    print(f'Best individual: Sharpe {best_ind_sharpe:.3f}')
    print(f'Diversification benefit: {best_port_sharpe - best_ind_sharpe:+.3f} Sharpe improvement')

    # ── Portfolio equity curves ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    for name in strat_names:
        cap_a = aligned[name] / aligned[name].iloc[0]
        ax1.plot(cap_a.index, cap_a.values, ls='--', alpha=0.5, lw=1, label=f'{name} (ind.)')

    port_colors = {'Equal-Weight': 'tab:blue', 'Risk-Parity': 'tab:orange',
                   'Min-Variance': 'tab:green', 'Max-Sharpe': 'tab:red'}
    for name, cap in port_results.items():
        norm = cap / cap.iloc[0]
        ax1.plot(norm.index, norm.values, color=port_colors.get(name, 'black'), lw=2, label=name)

    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Return (log)')
    ax1.set_title('Portfolio Equity Curves vs Individual Strategies')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    for name, cap in port_results.items():
        dd = cap / cap.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, color=port_colors.get(name, 'black'), lw=1, label=name)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'comb_portfolio_equity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Conclusions ──
    print('\nCOMBINED MULTI-STRATEGY PORTFOLIO -- KEY FINDINGS')
    print('=' * 70)

    print(f'\n1. BEST PORTFOLIO METHOD: {best_port_name}')
    print(f'   Sharpe: {best_port_sharpe:.3f}')
    s = compute_stats(port_results[best_port_name])
    if s:
        print(f'   CAGR: {s["CAGR"]:.1%}, Vol: {s["Vol"]:.1%}, MaxDD: {s["MaxDD"]:.1%}')

    print(f'\n2. DIVERSIFICATION BENEFIT:')
    print(f'   Best individual Sharpe: {best_ind_sharpe:.3f}')
    print(f'   Best portfolio Sharpe:  {best_port_sharpe:.3f}')
    print(f'   Improvement: {best_port_sharpe - best_ind_sharpe:+.3f}')

    print(f'\n3. CROSS-STRATEGY CORRELATIONS:')
    for i in range(n):
        for j in range(i+1, n):
            rho = pearson.iloc[i, j]
            mi = mi_matrix.iloc[i, j]
            print(f'   {strat_names[i]:>15} vs {strat_names[j]:<15}: rho={rho:+.3f}, MI={mi:.4f}')

    print(f'\n4. PORTFOLIO RANKING (by Sharpe):')
    port_ranking = []
    for p_name, p_cap in port_results.items():
        p_s = compute_stats(p_cap)
        if p_s:
            port_ranking.append((p_name, p_s['Sharpe'], p_s['CAGR'], p_s['MaxDD']))
    port_ranking.sort(key=lambda x: x[1], reverse=True)
    for rank, (p_name, sharpe, cagr, maxdd) in enumerate(port_ranking, 1):
        print(f'   {rank}. {p_name:>15}: Sharpe={sharpe:.3f}, CAGR={cagr:.1%}, MaxDD={maxdd:.1%}')


if __name__ == '__main__':
    main()
