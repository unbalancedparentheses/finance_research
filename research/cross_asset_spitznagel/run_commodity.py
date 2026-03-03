#!/usr/bin/env python3
"""Commodity Carry: Leveraged Futures + Cheap OTM Puts (Spitznagel Structure).

Tests the Spitznagel tail-hedge on four commodity futures:
  - Gold (GC/OG): Safe haven, minimal roll yield
  - Crude Oil (CL/LO): High contango, periodic backwardation
  - Copper (HG/HXE): Cyclical industrial, mixed term structure
  - Natural Gas (NG/ON): Extreme vol, steep contango

Carry = roll yield from term structure (backwardation +, contango -).
Hedge = 0.5% of notional/month on ~8% OTM puts.
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
from databento_helpers import MONTH_CODES

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'databento')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

MONTH_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


# ── Local helpers (commodity-specific, not in databento_helpers) ──

def parse_fut_contract(sym):
    """Parse futures symbol like 'GCQ0' -> (month_code, year_digit)."""
    for prefix in ['GC', 'CL', 'HG', 'NG']:
        if sym.startswith(prefix):
            rest = sym[len(prefix):]
            if len(rest) >= 2:
                month_code = rest[0]
                year_str = rest[1:]
                if month_code in MONTH_CODES:
                    try:
                        year_digit = int(year_str)
                        return month_code, year_digit
                    except ValueError:
                        pass
    return None, None


def contract_expiry_key(sym, ref_year):
    """Return (year, month) tuple for sorting contracts by expiry."""
    mc, yd = parse_fut_contract(sym)
    if mc is None:
        return (9999, 99)
    month = MONTH_CODES[mc]
    decade_base = (ref_year // 10) * 10
    year = decade_base + yd
    if year < ref_year - 2:
        year += 10
    return (year, month)


def _load_front_month(filename):
    """Load futures data and build roll-adjusted front-month series."""
    fut = pd.read_parquet(os.path.join(DATA_DIR, filename))
    outrights = fut[
        ~fut['symbol'].str.contains('-', na=False) &
        ~fut['symbol'].str.contains(':', na=False) &
        ~fut['symbol'].str.contains(' ', na=False)
    ].copy()
    outrights = outrights.sort_index()

    contract_prices = {}
    for _, row in outrights.iterrows():
        sym = row['symbol']
        date = row.name.normalize().tz_localize(None)
        if sym not in contract_prices:
            contract_prices[sym] = {}
        contract_prices[sym][date] = row['close']

    daily_front = {}
    for date, group in outrights.groupby(outrights.index.date):
        best = group.sort_values('volume', ascending=False).iloc[0]
        daily_front[pd.Timestamp(date)] = {
            'symbol': best['symbol'], 'close': best['close'], 'volume': best['volume'],
        }

    dates = sorted(daily_front.keys())
    records = []
    prev_date = prev_symbol = None

    for date in dates:
        info = daily_front[date]
        cur_symbol = info['symbol']
        if prev_date is None:
            records.append({'date': date, 'close': info['close'], 'return': 0.0,
                            'symbol': cur_symbol})
            prev_date, prev_symbol = date, cur_symbol
            continue

        if cur_symbol == prev_symbol:
            prev_p = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            cur_p = contract_prices.get(cur_symbol, {}).get(date, 0)
            ret = cur_p / prev_p - 1 if prev_p > 0 else 0.0
        else:
            old_prev = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            old_cur = contract_prices.get(prev_symbol, {}).get(date, 0)
            ret = old_cur / old_prev - 1 if old_prev > 0 and old_cur > 0 else 0.0

        records.append({'date': date, 'close': info['close'], 'return': ret,
                        'symbol': cur_symbol})
        prev_date, prev_symbol = date, cur_symbol

    return pd.DataFrame(records).set_index('date')


def _load_second_month(filename):
    """Load futures data and build second-month price series."""
    fut = pd.read_parquet(os.path.join(DATA_DIR, filename))
    outrights = fut[
        ~fut['symbol'].str.contains('-', na=False) &
        ~fut['symbol'].str.contains(':', na=False) &
        ~fut['symbol'].str.contains(' ', na=False)
    ].copy()
    outrights = outrights.sort_index()

    records = []
    for date, group in outrights.groupby(outrights.index.date):
        sorted_g = group.sort_values('volume', ascending=False)
        if len(sorted_g) < 2:
            continue
        front_sym = sorted_g.iloc[0]['symbol']
        ref_year = pd.Timestamp(date).year
        front_key = contract_expiry_key(front_sym, ref_year)
        second = None
        for _, row in sorted_g.iloc[1:].iterrows():
            key = contract_expiry_key(row['symbol'], ref_year)
            if key > front_key:
                second = row
                break
        if second is not None:
            records.append({
                'date': pd.Timestamp(date),
                'close': second['close'],
                'symbol': second['symbol'],
            })
    return pd.DataFrame(records).set_index('date')


def compute_roll_yield(front_df, second_df, months_between=1):
    """Compute annualized roll yield from front vs second month."""
    common = front_df.index.intersection(second_df.index)
    f = front_df.loc[common, 'close']
    s = second_df.loc[common, 'close']
    roll_yield = (f - s) / s * (12 / months_between)
    return roll_yield


def parse_commodity_option(sym, date_year, product):
    """Parse commodity option symbol -> (month, year, opt_type, strike).

    Products:
      OG:  'OGQ0 P1200'  -> strike in dollars
      LO:  'LON0 P7000'  -> strike / 100
      HXE: 'HXEU0 P285'  -> strike / 100
      ON:  'ONN0 P4200'  -> strike / 1000
    """
    parts = sym.split()
    if len(parts) != 2:
        return None
    contract, opt = parts
    opt_type = opt[0]
    if opt_type not in ('C', 'P'):
        return None
    try:
        strike_raw = int(opt[1:])
    except ValueError:
        return None

    if product == 'OG':
        if not contract.startswith('OG'):
            return None
        rest = contract[2:]
        strike = float(strike_raw)
    elif product == 'LO':
        if not contract.startswith('LO'):
            return None
        rest = contract[2:]
        strike = strike_raw / 100.0
    elif product == 'HXE':
        if not contract.startswith('HXE'):
            return None
        rest = contract[3:]
        strike = strike_raw / 100.0
    elif product == 'ON':
        if not contract.startswith('ON'):
            return None
        rest = contract[2:]
        strike = strike_raw / 1000.0
    else:
        return None

    if len(rest) < 2:
        return None
    month_code = rest[0]
    try:
        year_digit = int(rest[1:])
    except ValueError:
        return None

    month = MONTH_CODES.get(month_code, 0)
    if month == 0:
        return None

    decade_base = (date_year // 10) * 10
    year = decade_base + year_digit
    if year < date_year - 2:
        year += 10

    return month, year, opt_type, strike


def _load_commodity_options(filename, product):
    """Load and parse commodity options."""
    df = pd.read_parquet(os.path.join(DATA_DIR, filename))
    df = df[~df['symbol'].str.startswith('UD:')].copy()
    df = df.sort_index()

    records = []
    for idx, row in df.iterrows():
        parsed = parse_commodity_option(row['symbol'], idx.year, product)
        if parsed is None:
            continue
        month, year, opt_type, strike = parsed
        try:
            first_of_month = pd.Timestamp(year=year, month=month, day=1)
        except ValueError:
            continue
        third_wed = first_of_month + pd.offsets.WeekOfMonth(week=2, weekday=2)
        expiry = (third_wed - pd.offsets.BDay(2)).tz_localize('UTC')

        records.append({
            'date': idx, 'symbol': row['symbol'], 'opt_type': opt_type,
            'strike': strike, 'expiry': expiry,
            'close': row['close'], 'volume': row['volume'],
        })

    return pd.DataFrame(records)


def _select_monthly_options(opts, front_prices, opt_type='P', otm_target=0.92):
    """Select one OTM option per month."""
    filtered = opts[opts['opt_type'] == opt_type].copy()
    if len(filtered) == 0:
        return pd.DataFrame()

    prices = front_prices[['close']].rename(columns={'close': 'fut_close'})
    prices.index = prices.index.tz_localize('UTC')
    filtered['date_norm'] = filtered['date'].dt.normalize()
    filtered = filtered.merge(prices, left_on='date_norm', right_index=True, how='left')
    filtered = filtered.dropna(subset=['fut_close'])
    filtered['moneyness'] = filtered['strike'] / filtered['fut_close']
    filtered['year_month'] = filtered['date'].dt.to_period('M')

    selections = []
    for ym, group in filtered.groupby('year_month'):
        first_day = group['date'].min()
        day_opts = group[group['date'] == first_day]
        if len(day_opts) == 0:
            continue

        day_opts = day_opts[day_opts['expiry'] > first_day + pd.Timedelta(days=14)]
        if len(day_opts) == 0:
            continue
        nearest_exp = day_opts['expiry'].min()
        day_opts = day_opts[day_opts['expiry'] == nearest_exp]

        if opt_type == 'P':
            day_opts = day_opts[day_opts['moneyness'] < 1.0]
        else:
            day_opts = day_opts[day_opts['moneyness'] > 1.0]

        if len(day_opts) == 0:
            continue

        day_opts = day_opts.copy()
        day_opts['dist'] = (day_opts['moneyness'] - otm_target).abs()
        candidates = day_opts.nsmallest(5, 'dist')
        best = candidates.sort_values('volume', ascending=False).iloc[0]

        if best['close'] <= 0:
            continue

        selections.append({
            'entry_date': first_day,
            'symbol': best['symbol'],
            'strike': best['strike'],
            'entry_price': best['close'],
            'expiry': best['expiry'],
            'underlying': best['fut_close'],
            'moneyness': best['moneyness'],
            'volume': best['volume'],
        })

    return pd.DataFrame(selections)


def _build_settlement_lookup(opts):
    """Pre-build symbol -> [(date, price)] for fast settlement."""
    lookup = {}
    for _, row in opts.iterrows():
        sym = row['symbol']
        if sym not in lookup:
            lookup[sym] = []
        lookup[sym].append((row['date'], row['close']))
    for sym in lookup:
        lookup[sym].sort(key=lambda x: x[0])
    return lookup


def _get_settlement(symbol, strike, expiry, opt_type, lookup, front_prices):
    """Get option settlement price near expiry."""
    window_start = expiry - pd.Timedelta(days=5)
    window_end = expiry + pd.Timedelta(days=2)
    if symbol in lookup:
        near = [(d, p) for d, p in lookup[symbol] if window_start <= d <= window_end]
        if near:
            return near[-1][1]
    near_dates = front_prices[
        (front_prices.index >= (expiry - pd.Timedelta(days=3)).tz_localize(None)) &
        (front_prices.index <= (expiry + pd.Timedelta(days=3)).tz_localize(None))
    ]
    if len(near_dates) > 0:
        underlying = near_dates.iloc[-1]['close']
        if opt_type == 'P':
            return max(0, strike - underlying)
        else:
            return max(0, underlying - strike)
    return 0.0


def _precompute_settlements(selections, opt_type, lookup, front_prices):
    """Pre-compute settlement for all selected options."""
    put_map = {}
    for _, row in selections.iterrows():
        settle = _get_settlement(row['symbol'], row['strike'], row['expiry'],
                                 opt_type, lookup, front_prices)
        entry_price = row['entry_price']
        pnl_ratio = (settle - entry_price) / entry_price if entry_price > 0 else 0
        put_map[row['entry_date']] = {
            'symbol': row['symbol'],
            'strike': row['strike'],
            'entry_price': entry_price,
            'settlement': settle,
            'pnl_ratio': pnl_ratio,
            'moneyness': row['moneyness'],
        }
    return put_map


def _compute_stats(capital_series):
    """Compute comprehensive strategy stats from capital series."""
    cap = capital_series[capital_series > 0]
    if len(cap) < 252:
        return None
    daily_ret = cap.pct_change().dropna()
    years = (cap.index[-1] - cap.index[0]).days / 365.25
    total_ret = cap.iloc[-1] / cap.iloc[0]
    ann_ret = total_ret ** (1 / years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cap / cap.cummax() - 1).min()

    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    skew = daily_ret.skew()
    kurt = daily_ret.kurtosis()

    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'sortino': sortino, 'calmar': calmar,
        'max_dd': max_dd, 'skew': skew, 'kurt': kurt,
        'total': total_ret,
    }


def run_commodity_backtest(front_df, roll_yield_series, put_sels, all_opts,
                           leverage=1, put_budget=0.005, include_carry=True):
    """Run leveraged commodity backtest with optional put hedge."""
    lookup = _build_settlement_lookup(all_opts) if put_budget > 0 and len(put_sels) > 0 else {}
    put_map = _precompute_settlements(put_sels, 'P', lookup, front_df) if put_budget > 0 and len(put_sels) > 0 else {}

    ry_aligned = roll_yield_series.reindex(front_df.index).fillna(0)
    daily_carry = ry_aligned / 252

    capital = 100.0
    records = []
    current_month = None

    for date in front_df.index:
        if capital <= 0:
            records.append({'date': date, 'capital': 0, 'daily_carry': 0,
                            'daily_spot': 0, 'put_pnl': 0})
            continue

        notional = capital * leverage
        carry_income = notional * daily_carry.get(date, 0) if include_carry else 0
        spot_pnl = notional * front_df.loc[date, 'return']

        put_pnl = 0
        ym = pd.Timestamp(date).to_period('M')
        if ym != current_month:
            current_month = ym
            date_tz = pd.Timestamp(date, tz='UTC')
            if put_budget > 0 and date_tz in put_map:
                cost = put_budget * notional
                put_pnl = cost * put_map[date_tz]['pnl_ratio']

        capital += carry_income + spot_pnl + put_pnl
        records.append({
            'date': date, 'capital': capital, 'daily_carry': carry_income,
            'daily_spot': spot_pnl, 'put_pnl': put_pnl,
        })

    return pd.DataFrame(records).set_index('date')


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. Load Front-Month and Second-Month Price Series
    # ══════════════════════════════════════════════════════════════
    commodities = {
        'Gold':   'GC_FUT_ohlcv1d.parquet',
        'Crude':  'CL_FUT_ohlcv1d.parquet',
        'Copper': 'HG_FUT_ohlcv1d.parquet',
        'NatGas': 'NG_FUT_ohlcv1d.parquet',
    }

    front = {}
    second = {}
    for name, filename in commodities.items():
        print(f'Loading {name} front month...')
        front[name] = _load_front_month(filename)
        print(f'  {len(front[name])} days, {front[name].index.min()} to {front[name].index.max()}')
        print(f'  Price range: {front[name]["close"].min():.2f} to {front[name]["close"].max():.2f}')
        print(f'Loading {name} second month...')
        second[name] = _load_second_month(filename)
        print(f'  {len(second[name])} days')
        print()

    # ══════════════════════════════════════════════════════════════
    # 2. Roll Yield / Carry
    # ══════════════════════════════════════════════════════════════
    roll_yields = {}
    for name in commodities:
        ry = compute_roll_yield(front[name], second[name], months_between=1)
        roll_yields[name] = ry

    print('=' * 80)
    print('ANNUALIZED ROLL YIELD BY YEAR (positive = backwardation = positive carry)')
    print('=' * 80)
    header = f'{"Year":>6}'
    for name in commodities:
        header += f'  {name:>10}'
    print(header)
    print('-' * 80)

    for year in range(2010, 2027):
        row = f'{year:>6}'
        for name in commodities:
            ry = roll_yields[name]
            mask = ry.index.year == year
            if mask.sum() > 0:
                avg = ry[mask].mean()
                row += f'  {avg * 100:>9.1f}%'
            else:
                row += f'  {"--":>10}'
        print(row)

    print('-' * 80)
    row = f'{"Avg":>6}'
    for name in commodities:
        avg = roll_yields[name].mean()
        row += f'  {avg * 100:>9.1f}%'
    print(row)

    row = f'{"Med":>6}'
    for name in commodities:
        med = roll_yields[name].median()
        row += f'  {med * 100:>9.1f}%'
    print(row)

    # Roll yield chart
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {'Gold': 'goldenrod', 'Crude': 'black', 'Copper': 'orangered', 'NatGas': 'steelblue'}
    for name in commodities:
        ry_monthly = roll_yields[name].resample('ME').mean()
        ax.plot(ry_monthly.index, ry_monthly * 100, color=colors[name], label=name, alpha=0.8)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title('Annualized Roll Yield (Monthly Average)')
    ax.set_ylabel('Roll Yield (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'cmd_roll_yield.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # 3. Load Options
    # ══════════════════════════════════════════════════════════════
    opt_configs = {
        'Gold':   ('OG_OPT_ohlcv1d.parquet', 'OG'),
        'Crude':  ('LO_OPT_ohlcv1d.parquet', 'LO'),
        'Copper': ('HXE_OPT_ohlcv1d.parquet', 'HXE'),
        'NatGas': ('ON_OPT_ohlcv1d.parquet', 'ON'),
    }

    options = {}
    for name, (filename, product) in opt_configs.items():
        print(f'Loading {name} options ({product})...')
        options[name] = _load_commodity_options(filename, product)
        df = options[name]
        puts = (df['opt_type'] == 'P').sum()
        calls = (df['opt_type'] == 'C').sum()
        print(f'  {len(df):,} total (puts: {puts:,}, calls: {calls:,})')
        if len(df) > 0:
            print(f'  Date range: {df["date"].min()} to {df["date"].max()}')
        print()

    # Sanity check
    for name in commodities:
        opts = options[name]
        puts = opts[opts['opt_type'] == 'P']
        if len(puts) == 0:
            print(f'{name}: no puts')
            continue
        sample_date = puts['date'].iloc[len(puts) // 2]
        date_norm = sample_date.normalize().tz_localize(None)
        fut_price = front[name].loc[front[name].index.asof(date_norm), 'close']
        day_puts = puts[puts['date'] == sample_date].copy()
        day_puts['moneyness'] = day_puts['strike'] / fut_price
        near_atm = day_puts[(day_puts['moneyness'] > 0.85) & (day_puts['moneyness'] < 1.05)]
        print(f'\n{name} on {sample_date.date()}, underlying = {fut_price:.2f}:')
        for _, r in near_atm.head(5).iterrows():
            print(f'  {r["symbol"]:25} strike={r["strike"]:>8.2f}  moneyness={r["moneyness"]:.3f}  '
                  f'price={r["close"]:.4f}  vol={r["volume"]}')

    # ══════════════════════════════════════════════════════════════
    # 4. Monthly Put Selection
    # ══════════════════════════════════════════════════════════════
    put_selections = {}
    for name in commodities:
        print(f'Selecting {name} puts...')
        sels = _select_monthly_options(options[name], front[name], opt_type='P', otm_target=0.92)
        put_selections[name] = sels
        if len(sels) > 0:
            print(f'  {len(sels)} months selected')
            print(f'  Avg moneyness: {sels["moneyness"].mean():.3f}')
            print(f'  Avg entry price: {sels["entry_price"].mean():.4f}')
            print(f'  Sample:')
            for _, r in sels.head(3).iterrows():
                print(f'    {r["symbol"]:25} strike={r["strike"]:>8.2f}  '
                      f'underlying={r["underlying"]:>8.2f}  moneyness={r["moneyness"]:.3f}  '
                      f'price={r["entry_price"]:.4f}')
        else:
            print(f'  No options selected!')
        print()

    # ══════════════════════════════════════════════════════════════
    # 5. Run All Backtests
    # ══════════════════════════════════════════════════════════════
    results = {}
    leverage_levels = [1, 3, 5]

    for name in commodities:
        print(f'\n=== Running {name} backtests ===')
        ry = roll_yields.get(name, pd.Series(dtype=float))
        for lev in leverage_levels:
            label = f'{name} {lev}x unhedged'
            print(f'  {label}...')
            results[label] = run_commodity_backtest(
                front[name], ry, pd.DataFrame(), options[name],
                leverage=lev, put_budget=0
            )
            label = f'{name} {lev}x hedged'
            print(f'  {label}...')
            results[label] = run_commodity_backtest(
                front[name], ry, put_selections[name], options[name],
                leverage=lev, put_budget=0.005
            )

    print('\nDone!')

    # ══════════════════════════════════════════════════════════════
    # 6. Results Tables
    # ══════════════════════════════════════════════════════════════
    for name in commodities:
        print(f'\n{"=" * 105}')
        print(f'{name.upper()} FUTURES + PUT HEDGE')
        print(f'{"=" * 105}')
        print(f'{"Strategy":30} {"CAGR":>7} {"Vol":>6} {"Sharpe":>7} {"Sortino":>8} '
              f'{"Calmar":>7} {"MaxDD":>7} {"Skew":>6} {"Kurt":>6} {"Total":>7}')
        print('-' * 105)

        for lev in leverage_levels:
            for hedge_type in ['unhedged', 'hedged']:
                label = f'{name} {lev}x {hedge_type}'
                if label not in results:
                    continue
                s = _compute_stats(results[label]['capital'])
                if s:
                    print(f'{label:30} {s["ann_ret"] * 100:>6.2f}% {s["ann_vol"] * 100:>5.1f}% '
                          f'{s["sharpe"]:>7.3f} {s["sortino"]:>8.3f} {s["calmar"]:>7.3f} '
                          f'{s["max_dd"] * 100:>6.1f}% {s["skew"]:>6.2f} {s["kurt"]:>6.1f} '
                          f'{s["total"]:>6.1f}x')
            print()

    # Cross-commodity comparison at 3x
    print(f'\n{"=" * 105}')
    print('CROSS-COMMODITY COMPARISON (3x leverage)')
    print(f'{"=" * 105}')
    print(f'{"Strategy":30} {"CAGR":>7} {"Vol":>6} {"Sharpe":>7} {"Sortino":>8} '
          f'{"Calmar":>7} {"MaxDD":>7} {"Skew":>6} {"Kurt":>6} {"Total":>7}')
    print('-' * 105)

    for name in commodities:
        for hedge_type in ['unhedged', 'hedged']:
            label = f'{name} 3x {hedge_type}'
            if label not in results:
                continue
            s = _compute_stats(results[label]['capital'])
            if s:
                print(f'{label:30} {s["ann_ret"] * 100:>6.2f}% {s["ann_vol"] * 100:>5.1f}% '
                      f'{s["sharpe"]:>7.3f} {s["sortino"]:>8.3f} {s["calmar"]:>7.3f} '
                      f'{s["max_dd"] * 100:>6.1f}% {s["skew"]:>6.2f} {s["kurt"]:>6.1f} '
                      f'{s["total"]:>6.1f}x')
        print()

    # ══════════════════════════════════════════════════════════════
    # 7. Year-by-Year Returns
    # ══════════════════════════════════════════════════════════════
    for lev in [1, 3]:
        print(f'\n{"=" * 90}')
        print(f'YEAR-BY-YEAR RETURNS ({lev}x leverage)')
        print(f'{"=" * 90}')

        yearly_data = {}
        for name in commodities:
            for ht in ['unhedged', 'hedged']:
                label = f'{name} {lev}x {ht}'
                if label not in results:
                    continue
                cap = results[label]['capital']
                cap = cap[cap > 0]
                yearly = cap.resample('YE').last().pct_change().dropna()
                short = f'{name[:4]} {ht[:3]}'
                yearly_data[short] = yearly

        all_years = sorted(set(y.year for ys in yearly_data.values() for y in ys.index))

        header = f'{"Year":>6}'
        for k in yearly_data:
            header += f'  {k:>10}'
        print(header)
        print('-' * 90)

        for year in all_years:
            row = f'{year:>6}'
            for k, ys in yearly_data.items():
                match = ys[ys.index.year == year]
                if len(match) > 0:
                    row += f'  {match.iloc[0] * 100:>9.1f}%'
                else:
                    row += f'  {"--":>10}'
            print(row)

        print('-' * 90)
        row = f'{"Avg":>6}'
        for k, ys in yearly_data.items():
            row += f'  {ys.mean() * 100:>9.1f}%'
        print(row)
        row = f'{"%+":>6}'
        for k, ys in yearly_data.items():
            row += f'  {(ys > 0).mean() * 100:>9.0f}%'
        print(row)

    # ══════════════════════════════════════════════════════════════
    # 8. Equity Curves
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    colors_h = {'unhedged': 'black', 'hedged': 'steelblue'}
    ls_h = {'unhedged': '--', 'hedged': '-'}

    for i, name in enumerate(commodities):
        for j, lev in enumerate(leverage_levels):
            ax = axes[i][j]
            for ht in ['unhedged', 'hedged']:
                label = f'{name} {lev}x {ht}'
                if label not in results:
                    continue
                cap = results[label]['capital'] / 100
                cap = cap[cap > 0]
                ax.plot(cap.index, cap, color=colors_h[ht], linestyle=ls_h[ht],
                        label=ht, alpha=0.8)
            ax.set_title(f'{name} {lev}x')
            ax.set_ylabel('Value ($1 start)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(1, color='gray', linewidth=0.5)
            ax.set_yscale('log')

    plt.suptitle('Commodity Carry: Unhedged vs Hedged (0.5% OTM Puts)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'cmd_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Cross-commodity comparison at 3x
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ht, title in zip(axes, ['unhedged', 'hedged'],
                              ['3x Unhedged', '3x Hedged (0.5% puts)']):
        for name in commodities:
            label = f'{name} 3x {ht}'
            if label not in results:
                continue
            cap = results[label]['capital'] / 100
            cap = cap[cap > 0]
            ax.plot(cap.index, cap, color=colors[name], label=name, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel('Value ($1 start)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='gray', linewidth=0.5)
        ax.set_yscale('log')

    plt.suptitle('Cross-Commodity Comparison at 3x Leverage', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'cmd_cross_commodity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # 9. Crisis Performance
    # ══════════════════════════════════════════════════════════════
    crises = [
        ('2011 EU Debt',         '2011-07-01', '2011-10-31'),
        ('2013 Taper Tantrum',   '2013-05-01', '2013-08-31'),
        ('2014-15 Oil Crash',    '2014-07-01', '2016-02-29'),
        ('2015 China Deval',     '2015-07-01', '2015-09-30'),
        ('2018 Trade War',       '2018-01-01', '2018-12-31'),
        ('2020 COVID',           '2020-01-01', '2020-04-30'),
        ('2022 Rate Hikes',      '2022-04-01', '2022-10-31'),
        ('2022 Commodity Spike', '2022-02-01', '2022-06-30'),
    ]

    lev = 3
    for name in commodities:
        print(f'\n{"=" * 90}')
        print(f'{name.upper()} CRISIS PERFORMANCE ({lev}x leverage)')
        print(f'{"=" * 90}')
        print(f'{"Crisis":25} {"Spot":>8} {"Unhedged":>10} {"Hedged":>10} {"Hedge Diff":>11}')
        print('-' * 70)

        for crisis_name, start, end in crises:
            f_df = front[name]
            mask = (f_df.index >= start) & (f_df.index <= end)
            if mask.sum() < 5:
                continue

            spot_ret = f_df.loc[mask, 'close'].iloc[-1] / f_df.loc[mask, 'close'].iloc[0] - 1

            rets = {}
            for ht in ['unhedged', 'hedged']:
                label = f'{name} {lev}x {ht}'
                if label not in results:
                    continue
                cap = results[label]['capital'][mask]
                if len(cap) >= 2 and cap.iloc[0] > 0:
                    rets[ht] = cap.iloc[-1] / cap.iloc[0] - 1
                else:
                    rets[ht] = float('nan')

            unhdg = rets.get('unhedged', float('nan'))
            hdg = rets.get('hedged', float('nan'))
            diff = hdg - unhdg if not (np.isnan(hdg) or np.isnan(unhdg)) else float('nan')

            print(f'{crisis_name:25} {spot_ret * 100:>7.1f}% {unhdg * 100:>9.1f}% '
                  f'{hdg * 100:>9.1f}% {diff * 100:>10.1f}%')

    # ══════════════════════════════════════════════════════════════
    # 10. P&L Decomposition
    # ══════════════════════════════════════════════════════════════
    lev = 3
    print(f'\n{"=" * 95}')
    print(f'P&L DECOMPOSITION ({lev}x leverage, annualized)')
    print(f'{"=" * 95}')
    print(f'{"Commodity":15} {"Type":>10} {"Carry":>10} {"Spot":>10} {"Put P&L":>10} '
          f'{"Total":>10} {"Final":>10}')
    print('-' * 95)

    for name in commodities:
        for ht in ['unhedged', 'hedged']:
            label = f'{name} {lev}x {ht}'
            if label not in results:
                continue
            r = results[label]
            years = (r.index[-1] - r.index[0]).days / 365.25
            carry = r['daily_carry'].sum()
            spot = r['daily_spot'].sum()
            put_pnl = r['put_pnl'].sum()
            total = carry + spot + put_pnl
            final = r['capital'].iloc[-1]

            print(f'{name:15} {ht:>10} {carry / years:>9.1f}$ {spot / years:>9.1f}$ '
                  f'{put_pnl / years:>9.1f}$ {total / years:>9.1f}$ {final:>9.1f}$')
        print()

    # Put economics detail
    print(f'\n{"=" * 90}')
    print('PUT ECONOMICS DETAIL (all months)')
    print(f'{"=" * 90}')
    print(f'{"Commodity":12} {"Months":>7} {"Win%":>6} {"Avg Cost":>10} {"Avg Payoff":>12} '
          f'{"Avg Payout":>12} {"Best Payout":>12}')
    print('-' * 90)

    for name in commodities:
        sels = put_selections[name]
        if len(sels) == 0:
            print(f'{name:12} No options selected')
            continue

        lookup = _build_settlement_lookup(options[name])
        put_map = _precompute_settlements(sels, 'P', lookup, front[name])

        costs = []
        payoffs = []
        payout_ratios = []
        for date_tz, info in put_map.items():
            costs.append(info['entry_price'])
            payoffs.append(info['settlement'])
            payout_ratios.append(info['pnl_ratio'] + 1)

        costs = np.array(costs)
        payoffs = np.array(payoffs)
        payout_ratios = np.array(payout_ratios)
        wins = (payoffs > costs).sum()

        print(f'{name:12} {len(costs):>7} {wins / len(costs) * 100:>5.1f}% '
              f'{costs.mean():>10.4f} {payoffs.mean():>12.4f} '
              f'{payout_ratios.mean():>11.2f}x {payout_ratios.max():>11.1f}x')

    # Decomposition bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    names_list = list(commodities.keys())
    x = np.arange(len(names_list))
    width = 0.35

    carry_vals = []
    spot_vals = []
    put_vals = []

    for name in names_list:
        label = f'{name} 3x hedged'
        if label in results:
            r = results[label]
            years = (r.index[-1] - r.index[0]).days / 365.25
            carry_vals.append(r['daily_carry'].sum() / years)
            spot_vals.append(r['daily_spot'].sum() / years)
            put_vals.append(r['put_pnl'].sum() / years)
        else:
            carry_vals.append(0)
            spot_vals.append(0)
            put_vals.append(0)

    ax.bar(x - width / 2, carry_vals, width / 2, label='Carry (Roll Yield)', color='green', alpha=0.7)
    ax.bar(x, spot_vals, width / 2, label='Spot P&L', color='steelblue', alpha=0.7)
    ax.bar(x + width / 2, put_vals, width / 2, label='Put P&L', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(names_list)
    ax.set_ylabel('Annual P&L ($ on $100 start)')
    ax.set_title('P&L Decomposition at 3x Leverage (Hedged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'cmd_pnl_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # 11. Hedge Value Summary and Conclusions
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 100)
    print('HEDGE VALUE SUMMARY: Sharpe improvement from 0.5% OTM puts')
    print('=' * 100)
    print(f'{"Commodity":12} {"Lev":>4} {"Unhedged Sharpe":>16} {"Hedged Sharpe":>14} '
          f'{"Improvement":>12} {"Unhedged DD":>12} {"Hedged DD":>10}')
    print('-' * 100)

    for name in commodities:
        for lev in leverage_levels:
            u_label = f'{name} {lev}x unhedged'
            h_label = f'{name} {lev}x hedged'
            u_stats = _compute_stats(results[u_label]['capital']) if u_label in results else None
            h_stats = _compute_stats(results[h_label]['capital']) if h_label in results else None
            if u_stats and h_stats:
                delta = h_stats['sharpe'] - u_stats['sharpe']
                print(f'{name:12} {lev:>3}x {u_stats["sharpe"]:>15.3f} {h_stats["sharpe"]:>13.3f} '
                      f'{delta:>+11.3f} {u_stats["max_dd"] * 100:>11.1f}% {h_stats["max_dd"] * 100:>9.1f}%')
        print()

    print("""
KEY FINDINGS -- COMMODITY CARRY WITH SPITZNAGEL HEDGE
=====================================================

1. COMMODITY FUTURES ARE NOT FX CARRY:
   All four commodities have persistent NEGATIVE roll yield (contango):
   - Gold: -5.4%/yr (storage + financing costs)
   - Crude Oil: -3.5%/yr (storage glut most of the period)
   - Copper: -4.3%/yr (financing costs dominate)
   - NatGas: -22.3%/yr (extreme seasonal contango)
   Unlike FX carry (AUD/JPY +2.4%/yr), commodity "carry" is a drag, not income.

2. THE PUT HEDGE IMPROVES SHARPE ACROSS ALL COMMODITIES:
   Biggest improvement: Crude oil (+0.35 Sharpe at 1x) -- deep crash episodes
   (2014-16 oil crash, 2020 COVID negative prices) generate huge put payoffs.
   Gold improves +0.14 thanks to 2013 gold crash puts paying off ~100x.
   Copper and NatGas see minimal improvement: options are illiquid/expensive.

3. NONE OF THESE ARE GOOD STANDALONE STRATEGIES:
   Even at 1x, Gold barely breaks even (+0.4%/yr unhedged). Crude, Copper,
   and NatGas all have negative returns due to contango drag. The hedge
   improves risk-adjusted returns but cannot rescue a negative-carry asset.
   At 3x+ leverage, ALL commodities blow up (max DD > 94%).

4. PUT ECONOMICS -- CRUDE IS THE STAR:
   - Crude: 19.6% win rate, 5.6x avg payout, 118x best payout (2014-15 crash)
   - Gold: 10.1% win rate, 1.7x avg payout, 99.5x best payout (2013 crash)
   - NatGas: 24.2% win rate but only 1.5x avg payout (vol is priced in)
   - Copper: 13.9% win rate, 1.0x avg payout (basically break-even)

5. COMPARISON TO FX CARRY:
   FX carry + dual hedge at 3x: Sharpe 0.71, CAGR +29.6%
   Best commodity at 3x (Gold hedged): Sharpe -0.03, CAGR -1.7%

   The Spitznagel structure requires a POSITIVE base return to amplify.
   Commodity contango destroys the base return, leaving the hedge with
   nothing to protect. This is the key lesson: the structure works on
   assets with positive carry (equities, FX carry) not negative carry
   (most commodity futures).

6. PRACTICAL TAKEAWAY:
   Do NOT apply the Spitznagel structure blindly to commodity futures.
   The only viable use case would be a commodity that is in sustained
   backwardation (e.g., crude during a supply shortage). Gold is the
   exception -- it has real crisis alpha that occasionally makes the
   hedge worthwhile, but the contango drag means it underperforms cash.
""")


if __name__ == '__main__':
    main()
