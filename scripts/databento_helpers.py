"""Shared helper functions for Databento data build scripts.

Extracted from build_combined_portfolio_nb.py, build_equity_nb.py,
build_treasury_nb.py and related notebooks. Provides futures/options
parsing, roll-adjusted front-month loading, FX option selection,
settlement logic, and portfolio statistics.
"""

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_CODES = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}

STRIKE_DIVISORS = {
    'AUD': 1000, 'GBP': 1000, 'CAD': 1000, 'EUR': 1000,
    'CHF': 1000, 'NZD': 1000, 'MXN': 10000, 'JPY': 100000,
}

OPT_CONFIGS = {
    'AUD': ('6A_OPT_ohlcv1d.parquet', 'ADU_OPT_ohlcv1d.parquet',
            ['6A'], ['ADU'], '2016-08-23'),
    'GBP': ('6B_OPT_ohlcv1d.parquet', 'GBU_OPT_ohlcv1d.parquet',
            ['6B'], ['GBU'], '2016-08-23'),
    'CAD': ('6C_OPT_ohlcv1d.parquet', 'CAU_OPT_ohlcv1d.parquet',
            ['6C'], ['CAU'], '2016-08-22'),
    'EUR': ('6E_OPT_ohlcv1d.parquet', 'EUU_OPT_ohlcv1d.parquet',
            ['6E'], ['EUU'], '2016-08-09'),
    'CHF': ('6S_OPT_ohlcv1d.parquet', None,
            ['6S'], None, None),
    'MXN': ('6M_OPT_ohlcv1d.parquet', None,
            ['6M'], None, None),
}

# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------


def compute_stats(cap):
    """Compute strategy statistics from a capital series."""
    cap = cap[cap > 0]
    if len(cap) < 60:
        return None
    daily_ret = cap.pct_change().dropna()
    if len(daily_ret) < 2:
        return None
    years = (cap.index[-1] - cap.index[0]).days / 365.25
    if years < 0.5:
        return None
    total = cap.iloc[-1] / cap.iloc[0]
    ann_ret = total ** (1 / years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = cap / cap.cummax() - 1
    max_dd = dd.min()
    downside = daily_ret[daily_ret < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        'CAGR': ann_ret, 'Vol': ann_vol, 'Sharpe': sharpe,
        'Sortino': sortino, 'Calmar': calmar, 'MaxDD': max_dd,
        'Skew': daily_ret.skew(), 'Kurt': daily_ret.kurtosis(),
        'Total': total,
    }


# ---------------------------------------------------------------------------
# Front-month futures loading
# ---------------------------------------------------------------------------


def load_front_month(data_dir, filename):
    """Load Databento futures -> roll-adjusted front-month series.

    Front month = highest-volume contract per day.
    On roll days uses OLD contract's return (not price gap).
    """
    raw = pd.read_parquet(f'{data_dir}/{filename}')
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
    outrights = raw[~raw['symbol'].str.contains('-', na=False)].copy()
    outrights = outrights[~outrights['symbol'].str.startswith('UD:', na=False)]
    outrights = outrights.dropna(subset=['close'])
    outrights = outrights[outrights['close'] > 0]
    outrights = outrights.sort_index()

    contract_prices = {}
    for _, row in outrights.iterrows():
        sym = row['symbol']
        date = row.name.normalize()
        if sym not in contract_prices:
            contract_prices[sym] = {}
        contract_prices[sym][date] = row['close']

    daily_front = {}
    for ts, grp in outrights.groupby(outrights.index):
        best = grp.iloc[grp['volume'].values.argmax()]
        date = ts.normalize()
        daily_front[date] = {'symbol': str(best['symbol']),
                             'close': float(best['close']),
                             'volume': float(best['volume'])}

    dates = sorted(daily_front.keys())
    records = []
    prev_date = None
    prev_symbol = None

    for date in dates:
        info = daily_front[date]
        cur_symbol = info['symbol']
        if prev_date is None:
            records.append({'date': date, 'close': info['close'],
                            'volume': info['volume'], 'symbol': cur_symbol,
                            'return': 0.0})
            prev_date = date
            prev_symbol = cur_symbol
            continue

        if cur_symbol == prev_symbol:
            prev_price = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            cur_price = contract_prices.get(cur_symbol, {}).get(date, 0)
            ret = cur_price / prev_price - 1 if prev_price > 0 else 0.0
        else:
            old_prev = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            old_cur = contract_prices.get(prev_symbol, {}).get(date, 0)
            ret = old_cur / old_prev - 1 if old_prev > 0 and old_cur > 0 else 0.0

        records.append({'date': date, 'close': info['close'],
                        'volume': info['volume'], 'symbol': cur_symbol,
                        'return': ret})
        prev_date = date
        prev_symbol = cur_symbol

    df = pd.DataFrame(records).set_index('date')
    df = df[~df.index.duplicated(keep='first')]
    return df


# ---------------------------------------------------------------------------
# Option symbol parsers
# ---------------------------------------------------------------------------


def parse_es_option(symbol):
    """Parse ES option symbol like 'ESM5 P4200' or 'EW1M5 P4200'.

    ES options use whole-number strikes (no divisor needed).
    """
    parts = symbol.split()
    if len(parts) != 2:
        return None
    root, strike_str = parts

    if strike_str.startswith('P'):
        opt_type = 'P'
        strike_val = float(strike_str[1:])
    elif strike_str.startswith('C'):
        opt_type = 'C'
        strike_val = float(strike_str[1:])
    else:
        return None

    strike = strike_val

    suffix = None
    for prefix in ['EW1', 'EW2', 'EW3', 'EW4', 'E1A', 'E2A', 'E3A', 'E4A', 'ES']:
        if root.startswith(prefix):
            suffix = root[len(prefix):]
            break
    if suffix is None or len(suffix) < 2:
        return None

    month_char = suffix[0]
    year_digit = suffix[1]

    if month_char not in MONTH_CODES:
        return None
    try:
        yr = int(year_digit)
    except ValueError:
        return None

    month = MONTH_CODES[month_char]
    year = 2010 + yr if yr >= 0 else 2020 + yr
    if year < 2010:
        year += 10

    first_day = pd.Timestamp(year, month, 1)
    day_of_week = first_day.dayofweek
    first_friday = first_day + pd.Timedelta(days=(4 - day_of_week) % 7)
    third_friday = first_friday + pd.Timedelta(days=14)

    return {
        'opt_type': opt_type,
        'strike': strike,
        'month': month,
        'year': year,
        'expiry': third_friday,
    }


def parse_treasury_option(symbol, prefix='OZN', strike_divisor=10):
    """Parse treasury option symbol like 'OZNF1 P1290'.

    OZN uses strike_divisor=10 (1290 -> 129.0).
    OZB uses strike_divisor=1 (1190 -> 1190).
    """
    pat = re.compile(rf'^{prefix}\s*([FGHJKMNQUVXZ])(\d)\s+([PC])(\d+)')
    m = pat.match(str(symbol))
    if not m:
        return None
    month_code, year_digit, pc, strike_str = m.groups()
    return {
        'opt_month': MONTH_CODES[month_code],
        'opt_year_digit': int(year_digit),
        'opt_type': pc,
        'strike_raw': int(strike_str),
        'strike': int(strike_str) / strike_divisor,
    }


def parse_option_generic(sym, date_year, prefixes, strike_div):
    """Parse CME FX option symbol -> (month, year, opt_type, strike)."""
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
    strike = strike_raw / strike_div

    month_code = None
    year_digit = None
    for pfx in prefixes:
        if contract.startswith(pfx):
            rest = contract[len(pfx):]
            if len(rest) >= 2:
                month_code = rest[0]
                try:
                    year_digit = int(rest[1])
                except ValueError:
                    continue
                break

    if month_code is None or year_digit is None:
        return None
    month = MONTH_CODES.get(month_code, 0)
    if month == 0:
        return None

    decade_base = (date_year // 10) * 10
    year = decade_base + year_digit
    if year < date_year - 2:
        year += 10
    return month, year, opt_type, strike


# ---------------------------------------------------------------------------
# FX options loading and selection
# ---------------------------------------------------------------------------


def load_fx_options(data_dir, ccy):
    """Load FX options for a currency, merging old + new format files."""
    config = OPT_CONFIGS[ccy]
    old_file, new_file, old_prefixes, new_prefixes, cutoff = config
    strike_div = STRIKE_DIVISORS[ccy]

    old = pd.read_parquet(f'{data_dir}/{old_file}')
    old = old[~old['symbol'].str.contains('UD:', na=False)].copy()

    if new_file is not None:
        new = pd.read_parquet(f'{data_dir}/{new_file}')
        new = new[~new['symbol'].str.contains('UD:', na=False)].copy()
        cutoff_ts = pd.Timestamp(cutoff, tz='UTC')
        old = old[old.index < cutoff_ts]
        combined = pd.concat([old, new]).sort_index()
        all_prefixes = old_prefixes + new_prefixes
    else:
        combined = old.sort_index()
        all_prefixes = old_prefixes

    records = []
    for idx, row in combined.iterrows():
        parsed = parse_option_generic(row['symbol'], idx.year, all_prefixes, strike_div)
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


def select_monthly_puts(opts, front_prices, otm_target=0.92):
    """Select one OTM put per month for FX hedging."""
    filtered = opts[opts['opt_type'] == 'P'].copy()
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
        day_opts = day_opts[day_opts['moneyness'] < 1.0]
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


# ---------------------------------------------------------------------------
# Settlement helpers
# ---------------------------------------------------------------------------


def build_settlement_lookup(opts):
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


def get_settlement(symbol, strike, expiry, lookup, front_prices):
    """Get option settlement price from market data or intrinsic value."""
    window_start = expiry - pd.Timedelta(days=5)
    window_end = expiry + pd.Timedelta(days=2)
    if symbol in lookup:
        near = [(d, p) for d, p in lookup[symbol] if window_start <= d <= window_end]
        if near:
            return near[-1][1]
    fp_idx = front_prices.index
    if fp_idx.tz is not None:
        near_dates = front_prices[
            (fp_idx >= (expiry - pd.Timedelta(days=3))) &
            (fp_idx <= (expiry + pd.Timedelta(days=3)))
        ]
    else:
        exp_naive = expiry.tz_localize(None) if hasattr(expiry, 'tz_localize') and expiry.tzinfo else expiry
        near_dates = front_prices[
            (fp_idx >= (exp_naive - pd.Timedelta(days=3))) &
            (fp_idx <= (exp_naive + pd.Timedelta(days=3)))
        ]
    if len(near_dates) > 0:
        underlying = near_dates.iloc[-1]['close']
        return max(0, strike - underlying)
    return 0.0


def precompute_settlements(selections, lookup, front_prices):
    """Pre-compute settlement for all selected puts."""
    put_map = {}
    for _, row in selections.iterrows():
        settle = get_settlement(row['symbol'], row['strike'], row['expiry'],
                                lookup, front_prices)
        entry_price = row['entry_price']
        pnl_ratio = (settle - entry_price) / entry_price if entry_price > 0 else 0
        put_map[row['entry_date']] = {
            'symbol': row['symbol'], 'strike': row['strike'],
            'entry_price': entry_price, 'settlement': settle,
            'pnl_ratio': pnl_ratio, 'moneyness': row['moneyness'],
        }
    return put_map


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------


def mutual_info(x, y, bins=50):
    """Mutual information via 2D histogram (no sklearn needed)."""
    from scipy.stats import entropy
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = entropy(px[px > 0])
    hy = entropy(py[py > 0])
    hxy = entropy(pxy.flatten()[pxy.flatten() > 0])
    return hx + hy - hxy
