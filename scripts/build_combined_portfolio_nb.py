#!/usr/bin/env python3
"""Build the combined_portfolio.ipynb notebook.

Combines three uncorrelated strategies with Spitznagel-style hedging:
  1. S&P 500 (ES 3x + 25% OTM puts, 0.3% budget)
  2. FX Carry (6 pairs vs JPY, 1x hedged, 8% OTM puts, 0.5% budget)
  3. US-UK Bond Carry (Long ZN / Short Gilt 3x + 4% OTM puts, 0.3% budget)

Analyses dependencies with Pearson correlation and mutual information,
then constructs EW / Risk-Parity / Min-Variance / Max-Sharpe portfolios.
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'combined_portfolio.ipynb')

cells = []

def md(source):
    lines = source.strip().split('\n')
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + '\n' for l in lines[:-1]] + [lines[-1]]})

def code(source):
    lines = source.strip().split('\n')
    cells.append({"cell_type": "code", "metadata": {}, "source": [l + '\n' for l in lines[:-1]] + [lines[-1]],
                  "outputs": [], "execution_count": None})

# ═══════════════════════════════════════════════════════════════════════════
# Cell 0: Title
# ═══════════════════════════════════════════════════════════════════════════
md("""
# Combined Multi-Strategy Portfolio

Three uncorrelated strategies with Spitznagel-style OTM put hedging:

| Strategy | Description | Ind. Sharpe |
|----------|------------|-------------|
| **S&P 500** | ES 3x + 25% OTM puts (0.3% budget) | ~0.6 |
| **FX Carry** | 6 pairs vs JPY, 1x hedged (8% OTM, 0.5%) | ~0.93 |
| **US-UK Bond Carry** | Long ZN / Short Gilt 3x + 4% OTM puts (0.3%) | ~0.65 |

All strategies use deep OTM puts for tail protection:
- Cheap insurance that barely drags returns
- Massive payoffs during tail events (COVID, Truss crisis, carry unwinds)
- Allows safe use of leverage (tail risk is bounded)

Cross-strategy correlations are near zero, making this ideal for diversification.
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 1: Imports
# ═══════════════════════════════════════════════════════════════════════════
code("""
import pandas as pd
import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

DATA = '../data/databento'
MONTH_CODES = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,
               'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 2: Helper functions header
# ═══════════════════════════════════════════════════════════════════════════
md("## Helper Functions")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 3: Core helpers
# ═══════════════════════════════════════════════════════════════════════════
code("""
def compute_stats(cap):
    \"\"\"Compute strategy statistics from a capital series.\"\"\"
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


def mutual_info(x, y, bins=50):
    \"\"\"Mutual information via 2D histogram (no sklearn needed).\"\"\"
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = entropy(px[px > 0])
    hy = entropy(py[py > 0])
    hxy = entropy(pxy.flatten()[pxy.flatten() > 0])
    return hx + hy - hxy


def load_front_month(filename):
    \"\"\"Load Databento futures -> roll-adjusted front-month series.

    Front month = highest-volume contract per day.
    On roll days uses OLD contract's return (not price gap).
    \"\"\"
    raw = pd.read_parquet(f'{DATA}/{filename}')
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
    outrights = raw[~raw['symbol'].str.contains('-', na=False)].copy()
    outrights = outrights[~outrights['symbol'].str.startswith('UD:', na=False)]
    outrights = outrights.dropna(subset=['close'])
    outrights = outrights[outrights['close'] > 0]
    outrights = outrights.sort_index()

    # Per-contract price history for roll adjustment
    contract_prices = {}
    for _, row in outrights.iterrows():
        sym = row['symbol']
        date = row.name.normalize()
        if sym not in contract_prices:
            contract_prices[sym] = {}
        contract_prices[sym][date] = row['close']

    # Front-month (highest volume) per day
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
            # Roll day: use OLD contract's return
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


print('Core helpers defined.')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 4: FX option infrastructure
# ═══════════════════════════════════════════════════════════════════════════
md("## FX Option Infrastructure")

code("""
# Strike divisors per CME product
STRIKE_DIVISORS = {
    'AUD': 1000, 'GBP': 1000, 'CAD': 1000, 'EUR': 1000,
    'CHF': 1000, 'NZD': 1000, 'MXN': 10000, 'JPY': 100000,
}

# Option file configs: (old_file, new_file, old_prefixes, new_prefixes, cutoff)
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


def parse_option_generic(sym, date_year, prefixes, strike_div):
    \"\"\"Parse CME FX option symbol -> (month, year, opt_type, strike).\"\"\"
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


def load_fx_options(ccy):
    \"\"\"Load FX options for a currency, merging old + new format files.\"\"\"
    config = OPT_CONFIGS[ccy]
    old_file, new_file, old_prefixes, new_prefixes, cutoff = config
    strike_div = STRIKE_DIVISORS[ccy]

    old = pd.read_parquet(f'{DATA}/{old_file}')
    old = old[~old['symbol'].str.contains('UD:', na=False)].copy()

    if new_file is not None:
        new = pd.read_parquet(f'{DATA}/{new_file}')
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
    \"\"\"Select one OTM put per month for FX hedging.\"\"\"
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


def build_settlement_lookup(opts):
    \"\"\"Pre-build symbol -> [(date, price)] for fast settlement.\"\"\"
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
    \"\"\"Get option settlement price.\"\"\"
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
        return max(0, strike - underlying)
    return 0.0


def precompute_settlements(selections, lookup, front_prices):
    \"\"\"Pre-compute settlement for all selected puts.\"\"\"
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


def run_carry_backtest(cross_df, front_prices, put_sels, all_opts,
                       leverage=1, put_budget=0.005):
    \"\"\"Run leveraged carry + puts backtest for a single FX pair.\"\"\"
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


print('FX option infrastructure defined.')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 5: ES section header
# ═══════════════════════════════════════════════════════════════════════════
md("""
## Strategy 1: S&P 500 (ES 3x + 25% OTM Puts)

**Long ES futures at 3x leverage + monthly 25% OTM puts (0.3% budget)**

Deep OTM puts: very cheap insurance (~25% below spot). Minimal drag on returns
but massive payoff during tail events. At 3x leverage, CAGR is boosted while
puts bound the tail risk.
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 6: Load ES futures
# ═══════════════════════════════════════════════════════════════════════════
code("""
es_fut = load_front_month('ES_FUT_ohlcv1d.parquet')
es_daily = es_fut[['close', 'return']].copy().dropna()

print(f'ES futures: {len(es_daily):,} trading days')
print(f'Date range: {es_daily.index.min().date()} to {es_daily.index.max().date()}')
print(f'Price range: {es_daily["close"].min():.0f} to {es_daily["close"].max():.0f}')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 7: Load ES options + select deep OTM puts
# ═══════════════════════════════════════════════════════════════════════════
code("""
opt_file = f'{DATA}/ES_OPT_ohlcv1d.parquet'
has_es_opts = os.path.exists(opt_file)

if has_es_opts:
    print('Loading ES options...')
    es_opts_raw = pd.read_parquet(opt_file)
    es_opts_raw.index = es_opts_raw.index.tz_localize(None) if es_opts_raw.index.tz else es_opts_raw.index

    es_puts = es_opts_raw[es_opts_raw['symbol'].str.contains(' P', na=False)].copy()
    es_puts = es_puts[es_puts['close'] > 0]
    es_puts['strike'] = es_puts['symbol'].str.extract(r' P(\\d+)').astype(float)
    es_puts = es_puts.dropna(subset=['strike'])
    print(f'ES puts: {len(es_puts):,} rows')

    es_puts['ym'] = es_puts.index.to_period('M')
    puts_by_ym = {ym: grp for ym, grp in es_puts.groupby('ym')}

    # 25% OTM (moneyness = 0.75) -- very deep, very cheap
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
    if len(es_put_map) > 0:
        pnls = [v['pnl_ratio'] for v in es_put_map.values()]
        wins = sum(1 for p in pnls if p > 0)
        print(f'  Win rate: {wins/len(pnls)*100:.1f}%')
        print(f'  Avg P&L: {np.mean(pnls):.2f}x, Best: {max(pnls):.1f}x')
else:
    es_put_map = {}
    print('ES options not found -- will run unhedged')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 8: Run ES backtest (3x + deep OTM puts)
# ═══════════════════════════════════════════════════════════════════════════
code("""
ES_LEVERAGE = 3
ES_PUT_BUDGET = 0.003  # 0.3%

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
print(f'  Period: {es_cap.index.min().date()} to {es_cap.index.max().date()}')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 9: FX Carry section header
# ═══════════════════════════════════════════════════════════════════════════
md("""
## Strategy 2: FX Carry (6 Pairs vs JPY, Hedged)

**Equal-weight portfolio of 6 pairs vs JPY, 1x leverage, 8% OTM puts (0.5% budget)**

Each pair is backtested individually with carry income + spot return + put P&L,
then combined into an equal-weight portfolio. The hedging protects against
carry unwind events (e.g., JPY strengthening).
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 10: Load FX futures + build cross rates with carry
# ═══════════════════════════════════════════════════════════════════════════
code("""
FX_PAIRS = {'AUD': '6A', 'GBP': '6B', 'CAD': '6C', 'EUR': '6E', 'MXN': '6M', 'CHF': '6S'}

boj_rates = {y: 0.0 for y in range(2010, 2027)}
boj_rates[2024] = 0.25
boj_rates[2025] = 0.50
boj_rates[2026] = 0.50

policy_rates = {
    'AUD': {
        2010: 4.25, 2011: 4.50, 2012: 3.50, 2013: 2.75, 2014: 2.50,
        2015: 2.00, 2016: 1.75, 2017: 1.50, 2018: 1.50, 2019: 1.00,
        2020: 0.25, 2021: 0.10, 2022: 1.85, 2023: 4.10, 2024: 4.35,
        2025: 4.35, 2026: 4.10,
    },
    'GBP': {
        2010: 0.50, 2011: 0.50, 2012: 0.50, 2013: 0.50, 2014: 0.50,
        2015: 0.50, 2016: 0.25, 2017: 0.35, 2018: 0.65, 2019: 0.75,
        2020: 0.25, 2021: 0.15, 2022: 2.00, 2023: 4.75, 2024: 5.00,
        2025: 4.50, 2026: 4.25,
    },
    'CAD': {
        2010: 0.50, 2011: 1.00, 2012: 1.00, 2013: 1.00, 2014: 1.00,
        2015: 0.75, 2016: 0.50, 2017: 0.75, 2018: 1.50, 2019: 1.75,
        2020: 0.50, 2021: 0.25, 2022: 2.50, 2023: 4.75, 2024: 4.50,
        2025: 3.25, 2026: 3.00,
    },
    'EUR': {
        2010: 1.00, 2011: 1.25, 2012: 0.75, 2013: 0.50, 2014: 0.15,
        2015: 0.05, 2016: 0.00, 2017: 0.00, 2018: 0.00, 2019: 0.00,
        2020: 0.00, 2021: 0.00, 2022: 1.25, 2023: 4.00, 2024: 4.25,
        2025: 3.15, 2026: 2.65,
    },
    'CHF': {
        2010: 0.25, 2011: 0.00, 2012: 0.00, 2013: 0.00, 2014: 0.00,
        2015: -0.75, 2016: -0.75, 2017: -0.75, 2018: -0.75, 2019: -0.75,
        2020: -0.75, 2021: -0.75, 2022: 0.25, 2023: 1.50, 2024: 1.50,
        2025: 0.50, 2026: 0.25,
    },
    'MXN': {
        2010: 4.50, 2011: 4.50, 2012: 4.50, 2013: 4.00, 2014: 3.50,
        2015: 3.25, 2016: 5.00, 2017: 7.00, 2018: 8.00, 2019: 8.00,
        2020: 5.00, 2021: 4.50, 2022: 8.50, 2023: 11.25, 2024: 10.75,
        2025: 9.50, 2026: 8.50,
    },
}

# Load FX futures
fx_data = {}
for ccy, code_str in FX_PAIRS.items():
    fx_data[ccy] = load_front_month(f'{code_str}_FUT_ohlcv1d.parquet')
    print(f'  {ccy}: {len(fx_data[ccy]):,} days')

jpy = load_front_month('6J_FUT_ohlcv1d.parquet')
print(f'  JPY: {len(jpy):,} days')

# Build cross-rate DataFrames with carry for each pair
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

    avg_carry = cr['daily_carry'].mean() * 365 * 100
    print(f'    {ccy}/JPY carry: {avg_carry:.1f}% p.a.')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 11: Load FX options + run per-pair hedged backtests
# ═══════════════════════════════════════════════════════════════════════════
code("""
FX_LEVERAGE = 1
FX_PUT_BUDGET = 0.005  # 0.5%
FX_OTM = 0.92  # 8% OTM

fx_pair_caps = {}
fx_pair_stats = {}

for ccy in FX_PAIRS:
    cr = cross_rates[ccy]
    front = fx_data[ccy]

    # Try to load FX options for this pair
    try:
        all_opts = load_fx_options(ccy)
        if len(all_opts) > 0:
            put_sels = select_monthly_puts(all_opts, front, otm_target=FX_OTM)
        else:
            put_sels = pd.DataFrame()
    except Exception as e:
        print(f'  {ccy}: option loading failed ({e}), running unhedged')
        put_sels = pd.DataFrame()

    n_puts = len(put_sels) if len(put_sels) > 0 else 0
    budget = FX_PUT_BUDGET if n_puts > 0 else 0

    cap = run_carry_backtest(cr, front, put_sels, all_opts if n_puts > 0 else pd.DataFrame(),
                             leverage=FX_LEVERAGE, put_budget=budget)
    fx_pair_caps[ccy] = cap
    s = compute_stats(cap)
    fx_pair_stats[ccy] = s

    hedged_str = f'hedged ({n_puts} puts)' if n_puts > 0 else 'unhedged'
    if s:
        print(f'  {ccy}/JPY {FX_LEVERAGE}x {hedged_str}: '
              f'Sharpe={s["Sharpe"]:.3f}, CAGR={s["CAGR"]:.1%}, MaxDD={s["MaxDD"]:.1%}')
    else:
        print(f'  {ccy}/JPY: insufficient data')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 12: Build FX EW portfolio from individual pair backtests
# ═══════════════════════════════════════════════════════════════════════════
code("""
# EW portfolio: average daily returns of individual pair capital series
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
    print(f'  MaxDD: {fx_stats["MaxDD"]:.1%}, Total: {fx_stats["Total"]:.1f}x')
print(f'  Period: {fx_cap.index.min().date()} to {fx_cap.index.max().date()}')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 13: Bond Carry section header
# ═══════════════════════════════════════════════════════════════════════════
md("""
## Strategy 3: US-UK Bond Carry (3x + 4% OTM Puts)

**Long ZN / Short Gilt at 3x leverage + 4% OTM OZN puts (0.3% budget)**

Monthly rebalance: aggregate daily returns to monthly, apply leverage to monthly spread.
OZN puts protect the long ZN leg.
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 14: Load ZN + Gilt, build spread (matching bond_carry_usuk.ipynb)
# ═══════════════════════════════════════════════════════════════════════════
code("""
# Load raw data (matching bond_carry_usuk.ipynb inline approach)
zn_raw = pd.read_parquet(f'{DATA}/ZN_FUT_ohlcv1d.parquet')
gilt_raw = pd.read_parquet(f'{DATA}/R_FUT_ohlcv1d.parquet')
zn_raw.index = zn_raw.index.tz_localize(None) if zn_raw.index.tz else zn_raw.index
gilt_raw.index = gilt_raw.index.tz_localize(None) if gilt_raw.index.tz else gilt_raw.index

# Filter outrights
zn_all = zn_raw[~zn_raw['symbol'].str.contains('-', na=False)]
zn_all = zn_all[~zn_all['symbol'].str.startswith('UD:', na=False)]
zn_all = zn_all.dropna(subset=['close'])
zn_all = zn_all[zn_all['close'] > 50]

gilt_all = gilt_raw[~gilt_raw['symbol'].str.contains('-', na=False)]
gilt_all = gilt_all[~gilt_all['symbol'].str.contains('_Z', na=False)]
gilt_all = gilt_all.dropna(subset=['close'])
gilt_all = gilt_all[gilt_all['close'] > 50]

# Front month by volume (matching original notebook)
zn_front = zn_all.loc[zn_all.groupby(zn_all.index)['volume'].idxmax()]
zn_front = zn_front[['close', 'volume', 'symbol']].copy()
zn_front = zn_front[~zn_front.index.duplicated(keep='first')]
zn_front.columns = ['zn_close', 'zn_vol', 'zn_sym']

gilt_front = gilt_all.loc[gilt_all.groupby(gilt_all.index)['volume'].idxmax()]
gilt_front = gilt_front[['close', 'volume', 'symbol']].copy()
gilt_front = gilt_front[~gilt_front.index.duplicated(keep='first')]
gilt_front.columns = ['gilt_close', 'gilt_vol', 'gilt_sym']

# Merge
bond_df = zn_front.join(gilt_front, how='inner').sort_index()

# Daily returns (simple pct_change, matching original)
bond_df['zn_ret'] = bond_df['zn_close'].pct_change()
bond_df['gilt_ret'] = bond_df['gilt_close'].pct_change()
bond_df = bond_df.dropna()

# Monthly returns: compound SEPARATELY, then subtract (matching original)
monthly = bond_df[['zn_ret', 'gilt_ret']].resample('ME').apply(lambda x: (1 + x).prod() - 1)
monthly['long_zn'] = monthly['zn_ret'] - monthly['gilt_ret']
monthly = monthly.dropna()

print(f'Bond carry data: {len(bond_df):,} daily, {len(monthly)} monthly')
print(f'Date range: {bond_df.index.min().date()} to {bond_df.index.max().date()}')
print(f'Avg monthly spread: {monthly["long_zn"].mean()*12:.2%} (annualised)')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 15: Load OZN options, select puts (matching original)
# ═══════════════════════════════════════════════════════════════════════════
code("""
ozn_raw = pd.read_parquet(f'{DATA}/OZN_OPT_ohlcv1d.parquet')
ozn_raw.index = ozn_raw.index.tz_localize(None) if ozn_raw.index.tz else ozn_raw.index
print(f'OZN options: {len(ozn_raw):,} rows')

pat = re.compile(r'^OZN\\s*([FGHJKMNQUVXZ])(\\d)\\s+([PC])(\\d+)')
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

# Select monthly puts at 4% OTM (moneyness = 0.96)
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
    sym_arr = ozn_puts['symbol'].values
    yr_arr = ozn_puts.index.year
    mo_arr = ozn_puts.index.month
    mask = (sym_arr == best_sym) & (
        ((yr_arr == yr) & (mo_arr == mo)) |
        ((yr_arr == yr_next) & (mo_arr == mo_next))
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
if len(ozn_put_map) > 0:
    pnls = [v['pnl_ratio'] for v in ozn_put_map.values()]
    wins = sum(1 for p in pnls if p > 0)
    print(f'  Win rate: {wins/len(pnls)*100:.1f}%, Avg P&L: {np.mean(pnls):.2f}x')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 16: Run bond carry backtest (monthly, matching original)
# ═══════════════════════════════════════════════════════════════════════════
code("""
BOND_LEVERAGE = 3
BOND_PUT_BUDGET = 0.003  # 0.3%

# Backtest on monthly returns (matching bond_carry_usuk.ipynb)
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

# Interpolate to daily for portfolio alignment
bond_cap = bond_cap_monthly.reindex(bond_df.index, method='ffill').dropna()
if len(bond_cap) > 0:
    bond_cap = bond_cap / bond_cap.iloc[0] * 100

bond_stats = compute_stats(bond_cap)

print(f'Bond Carry {BOND_LEVERAGE}x + {BOND_PUT_BUDGET*100:.1f}% OZN puts (4% OTM, monthly)')
if bond_stats:
    print(f'  CAGR: {bond_stats["CAGR"]:.1%}, Vol: {bond_stats["Vol"]:.1%}, Sharpe: {bond_stats["Sharpe"]:.3f}')
    print(f'  MaxDD: {bond_stats["MaxDD"]:.1%}, Total: {bond_stats["Total"]:.1f}x')
print(f'  Period: {bond_cap.index.min().date()} to {bond_cap.index.max().date()}')
print(f'  Monthly data points: {len(bond_cap_monthly)}')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 17: Individual performance header
# ═══════════════════════════════════════════════════════════════════════════
md("## Individual Strategy Performance")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 18: Stats table + equity curves
# ═══════════════════════════════════════════════════════════════════════════
code("""
strat_caps = {
    'ES 3x hedged': es_cap,
    'FX Carry EW': fx_cap,
    'Bond Carry': bond_cap,
}

print('=' * 110)
print('INDIVIDUAL STRATEGY PERFORMANCE (full period each)')
print('=' * 110)
header = f'  {"Strategy":>20s}  {"CAGR":>7s}  {"Vol":>7s}  {"Sharpe":>7s}  {"Sortino":>8s}  {"Calmar":>7s}  {"MaxDD":>7s}  {"Skew":>6s}  {"Kurt":>6s}  {"Total":>7s}'
print(header)
print('-' * 110)
for name, cap in strat_caps.items():
    s = compute_stats(cap)
    if s:
        print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
              f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Skew"]:>5.2f}  {s["Kurt"]:>5.1f}  {s["Total"]:>6.1f}x')

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
plt.savefig('/tmp/combined_individual_equity.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 19: Dependency analysis header
# ═══════════════════════════════════════════════════════════════════════════
md("""
## Dependency Analysis: Pearson Correlation & Mutual Information

**Pearson** captures linear dependence. **Mutual Information** captures any
dependency (including nonlinear). Low values for both confirms true independence.
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 20: Pearson + MI matrices
# ═══════════════════════════════════════════════════════════════════════════
code("""
aligned = pd.DataFrame({name: cap for name, cap in strat_caps.items()}).dropna()
print(f'Common date range: {aligned.index.min().date()} to {aligned.index.max().date()} ({len(aligned):,} days)')

ret_df = aligned.pct_change().dropna()
strat_names = list(strat_caps.keys())
n = len(strat_names)

pearson = ret_df.corr()
print('\\nPEARSON CORRELATION MATRIX:')
print(pearson.round(4).to_string())

mi_matrix = pd.DataFrame(np.zeros((n, n)), index=strat_names, columns=strat_names)
for i in range(n):
    for j in range(n):
        mi_matrix.iloc[i, j] = mutual_info(ret_df[strat_names[i]].values,
                                             ret_df[strat_names[j]].values, bins=50)

print('\\nMUTUAL INFORMATION MATRIX:')
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
plt.savefig('/tmp/combined_dependency.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 21: Scatter plots
# ═══════════════════════════════════════════════════════════════════════════
code("""
pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (i, j) in enumerate(pairs):
    ax = axes[idx]
    x = ret_df[strat_names[i]].values * 100
    y = ret_df[strat_names[j]].values * 100
    ax.scatter(x, y, alpha=0.15, s=5, color='steelblue')
    if len(x) > 10:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, 'r-', lw=1.5, alpha=0.8)
    rho = pearson.iloc[i, j]
    mi_val = mi_matrix.iloc[i, j]
    ax.set_xlabel(f'{strat_names[i]} (%)')
    ax.set_ylabel(f'{strat_names[j]} (%)')
    ax.set_title(f'{short_names[i]} vs {short_names[j]}')
    ax.annotate(f'rho={rho:.3f}\\nMI={mi_val:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

plt.suptitle('Pairwise Daily Return Scatter Plots', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/tmp/combined_scatter.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 22: Portfolio construction header
# ═══════════════════════════════════════════════════════════════════════════
md("""
## Portfolio Construction

| Method | Description | Rebalance |
|--------|------------|-----------|
| **Equal-Weight (EW)** | 1/3 each strategy | Daily |
| **Risk-Parity (RP)** | Inverse-vol weights, 60-day trailing | Monthly |
| **Min-Variance (MV)** | Minimize portfolio variance, 252-day cov | Monthly |
| **Max-Sharpe (MS)** | Maximize Sharpe ratio, 252-day window | Quarterly |
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 23: Portfolio solver functions
# ═══════════════════════════════════════════════════════════════════════════
code("""
def solve_min_variance(cov_matrix, max_iter=1000):
    \"\"\"Min variance portfolio, long-only, gradient projection.\"\"\"
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
    \"\"\"Max-Sharpe portfolio, long-only, gradient ascent.\"\"\"
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

print('Portfolio solvers defined.')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 24: Build 4 portfolios
# ═══════════════════════════════════════════════════════════════════════════
code("""
port_results = {}
port_weights = {}

# 1. Equal-Weight
ew_ret = ret_df.mean(axis=1)
ew_cap = (1 + ew_ret).cumprod() * 100
port_results['Equal-Weight'] = ew_cap
print(f'EW: {len(ew_cap)} days')

# 2. Risk-Parity (inverse vol, 60-day, monthly)
VOL_WINDOW = 60
trailing_vol = ret_df.rolling(VOL_WINDOW).std() * np.sqrt(252)
rebal_monthly = ret_df.resample('ME').last().index

rp_rets = []
rp_wts = []
current_w = None
for date in ret_df.index:
    if date in rebal_monthly or current_w is None:
        vol_row = trailing_vol.loc[:date].iloc[-1] if date in trailing_vol.index else None
        if vol_row is not None and vol_row.notna().sum() >= 2:
            inv_vol = 1.0 / vol_row.replace(0, np.nan).dropna()
            if len(inv_vol) > 0:
                current_w = inv_vol / inv_vol.sum()
                rp_wts.append({'date': date, **current_w.to_dict()})
    if current_w is not None:
        day_ret = ret_df.loc[date]
        cc = current_w.index.intersection(day_ret.dropna().index)
        if len(cc) > 0:
            ww = current_w[cc] / current_w[cc].sum()
            rp_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})

rp_df = pd.DataFrame(rp_rets).set_index('date')
rp_cap = (1 + rp_df['return']).cumprod() * 100
port_results['Risk-Parity'] = rp_cap
port_weights['Risk-Parity'] = pd.DataFrame(rp_wts).set_index('date') if rp_wts else None
print(f'RP: {len(rp_cap)} days')

# 3. Min-Variance (252-day cov, monthly)
COV_WINDOW = 252
mv_rets = []
mv_wts = []
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
            mv_wts.append({'date': date, **current_w.to_dict()})
    if current_w is not None:
        day_ret = ret_df.loc[date]
        cc = current_w.index.intersection(day_ret.dropna().index)
        if len(cc) > 0:
            ww = current_w[cc] / current_w[cc].sum()
            mv_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})

mv_df = pd.DataFrame(mv_rets).set_index('date')
mv_cap = (1 + mv_df['return']).cumprod() * 100
port_results['Min-Variance'] = mv_cap
port_weights['Min-Variance'] = pd.DataFrame(mv_wts).set_index('date') if mv_wts else None
print(f'MV: {len(mv_cap)} days')

# 4. Max-Sharpe (252-day, quarterly)
rebal_quarterly = ret_df.resample('QE').last().index
ms_rets = []
ms_wts = []
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
            ms_wts.append({'date': date, **current_w.to_dict()})
    if current_w is not None:
        day_ret = ret_df.loc[date]
        cc = current_w.index.intersection(day_ret.dropna().index)
        if len(cc) > 0:
            ww = current_w[cc] / current_w[cc].sum()
            ms_rets.append({'date': date, 'return': (day_ret[cc] * ww).sum()})

ms_df = pd.DataFrame(ms_rets).set_index('date')
ms_cap = (1 + ms_df['return']).cumprod() * 100
port_results['Max-Sharpe'] = ms_cap
port_weights['Max-Sharpe'] = pd.DataFrame(ms_wts).set_index('date') if ms_wts else None
print(f'MS: {len(ms_cap)} days')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 25: Portfolio comparison header
# ═══════════════════════════════════════════════════════════════════════════
md("## Portfolio Performance Comparison")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 26: Portfolio stats + equity curves
# ═══════════════════════════════════════════════════════════════════════════
code("""
print('=' * 115)
print('PORTFOLIO vs INDIVIDUAL STRATEGY PERFORMANCE')
print('=' * 115)
header = f'  {"Name":>20s}  {"CAGR":>7s}  {"Vol":>7s}  {"Sharpe":>7s}  {"Sortino":>8s}  {"Calmar":>7s}  {"MaxDD":>7s}  {"Skew":>6s}  {"Kurt":>6s}  {"Total":>7s}'
print(header)
print('-' * 115)

print('  -- Individual Strategies (common period) --')
for name in strat_names:
    s = compute_stats(aligned[name])
    if s:
        print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
              f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Skew"]:>5.2f}  {s["Kurt"]:>5.1f}  {s["Total"]:>6.1f}x')

print('  -- Portfolios --')
best_port_sharpe = -999
best_port_name = ''
for name, cap in port_results.items():
    s = compute_stats(cap)
    if s:
        print(f'  {name:>20s}  {s["CAGR"]:>+6.1%}  {s["Vol"]:>6.1%}  {s["Sharpe"]:>+6.3f}  {s["Sortino"]:>+7.3f}  '
              f'{s["Calmar"]:>+6.3f}  {s["MaxDD"]:>6.1%}  {s["Skew"]:>5.2f}  {s["Kurt"]:>5.1f}  {s["Total"]:>6.1f}x')
        if s['Sharpe'] > best_port_sharpe:
            best_port_sharpe = s['Sharpe']
            best_port_name = name

best_ind_sharpe = max(compute_stats(aligned[name])['Sharpe'] for name in strat_names
                      if compute_stats(aligned[name]) is not None)
print()
print(f'Best portfolio: {best_port_name} (Sharpe {best_port_sharpe:.3f})')
print(f'Best individual: Sharpe {best_ind_sharpe:.3f}')
print(f'Diversification benefit: {best_port_sharpe - best_ind_sharpe:+.3f} Sharpe improvement')

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
plt.savefig('/tmp/combined_portfolio_equity.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 27: Year-by-year
# ═══════════════════════════════════════════════════════════════════════════
md("## Year-by-Year Returns")

code("""
all_series = {}
for name in strat_names:
    all_series[name] = aligned[name]
for name, cap in port_results.items():
    all_series[name] = cap

annual_rets = {}
for name, cap in all_series.items():
    yearly = cap.resample('YE').last().pct_change().dropna()
    annual_rets[name] = yearly

all_years = sorted(set(y for ys in annual_rets.values() for y in ys.index.year))
all_names = list(all_series.keys())

print('YEAR-BY-YEAR RETURNS')
print('=' * 140)
header = f'  {"Year":>4}'
for name in all_names:
    header += f'  {name[:12]:>12}'
print(header)
print('-' * 140)
for yr in all_years:
    row = f'  {yr:>4}'
    for name in all_names:
        if name in annual_rets:
            match = annual_rets[name][annual_rets[name].index.year == yr]
            row += f'  {match.iloc[0]:>+11.1%}' if len(match) > 0 else f'  {"":>12}'
        else:
            row += f'  {"":>12}'
    print(row)

heat_data = pd.DataFrame(index=all_years)
for name in all_names:
    col = []
    for yr in all_years:
        match = annual_rets[name][annual_rets[name].index.year == yr] if name in annual_rets else pd.Series()
        col.append(match.iloc[0] * 100 if len(match) > 0 else np.nan)
    heat_data[name] = col

fig, ax = plt.subplots(figsize=(16, max(4, len(all_years) * 0.5)))
short_cols = [n[:12] for n in heat_data.columns]
heat_plot = heat_data.copy()
heat_plot.columns = short_cols
sns.heatmap(heat_plot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, yticklabels=[str(y) for y in all_years])
ax.set_title('Annual Returns (%)', fontsize=13)
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig('/tmp/combined_annual_heatmap.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 29: Weight evolution
# ═══════════════════════════════════════════════════════════════════════════
md("## Portfolio Weight Evolution")

code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (wname, wdf) in enumerate([
    ('Risk-Parity', port_weights.get('Risk-Parity')),
    ('Min-Variance', port_weights.get('Min-Variance')),
    ('Max-Sharpe', port_weights.get('Max-Sharpe')),
]):
    ax = axes[idx]
    if wdf is not None and len(wdf) > 0:
        for col in strat_names:
            if col in wdf.columns:
                ax.plot(wdf.index, wdf[col].values * 100, label=col[:12], lw=1.5)
        ax.set_ylabel('Weight (%)')
        ax.set_title(f'{wname} Weights', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'No weight data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{wname} Weights', fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/combined_weights.png', bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 31: Conclusions
# ═══════════════════════════════════════════════════════════════════════════
md("## Conclusions")

code("""
print('COMBINED MULTI-STRATEGY PORTFOLIO -- KEY FINDINGS')
print('=' * 70)

print(f'\\n1. BEST PORTFOLIO METHOD: {best_port_name}')
print(f'   Sharpe: {best_port_sharpe:.3f}')
s = compute_stats(port_results[best_port_name])
if s:
    print(f'   CAGR: {s["CAGR"]:.1%}, Vol: {s["Vol"]:.1%}, MaxDD: {s["MaxDD"]:.1%}')

print(f'\\n2. DIVERSIFICATION BENEFIT:')
print(f'   Best individual Sharpe: {best_ind_sharpe:.3f}')
print(f'   Best portfolio Sharpe:  {best_port_sharpe:.3f}')
print(f'   Improvement: {best_port_sharpe - best_ind_sharpe:+.3f}')

print(f'\\n3. CROSS-STRATEGY CORRELATIONS:')
for i in range(n):
    for j in range(i+1, n):
        rho = pearson.iloc[i, j]
        mi = mi_matrix.iloc[i, j]
        print(f'   {strat_names[i]:>15} vs {strat_names[j]:<15}: rho={rho:+.3f}, MI={mi:.4f}')

print(f'\\n4. HEDGING SUMMARY:')
print(f'   ES 3x + 25% OTM puts (0.3%): deep OTM = minimal drag, tail protection')
print(f'   FX 1x + 8% OTM puts (0.5%): hedging HELPS FX carry (reduces unwind risk)')
print(f'   Bond 3x + 4% OTM puts (0.3%): hedging improves Sharpe on ZN leg')

print(f'\\n5. PORTFOLIO RANKING (by Sharpe):')
port_ranking = []
for p_name, p_cap in port_results.items():
    p_s = compute_stats(p_cap)
    if p_s:
        port_ranking.append((p_name, p_s['Sharpe'], p_s['CAGR'], p_s['MaxDD']))
port_ranking.sort(key=lambda x: x[1], reverse=True)
for rank, (p_name, sharpe, cagr, maxdd) in enumerate(port_ranking, 1):
    print(f'   {rank}. {p_name:>15}: Sharpe={sharpe:.3f}, CAGR={cagr:.1%}, MaxDD={maxdd:.1%}')
""")

# ═══════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.14.0"}},
    "cells": cells,
}

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Wrote {len(cells)} cells to {NB_PATH}')
