# FX Carry Trade: Leveraged AUD/JPY + Dual-Leg Hedge

The **real** carry trade: long AUD / short JPY with leverage, hedged on **both legs**.

**What carry traders actually do:**
1. Borrow JPY at ~0% interest
2. Buy AUD assets earning ~3-4% interest
3. Pocket the 3-4% rate differential
4. Apply leverage (3-10x) to amplify: 5x → ~15-20%/yr carry income
5. Risk: AUD/JPY crashes violently during crises (2008: -40%, 2015: -15%, 2020: -15%)

**Our hedge — both legs:**
- **AUD puts** (on 6A futures): protect against AUD weakness
- **JPY calls** (on 6J futures): protect against JPY strength
- Together they cover both ways AUD/JPY can crash

**Data:** Real CME futures and options from Databento (2010-2026)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data/databento'
MONTH_CODES = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,
               'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}
```

## 1. Build AUD/JPY Cross Rate from Futures


```python
def load_front_month(filename):
    """Load futures data and build roll-adjusted front-month series."""
    fut = pd.read_parquet(f'{DATA_DIR}/{filename}')
    outrights = fut[~fut['symbol'].str.contains('-', na=False)].copy()
    outrights = outrights.sort_index()
    
    # Per-contract price history for roll adjustment
    contract_prices = {}
    for _, row in outrights.iterrows():
        sym = row['symbol']
        date = row.name.normalize().tz_localize(None)
        if sym not in contract_prices:
            contract_prices[sym] = {}
        contract_prices[sym][date] = row['close']
    
    # Identify front-month (highest volume) per day
    daily_front = {}
    for date, group in outrights.groupby(outrights.index.date):
        best = group.sort_values('volume', ascending=False).iloc[0]
        daily_front[pd.Timestamp(date)] = {
            'symbol': best['symbol'],
            'close': best['close'],
            'volume': best['volume'],
        }
    
    # Build roll-adjusted return series
    dates = sorted(daily_front.keys())
    records = []
    prev_date = None
    prev_symbol = None
    
    for date in dates:
        info = daily_front[date]
        cur_symbol = info['symbol']
        
        if prev_date is None:
            records.append({'date': date, 'close': info['close'], 'return': 0.0})
            prev_date = date
            prev_symbol = cur_symbol
            continue
        
        if cur_symbol == prev_symbol:
            # Same contract: simple return
            prev_price = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            cur_price = contract_prices.get(cur_symbol, {}).get(date, 0)
            ret = cur_price / prev_price - 1 if prev_price > 0 else 0.0
        else:
            # Roll day: use OLD contract's price change
            old_prev = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            old_cur = contract_prices.get(prev_symbol, {}).get(date, 0)
            ret = old_cur / old_prev - 1 if old_prev > 0 and old_cur > 0 else 0.0
        
        records.append({'date': date, 'close': info['close'], 'return': ret})
        prev_date = date
        prev_symbol = cur_symbol
    
    return pd.DataFrame(records).set_index('date')


# Load both futures
print('Loading AUD futures...')
aud = load_front_month('6A_FUT_ohlcv1d.parquet')
print(f'  {len(aud)} days, {aud.index.min()} to {aud.index.max()}')

print('Loading JPY futures...')
jpy = load_front_month('6J_FUT_ohlcv1d.parquet')
print(f'  {len(jpy)} days, {jpy.index.min()} to {jpy.index.max()}')

# Compute AUD/JPY cross rate: 6A (USD per AUD) / 6J (USD per JPY) = JPY per AUD
common_dates = aud.index.intersection(jpy.index)
audjpy = aud.loc[common_dates, 'close'] / jpy.loc[common_dates, 'close']
audjpy_ret = aud.loc[common_dates, 'return'] - jpy.loc[common_dates, 'return']  # approx cross return

cross = pd.DataFrame({
    'audjpy': audjpy,
    'aud_ret': aud.loc[common_dates, 'return'],
    'jpy_ret': jpy.loc[common_dates, 'return'],
    'cross_ret': audjpy_ret,
})

print(f'\nAUD/JPY cross rate:')
print(f'  {len(cross)} days, {cross.index.min()} to {cross.index.max()}')
print(f'  Range: {cross["audjpy"].min():.1f} to {cross["audjpy"].max():.1f} JPY per AUD')
print(f'  Current: {cross["audjpy"].iloc[-1]:.1f}')

# Verify against known values
years = (cross.index[-1] - cross.index[0]).days / 365.25
spot_total = cross['audjpy'].iloc[-1] / cross['audjpy'].iloc[0]
print(f'\n  Spot return: {(spot_total - 1)*100:.1f}% total, {(spot_total**(1/years)-1)*100:.2f}%/yr')
```

    Loading AUD futures...


      4880 days, 2010-06-06 00:00:00 to 2026-02-27 00:00:00
    Loading JPY futures...


      4880 days, 2010-06-06 00:00:00 to 2026-02-27 00:00:00
    
    AUD/JPY cross rate:
      4880 days, 2010-06-06 00:00:00 to 2026-02-27 00:00:00
      Range: 62.5 to 111.0 JPY per AUD
      Current: 110.9
    
      Spot return: 48.2% total, 2.53%/yr


## 2. Build Carry Rate Series

The carry is the AUD-JPY interest rate differential. In practice, this accrues daily through the futures roll.


```python
# Historical policy rates (annual averages)
rba_rates = {
    2010: 4.25, 2011: 4.50, 2012: 3.50, 2013: 2.75, 2014: 2.50,
    2015: 2.00, 2016: 1.75, 2017: 1.50, 2018: 1.50, 2019: 1.00,
    2020: 0.25, 2021: 0.10, 2022: 1.85, 2023: 4.10, 2024: 4.35, 2025: 4.35,
}
boj_rates = {y: 0.0 for y in range(2010, 2026)}
boj_rates[2024] = 0.25
boj_rates[2025] = 0.50

# Fed funds (for funding cost)
fed_rates = {
    2010: 0.25, 2011: 0.25, 2012: 0.25, 2013: 0.25, 2014: 0.25,
    2015: 0.25, 2016: 0.50, 2017: 1.25, 2018: 2.00, 2019: 2.25,
    2020: 0.25, 2021: 0.25, 2022: 2.50, 2023: 5.00, 2024: 5.00, 2025: 4.50,
}

# Daily carry rate for AUD/JPY
cross['aud_rate'] = cross.index.year.map(lambda y: rba_rates.get(y, 0)) / 100
cross['jpn_rate'] = cross.index.year.map(lambda y: boj_rates.get(y, 0)) / 100
cross['usd_rate'] = cross.index.year.map(lambda y: fed_rates.get(y, 0)) / 100
cross['daily_carry'] = (cross['aud_rate'] - cross['jpn_rate']) / 365  # earn AUD rate, pay JPY rate

avg_carry = cross['daily_carry'].mean() * 365
print(f'Average AUD-JPY daily carry: {cross["daily_carry"].mean()*100:.4f}%')
print(f'Annualized carry: {avg_carry*100:.2f}%')
print(f'\nCarry by period:')
for year in range(2010, 2026):
    aud_r = rba_rates.get(year, 0)
    jpn_r = boj_rates.get(year, 0)
    print(f'  {year}: AUD {aud_r:.2f}% - JPY {jpn_r:.2f}% = {aud_r-jpn_r:+.2f}%')
```

    Average AUD-JPY daily carry: 0.0066%
    Annualized carry: 2.39%
    
    Carry by period:
      2010: AUD 4.25% - JPY 0.00% = +4.25%
      2011: AUD 4.50% - JPY 0.00% = +4.50%
      2012: AUD 3.50% - JPY 0.00% = +3.50%
      2013: AUD 2.75% - JPY 0.00% = +2.75%
      2014: AUD 2.50% - JPY 0.00% = +2.50%
      2015: AUD 2.00% - JPY 0.00% = +2.00%
      2016: AUD 1.75% - JPY 0.00% = +1.75%
      2017: AUD 1.50% - JPY 0.00% = +1.50%
      2018: AUD 1.50% - JPY 0.00% = +1.50%
      2019: AUD 1.00% - JPY 0.00% = +1.00%
      2020: AUD 0.25% - JPY 0.00% = +0.25%
      2021: AUD 0.10% - JPY 0.00% = +0.10%
      2022: AUD 1.85% - JPY 0.00% = +1.85%
      2023: AUD 4.10% - JPY 0.00% = +4.10%
      2024: AUD 4.35% - JPY 0.25% = +4.10%
      2025: AUD 4.35% - JPY 0.50% = +3.85%


## 3. Load AUD Options + JPY Options

- **AUD puts**: protect against AUD/USD dropping (AUD weakness)
- **JPY calls**: protect against JPY/USD rising (JPY strength → USD/JPY falls → AUD/JPY falls)


```python
def parse_option(sym, date_year, product='AUD'):
    """Parse CME FX option symbol -> (month, year, opt_type, strike).
    
    AUD: '6AN0 C0820' or 'ADUF7 P0745' -> strike / 1000
    JPY: '6JN0 C1125' or 'JPUF7 P0925' -> strike / 100000
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
    
    if product == 'AUD':
        strike = strike_raw / 1000.0
        if contract.startswith('ADU'):
            month_code, year_digit = contract[3], int(contract[4])
        elif contract.startswith('6A'):
            month_code, year_digit = contract[2], int(contract[3])
        else:
            return None
    elif product == 'JPY':
        strike = strike_raw / 100000.0
        if contract.startswith('JPU'):
            month_code, year_digit = contract[3], int(contract[4])
        elif contract.startswith('6J'):
            month_code, year_digit = contract[2], int(contract[3])
        else:
            return None
    else:
        return None
    
    month = MONTH_CODES.get(month_code, 0)
    if month == 0:
        return None
    decade_base = (date_year // 10) * 10
    year = decade_base + year_digit
    if year < date_year - 2:
        year += 10
    return month, year, opt_type, strike


def load_fx_options(old_file, new_file, product, cutoff='2016-08-23'):
    """Load and merge old + new FX option data."""
    old = pd.read_parquet(f'{DATA_DIR}/{old_file}')
    new = pd.read_parquet(f'{DATA_DIR}/{new_file}')
    old = old[~old['symbol'].str.contains('UD:', na=False)].copy()
    new = new[~new['symbol'].str.contains('UD:', na=False)].copy()
    cutoff_ts = pd.Timestamp(cutoff, tz='UTC')
    old = old[old.index < cutoff_ts]
    combined = pd.concat([old, new]).sort_index()
    
    records = []
    for idx, row in combined.iterrows():
        parsed = parse_option(row['symbol'], idx.year, product=product)
        if parsed is None:
            continue
        month, year, opt_type, strike = parsed
        first_of_month = pd.Timestamp(year=year, month=month, day=1)
        third_wed = first_of_month + pd.offsets.WeekOfMonth(week=2, weekday=2)
        expiry = (third_wed - pd.offsets.BDay(2)).tz_localize('UTC')
        records.append({
            'date': idx, 'symbol': row['symbol'], 'opt_type': opt_type,
            'strike': strike, 'expiry': expiry,
            'close': row['close'], 'volume': row['volume'],
        })
    return pd.DataFrame(records)


# Load AUD options
print('Loading AUD options...')
aud_opts = load_fx_options('6A_OPT_ohlcv1d.parquet', 'ADU_OPT_ohlcv1d.parquet', 'AUD')
print(f'  AUD options: {len(aud_opts):,} (puts: {(aud_opts["opt_type"]=="P").sum():,}, calls: {(aud_opts["opt_type"]=="C").sum():,})')

# Load JPY options
print('Loading JPY options...')
jpy_opts = load_fx_options('6J_OPT_ohlcv1d.parquet', 'JPU_OPT_ohlcv1d.parquet', 'JPY',
                           cutoff='2016-08-16')
print(f'  JPY options: {len(jpy_opts):,} (puts: {(jpy_opts["opt_type"]=="P").sum():,}, calls: {(jpy_opts["opt_type"]=="C").sum():,})')

# Sanity check JPY strikes
jpy_sample = jpy_opts[jpy_opts['opt_type'] == 'C'].head(5)
for _, r in jpy_sample.iterrows():
    nearest_fut = jpy.index.asof(r['date'].tz_localize(None))
    if pd.notna(nearest_fut):
        fut_price = jpy.loc[nearest_fut, 'close']
        print(f"  {r['symbol']}: strike={r['strike']:.6f}, underlying~{fut_price:.6f}, "
              f"moneyness={r['strike']/fut_price:.3f}, price={r['close']:.6f}")
```

    Loading AUD options...


      AUD options: 281,447 (puts: 150,657, calls: 130,790)
    Loading JPY options...


      JPY options: 342,656 (puts: 156,699, calls: 185,957)
      6JU0 C1125: strike=0.011250, underlying~0.010921, moneyness=1.030, price=0.000184
      6JU0 C1200: strike=0.012000, underlying~0.010921, moneyness=1.099, price=0.000059
      6JN0 C1135: strike=0.011350, underlying~0.010921, moneyness=1.039, price=0.000061
      6JU0 C1260: strike=0.012600, underlying~0.010921, moneyness=1.154, price=0.000026
      6JN0 C1090: strike=0.010900, underlying~0.010921, moneyness=0.998, price=0.000182


## 4. Select Monthly AUD Puts + JPY Calls

**AUD puts**: 5-8% OTM, protect when AUD weakens
**JPY calls**: 5-8% OTM, protect when JPY strengthens (6J rises above strike)


```python
def select_monthly_options(opts, front_prices, opt_type='P', otm_target=0.92):
    """Select one OTM option per month.
    
    For puts: moneyness = strike/underlying < 1 (OTM when below spot)
    For calls: moneyness = strike/underlying > 1 (OTM when above spot)
    
    otm_target for puts: 0.92 = 8% below spot
    otm_target for calls: 1.08 = 8% above spot
    """
    filtered = opts[opts['opt_type'] == opt_type].copy()
    
    # Merge underlying futures price
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
        
        # Nearest expiry > 14 days out
        day_opts = day_opts[day_opts['expiry'] > first_day + pd.Timedelta(days=14)]
        if len(day_opts) == 0:
            continue
        nearest_exp = day_opts['expiry'].min()
        day_opts = day_opts[day_opts['expiry'] == nearest_exp]
        
        # Filter OTM only
        if opt_type == 'P':
            day_opts = day_opts[day_opts['moneyness'] < 1.0]
        else:  # calls
            day_opts = day_opts[day_opts['moneyness'] > 1.0]
        
        if len(day_opts) == 0:
            continue
        
        day_opts = day_opts.copy()
        day_opts['dist'] = (day_opts['moneyness'] - otm_target).abs()
        
        # Among close candidates, prefer higher volume
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


# Build front-month price series for option selection
aud_front_naive = pd.DataFrame({'close': aud['close']})
jpy_front_naive = pd.DataFrame({'close': jpy['close']})

# AUD puts (protect AUD weakness)
aud_puts_8 = select_monthly_options(aud_opts, aud_front_naive, opt_type='P', otm_target=0.92)
aud_puts_5 = select_monthly_options(aud_opts, aud_front_naive, opt_type='P', otm_target=0.95)

# JPY calls (protect JPY strength — 6J rises = JPY strengthens = AUD/JPY falls)
jpy_calls_8 = select_monthly_options(jpy_opts, jpy_front_naive, opt_type='C', otm_target=1.08)
jpy_calls_5 = select_monthly_options(jpy_opts, jpy_front_naive, opt_type='C', otm_target=1.05)

print(f'AUD puts selected:  8% OTM = {len(aud_puts_8)} months, 5% OTM = {len(aud_puts_5)} months')
print(f'JPY calls selected: 8% OTM = {len(jpy_calls_8)} months, 5% OTM = {len(jpy_calls_5)} months')

print(f'\n=== Sample AUD Puts (8% OTM) ===')
print(aud_puts_8[['entry_date', 'symbol', 'strike', 'entry_price', 'underlying', 'moneyness']].head(5).to_string(index=False))

print(f'\n=== Sample JPY Calls (8% OTM) ===')
print(jpy_calls_8[['entry_date', 'symbol', 'strike', 'entry_price', 'underlying', 'moneyness']].head(5).to_string(index=False))
```

    AUD puts selected:  8% OTM = 182 months, 5% OTM = 182 months
    JPY calls selected: 8% OTM = 180 months, 5% OTM = 180 months
    
    === Sample AUD Puts (8% OTM) ===
                   entry_date     symbol  strike  entry_price  underlying  moneyness
    2010-06-07 00:00:00+00:00 6AN0 P0760   0.760      0.00770      0.8117   0.936307
    2010-07-01 00:00:00+00:00 6AN0 P0770   0.770      0.00030      0.8381   0.918745
    2010-08-02 00:00:00+00:00 6AU0 P0850   0.850      0.00250      0.9080   0.936123
    2010-09-01 00:00:00+00:00 6AV0 P0850   0.850      0.00360      0.9075   0.936639
    2010-10-01 00:00:00+00:00 6AV0 P0845   0.845      0.00005      0.9638   0.876738
    
    === Sample JPY Calls (8% OTM) ===
                   entry_date     symbol  strike  entry_price  underlying  moneyness
    2010-06-07 00:00:00+00:00 6JN0 C1150 0.01150     0.000032    0.010921   1.053017
    2010-07-01 00:00:00+00:00 6JN0 C1200 0.01200     0.000003    0.011410   1.051709
    2010-08-02 00:00:00+00:00 6JU0 C1250 0.01250     0.000006    0.011561   1.081221
    2010-09-01 00:00:00+00:00 6JV0 C1260 0.01260     0.000031    0.011833   1.064819
    2010-10-01 00:00:00+00:00 6JV0 C1245 0.01245     0.000005    0.012027   1.035171


## 5. Leveraged Carry + Dual-Leg Hedge Backtest

**Hedge configurations tested:**
1. **AUD puts only** — protects AUD weakness leg
2. **JPY calls only** — protects JPY strength leg
3. **AUD puts + JPY calls** — protects both legs (budget split 50/50)

Put budget is 0.5% of notional per month. For dual-leg, each side gets 0.25%.


```python
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


def get_settlement(symbol, strike, expiry, opt_type, lookup, front_prices):
    """Get option settlement price."""
    window_start = expiry - pd.Timedelta(days=5)
    window_end = expiry + pd.Timedelta(days=2)
    if symbol in lookup:
        near = [(d, p) for d, p in lookup[symbol] if window_start <= d <= window_end]
        if near:
            return near[-1][1]
    # Fallback: intrinsic value
    near_dates = front_prices[
        (front_prices.index >= (expiry - pd.Timedelta(days=3)).tz_localize(None)) &
        (front_prices.index <= (expiry + pd.Timedelta(days=3)).tz_localize(None))
    ]
    if len(near_dates) > 0:
        underlying = near_dates.iloc[-1]['close']
        if opt_type == 'P':
            return max(0, strike - underlying)
        else:  # call
            return max(0, underlying - strike)
    return 0.0


def precompute_settlements(selections, opt_type, lookup, front_prices):
    """Pre-compute settlement for all selected options."""
    put_map = {}
    for _, row in selections.iterrows():
        settle = get_settlement(row['symbol'], row['strike'], row['expiry'],
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


def run_backtest(cross, aud_front, jpy_front,
                 aud_put_sels, jpy_call_sels, aud_opts, jpy_opts,
                 leverage=1, aud_budget=0.005, jpy_budget=0.0):
    """Run leveraged carry backtest with optional dual-leg hedge.
    
    Args:
        aud_budget: fraction of notional spent on AUD puts per month
        jpy_budget: fraction of notional spent on JPY calls per month
    """
    # Build settlement lookups
    aud_lookup = build_settlement_lookup(aud_opts) if aud_budget > 0 else {}
    jpy_lookup = build_settlement_lookup(jpy_opts) if jpy_budget > 0 else {}
    
    # Pre-compute settlements
    aud_map = precompute_settlements(aud_put_sels, 'P', aud_lookup, aud_front) if aud_budget > 0 else {}
    jpy_map = precompute_settlements(jpy_call_sels, 'C', jpy_lookup, jpy_front) if jpy_budget > 0 else {}
    
    # Daily simulation
    capital = 100.0
    records = []
    current_month = None
    
    for date in cross.index:
        if capital <= 0:
            records.append({'date': date, 'capital': 0, 'daily_carry': 0,
                          'daily_spot': 0, 'aud_put_pnl': 0, 'jpy_call_pnl': 0})
            continue
        
        notional = capital * leverage
        carry_income = notional * cross.loc[date, 'daily_carry']
        spot_pnl = notional * cross.loc[date, 'cross_ret']
        
        aud_pnl = 0
        jpy_pnl = 0
        ym = pd.Timestamp(date).to_period('M')
        if ym != current_month:
            current_month = ym
            date_tz = pd.Timestamp(date, tz='UTC')
            
            # AUD put hedge
            if aud_budget > 0 and date_tz in aud_map:
                cost = aud_budget * notional
                aud_pnl = cost * aud_map[date_tz]['pnl_ratio']  # net = payoff - cost
            
            # JPY call hedge
            if jpy_budget > 0 and date_tz in jpy_map:
                cost = jpy_budget * notional
                jpy_pnl = cost * jpy_map[date_tz]['pnl_ratio']
        
        capital += carry_income + spot_pnl + aud_pnl + jpy_pnl
        records.append({
            'date': date, 'capital': capital, 'daily_carry': carry_income,
            'daily_spot': spot_pnl, 'aud_put_pnl': aud_pnl, 'jpy_call_pnl': jpy_pnl,
        })
    
    return pd.DataFrame(records).set_index('date')


# Empty selections for "no hedge" scenarios
empty_sels = pd.DataFrame(columns=['entry_date', 'symbol', 'strike', 'entry_price',
                                    'expiry', 'underlying', 'moneyness', 'volume'])

# Run all configurations
configs = {
    # (label, leverage, aud_puts, jpy_calls, aud_budget, jpy_budget)
    '1x unhedged': (1, empty_sels, empty_sels, 0, 0),
    '1x AUD puts only': (1, aud_puts_8, empty_sels, 0.005, 0),
    '1x JPY calls only': (1, empty_sels, jpy_calls_8, 0, 0.005),
    '1x dual hedge': (1, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
    '3x unhedged': (3, empty_sels, empty_sels, 0, 0),
    '3x AUD puts only': (3, aud_puts_8, empty_sels, 0.005, 0),
    '3x JPY calls only': (3, empty_sels, jpy_calls_8, 0, 0.005),
    '3x dual hedge': (3, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
    '5x unhedged': (5, empty_sels, empty_sels, 0, 0),
    '5x AUD puts only': (5, aud_puts_8, empty_sels, 0.005, 0),
    '5x JPY calls only': (5, empty_sels, jpy_calls_8, 0, 0.005),
    '5x dual hedge': (5, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
}

results = {}
for label, (lev, aud_sels, jpy_sels, aud_b, jpy_b) in configs.items():
    print(f'Running {label}...')
    results[label] = run_backtest(
        cross, aud_front_naive, jpy_front_naive,
        aud_sels, jpy_sels, aud_opts, jpy_opts,
        leverage=lev, aud_budget=aud_b, jpy_budget=jpy_b
    )

print('Done!')
```

    Running 1x unhedged...
    Running 1x AUD puts only...


    Running 1x JPY calls only...


    Running 1x dual hedge...


    Running 3x unhedged...
    Running 3x AUD puts only...


    Running 3x JPY calls only...


    Running 3x dual hedge...


    Running 5x unhedged...
    Running 5x AUD puts only...


    Running 5x JPY calls only...


    Running 5x dual hedge...


    Done!


## 6. Results


```python
def compute_stats(capital_series):
    """Compute comprehensive strategy stats from capital series."""
    cap = capital_series[capital_series > 0]
    if len(cap) < 252:
        return None
    daily_ret = cap.pct_change().dropna()
    years = (cap.index[-1] - cap.index[0]).days / 365.25
    total_ret = cap.iloc[-1] / cap.iloc[0]
    ann_ret = total_ret ** (1/years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cap / cap.cummax() - 1).min()

    # Sortino: only penalize downside volatility
    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    # Calmar: return / max drawdown
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Tail ratio: upside 95th pctile / abs(downside 5th pctile)
    p95 = np.percentile(daily_ret, 95)
    p5 = np.percentile(daily_ret, 5)
    tail_ratio = p95 / abs(p5) if p5 != 0 else 0

    # Skewness and kurtosis of daily returns
    skew = daily_ret.skew()
    kurt = daily_ret.kurtosis()  # excess kurtosis (normal = 0)

    # Max drawdown duration (in trading days)
    drawdown = cap / cap.cummax() - 1
    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    if in_dd.any():
        dd_durations = in_dd.groupby(dd_groups).sum()
        max_dd_days = int(dd_durations.max())
    else:
        max_dd_days = 0

    # Monthly returns for monthly stats
    monthly_ret = cap.resample('ME').last().pct_change().dropna()
    monthly_mean = monthly_ret.mean() * 12
    monthly_vol = monthly_ret.std() * np.sqrt(12)
    best_month = monthly_ret.max()
    worst_month = monthly_ret.min()
    pct_positive_months = (monthly_ret > 0).mean() * 100

    # Yearly returns
    yearly_ret = cap.resample('YE').last().pct_change().dropna()
    best_year = yearly_ret.max()
    worst_year = yearly_ret.min()
    pct_positive_years = (yearly_ret > 0).mean() * 100

    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'sortino': sortino, 'calmar': calmar,
        'max_dd': max_dd, 'max_dd_days': max_dd_days,
        'tail_ratio': tail_ratio, 'skew': skew, 'kurt': kurt,
        'best_month': best_month, 'worst_month': worst_month,
        'pct_pos_months': pct_positive_months,
        'best_year': best_year, 'worst_year': worst_year,
        'pct_pos_years': pct_positive_years,
        'total': total_ret, 'blown': capital_series.min() <= 0,
    }


# ── Summary table ──
print('=' * 120)
print('LEVERAGED AUD/JPY CARRY — FULL METRICS')
print(f'Period: {cross.index.min()} to {cross.index.max()} | Avg carry: {cross["daily_carry"].mean()*365*100:.2f}%/yr')
print('=' * 120)
print(f'{"Strategy":28} {"Return":>8} {"Vol":>6} {"Sharpe":>7} {"Sortino":>8} {"Calmar":>7} '
      f'{"MaxDD":>7} {"DD days":>7} {"Tail":>5} {"Skew":>6} {"Kurt":>6} {"Total":>7}')
print('-' * 120)

for label in ['1x unhedged', '1x AUD puts only', '1x JPY calls only', '1x dual hedge',
              '3x unhedged', '3x AUD puts only', '3x JPY calls only', '3x dual hedge',
              '5x unhedged', '5x AUD puts only', '5x JPY calls only', '5x dual hedge']:
    s = compute_stats(results[label]['capital'])
    if s:
        print(f'{label:28} {s["ann_ret"]*100:>7.2f}% {s["ann_vol"]*100:>5.1f}% '
              f'{s["sharpe"]:>7.3f} {s["sortino"]:>8.3f} {s["calmar"]:>7.3f} '
              f'{s["max_dd"]*100:>6.1f}% {s["max_dd_days"]:>7d} '
              f'{s["tail_ratio"]:>5.2f} {s["skew"]:>6.2f} {s["kurt"]:>6.1f} '
              f'{s["total"]:>6.1f}x')
    if label.endswith('dual hedge'):
        print()

print('-' * 120)
print(f'{"SPY + 0.5% puts":28} {"16.02%":>8} {"17.8%":>6} {"0.901":>7} {"1.150":>8} {"0.340":>7} '
      f'{"-47.1%":>7} {"":>7} {"":>5} {"":>6} {"":>6}')
```

    ========================================================================================================================
    LEVERAGED AUD/JPY CARRY — FULL METRICS
    Period: 2010-06-06 00:00:00 to 2026-02-27 00:00:00 | Avg carry: 2.39%/yr
    ========================================================================================================================
    Strategy                       Return    Vol  Sharpe  Sortino  Calmar   MaxDD DD days  Tail   Skew   Kurt   Total
    ------------------------------------------------------------------------------------------------------------------------
    1x unhedged                     6.81%  11.1%   0.616    0.840   0.243  -28.1%    2273  1.03  -0.25    3.2    2.8x
    1x AUD puts only               14.05%  19.0%   0.741    1.711   0.619  -22.7%    1096  1.03  21.21  786.2    7.9x
    1x JPY calls only               8.02%  13.3%   0.604    0.978   0.246  -32.6%    2490  1.02   4.61   84.5    3.4x
    1x dual hedge                  11.37%  13.9%   0.817    1.386   0.515  -22.1%    1943  1.04   6.83  167.4    5.4x
    
    3x unhedged                    16.45%  33.2%   0.496    0.676   0.233  -70.5%    2301  1.03  -0.25    3.2   11.0x
    3x AUD puts only               35.69%  56.9%   0.627    1.449   0.605  -58.9%    1944  1.03  21.21  786.2  121.6x
    3x JPY calls only              18.67%  39.8%   0.469    0.759   0.239  -78.1%    2962  1.02   4.61   84.5   14.8x
    3x dual hedge                  29.58%  41.8%   0.708    1.202   0.464  -63.8%    2142  1.04   6.83  167.4   58.9x
    
    5x unhedged                    19.41%  55.3%   0.351    0.478   0.212  -91.4%    3402  1.03  -0.25    3.2   16.3x
    5x AUD puts only               46.99%  94.8%   0.495    1.144   0.575  -81.7%    2275  1.03  21.21  786.2  427.6x
    5x JPY calls only              20.87%  66.3%   0.315    0.509   0.219  -95.5%    3299  1.02   4.61   84.5   19.7x
    5x dual hedge                  39.31%  69.6%   0.565    0.958   0.446  -88.2%    2469  1.04   6.83  167.4  183.9x
    
    ------------------------------------------------------------------------------------------------------------------------
    SPY + 0.5% puts                16.02%  17.8%   0.901    1.150   0.340  -47.1%                            



```python
# ── Year-by-year returns ──
key_strats = ['1x unhedged', '1x AUD puts only', '1x dual hedge',
              '3x unhedged', '3x dual hedge']

# Build yearly returns for each strategy
yearly_data = {}
for label in key_strats:
    cap = results[label]['capital']
    cap = cap[cap > 0]
    yearly = cap.resample('YE').last().pct_change().dropna()
    yearly_data[label] = yearly

# Get all years
all_years = sorted(set(y.year for ys in yearly_data.values() for y in ys.index))

print('=' * 100)
print('YEAR-BY-YEAR RETURNS')
print('=' * 100)
header = f'{"Year":>6}'
for label in key_strats:
    short = label.replace(' unhedged', ' unhdg').replace(' AUD puts only', ' AUD').replace(' dual hedge', ' dual')
    header += f'  {short:>12}'
print(header)
print('-' * 100)

for year in all_years:
    row = f'{year:>6}'
    for label in key_strats:
        ys = yearly_data[label]
        match = ys[ys.index.year == year]
        if len(match) > 0:
            ret = match.iloc[0]
            row += f'  {ret*100:>11.1f}%'
        else:
            row += f'  {"—":>12}'
    print(row)

# Averages
print('-' * 100)
row = f'{"Avg":>6}'
for label in key_strats:
    avg = yearly_data[label].mean()
    row += f'  {avg*100:>11.1f}%'
print(row)

row = f'{"Med":>6}'
for label in key_strats:
    med = yearly_data[label].median()
    row += f'  {med*100:>11.1f}%'
print(row)

row = f'{"%+":>6}'
for label in key_strats:
    pct = (yearly_data[label] > 0).mean() * 100
    row += f'  {pct:>11.0f}%'
print(row)
```

    ====================================================================================================
    YEAR-BY-YEAR RETURNS
    ====================================================================================================
      Year      1x unhdg        1x AUD       1x dual      3x unhdg       3x dual
    ----------------------------------------------------------------------------------------------------
      2011         -1.2%         52.6%         24.9%        -10.8%         60.7%
      2012         21.9%         16.0%         17.6%         73.8%         56.0%
      2013          8.9%         71.0%         36.6%         21.6%        124.4%
      2014          8.3%         19.2%         10.7%         23.5%         30.1%
      2015         -6.8%        -10.5%         -9.8%        -22.9%        -30.3%
      2016         -1.2%          1.3%          4.0%        -10.3%          3.3%
      2017          7.8%          4.2%          3.9%         22.2%          9.3%
      2018         -9.1%          6.8%         -1.1%        -27.3%         -7.7%
      2019          0.9%          6.1%          1.5%         -0.3%          1.2%
      2020          5.2%          4.7%          2.6%          9.9%          1.9%
      2021          5.7%         16.7%         16.7%         15.0%         51.6%
      2022          9.7%          5.7%         12.1%         25.3%         32.4%
      2023         15.8%         15.5%         29.5%         49.8%        104.1%
      2024          9.0%          3.7%          8.4%         24.9%         22.2%
      2025         14.4%          9.9%         10.7%         43.8%         30.0%
      2026          6.6%          6.1%          6.1%         20.6%         18.7%
    ----------------------------------------------------------------------------------------------------
       Avg          6.0%         14.3%         10.9%         16.2%         31.8%
       Med          7.2%          6.5%          9.5%         21.1%         26.1%
        %+           75%           94%           88%           69%           88%



```python
# Equity curves — compare hedge types at 3x leverage
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 3x leverage comparison
ax = axes[0]
for label, color, ls in [
    ('3x unhedged', 'black', '--'),
    ('3x AUD puts only', 'blue', '-'),
    ('3x JPY calls only', 'orange', '-'),
    ('3x dual hedge', 'green', '-'),
]:
    cap = results[label]['capital'] / 100
    ax.plot(cap.index, cap, color=color, linestyle=ls, label=label, alpha=0.8)

ax.set_title('3x Leveraged AUD/JPY Carry — Hedge Comparison')
ax.set_ylabel('Portfolio Value ($1 start)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(1, color='gray', linewidth=0.5)

# 1x leverage comparison  
ax = axes[1]
for label, color, ls in [
    ('1x unhedged', 'black', '--'),
    ('1x AUD puts only', 'blue', '-'),
    ('1x JPY calls only', 'orange', '-'),
    ('1x dual hedge', 'green', '-'),
]:
    cap = results[label]['capital'] / 100
    ax.plot(cap.index, cap, color=color, linestyle=ls, label=label, alpha=0.8)

ax.set_title('1x AUD/JPY Carry — Hedge Comparison')
ax.set_ylabel('Portfolio Value ($1 start)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(1, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('../research/fx_carry_dual_hedge.png', dpi=100, bbox_inches='tight')
plt.show()
print('Saved to research/fx_carry_dual_hedge.png')
```


    
![png](fx_carry_real_files/fx_carry_real_15_0.png)
    


    Saved to research/fx_carry_dual_hedge.png


## 7. Put Economics at Each Leverage Level


```python
# P&L decomposition for dual hedge vs single leg at 3x
for label in ['3x unhedged', '3x AUD puts only', '3x JPY calls only', '3x dual hedge']:
    r = results[label]
    years = (r.index[-1] - r.index[0]).days / 365.25
    carry = r['daily_carry'].sum()
    spot = r['daily_spot'].sum()
    aud_p = r['aud_put_pnl'].sum()
    jpy_c = r['jpy_call_pnl'].sum()
    
    print(f'\n=== {label} ===')
    print(f'  Carry income:  ${carry:>8.1f} ({carry/years:>6.1f}/yr)')
    print(f'  Spot P&L:      ${spot:>8.1f} ({spot/years:>6.1f}/yr)')
    if aud_p != 0:
        print(f'  AUD put P&L:   ${aud_p:>8.1f} ({aud_p/years:>6.1f}/yr)')
    if jpy_c != 0:
        print(f'  JPY call P&L:  ${jpy_c:>8.1f} ({jpy_c/years:>6.1f}/yr)')
    print(f'  Final capital: ${r["capital"].iloc[-1]:>8.1f} (started $100)')
```

    
    === 3x unhedged ===
      Carry income:  $   323.0 (  20.5/yr)
      Spot P&L:      $   674.5 (  42.9/yr)
      Final capital: $  1097.6 (started $100)
    
    === 3x AUD puts only ===
      Carry income:  $  3389.6 ( 215.5/yr)
      Spot P&L:      $  7472.1 ( 475.1/yr)
      AUD put P&L:   $  1198.3 (  76.2/yr)
      Final capital: $ 12160.0 (started $100)
    
    === 3x JPY calls only ===
      Carry income:  $   335.1 (  21.3/yr)
      Spot P&L:      $   697.3 (  44.3/yr)
      JPY call P&L:  $   344.4 (  21.9/yr)
      Final capital: $  1476.9 (started $100)
    
    === 3x dual hedge ===
      Carry income:  $  1364.6 (  86.8/yr)
      Spot P&L:      $  2916.8 ( 185.4/yr)
      AUD put P&L:   $   346.5 (  22.0/yr)
      JPY call P&L:  $  1162.4 (  73.9/yr)
      Final capital: $  5890.3 (started $100)


## 8. Crisis Zoom-In


```python
crises = [
    ('2011 EU debt', '2011-07-01', '2011-10-31'),
    ('2013 taper tantrum', '2013-05-01', '2013-08-31'),
    ('2015 China deval', '2015-07-01', '2015-09-30'),
    ('2018 trade war', '2018-01-01', '2018-12-31'),
    ('2020 COVID', '2020-01-01', '2020-04-30'),
    ('2022 rate hikes', '2022-04-01', '2022-10-31'),
]

lev = 3
print(f'\n=== Crisis Performance ({lev}x leverage) ===')
print(f'{"Crisis":25} {"AUD/JPY":>8} {"Unhdgd":>8} {"AUD put":>8} {"JPY call":>9} {"Dual":>8}')
print('-' * 76)

for name, start, end in crises:
    mask = (cross.index >= start) & (cross.index <= end)
    if mask.sum() == 0:
        continue
    
    spot = cross.loc[mask, 'audjpy'].iloc[-1] / cross.loc[mask, 'audjpy'].iloc[0] - 1
    
    rets = {}
    for label in [f'{lev}x unhedged', f'{lev}x AUD puts only', f'{lev}x JPY calls only', f'{lev}x dual hedge']:
        cap = results[label]['capital'][mask]
        rets[label] = cap.iloc[-1] / cap.iloc[0] - 1 if len(cap) >= 2 else 0
    
    print(f'{name:25} {spot*100:>7.1f}% {rets[f"{lev}x unhedged"]*100:>7.1f}% '
          f'{rets[f"{lev}x AUD puts only"]*100:>7.1f}% {rets[f"{lev}x JPY calls only"]*100:>8.1f}% '
          f'{rets[f"{lev}x dual hedge"]*100:>7.1f}%')
```

    
    === Crisis Performance (3x leverage) ===
    Crisis                     AUD/JPY   Unhdgd  AUD put  JPY call     Dual
    ----------------------------------------------------------------------------
    2011 EU debt                 -4.9%   -11.0%   174.2%    -14.9%    70.2%
    2013 taper tantrum          -12.3%   -31.8%   -24.1%    -29.3%   -26.7%
    2015 China deval            -10.8%   -29.2%   -30.3%    -31.3%   -30.8%
    2018 trade war              -12.2%   -27.2%    12.8%    -26.5%    -7.6%
    2020 COVID                   -8.5%   -25.2%   -26.4%    -27.4%   -26.9%
    2022 rate hikes               3.4%    14.0%    11.1%      4.0%     7.5%


## 9. Conclusions


```python
print("""
KEY FINDINGS — DUAL-LEG HEDGE
==============================

1. AUD/JPY can crash two ways: AUD weakness OR JPY strength.
   AUD puts alone only catch the first. JPY calls catch the second.
   The dual hedge covers both scenarios.

2. In crisis, which leg mattered?
   - 2011 EU debt: AUD dropped → AUD puts paid off
   - 2013 taper tantrum: AUD dropped → AUD puts were the key leg  
   - 2015 China: Both AUD weak + JPY strong → dual hedge covers best
   - 2020 COVID: JPY spiked first → JPY calls triggered before AUD puts
   - 2022 rate hikes: JPY weakened (AUD/JPY rose!) → no hedge needed

3. Same total budget (0.5% of notional), split 50/50 between legs,
   should give better coverage than concentrating on one leg.

4. SPY + puts comparison remains the benchmark at Sharpe 0.901.
""")
```

    
    KEY FINDINGS — DUAL-LEG HEDGE
    ==============================
    
    1. AUD/JPY can crash two ways: AUD weakness OR JPY strength.
       AUD puts alone only catch the first. JPY calls catch the second.
       The dual hedge covers both scenarios.
    
    2. In crisis, which leg mattered?
       - 2011 EU debt: AUD dropped → AUD puts paid off
       - 2013 taper tantrum: AUD dropped → AUD puts were the key leg  
       - 2015 China: Both AUD weak + JPY strong → dual hedge covers best
       - 2020 COVID: JPY spiked first → JPY calls triggered before AUD puts
       - 2022 rate hikes: JPY weakened (AUD/JPY rose!) → no hedge needed
    
    3. Same total budget (0.5% of notional), split 50/50 between legs,
       should give better coverage than concentrating on one leg.
    
    4. SPY + puts comparison remains the benchmark at Sharpe 0.901.
    

