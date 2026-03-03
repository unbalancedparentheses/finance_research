#!/usr/bin/env python3
"""FX carry + tail hedge analysis: single pair, multi-asset, portfolio, and leverage."""
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from databento_helpers import (
    MONTH_CODES, STRIKE_DIVISORS, OPT_CONFIGS,
    compute_stats, load_front_month, parse_option_generic,
    load_fx_options, select_monthly_puts,
    build_settlement_lookup, get_settlement, precompute_settlements,
)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'databento')
CHART_DIR = os.path.join(SCRIPT_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# ── Shared constants ─────────────────────────────────────────────────────────

CARRY_PAIRS = ['AUD', 'GBP', 'CAD', 'EUR', 'CHF', 'MXN']
ANALYSIS_PAIRS = ['AUD', 'GBP', 'CAD', 'EUR', 'MXN']
LEVERAGE_LEVELS = [1, 2, 3, 5, 7, 10]
COLORS = {'AUD': '#1f77b4', 'GBP': '#d62728', 'CAD': '#2ca02c',
          'EUR': '#9467bd', 'MXN': '#ff7f0e', 'CHF': '#8c564b'}

FUT_FILES = {
    'AUD': '6A_FUT_ohlcv1d.parquet',
    'GBP': '6B_FUT_ohlcv1d.parquet',
    'CAD': '6C_FUT_ohlcv1d.parquet',
    'EUR': '6E_FUT_ohlcv1d.parquet',
    'CHF': '6S_FUT_ohlcv1d.parquet',
    'MXN': '6M_FUT_ohlcv1d.parquet',
    'NZD': '6N_FUT_ohlcv1d.parquet',
    'JPY': '6J_FUT_ohlcv1d.parquet',
}

# ── Policy rate tables ───────────────────────────────────────────────────────

BOJ_RATES = {y: 0.0 for y in range(2010, 2027)}
BOJ_RATES[2024] = 0.25
BOJ_RATES[2025] = 0.50
BOJ_RATES[2026] = 0.50

POLICY_RATES = {
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
    'NZD': {
        2010: 3.00, 2011: 2.50, 2012: 2.50, 2013: 2.50, 2014: 3.25,
        2015: 2.75, 2016: 2.00, 2017: 1.75, 2018: 1.75, 2019: 1.25,
        2020: 0.50, 2021: 0.50, 2022: 2.75, 2023: 5.50, 2024: 5.50,
        2025: 4.25, 2026: 3.50,
    },
}

FED_RATES = {
    2010: 0.25, 2011: 0.25, 2012: 0.25, 2013: 0.25, 2014: 0.25,
    2015: 0.25, 2016: 0.50, 2017: 1.25, 2018: 2.00, 2019: 2.25,
    2020: 0.25, 2021: 0.25, 2022: 2.50, 2023: 5.00, 2024: 5.00,
    2025: 4.50,
}


# ── Shared helper functions (unique to this analysis) ────────────────────────

def _select_monthly_options(opts, front_prices, opt_type='P', otm_target=0.92):
    """Select one OTM option per month (supports puts and calls)."""
    filtered = opts[opts['opt_type'] == opt_type].copy()
    if len(filtered) == 0:
        return pd.DataFrame()

    prices = front_prices[['close']].rename(columns={'close': 'fut_close'})
    if prices.index.tz is None:
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
            'entry_date': first_day, 'symbol': best['symbol'],
            'strike': best['strike'], 'entry_price': best['close'],
            'expiry': best['expiry'], 'underlying': best['fut_close'],
            'moneyness': best['moneyness'], 'volume': best['volume'],
        })
    return pd.DataFrame(selections)


def _get_settlement_with_type(symbol, strike, expiry, opt_type, lookup, front_prices):
    """Get option settlement price (supports calls and puts via opt_type)."""
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
        if opt_type == 'P':
            return max(0, strike - underlying)
        else:
            return max(0, underlying - strike)
    return 0.0


def _precompute_settlements_typed(selections, opt_type, lookup, front_prices):
    """Pre-compute settlement for all selected options (puts or calls)."""
    put_map = {}
    for _, row in selections.iterrows():
        settle = _get_settlement_with_type(
            row['symbol'], row['strike'], row['expiry'],
            opt_type, lookup, front_prices)
        entry_price = row['entry_price']
        pnl_ratio = (settle - entry_price) / entry_price if entry_price > 0 else 0
        put_map[row['entry_date']] = {
            'symbol': row['symbol'], 'strike': row['strike'],
            'entry_price': entry_price, 'settlement': settle,
            'pnl_ratio': pnl_ratio, 'moneyness': row['moneyness'],
        }
    return put_map


def _load_all_futures(data_dir, pairs=None):
    """Load front-month futures for all FX pairs."""
    if pairs is None:
        pairs = FUT_FILES
    fx_futures = {}
    for ccy, filename in pairs.items():
        print(f'Loading {ccy} futures...')
        fx_futures[ccy] = load_front_month(data_dir, filename)
        f = fx_futures[ccy]
        print(f'  {len(f)} days, {f.index.min().date()} to {f.index.max().date()}, '
              f'close: {f["close"].iloc[-1]:.6f}')
    return fx_futures


def _build_cross_rates(fx_futures, carry_pairs):
    """Build cross rates vs JPY for each carry pair."""
    jpy = fx_futures['JPY']
    cross_rates = {}
    for ccy in carry_pairs:
        f = fx_futures[ccy]
        common = f.index.intersection(jpy.index)
        cross = pd.DataFrame({
            f'{ccy.lower()}_jpy': f.loc[common, 'close'] / jpy.loc[common, 'close'],
            'ccy_ret': f.loc[common, 'return'],
            'jpy_ret': jpy.loc[common, 'return'],
            'cross_ret': f.loc[common, 'return'] - jpy.loc[common, 'return'],
        }, index=common)
        cross_rates[ccy] = cross

        spot_col = f'{ccy.lower()}_jpy'
        years = (cross.index[-1] - cross.index[0]).days / 365.25
        total = cross[spot_col].iloc[-1] / cross[spot_col].iloc[0]
        cagr = total**(1/years) - 1
        print(f'{ccy}/JPY: {len(cross)} days, range {cross[spot_col].min():.2f}-'
              f'{cross[spot_col].max():.2f}, spot CAGR {cagr*100:.2f}%')
    return cross_rates


def _attach_carry(cross_rates):
    """Attach daily carry rate to each cross-rate DataFrame."""
    for ccy in cross_rates:
        cr = cross_rates[ccy]
        cr['ccy_rate'] = cr.index.year.map(lambda y, c=ccy: POLICY_RATES[c].get(y, 0)) / 100
        cr['jpn_rate'] = cr.index.year.map(lambda y: BOJ_RATES.get(y, 0)) / 100
        cr['daily_carry'] = (cr['ccy_rate'] - cr['jpn_rate']) / 365


def _load_all_options(data_dir, pairs):
    """Load FX options for all specified pairs."""
    fx_options = {}
    for ccy in pairs:
        if ccy not in OPT_CONFIGS:
            continue
        print(f'Loading {ccy} options...')
        opts = load_fx_options(data_dir, ccy)
        fx_options[ccy] = opts
        puts = (opts['opt_type'] == 'P').sum()
        calls = (opts['opt_type'] == 'C').sum()
        print(f'  {len(opts):,} total (puts: {puts:,}, calls: {calls:,}), '
              f'{opts["date"].min().date()} to {opts["date"].max().date()}')
    return fx_options


def _select_all_puts(fx_options, fx_futures, pairs):
    """Select monthly 8% OTM puts for all pairs."""
    put_selections = {}
    for ccy in pairs:
        if ccy not in fx_options:
            continue
        front = pd.DataFrame({'close': fx_futures[ccy]['close']})
        sels = select_monthly_puts(fx_options[ccy], front, otm_target=0.92)
        put_selections[ccy] = sels
        if len(sels) > 0:
            print(f'{ccy} puts: {len(sels)} months, '
                  f'{sels["entry_date"].min().date()} to {sels["entry_date"].max().date()}, '
                  f'avg moneyness {sels["moneyness"].mean():.3f}')
        else:
            print(f'{ccy} puts: no selections')
    return put_selections


def _run_carry_backtest(cross_df, front_prices, put_sels, all_opts,
                        leverage=1, put_budget=0.005):
    """Run leveraged carry + puts backtest for a single pair."""
    empty_sels = pd.DataFrame(columns=['entry_date', 'symbol', 'strike', 'entry_price',
                                        'expiry', 'underlying', 'moneyness', 'volume'])
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
            records.append({'date': date, 'capital': 0, 'daily_carry': 0,
                            'daily_spot': 0, 'put_pnl': 0})
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
        records.append({
            'date': date, 'capital': capital, 'daily_carry': carry_income,
            'daily_spot': spot_pnl, 'put_pnl': put_pnl,
        })

    return pd.DataFrame(records).set_index('date')


def _run_carry_backtest_fast(cross_df, put_map, leverage=1, put_budget=0.005):
    """Faster backtest using pre-computed put settlements."""
    capital = 100.0
    records = []
    current_month = None
    daily_carry_col = cross_df['daily_carry'].values
    cross_ret_col = cross_df['cross_ret'].values
    dates = cross_df.index

    for i in range(len(dates)):
        date = dates[i]
        if capital <= 0:
            records.append({'date': date, 'capital': 0.0})
            continue

        notional = capital * leverage
        carry_income = notional * daily_carry_col[i]
        spot_pnl = notional * cross_ret_col[i]

        put_pnl = 0.0
        ym = pd.Timestamp(date).to_period('M')
        if ym != current_month:
            current_month = ym
            if put_budget > 0 and put_map:
                date_tz = pd.Timestamp(date, tz='UTC')
                if date_tz in put_map:
                    cost = put_budget * notional
                    put_pnl = cost * put_map[date_tz]['pnl_ratio']

        capital += carry_income + spot_pnl + put_pnl
        records.append({'date': date, 'capital': capital})

    return pd.DataFrame(records).set_index('date')


def _empty_sels():
    return pd.DataFrame(columns=['entry_date', 'symbol', 'strike', 'entry_price',
                                  'expiry', 'underlying', 'moneyness', 'volume'])


# ── Portfolio construction helpers (unique to portfolio/leverage analysis) ────

def _build_equal_weight_portfolio(results_dict, pairs, leverage, hedged):
    """Build equal-weight portfolio from multiple carry pair backtests."""
    daily_rets = {}
    for ccy in pairs:
        key = (ccy, leverage, hedged)
        if key not in results_dict:
            key = (ccy, leverage, False)
        if key not in results_dict:
            continue
        cap = results_dict[key]['capital']
        cap = cap[cap > 0]
        daily_rets[ccy] = cap.pct_change().fillna(0)

    if not daily_rets:
        return None

    ret_df = pd.DataFrame(daily_rets)
    ret_df = ret_df.dropna()
    port_ret = ret_df.mean(axis=1)
    capital = (1 + port_ret).cumprod() * 100
    return capital


def _get_daily_returns(results_dict, pairs, leverage, hedged):
    """Extract daily returns for all available pairs at given leverage/hedge."""
    rets = {}
    for ccy in pairs:
        key = (ccy, leverage, hedged)
        if key not in results_dict:
            continue
        cap = results_dict[key]['capital']
        cap = cap[cap > 0]
        rets[ccy] = cap.pct_change().fillna(0)
    return pd.DataFrame(rets)


def _solve_min_variance(cov_matrix, max_iter=1000):
    """Solve minimum variance portfolio with long-only constraint."""
    n = cov_matrix.shape[0]
    if n == 0:
        return np.array([])
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        return np.ones(n) / n

    w = np.ones(n) / n
    lr = 0.5
    for _ in range(max_iter):
        grad = 2 * cov_matrix @ w
        w_new = w - lr * grad
        w_new = np.maximum(w_new, 0)
        s = w_new.sum()
        if s > 0:
            w_new = w_new / s
        else:
            w_new = np.ones(n) / n
        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new
    return w


def _solve_max_sharpe(mean_ret, cov_matrix, rf=0.0, max_iter=2000):
    """Solve max-Sharpe portfolio with long-only constraint."""
    n = len(mean_ret)
    if n == 0:
        return np.array([])
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        return np.ones(n) / n

    excess = mean_ret - rf
    w = np.ones(n) / n
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
        if s > 0:
            w_new = w_new / s
        else:
            w_new = np.ones(n) / n
        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new
    return w


def _build_risk_parity_portfolio(results_dict, pairs, leverage, hedged,
                                  vol_window=60, rebal_freq='ME'):
    """Build risk-parity (inverse vol) portfolio with monthly rebalancing."""
    daily_rets = {}
    for ccy in pairs:
        key = (ccy, leverage, hedged)
        if key not in results_dict:
            key = (ccy, leverage, False)
        if key not in results_dict:
            continue
        cap = results_dict[key]['capital']
        cap = cap[cap > 0]
        daily_rets[ccy] = cap.pct_change().fillna(0)

    if not daily_rets:
        return None, None

    ret_df = pd.DataFrame(daily_rets)
    trailing_vol = ret_df.rolling(vol_window).std() * np.sqrt(252)
    rebal_dates = ret_df.resample(rebal_freq).last().index

    weights_history = []
    port_rets = []
    current_weights = None

    for date in ret_df.index:
        if date in rebal_dates or current_weights is None:
            vol_row = trailing_vol.loc[:date].iloc[-1] if date in trailing_vol.index else None
            if vol_row is not None and vol_row.notna().sum() >= 2:
                inv_vol = 1.0 / vol_row.replace(0, np.nan)
                inv_vol = inv_vol.dropna()
                if len(inv_vol) > 0:
                    current_weights = inv_vol / inv_vol.sum()
                    weights_history.append({'date': date, **current_weights.to_dict()})

        if current_weights is not None:
            day_ret = ret_df.loc[date]
            common = current_weights.index.intersection(day_ret.dropna().index)
            if len(common) > 0:
                w = current_weights[common]
                w = w / w.sum()
                pr = (day_ret[common] * w).sum()
                port_rets.append({'date': date, 'return': pr})

    if not port_rets:
        return None, None

    port_df = pd.DataFrame(port_rets).set_index('date')
    capital = (1 + port_df['return']).cumprod() * 100
    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else None
    return capital, weights_df


def _build_min_variance_portfolio(results_dict, pairs, leverage, hedged,
                                   cov_window=252, rebal_freq='ME'):
    """Build minimum-variance portfolio with monthly rebalancing."""
    daily_rets = {}
    for ccy in pairs:
        key = (ccy, leverage, hedged)
        if key not in results_dict:
            key = (ccy, leverage, False)
        if key not in results_dict:
            continue
        cap = results_dict[key]['capital']
        cap = cap[cap > 0]
        daily_rets[ccy] = cap.pct_change().fillna(0)

    if not daily_rets:
        return None, None

    ret_df = pd.DataFrame(daily_rets)
    rebal_dates = ret_df.resample(rebal_freq).last().index
    weights_history = []
    port_rets = []
    current_weights = None

    for date in ret_df.index:
        if date in rebal_dates or current_weights is None:
            lookback = ret_df.loc[:date].tail(cov_window)
            available = lookback.dropna(axis=1, how='all')
            available = available.loc[:, available.notna().sum() >= cov_window // 2]
            if available.shape[1] >= 2:
                cov = available.cov().values
                cols = list(available.columns)
                w = _solve_min_variance(cov)
                current_weights = pd.Series(w, index=cols)
                weights_history.append({'date': date, **current_weights.to_dict()})

        if current_weights is not None:
            day_ret = ret_df.loc[date]
            common = current_weights.index.intersection(day_ret.dropna().index)
            if len(common) > 0:
                ww = current_weights[common]
                ww = ww / ww.sum()
                pr = (day_ret[common] * ww).sum()
                port_rets.append({'date': date, 'return': pr})

    if not port_rets:
        return None, None

    port_df = pd.DataFrame(port_rets).set_index('date')
    capital = (1 + port_df['return']).cumprod() * 100
    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else None
    return capital, weights_df


def _build_max_sharpe_portfolio(results_dict, pairs, leverage, hedged,
                                 window=252, rebal_freq='QE'):
    """Build max-Sharpe portfolio with quarterly rebalancing."""
    daily_rets = {}
    for ccy in pairs:
        key = (ccy, leverage, hedged)
        if key not in results_dict:
            key = (ccy, leverage, False)
        if key not in results_dict:
            continue
        cap = results_dict[key]['capital']
        cap = cap[cap > 0]
        daily_rets[ccy] = cap.pct_change().fillna(0)

    if not daily_rets:
        return None, None

    ret_df = pd.DataFrame(daily_rets)
    rebal_dates = ret_df.resample(rebal_freq).last().index
    weights_history = []
    port_rets = []
    current_weights = None

    for date in ret_df.index:
        if date in rebal_dates or current_weights is None:
            lookback = ret_df.loc[:date].tail(window)
            available = lookback.dropna(axis=1, how='all')
            available = available.loc[:, available.notna().sum() >= window // 2]
            if available.shape[1] >= 2:
                mean_r = available.mean().values * 252
                cov = available.cov().values * 252
                cols = list(available.columns)
                w = _solve_max_sharpe(mean_r, cov)
                current_weights = pd.Series(w, index=cols)
                weights_history.append({'date': date, **current_weights.to_dict()})

        if current_weights is not None:
            day_ret = ret_df.loc[date]
            common = current_weights.index.intersection(day_ret.dropna().index)
            if len(common) > 0:
                ww = current_weights[common]
                ww = ww / ww.sum()
                pr = (day_ret[common] * ww).sum()
                port_rets.append({'date': date, 'return': pr})

    if not port_rets:
        return None, None

    port_df = pd.DataFrame(port_rets).set_index('date')
    capital = (1 + port_df['return']).cumprod() * 100
    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else None
    return capital, weights_df


def _print_stats_row(label, s, wide=False):
    """Print a formatted stats row."""
    if s is None:
        print(f'{label:>28} — insufficient data')
        return
    if wide:
        print(f'{label:>28} {s["CAGR"]*100:>7.2f}% {s["Vol"]*100:>5.1f}% '
              f'{s["Sharpe"]:>7.3f} {s["Sortino"]:>8.3f} {s["Calmar"]:>7.3f} '
              f'{s["MaxDD"]*100:>6.1f}% {s["Skew"]:>6.2f} {s["Kurt"]:>6.1f} '
              f'{s["Total"]:>6.1f}x')
    else:
        print(f'{label:>28} {s["CAGR"]*100:>7.2f}% {s["Vol"]*100:>5.1f}% '
              f'{s["Sharpe"]:>7.3f} {s["MaxDD"]*100:>6.1f}% {s["Total"]:>6.1f}x')


# ═══════════════════════════════════════════════════════════════════════════════
#  Function 1: AUD/JPY Carry + Dual-Leg Hedge
# ═══════════════════════════════════════════════════════════════════════════════

def run_fx_carry_real(data_dir):
    """AUD/JPY carry with dual-leg hedge (AUD puts + JPY calls)."""

    # 1. Load futures
    print('\n--- Loading AUD and JPY futures ---')
    aud = load_front_month(data_dir, '6A_FUT_ohlcv1d.parquet')
    jpy = load_front_month(data_dir, '6J_FUT_ohlcv1d.parquet')
    print(f'AUD: {len(aud)} days, {aud.index.min().date()} to {aud.index.max().date()}')
    print(f'JPY: {len(jpy)} days, {jpy.index.min().date()} to {jpy.index.max().date()}')

    # 2. Build AUD/JPY cross rate
    common_dates = aud.index.intersection(jpy.index)
    cross = pd.DataFrame({
        'audjpy': aud.loc[common_dates, 'close'] / jpy.loc[common_dates, 'close'],
        'aud_ret': aud.loc[common_dates, 'return'],
        'jpy_ret': jpy.loc[common_dates, 'return'],
        'cross_ret': aud.loc[common_dates, 'return'] - jpy.loc[common_dates, 'return'],
    })
    print(f'AUD/JPY cross: {len(cross)} days, range {cross["audjpy"].min():.1f}'
          f'-{cross["audjpy"].max():.1f}')

    # 3. Carry rates
    rba_rates = POLICY_RATES['AUD']
    cross['aud_rate'] = cross.index.year.map(lambda y: rba_rates.get(y, 0)) / 100
    cross['jpn_rate'] = cross.index.year.map(lambda y: BOJ_RATES.get(y, 0)) / 100
    cross['usd_rate'] = cross.index.year.map(lambda y: FED_RATES.get(y, 0)) / 100
    cross['daily_carry'] = (cross['aud_rate'] - cross['jpn_rate']) / 365
    avg_carry = cross['daily_carry'].mean() * 365
    print(f'Average AUD-JPY carry: {avg_carry*100:.2f}%/yr')

    # 4. Load AUD + JPY options
    print('\n--- Loading options ---')
    aud_opts = load_fx_options(data_dir, 'AUD')
    print(f'AUD options: {len(aud_opts):,}')

    # JPY options need special handling: load old 6J + new JPU
    old_jpy = pd.read_parquet(f'{data_dir}/6J_OPT_ohlcv1d.parquet')
    old_jpy = old_jpy[~old_jpy['symbol'].str.contains('UD:', na=False)].copy()
    new_jpy = pd.read_parquet(f'{data_dir}/JPU_OPT_ohlcv1d.parquet')
    new_jpy = new_jpy[~new_jpy['symbol'].str.contains('UD:', na=False)].copy()
    cutoff_ts = pd.Timestamp('2016-08-16', tz='UTC')
    old_jpy = old_jpy[old_jpy.index < cutoff_ts]
    combined_jpy = pd.concat([old_jpy, new_jpy]).sort_index()

    jpy_prefixes = ['6J', 'JPU']
    jpy_strike_div = 100000
    jpy_records = []
    for idx, row in combined_jpy.iterrows():
        parsed = parse_option_generic(row['symbol'], idx.year, jpy_prefixes, jpy_strike_div)
        if parsed is None:
            continue
        month, year, opt_type, strike = parsed
        try:
            first_of_month = pd.Timestamp(year=year, month=month, day=1)
        except ValueError:
            continue
        third_wed = first_of_month + pd.offsets.WeekOfMonth(week=2, weekday=2)
        expiry = (third_wed - pd.offsets.BDay(2)).tz_localize('UTC')
        jpy_records.append({
            'date': idx, 'symbol': row['symbol'], 'opt_type': opt_type,
            'strike': strike, 'expiry': expiry,
            'close': row['close'], 'volume': row['volume'],
        })
    jpy_opts = pd.DataFrame(jpy_records)
    print(f'JPY options: {len(jpy_opts):,}')

    # 5. Select monthly AUD puts + JPY calls
    aud_front = pd.DataFrame({'close': aud['close']})
    jpy_front = pd.DataFrame({'close': jpy['close']})

    aud_puts_8 = _select_monthly_options(aud_opts, aud_front, opt_type='P', otm_target=0.92)
    jpy_calls_8 = _select_monthly_options(jpy_opts, jpy_front, opt_type='C', otm_target=1.08)
    print(f'AUD puts (8% OTM): {len(aud_puts_8)} months')
    print(f'JPY calls (8% OTM): {len(jpy_calls_8)} months')

    # 6. Run dual-leg backtest
    def _run_dual_backtest(cross_df, aud_front_df, jpy_front_df,
                           aud_put_sels, jpy_call_sels, aud_all_opts, jpy_all_opts,
                           leverage=1, aud_budget=0.005, jpy_budget=0.0):
        aud_lookup = build_settlement_lookup(aud_all_opts) if aud_budget > 0 else {}
        jpy_lookup = build_settlement_lookup(jpy_all_opts) if jpy_budget > 0 else {}
        aud_map = _precompute_settlements_typed(aud_put_sels, 'P', aud_lookup, aud_front_df) if aud_budget > 0 and len(aud_put_sels) > 0 else {}
        jpy_map = _precompute_settlements_typed(jpy_call_sels, 'C', jpy_lookup, jpy_front_df) if jpy_budget > 0 and len(jpy_call_sels) > 0 else {}

        capital = 100.0
        records = []
        current_month = None
        for date in cross_df.index:
            if capital <= 0:
                records.append({'date': date, 'capital': 0, 'daily_carry': 0,
                                'daily_spot': 0, 'aud_put_pnl': 0, 'jpy_call_pnl': 0})
                continue
            notional = capital * leverage
            carry_income = notional * cross_df.loc[date, 'daily_carry']
            spot_pnl = notional * cross_df.loc[date, 'cross_ret']
            aud_pnl = 0
            jpy_pnl = 0
            ym = pd.Timestamp(date).to_period('M')
            if ym != current_month:
                current_month = ym
                date_tz = pd.Timestamp(date, tz='UTC')
                if aud_budget > 0 and date_tz in aud_map:
                    cost = aud_budget * notional
                    aud_pnl = cost * aud_map[date_tz]['pnl_ratio']
                if jpy_budget > 0 and date_tz in jpy_map:
                    cost = jpy_budget * notional
                    jpy_pnl = cost * jpy_map[date_tz]['pnl_ratio']
            capital += carry_income + spot_pnl + aud_pnl + jpy_pnl
            records.append({
                'date': date, 'capital': capital, 'daily_carry': carry_income,
                'daily_spot': spot_pnl, 'aud_put_pnl': aud_pnl, 'jpy_call_pnl': jpy_pnl,
            })
        return pd.DataFrame(records).set_index('date')

    empty = _empty_sels()
    configs = {
        '1x unhedged': (1, empty, empty, 0, 0),
        '1x AUD puts only': (1, aud_puts_8, empty, 0.005, 0),
        '1x dual hedge': (1, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
        '3x unhedged': (3, empty, empty, 0, 0),
        '3x AUD puts only': (3, aud_puts_8, empty, 0.005, 0),
        '3x dual hedge': (3, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
        '5x unhedged': (5, empty, empty, 0, 0),
        '5x AUD puts only': (5, aud_puts_8, empty, 0.005, 0),
        '5x dual hedge': (5, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
    }

    results = {}
    for label, (lev, aud_sels, jpy_sels, aud_b, jpy_b) in configs.items():
        print(f'Running {label}...')
        results[label] = _run_dual_backtest(
            cross, aud_front, jpy_front,
            aud_sels, jpy_sels, aud_opts, jpy_opts,
            leverage=lev, aud_budget=aud_b, jpy_budget=jpy_b)

    # 7. Print results
    print('\n' + '=' * 100)
    print('LEVERAGED AUD/JPY CARRY - DUAL-LEG HEDGE')
    print('=' * 100)
    print(f'{"Strategy":>28} {"CAGR":>8} {"Vol":>6} {"Sharpe":>7} {"MaxDD":>7} {"Total":>7}')
    print('-' * 100)

    for label in configs:
        s = compute_stats(results[label]['capital'])
        _print_stats_row(label, s)
        if label.endswith('dual hedge'):
            print()

    # 8. Equity curves chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, lev_label in [(axes[0], '3x'), (axes[1], '1x')]:
        for label, color, ls in [
            (f'{lev_label} unhedged', 'black', '--'),
            (f'{lev_label} AUD puts only', 'blue', '-'),
            (f'{lev_label} dual hedge', 'green', '-'),
        ]:
            if label in results:
                cap = results[label]['capital'] / 100
                ax.plot(cap.index, cap, color=color, linestyle=ls, label=label, alpha=0.8)
        ax.set_title(f'{lev_label} Leveraged AUD/JPY Carry')
        ax.set_ylabel('Portfolio Value ($1 start)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'fxr_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved fxr_equity_curves.png')

    # 9. Crisis performance
    crises = [
        ('2011 EU debt', '2011-07-01', '2011-10-31'),
        ('2015 China deval', '2015-07-01', '2015-09-30'),
        ('2018 trade war', '2018-01-01', '2018-12-31'),
        ('2020 COVID', '2020-01-01', '2020-04-30'),
    ]
    print(f'\n--- Crisis Performance (3x) ---')
    print(f'{"Crisis":>22} {"Unhdgd":>8} {"AUD put":>8} {"Dual":>8}')
    print('-' * 55)
    for name, start, end in crises:
        mask = (cross.index >= start) & (cross.index <= end)
        if mask.sum() == 0:
            continue
        rets = {}
        for label in ['3x unhedged', '3x AUD puts only', '3x dual hedge']:
            cap = results[label]['capital'][mask]
            rets[label] = cap.iloc[-1] / cap.iloc[0] - 1 if len(cap) >= 2 else 0
        print(f'{name:>22} {rets["3x unhedged"]*100:>7.1f}% '
              f'{rets["3x AUD puts only"]*100:>7.1f}% {rets["3x dual hedge"]*100:>7.1f}%')


# ═══════════════════════════════════════════════════════════════════════════════
#  Function 2: Multi-Asset FX Carry
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_asset(data_dir):
    """7 FX pairs vs JPY with monthly OTM puts."""

    # 1. Load all futures
    print('\n--- Loading FX futures ---')
    fx_futures = _load_all_futures(data_dir)

    # 2. Build cross rates
    print('\n--- Cross rates vs JPY ---')
    carry_pairs = ['AUD', 'GBP', 'CAD', 'EUR', 'CHF', 'MXN', 'NZD']
    cross_rates = _build_cross_rates(fx_futures, carry_pairs)
    _attach_carry(cross_rates)

    # Print carry summary
    print(f'\n{"Pair":>10} {"Avg carry":>10}')
    print('-' * 25)
    for ccy in carry_pairs:
        avg = cross_rates[ccy]['daily_carry'].mean() * 365 * 100
        print(f'{ccy}/JPY:  {avg:>8.2f}%')

    # 3. Load options
    print('\n--- Loading options ---')
    opt_pairs = ['AUD', 'GBP', 'CAD', 'EUR', 'CHF', 'MXN']
    fx_options = _load_all_options(data_dir, opt_pairs)

    # 4. Select puts
    print('\n--- Selecting monthly puts ---')
    put_selections = _select_all_puts(fx_options, fx_futures, opt_pairs)

    # 5. Run backtests for all pairs
    print('\n--- Running backtests ---')
    all_results = {}
    empty = _empty_sels()

    for ccy in opt_pairs:
        cr = cross_rates[ccy]
        front = pd.DataFrame({'close': fx_futures[ccy]['close']})
        sels = put_selections.get(ccy, empty)
        opts = fx_options.get(ccy, pd.DataFrame())
        has_hedge = len(sels) >= 12

        for lev in [1, 3]:
            print(f'Running {ccy}/JPY {lev}x unhedged...')
            all_results[(ccy, lev, False)] = _run_carry_backtest(
                cr, front, empty, opts, leverage=lev, put_budget=0)
            if has_hedge:
                print(f'Running {ccy}/JPY {lev}x hedged...')
                all_results[(ccy, lev, True)] = _run_carry_backtest(
                    cr, front, sels, opts, leverage=lev, put_budget=0.005)

    # 6. Print results
    for lev in [1, 3]:
        print(f'\n{"=" * 100}')
        print(f'MULTI-ASSET FX CARRY vs JPY - {lev}x LEVERAGE')
        print(f'{"=" * 100}')
        print(f'{"Pair":>12} {"Type":>10} {"CAGR":>8} {"Vol":>6} {"Sharpe":>7} '
              f'{"MaxDD":>7} {"Total":>7}')
        print('-' * 100)

        for ccy in opt_pairs:
            for hedged in [False, True]:
                key = (ccy, lev, hedged)
                if key not in all_results:
                    continue
                s = compute_stats(all_results[key]['capital'])
                if s is None:
                    continue
                label = f'{ccy}/JPY'
                htype = 'hedged' if hedged else 'unhedged'
                print(f'{label:>12} {htype:>10} {s["CAGR"]*100:>7.2f}% {s["Vol"]*100:>5.1f}% '
                      f'{s["Sharpe"]:>7.3f} {s["MaxDD"]*100:>6.1f}% {s["Total"]:>6.1f}x')
            print()

    # 7. Equal-weight portfolios
    core_pairs = ['AUD', 'GBP', 'CAD', 'EUR']
    all_6 = opt_pairs

    port_results = {}
    for lev in [1, 3]:
        for hedged in [False, True]:
            for pairs_list, label in [(core_pairs, 'core'), (all_6, 'all')]:
                cap = _build_equal_weight_portfolio(all_results, pairs_list, lev, hedged)
                if cap is not None:
                    tag = f'{label} {lev}x {"hedged" if hedged else "unhedged"}'
                    port_results[tag] = cap

    print(f'\n{"=" * 100}')
    print(f'EQUAL-WEIGHT CARRY PORTFOLIOS')
    print(f'{"=" * 100}')
    print(f'{"Portfolio":>30} {"CAGR":>8} {"Vol":>6} {"Sharpe":>7} {"MaxDD":>7} {"Total":>7}')
    print('-' * 100)
    for tag in sorted(port_results.keys()):
        s = compute_stats(port_results[tag])
        _print_stats_row(tag, s)

    # 8. Equity curves chart
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    colors = {'AUD': 'blue', 'GBP': 'red', 'CAD': 'green',
              'EUR': 'purple', 'CHF': 'brown', 'MXN': 'orange'}

    for ax, lev in [(axes[0], 1), (axes[1], 3)]:
        for ccy in opt_pairs:
            for hedged, ls, alpha in [(False, '--', 0.5), (True, '-', 0.9)]:
                key = (ccy, lev, hedged)
                if key in all_results:
                    cap = all_results[key]['capital'] / 100
                    hstr = 'hedged' if hedged else 'unhedged'
                    ax.plot(cap.index, cap, color=colors[ccy], linestyle=ls,
                            alpha=alpha, label=f'{ccy}/JPY {hstr}')
        ax.set_title(f'FX Carry vs JPY - {lev}x Leverage')
        ax.set_ylabel('Portfolio Value ($1 start)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'ma_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved ma_equity_curves.png')

    # 9. Correlation heatmap
    ret_matrix = pd.DataFrame()
    for ccy in opt_pairs:
        ret_matrix[f'{ccy}/JPY'] = cross_rates[ccy]['cross_ret']
    corr = ret_matrix.corr()
    avg_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().mean()
    print(f'\nAverage pairwise correlation: {avg_corr:.3f}')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdYlGn_r', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)
    ax.set_title('FX Carry Pair Correlation (daily returns)')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'ma_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved ma_correlation.png')


# ═══════════════════════════════════════════════════════════════════════════════
#  Function 3: Portfolio Construction
# ═══════════════════════════════════════════════════════════════════════════════

def run_portfolio(data_dir):
    """Portfolio construction: EW, risk-parity, min-var, max-Sharpe."""

    # 1. Load data (same as multi_asset)
    print('\n--- Loading data ---')
    fx_futures = _load_all_futures(data_dir, {k: v for k, v in FUT_FILES.items()
                                               if k in CARRY_PAIRS + ['JPY']})
    cross_rates = _build_cross_rates(fx_futures, CARRY_PAIRS)
    _attach_carry(cross_rates)

    fx_options = _load_all_options(data_dir, CARRY_PAIRS)
    put_selections = _select_all_puts(fx_options, fx_futures, CARRY_PAIRS)

    # 2. Run per-pair backtests
    print('\n--- Running per-pair backtests ---')
    all_results = {}
    empty = _empty_sels()

    for ccy in CARRY_PAIRS:
        cr = cross_rates[ccy]
        front = pd.DataFrame({'close': fx_futures[ccy]['close']})
        sels = put_selections.get(ccy, empty)
        opts = fx_options.get(ccy, pd.DataFrame())
        has_hedge = len(sels) >= 12

        for lev in [1, 3]:
            all_results[(ccy, lev, False)] = _run_carry_backtest(
                cr, front, empty, opts, leverage=lev, put_budget=0)
            if has_hedge:
                all_results[(ccy, lev, True)] = _run_carry_backtest(
                    cr, front, sels, opts, leverage=lev, put_budget=0.005)

    print(f'Total backtests: {len(all_results)}')

    # 3. Equal-weight portfolios
    print('\n--- Equal-Weight Portfolios ---')
    portfolio_configs = {
        'All-6 EW': CARRY_PAIRS,
        'Core (AUD+GBP+CAD)': ['AUD', 'GBP', 'CAD'],
        'High-Carry (AUD+MXN)': ['AUD', 'MXN'],
        'Diversified (AUD+EUR+CHF+MXN)': ['AUD', 'EUR', 'CHF', 'MXN'],
    }

    ew_results = {}
    for name, pairs_list in portfolio_configs.items():
        for lev in [1, 3]:
            for hedged in [False, True]:
                tag = f'{name} {lev}x {"H" if hedged else "U"}'
                cap = _build_equal_weight_portfolio(all_results, pairs_list, lev, hedged)
                if cap is not None and len(cap) > 252:
                    ew_results[tag] = cap

    print(f'Built {len(ew_results)} EW portfolios')

    # 4. Risk-parity portfolios
    print('\n--- Risk-Parity Portfolios ---')
    rp_results = {}
    for lev in [1, 3]:
        for hedged in [False, True]:
            tag = f'RP All-6 {lev}x {"H" if hedged else "U"}'
            cap, _ = _build_risk_parity_portfolio(all_results, CARRY_PAIRS, lev, hedged)
            if cap is not None and len(cap) > 252:
                rp_results[tag] = cap
    print(f'Built {len(rp_results)} RP portfolios')

    # 5. Min-variance portfolios
    print('\n--- Min-Variance Portfolios ---')
    mv_results = {}
    for lev in [1, 3]:
        for hedged in [False, True]:
            tag = f'MinVar All-6 {lev}x {"H" if hedged else "U"}'
            cap, _ = _build_min_variance_portfolio(all_results, CARRY_PAIRS, lev, hedged)
            if cap is not None and len(cap) > 252:
                mv_results[tag] = cap
    print(f'Built {len(mv_results)} MinVar portfolios')

    # 6. Max-Sharpe portfolios
    print('\n--- Max-Sharpe Portfolios ---')
    ms_results = {}
    for lev in [1, 3]:
        for hedged in [False, True]:
            tag = f'MaxSharpe All-6 {lev}x {"H" if hedged else "U"}'
            cap, _ = _build_max_sharpe_portfolio(all_results, CARRY_PAIRS, lev, hedged)
            if cap is not None and len(cap) > 252:
                ms_results[tag] = cap
    print(f'Built {len(ms_results)} MaxSharpe portfolios')

    # 7. Combined results table
    all_portfolios = {}
    all_portfolios.update(ew_results)
    all_portfolios.update(rp_results)
    all_portfolios.update(mv_results)
    all_portfolios.update(ms_results)

    rows = []
    for tag in sorted(all_portfolios.keys()):
        s = compute_stats(all_portfolios[tag])
        if s:
            s['Portfolio'] = tag
            rows.append(s)

    stats_df = pd.DataFrame(rows).set_index('Portfolio')
    stats_df = stats_df.sort_values('Sharpe', ascending=False)

    print(f'\n{"=" * 110}')
    print('ALL PORTFOLIO VARIANTS (sorted by Sharpe)')
    print(f'{"=" * 110}')
    print(f'{"Portfolio":>38} {"CAGR":>8} {"Vol":>6} {"Sharpe":>7} '
          f'{"MaxDD":>7} {"Total":>7}')
    print('-' * 110)
    for idx, row in stats_df.iterrows():
        print(f'{idx:>38} {row["CAGR"]*100:>7.2f}% {row["Vol"]*100:>5.1f}% '
              f'{row["Sharpe"]:>7.3f} {row["MaxDD"]*100:>6.1f}% {row["Total"]:>6.1f}x')

    print('\n--- TOP 5 BY SHARPE ---')
    for i, (idx, row) in enumerate(stats_df.head(5).iterrows()):
        print(f'  {i+1}. {idx}: Sharpe {row["Sharpe"]:.3f}, '
              f'CAGR {row["CAGR"]*100:.2f}%, MaxDD {row["MaxDD"]*100:.1f}%')

    # 8. Equity curves chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    colors_ew = {'All-6 EW': 'blue', 'Core (AUD+GBP+CAD)': 'red',
                 'High-Carry (AUD+MXN)': 'green', 'Diversified (AUD+EUR+CHF+MXN)': 'purple'}

    for name in portfolio_configs:
        for lev, ax in [(1, ax1), (3, ax2)]:
            for hedged, ls in [('U', '--'), ('H', '-')]:
                tag = f'{name} {lev}x {hedged}'
                if tag in ew_results:
                    cap = ew_results[tag] / 100
                    ax.plot(cap.index, cap, color=colors_ew[name], linestyle=ls,
                            alpha=0.8 if hedged == 'H' else 0.4, label=tag, linewidth=1.5)

    for ax, lev in [(ax1, '1x'), (ax2, '3x')]:
        ax.set_title(f'EW Carry Portfolios - {lev}')
        ax.set_ylabel('Portfolio Value ($1 start)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'pf_equity_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved pf_equity_curves.png')

    # 9. Correlation analysis chart
    ret_1x_u = _get_daily_returns(all_results, CARRY_PAIRS, 1, False)
    ret_1x_h = _get_daily_returns(all_results, CARRY_PAIRS, 1, True)
    corr_u = ret_1x_u.corr()
    corr_h = ret_1x_h.corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, corr, title in [(ax1, corr_u, '1x Unhedged'), (ax2, corr_h, '1x Hedged')]:
        im = ax.imshow(corr.values, cmap='RdYlGn_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.index)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
        ax.set_title(title)
    plt.colorbar(im, ax=[ax1, ax2], shrink=0.8)
    plt.suptitle('Carry Pair Return Correlations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'pf_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved pf_correlations.png')


# ═══════════════════════════════════════════════════════════════════════════════
#  Function 4: Leverage Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_leverage(data_dir):
    """Kelly-optimal leverage, blow-up frontier, put budget sensitivity."""

    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11,
    })

    # 1. Load data
    print('\n--- Loading data ---')
    fx_futures = _load_all_futures(data_dir)
    cross_rates = _build_cross_rates(fx_futures, ['AUD', 'GBP', 'CAD', 'EUR', 'CHF', 'MXN', 'NZD'])
    _attach_carry(cross_rates)

    fx_options = _load_all_options(data_dir, ANALYSIS_PAIRS)
    put_selections = _select_all_puts(fx_options, fx_futures, ANALYSIS_PAIRS)

    # 2. Pre-compute settlements
    print('\n--- Pre-computing settlements ---')
    precomputed_puts = {}
    for ccy in ANALYSIS_PAIRS:
        sels = put_selections.get(ccy, _empty_sels())
        opts = fx_options.get(ccy, pd.DataFrame())
        front = pd.DataFrame({'close': fx_futures[ccy]['close']})
        if len(sels) >= 12 and len(opts) > 0:
            lookup = build_settlement_lookup(opts)
            precomputed_puts[ccy] = precompute_settlements(sels, lookup, front)
            print(f'{ccy}: pre-computed {len(precomputed_puts[ccy])} put settlements')
        else:
            print(f'{ccy}: insufficient option data for hedging')

    # 3. Leverage sweep
    print('\n--- Leverage sweep ---')
    all_results = {}
    total = len(ANALYSIS_PAIRS) * len(LEVERAGE_LEVELS) * 2
    count = 0

    for ccy in ANALYSIS_PAIRS:
        cr = cross_rates[ccy]
        put_map = precomputed_puts.get(ccy, {})
        has_hedge = ccy in precomputed_puts

        for lev in LEVERAGE_LEVELS:
            count += 1
            print(f'[{count}/{total}] {ccy}/JPY {lev}x unhedged...')
            all_results[(ccy, lev, False)] = _run_carry_backtest_fast(
                cr, {}, leverage=lev, put_budget=0)

            count += 1
            if has_hedge:
                print(f'[{count}/{total}] {ccy}/JPY {lev}x hedged...')
                all_results[(ccy, lev, True)] = _run_carry_backtest_fast(
                    cr, put_map, leverage=lev, put_budget=0.005)

    print(f'Done! {len(all_results)} backtests completed.')

    # 4. Collect stats
    stats_rows = []
    for (ccy, lev, hedged), df in all_results.items():
        cap = df['capital']
        s = compute_stats(cap)
        if s is None:
            s = {'CAGR': np.nan, 'Vol': np.nan, 'Sharpe': np.nan,
                 'Sortino': np.nan, 'Calmar': np.nan, 'MaxDD': np.nan,
                 'Skew': np.nan, 'Kurt': np.nan, 'Total': np.nan}
        blew_up = (cap <= 0).any()
        stats_rows.append({
            'pair': f'{ccy}/JPY', 'ccy': ccy, 'leverage': lev,
            'hedged': hedged, 'blew_up': blew_up, **s,
        })
    stats_df = pd.DataFrame(stats_rows)

    # 5. Print per-pair leverage sweep
    for ccy in ANALYSIS_PAIRS:
        print(f'\n{"=" * 100}')
        print(f'{ccy}/JPY LEVERAGE SWEEP')
        print(f'{"=" * 100}')
        print(f'{"Lev":>4} {"Type":>10} {"CAGR":>8} {"Vol":>7} {"Sharpe":>7} '
              f'{"MaxDD":>7} {"Total":>8} {"Blowup":>7}')
        print('-' * 100)
        sub = stats_df[stats_df['ccy'] == ccy].sort_values(['leverage', 'hedged'])
        for _, row in sub.iterrows():
            htype = 'hedged' if row['hedged'] else 'unhedged'
            blowup = 'YES' if row['blew_up'] else 'no'
            if pd.isna(row['CAGR']):
                print(f'{row["leverage"]:>4}x {htype:>10} {"BLEW UP":>8}')
            else:
                print(f'{row["leverage"]:>4}x {htype:>10} {row["CAGR"]*100:>7.2f}% '
                      f'{row["Vol"]*100:>6.1f}% {row["Sharpe"]:>7.3f} '
                      f'{row["MaxDD"]*100:>6.1f}% {row["Total"]:>7.1f}x {blowup:>7}')

    # 6. Kelly-optimal leverage
    kelly_results = []
    for ccy in ANALYSIS_PAIRS:
        for hedged in [False, True]:
            levs = []
            cagrs = []
            for lev in LEVERAGE_LEVELS:
                sub = stats_df[(stats_df['ccy'] == ccy) &
                               (stats_df['leverage'] == lev) &
                               (stats_df['hedged'] == hedged)]
                if len(sub) == 0 or pd.isna(sub.iloc[0]['CAGR']):
                    continue
                levs.append(lev)
                cagrs.append(sub.iloc[0]['CAGR'] * 100)
            if levs:
                best_idx = np.argmax(cagrs)
                kelly_results.append({
                    'ccy': ccy, 'hedged': hedged,
                    'kelly_lev': levs[best_idx], 'max_cagr': cagrs[best_idx],
                })

    kelly_df = pd.DataFrame(kelly_results)
    print(f'\n{"=" * 60}')
    print('KELLY-OPTIMAL LEVERAGE')
    print(f'{"=" * 60}')
    print(f'{"Pair":>10} {"Type":>10} {"Opt Lev":>8} {"Max CAGR":>10}')
    print('-' * 60)
    for _, row in kelly_df.sort_values(['ccy', 'hedged']).iterrows():
        htype = 'hedged' if row['hedged'] else 'unhedged'
        print(f'{row["ccy"]+"/JPY":>10} {htype:>10} {row["kelly_lev"]:>7.0f}x '
              f'{row["max_cagr"]:>9.2f}%')

    # 7. CAGR vs leverage chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    for idx, ccy in enumerate(ANALYSIS_PAIRS):
        ax = axes_flat[idx]
        for hedged, ls, marker in [(False, '--', 'o'), (True, '-', 's')]:
            levs, cagrs = [], []
            for lev in LEVERAGE_LEVELS:
                sub = stats_df[(stats_df['ccy'] == ccy) &
                               (stats_df['leverage'] == lev) &
                               (stats_df['hedged'] == hedged)]
                if len(sub) == 0 or pd.isna(sub.iloc[0]['CAGR']):
                    continue
                levs.append(lev)
                cagrs.append(sub.iloc[0]['CAGR'] * 100)
            if levs:
                label = 'hedged' if hedged else 'unhedged'
                ax.plot(levs, cagrs, ls=ls, marker=marker, color=COLORS[ccy],
                        alpha=1.0 if hedged else 0.5, label=label, linewidth=2)
                best_idx = np.argmax(cagrs)
                ax.plot(levs[best_idx], cagrs[best_idx], '*', color=COLORS[ccy],
                        markersize=14, zorder=5)
        ax.set_title(f'{ccy}/JPY', fontsize=13, fontweight='bold')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('CAGR (%)')
        ax.legend(fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5)
    axes_flat[-1].set_visible(False)
    fig.suptitle('CAGR vs Leverage: Hedged vs Unhedged', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'lv_cagr_vs_leverage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved lv_cagr_vs_leverage.png')

    # 8. Sharpe heatmap
    fig, axes_hm = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (hedged, title) in enumerate([(False, 'Unhedged'), (True, 'Hedged')]):
        ax = axes_hm[ax_idx]
        matrix = np.full((len(ANALYSIS_PAIRS), len(LEVERAGE_LEVELS)), np.nan)
        for i, ccy in enumerate(ANALYSIS_PAIRS):
            for j, lev in enumerate(LEVERAGE_LEVELS):
                sub = stats_df[(stats_df['ccy'] == ccy) &
                               (stats_df['leverage'] == lev) &
                               (stats_df['hedged'] == hedged)]
                if len(sub) > 0 and not pd.isna(sub.iloc[0]['Sharpe']):
                    matrix[i, j] = sub.iloc[0]['Sharpe']
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(LEVERAGE_LEVELS)))
        ax.set_xticklabels([f'{l}x' for l in LEVERAGE_LEVELS])
        ax.set_yticks(range(len(ANALYSIS_PAIRS)))
        ax.set_yticklabels([f'{c}/JPY' for c in ANALYSIS_PAIRS])
        ax.set_xlabel('Leverage')
        ax.set_title(title, fontsize=13, fontweight='bold')
        for i in range(len(ANALYSIS_PAIRS)):
            for j in range(len(LEVERAGE_LEVELS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=color)
        plt.colorbar(im, ax=ax, label='Sharpe Ratio')
    fig.suptitle('Sharpe Ratio Heat Map', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'lv_sharpe_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved lv_sharpe_heatmap.png')

    # 9. Portfolio leverage
    port_results = {}
    port_stats = []
    for lev in LEVERAGE_LEVELS:
        for hedged in [False, True]:
            cap = _build_equal_weight_portfolio(all_results, ANALYSIS_PAIRS, lev, hedged)
            if cap is not None and len(cap) > 0:
                tag = f'{lev}x {"hedged" if hedged else "unhedged"}'
                port_results[tag] = cap
                s = compute_stats(cap)
                if s is not None:
                    port_stats.append({'leverage': lev, 'hedged': hedged, 'tag': tag, **s})

    port_stats_df = pd.DataFrame(port_stats)
    print(f'\n{"=" * 100}')
    print('EW PORTFOLIO ACROSS ALL CARRY PAIRS')
    print(f'{"=" * 100}')
    print(f'{"Portfolio":>22} {"CAGR":>8} {"Vol":>7} {"Sharpe":>7} {"MaxDD":>7} {"Total":>8}')
    print('-' * 100)
    for _, row in port_stats_df.sort_values(['leverage', 'hedged']).iterrows():
        if pd.isna(row['CAGR']):
            print(f'{row["tag"]:>22} BLEW UP')
        else:
            print(f'{row["tag"]:>22} {row["CAGR"]*100:>7.2f}% {row["Vol"]*100:>6.1f}% '
                  f'{row["Sharpe"]:>7.3f} {row["MaxDD"]*100:>6.1f}% {row["Total"]:>7.1f}x')

    # 10. Portfolio equity curves
    fig, axes_p = plt.subplots(1, 2, figsize=(18, 7))
    cmap = plt.cm.viridis
    lev_colors = {1: cmap(0.0), 2: cmap(0.2), 3: cmap(0.4),
                  5: cmap(0.6), 7: cmap(0.8), 10: cmap(1.0)}
    for ax_idx, hedged in enumerate([False, True]):
        ax = axes_p[ax_idx]
        for lev in LEVERAGE_LEVELS:
            tag = f'{lev}x {"hedged" if hedged else "unhedged"}'
            if tag in port_results:
                cap = port_results[tag] / 100
                ax.plot(cap.index, cap, color=lev_colors[lev], linewidth=2, label=f'{lev}x')
        ax.set_yscale('log')
        ax.set_title(f'Portfolio: {"Hedged" if hedged else "Unhedged"}',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Growth of $1 (log)')
        ax.legend(title='Leverage')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'${y:.1f}'))
    fig.suptitle('EW Portfolio Equity Curves at Various Leverage',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'lv_portfolio_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved lv_portfolio_curves.png')

    # 11. Put budget sensitivity
    print('\n--- Put budget sensitivity ---')
    PUT_BUDGETS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
    BUDGET_PAIRS = ['AUD', 'MXN']
    BUDGET_LEVERAGE = 3

    budget_results = {}
    for ccy in BUDGET_PAIRS:
        cr = cross_rates[ccy]
        put_map = precomputed_puts.get(ccy, {})
        if not put_map:
            continue

        print(f'{ccy}/JPY at {BUDGET_LEVERAGE}x:')
        bt = _run_carry_backtest_fast(cr, {}, leverage=BUDGET_LEVERAGE, put_budget=0)
        s = compute_stats(bt['capital'])
        if s:
            budget_results[(ccy, 0)] = s
            print(f'  budget=0.0% (unhedged): CAGR={s["CAGR"]*100:.2f}%, Sharpe={s["Sharpe"]:.3f}')

        for budget in PUT_BUDGETS:
            bt = _run_carry_backtest_fast(cr, put_map, leverage=BUDGET_LEVERAGE, put_budget=budget)
            s = compute_stats(bt['capital'])
            if s:
                budget_results[(ccy, budget)] = s
                print(f'  budget={budget*100:.1f}%: CAGR={s["CAGR"]*100:.2f}%, Sharpe={s["Sharpe"]:.3f}')

    # Put budget chart
    fig, axes_b = plt.subplots(1, 3, figsize=(18, 6))
    for ccy in BUDGET_PAIRS:
        budgets = [0] + PUT_BUDGETS
        cagrs, sharpes, max_dds = [], [], []
        for b in budgets:
            s = budget_results.get((ccy, b))
            if s is None:
                cagrs.append(np.nan); sharpes.append(np.nan); max_dds.append(np.nan)
            else:
                cagrs.append(s['CAGR'] * 100)
                sharpes.append(s['Sharpe'])
                max_dds.append(abs(s['MaxDD']) * 100)

        budget_pcts = [b * 100 for b in budgets]
        color = COLORS[ccy]
        axes_b[0].plot(budget_pcts, cagrs, '-o', color=color, linewidth=2, label=f'{ccy}/JPY')
        axes_b[1].plot(budget_pcts, sharpes, '-o', color=color, linewidth=2, label=f'{ccy}/JPY')
        axes_b[2].plot(budget_pcts, max_dds, '-o', color=color, linewidth=2, label=f'{ccy}/JPY')

        valid_sharpes = [(b, s) for b, s in zip(budget_pcts, sharpes) if not np.isnan(s)]
        if valid_sharpes:
            best_b, best_s = max(valid_sharpes, key=lambda x: x[1])
            axes_b[1].plot(best_b, best_s, '*', color=color, markersize=14, zorder=5)
            print(f'{ccy}/JPY: optimal Sharpe at budget={best_b:.1f}% (Sharpe={best_s:.3f})')

    axes_b[0].set_title(f'CAGR vs Put Budget ({BUDGET_LEVERAGE}x)', fontweight='bold')
    axes_b[0].set_xlabel('Monthly Put Budget (%)'); axes_b[0].set_ylabel('CAGR (%)')
    axes_b[0].legend()
    axes_b[1].set_title(f'Sharpe vs Put Budget ({BUDGET_LEVERAGE}x)', fontweight='bold')
    axes_b[1].set_xlabel('Monthly Put Budget (%)'); axes_b[1].set_ylabel('Sharpe')
    axes_b[1].legend()
    axes_b[2].set_title(f'Max DD vs Put Budget ({BUDGET_LEVERAGE}x)', fontweight='bold')
    axes_b[2].set_xlabel('Monthly Put Budget (%)'); axes_b[2].set_ylabel('Max DD (%)')
    axes_b[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'lv_put_budget.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved lv_put_budget.png')


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FX carry + tail hedge analysis')
    parser.add_argument('--analysis', default='all',
                        choices=['fx_carry_real', 'multi_asset', 'portfolio', 'leverage', 'all'])
    args = parser.parse_args()

    if args.analysis in ('fx_carry_real', 'all'):
        print('\n' + '='*80 + '\n  FX Carry Real: AUD/JPY + Dual Hedge\n' + '='*80)
        run_fx_carry_real(DATA_DIR)
    if args.analysis in ('multi_asset', 'all'):
        print('\n' + '='*80 + '\n  Multi-Asset FX Carry\n' + '='*80)
        run_multi_asset(DATA_DIR)
    if args.analysis in ('portfolio', 'all'):
        print('\n' + '='*80 + '\n  Portfolio Construction\n' + '='*80)
        run_portfolio(DATA_DIR)
    if args.analysis in ('leverage', 'all'):
        print('\n' + '='*80 + '\n  Leverage Analysis\n' + '='*80)
        run_leverage(DATA_DIR)


if __name__ == '__main__':
    main()
