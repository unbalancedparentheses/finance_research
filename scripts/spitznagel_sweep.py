"""Sweep Spitznagel parameters to find what actually works.

Questions to answer:
1. Is 6%/yr in puts too much? What's the sweet spot?
2. How much of our result comes from starting in 2008?
3. What DTE/delta/exit params work best?
4. Do the results hold in non-crisis periods?
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from options_portfolio_backtester import (
    BacktestEngine,
    Direction,
    OptionType,
    Stock,
    Strategy,
    StrategyLeg,
)
from options_portfolio_backtester.data.providers import (
    HistoricalOptionsData,
    TiingoData,
)

OPTIONS_PATH = "../options_backtester/data/processed/options.csv"
STOCKS_PATH = "../options_backtester/data/processed/stocks.csv"

INITIAL_CAPITAL = 1_000_000


def load_data():
    options = HistoricalOptionsData(OPTIONS_PATH)
    stocks = TiingoData(STOCKS_PATH)
    return options, stocks


def run_spitznagel(options, stocks, budget_pct, delta_min=-0.10, delta_max=-0.02,
                   dte_min=14, dte_max=60, exit_dte=7, profit_pct=math.inf):
    schema = options.schema
    leg = StrategyLeg("leg_1", schema, OptionType.PUT, Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= dte_min)
        & (schema.dte <= dte_max)
        & (schema.delta >= delta_min)
        & (schema.delta <= delta_max)
    )
    leg.entry_sort = ("delta", False)  # deepest OTM first
    leg.exit_filter = schema.dte <= exit_dte

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=profit_pct, loss_pct=math.inf)

    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=int(INITIAL_CAPITAL),
    )
    _bp = budget_pct
    bt.options_budget = lambda date, tc, bp=_bp: tc * bp
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = strategy
    bt.run(rebalance_freq=1, rebalance_unit="BMS")

    return bt.balance["total capital"]


def run_unhedged(stocks):
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=int(INITIAL_CAPITAL),
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    # Need a dummy strategy and options data — run without options
    return None


def compute_stats(balance, years):
    total_ret = (balance.iloc[-1] / balance.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = balance.cummax()
    dd = (balance - cummax) / cummax
    max_dd = dd.min() * 100
    daily_ret = balance.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = annual_ret / vol if vol > 0 else 0
    return {
        "annual_ret": annual_ret,
        "total_ret": total_ret,
        "max_dd": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "final": balance.iloc[-1],
    }


def spy_stats(stocks, start_date=None, end_date=None):
    stk = stocks._data.sort_values("date")
    if start_date:
        stk = stk[stk["date"] >= start_date]
    if end_date:
        stk = stk[stk["date"] <= end_date]
    prices = stk["adjClose"].values
    years = (stk["date"].iloc[-1] - stk["date"].iloc[0]).days / 365.25
    total_ret = (prices[-1] / prices[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    return annual_ret, years


def main():
    print("Loading data...")
    options, stocks = load_data()

    stk_df = stocks._data.sort_values("date")
    full_start = stk_df["date"].iloc[0]
    full_end = stk_df["date"].iloc[-1]
    full_years = (full_end - full_start).days / 365.25

    spy_ret, _ = spy_stats(stocks)
    print(f"Full range: {full_start.date()} to {full_end.date()} ({full_years:.1f} yr)")
    print(f"SPY B&H: {spy_ret:+.2f}%/yr\n")

    # ===================================================================
    # SWEEP 1: Budget percentage (the key question)
    # ===================================================================
    print("=" * 90)
    print("  SWEEP 1: Budget % (Spitznagel framing, DTE 14-60, delta -0.10)")
    print("=" * 90)
    print(f"  {'Budget':>8}  {'Annual%':>8}  {'vs SPY':>7}  {'MaxDD%':>7}  {'Vol%':>6}  {'Sharpe':>7}  {'Final':>14}")
    print("-" * 90)

    for bp in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
        bal = run_spitznagel(options, stocks, bp)
        s = compute_stats(bal, full_years)
        print(f"  {bp*100:>7.1f}%  {s['annual_ret']:>+7.2f}%  {s['annual_ret']-spy_ret:>+6.2f}%  "
              f"{s['max_dd']:>6.1f}%  {s['vol']:>5.1f}%  {s['sharpe']:>7.3f}  ${s['final']:>12,.0f}")

    # ===================================================================
    # SWEEP 2: DTE range (short vs long-dated puts)
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("  SWEEP 2: DTE range (budget=0.5%, delta -0.10)")
    print("=" * 90)
    print(f"  {'DTE range':>12}  {'Exit':>4}  {'Annual%':>8}  {'vs SPY':>7}  {'MaxDD%':>7}  {'Sharpe':>7}")
    print("-" * 90)

    for dte_min, dte_max, exit_dte in [
        (14, 45, 7),
        (14, 60, 7),
        (30, 90, 14),
        (60, 120, 30),
        (90, 180, 30),
        (90, 180, 14),
    ]:
        bal = run_spitznagel(options, stocks, 0.005, dte_min=dte_min, dte_max=dte_max, exit_dte=exit_dte)
        s = compute_stats(bal, full_years)
        print(f"  {dte_min:>3}-{dte_max:<3}      {exit_dte:>4}  {s['annual_ret']:>+7.2f}%  "
              f"{s['annual_ret']-spy_ret:>+6.2f}%  {s['max_dd']:>6.1f}%  {s['sharpe']:>7.3f}")

    # ===================================================================
    # SWEEP 3: Delta (how far OTM)
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("  SWEEP 3: Delta target (budget=0.5%, DTE 14-60)")
    print("=" * 90)
    print(f"  {'Delta':>12}  {'Annual%':>8}  {'vs SPY':>7}  {'MaxDD%':>7}  {'Sharpe':>7}")
    print("-" * 90)

    for delta_min, delta_max in [
        (-0.05, -0.01),
        (-0.10, -0.02),
        (-0.15, -0.05),
        (-0.20, -0.10),
        (-0.25, -0.10),
        (-0.30, -0.15),
    ]:
        bal = run_spitznagel(options, stocks, 0.005, delta_min=delta_min, delta_max=delta_max)
        s = compute_stats(bal, full_years)
        print(f"  {delta_min:>+.2f}/{delta_max:>+.2f}  {s['annual_ret']:>+7.2f}%  "
              f"{s['annual_ret']-spy_ret:>+6.2f}%  {s['max_dd']:>6.1f}%  {s['sharpe']:>7.3f}")

    # ===================================================================
    # SWEEP 4: Start date sensitivity (is 2008 driving everything?)
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("  SWEEP 4: Start date sensitivity (budget=0.5%, DTE 14-60, delta -0.10)")
    print("=" * 90)
    print(f"  {'Start':>12}  {'Hedged%':>8}  {'SPY%':>7}  {'Delta':>7}  {'MaxDD':>7}  {'Sharpe':>7}")
    print("-" * 90)

    for start_year in ["2008-01-01", "2010-01-01", "2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01"]:
        # Filter data to start from this date
        start_dt = pd.Timestamp(start_year)
        opts_filtered = HistoricalOptionsData(OPTIONS_PATH)
        opts_filtered._data = opts_filtered._data[opts_filtered._data["quotedate"] >= start_dt].reset_index(drop=True)

        stks_filtered = TiingoData(STOCKS_PATH)
        stks_filtered._data = stks_filtered._data[stks_filtered._data["date"] >= start_dt].reset_index(drop=True)

        if len(opts_filtered._data) == 0 or len(stks_filtered._data) == 0:
            continue

        sub_years = (stks_filtered._data["date"].max() - stks_filtered._data["date"].min()).days / 365.25
        if sub_years < 1:
            continue

        spy_sub, _ = spy_stats(stks_filtered)

        try:
            bal = run_spitznagel(opts_filtered, stks_filtered, 0.005)
            s = compute_stats(bal, sub_years)
            print(f"  {start_year:>12}  {s['annual_ret']:>+7.2f}%  {spy_sub:>+6.2f}%  "
                  f"{s['annual_ret']-spy_sub:>+6.2f}%  {s['max_dd']:>6.1f}%  {s['sharpe']:>7.3f}")
        except Exception as e:
            print(f"  {start_year:>12}  ERROR: {e}")

    # ===================================================================
    # SWEEP 5: Blog post parameters (DTE 90-180, exit 14)
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("  SWEEP 5: Blog post params (DTE 90-180, exit 14, delta -0.10)")
    print("=" * 90)
    print(f"  {'Budget':>8}  {'Annual%':>8}  {'vs SPY':>7}  {'MaxDD%':>7}  {'Sharpe':>7}")
    print("-" * 90)

    for bp in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03]:
        bal = run_spitznagel(options, stocks, bp, dte_min=90, dte_max=180, exit_dte=14)
        s = compute_stats(bal, full_years)
        print(f"  {bp*100:>7.1f}%  {s['annual_ret']:>+7.2f}%  {s['annual_ret']-spy_ret:>+6.2f}%  "
              f"{s['max_dd']:>6.1f}%  {s['sharpe']:>7.3f}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
