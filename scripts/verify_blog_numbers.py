"""Reproduce the blog post's Spitznagel numbers with the fixed backtester.

Blog parameters:
- DTE 90-180, delta (-0.10, -0.02), exit at DTE 14
- budget_fn = lambda date, tc: tc * budget_pct
- allocation: stocks=1.0, options=0.0, cash=0.0
- monthly rebalance (BMS)
- initial_capital = 1_000_000

Blog results (Spitznagel framing):
| Budget | Annual % | Excess % | Max DD % | Vol %  | Sharpe |
|--------|----------|----------|----------|--------|--------|
|  0.0%  |  11.05   |   0.00   |  -51.9   |  20.0  |  0.556 |
|  0.5%  |  16.02   |  +4.97   |  -47.1   |  17.8  |  0.901 |
|  1.0%  |  21.08   |  +10.03  |  -42.4   |  16.7  |  1.259 |
|  2.0%  |  31.73   |  +20.69  |  -32.0   |  17.7  |  1.790 |
|  3.3%  |  46.60   |  +35.55  |  -29.2   |  22.7  |  2.056 |

If the fix changed these numbers, the fix broke Spitznagel.
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "../options_backtester")
sys.path.insert(0, "../options_backtester/scripts")

from options_portfolio_backtester import (
    BacktestEngine,
    Direction,
    OptionType as Type,
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


def make_deep_otm_put_strategy(schema):
    """Exact same as blog's make_deep_otm_put_strategy with defaults."""
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 90)
        & (schema.dte <= 180)
        & (schema.delta >= -0.10)
        & (schema.delta <= -0.02)
    )
    leg.entry_sort = ("delta", False)  # deepest OTM first
    leg.exit_filter = schema.dte <= 14
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def run_spitznagel(options, stocks, schema, budget_pct):
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    _bp = budget_pct
    bt.options_budget = lambda date, tc, bp=_bp: tc * bp
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = make_deep_otm_put_strategy(schema)
    bt.run(rebalance_freq=1, rebalance_unit="BMS")
    return bt


def compute_stats(balance):
    bal = balance["total capital"]
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    total_ret = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = bal.cummax()
    dd = (bal - cummax) / cummax
    max_dd = dd.min() * 100
    daily_ret = bal.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = annual_ret / vol if vol > 0 else 0
    return {
        "annual": annual_ret,
        "max_dd": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "final": bal.iloc[-1],
    }


def main():
    print("Loading data...")
    options, stocks = load_data()
    schema = options.schema

    # SPY baseline
    stk = stocks._data.sort_values("date")
    prices = stk["adjClose"].values
    years = (stk["date"].iloc[-1] - stk["date"].iloc[0]).days / 365.25
    spy_ret = ((prices[-1] / prices[0]) ** (1 / years) - 1) * 100
    print(f"Date range: {stk['date'].iloc[0].date()} to {stk['date'].iloc[-1].date()} ({years:.1f} yr)")
    print(f"SPY B&H: {spy_ret:+.2f}%/yr")

    print(f"\n{'='*90}")
    print("  Reproducing blog post Spitznagel table")
    print(f"  Params: DTE 90-180, delta (-0.10, -0.02), exit DTE 14, monthly rebalance")
    print(f"{'='*90}")

    blog_results = {
        0.005: {"annual": 16.02, "max_dd": -47.1, "vol": 17.8, "sharpe": 0.901},
        0.01:  {"annual": 21.08, "max_dd": -42.4, "vol": 16.7, "sharpe": 1.259},
        0.02:  {"annual": 31.73, "max_dd": -32.0, "vol": 17.7, "sharpe": 1.790},
        0.033: {"annual": 46.60, "max_dd": -29.2, "vol": 22.7, "sharpe": 2.056},
    }

    print(f"\n  {'Budget':>8}  |  {'Blog':>8} {'Now':>8} {'Diff':>7}  |  "
          f"{'Blog DD':>8} {'Now DD':>8} {'Diff':>7}  |  "
          f"{'Blog Sh':>8} {'Now Sh':>8} {'Diff':>7}")
    print(f"  {'-'*8}  +  {'-'*27}  +  {'-'*27}  +  {'-'*27}")

    for bp in [0.005, 0.01, 0.02, 0.033]:
        bt = run_spitznagel(options, stocks, schema, bp)
        s = compute_stats(bt.balance)
        blog = blog_results[bp]

        ann_diff = s["annual"] - blog["annual"]
        dd_diff = s["max_dd"] - blog["max_dd"]
        sh_diff = s["sharpe"] - blog["sharpe"]

        flag = " " if abs(ann_diff) < 0.5 else " !!!"

        print(f"  {bp*100:>7.1f}%  |  {blog['annual']:>7.2f}% {s['annual']:>7.2f}% {ann_diff:>+6.2f}%  |  "
              f"{blog['max_dd']:>7.1f}% {s['max_dd']:>7.1f}% {dd_diff:>+6.1f}%  |  "
              f"{blog['sharpe']:>8.3f} {s['sharpe']:>8.3f} {sh_diff:>+6.3f}{flag}")

    # Also show the extended table with more budget levels
    print(f"\n{'='*90}")
    print("  Extended budget sweep (same params)")
    print(f"{'='*90}")
    print(f"  {'Budget':>8}  {'Annual%':>8}  {'Excess':>7}  {'MaxDD%':>7}  {'Vol%':>6}  {'Sharpe':>7}  {'Ret/1%Prem':>11}")
    print(f"  {'-'*75}")

    for bp in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.033]:
        bt = run_spitznagel(options, stocks, schema, bp)
        s = compute_stats(bt.balance)
        excess = s["annual"] - spy_ret
        ret_per_pct = excess / (bp * 100) if bp > 0 else 0
        print(f"  {bp*100:>7.2f}%  {s['annual']:>+7.2f}%  {excess:>+6.2f}%  "
              f"{s['max_dd']:>6.1f}%  {s['vol']:>5.1f}%  {s['sharpe']:>7.3f}  {ret_per_pct:>10.1f}x")

    print(f"\n  SPY baseline: {spy_ret:+.2f}%/yr")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
