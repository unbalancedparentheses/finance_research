"""Test tail hedge strategy during the calm 2010-2019 decade (no crash > -20%).

This is the honest stress test: 10 years of bull market where puts just bleed.
"""

import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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


def load_data(start="2010-01-01", end="2020-01-01"):
    options = HistoricalOptionsData(OPTIONS_PATH)
    stocks = TiingoData(STOCKS_PATH)

    # Filter to date range
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    options._data = options._data[
        (options._data["quotedate"] >= start_ts)
        & (options._data["quotedate"] < end_ts)
    ].copy()
    options.start_date = options._data["quotedate"].min()
    options.end_date = options._data["quotedate"].max()

    stocks._data = stocks._data[
        (stocks._data["date"] >= start_ts) & (stocks._data["date"] < end_ts)
    ].copy()
    stocks.start_date = stocks._data["date"].min()
    stocks.end_date = stocks._data["date"].max()

    return options, stocks


def make_strategy(schema, delta_min=-0.10, delta_max=-0.02):
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 90)
        & (schema.dte <= 180)
        & (schema.delta >= delta_min)
        & (schema.delta <= delta_max)
    )
    leg.entry_sort = ("delta", False)
    leg.exit_filter = schema.dte <= 14
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def run_no_leverage(options, stocks, schema, stock_pct, opt_pct):
    bt = BacktestEngine(
        {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = make_strategy(schema)
    bt.run(rebalance_freq=1, rebalance_unit="BMS")
    return bt


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
    bt.options_strategy = make_strategy(schema)
    bt.run(rebalance_freq=1, rebalance_unit="BMS")
    return bt


def stats(balance):
    bal = balance["total capital"]
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    total_ret = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = bal.cummax()
    max_dd = ((bal - cummax) / cummax).min() * 100
    daily_ret = bal.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = annual_ret / vol if vol > 0 else 0
    return annual_ret, max_dd, vol, sharpe


def main():
    periods = [
        ("2010-2020 (calm decade)", "2010-01-01", "2020-01-01"),
        ("2012-2018 (calmest stretch)", "2012-01-01", "2018-01-01"),
        ("2014-2020 (pre-COVID)", "2014-01-01", "2020-01-01"),
        ("2008-2025 (full sample)", "2008-01-01", "2026-01-01"),
    ]

    for period_name, start, end in periods:
        print(f"\n{'=' * 95}")
        print(f"  {period_name}")
        print(f"{'=' * 95}")

        options, stocks = load_data(start, end)
        schema = options.schema

        stk = stocks._data.sort_values("date")
        years = (stk["date"].iloc[-1] - stk["date"].iloc[0]).days / 365.25
        spy_ret = (
            (stk["adjClose"].iloc[-1] / stk["adjClose"].iloc[0]) ** (1 / years) - 1
        ) * 100
        spy_cummax = stk["adjClose"].cummax()
        spy_dd = ((stk["adjClose"] - spy_cummax) / spy_cummax).min() * 100

        print(f"  SPY B&H: {spy_ret:+.2f}%/yr, Max DD: {spy_dd:.1f}%")
        print()

        # Spitznagel framing
        print(
            f"  {'Strategy':<40} {'Annual%':>8} {'vs SPY':>7} {'MaxDD%':>7} {'Vol%':>6} {'Sharpe':>7}"
        )
        print("-" * 95)

        for label, fn in [
            (
                "Spitznagel 0.5%",
                lambda: run_spitznagel(options, stocks, schema, 0.005),
            ),
            (
                "Spitznagel 1.0%",
                lambda: run_spitznagel(options, stocks, schema, 0.01),
            ),
            (
                "Spitznagel 3.3%",
                lambda: run_spitznagel(options, stocks, schema, 0.033),
            ),
            (
                "No-leverage 0.5% (99.5/0.5)",
                lambda: run_no_leverage(options, stocks, schema, 0.995, 0.005),
            ),
            (
                "No-leverage 1.0% (99/1)",
                lambda: run_no_leverage(options, stocks, schema, 0.99, 0.01),
            ),
            (
                "No-leverage 3.3% (96.7/3.3)",
                lambda: run_no_leverage(options, stocks, schema, 0.967, 0.033),
            ),
        ]:
            bt = fn()
            ar, dd, vol, sh = stats(bt.balance)
            excess = ar - spy_ret
            print(
                f"  {label:<40} {ar:>+7.2f}% {excess:>+6.2f}% {dd:>6.1f}% {vol:>5.1f}% {sh:>7.3f}"
            )

    print()


if __name__ == "__main__":
    main()
