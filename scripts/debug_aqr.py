"""Debug AQR cash flow to understand why returns are so high after fix."""

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


def load_data():
    return HistoricalOptionsData(OPTIONS_PATH), TiingoData(STOCKS_PATH)


def make_strategy(schema):
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 90)
        & (schema.dte <= 180)
        & (schema.delta >= -0.10)
        & (schema.delta <= -0.02)
    )
    leg.entry_sort = ("delta", False)
    leg.exit_filter = schema.dte <= 14
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def run_and_trace(options, stocks, schema, stock_pct, opt_pct, label):
    bt = BacktestEngine(
        {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = make_strategy(schema)
    bt.run(rebalance_freq=1, rebalance_unit="BMS")

    bal = bt.balance
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"  Allocation: stocks={stock_pct}, options={opt_pct}")
    print(f"{'='*100}")

    # Check capital conservation
    component_sum = bal["cash"] + bal["stocks capital"] + bal["options capital"]
    total = bal["total capital"]
    max_diff = (component_sum - total).abs().max()
    print(f"  Capital conservation: max |components - total| = ${max_diff:.4f}")

    # Trace first 24 months
    print(f"\n  {'Date':<12} {'Cash':>10} {'Stocks':>12} {'Options':>10} {'Total':>12} {'Stk%':>6} {'Opt%':>6} {'Cash%':>6}")
    count = 0
    prev_month = None
    for dt in bal.index:
        month = dt.to_period('M')
        if month == prev_month:
            continue
        prev_month = month
        row = bal.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        c = row["cash"]
        s = row["stocks capital"]
        o = row["options capital"]
        t = row["total capital"]
        sp = s / t * 100 if t > 0 else 0
        op = o / t * 100 if t > 0 else 0
        cp = c / t * 100 if t > 0 else 0
        print(f"  {str(dt.date()):<12} {c:>10,.0f} {s:>12,.0f} {o:>10,.0f} {t:>12,.0f} {sp:>5.1f}% {op:>5.1f}% {cp:>5.1f}%")
        count += 1
        if count >= 24:
            break

    # Final stats
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    final = bal["total capital"].iloc[-1]
    tr = (final / bal["total capital"].iloc[0] - 1) * 100
    ar = ((1 + tr / 100) ** (1 / years) - 1) * 100
    cummax = bal["total capital"].cummax()
    dd = ((bal["total capital"] - cummax) / cummax).min() * 100
    print(f"\n  Annual: {ar:+.2f}%, MaxDD: {dd:.1f}%, Final: ${final:,.0f}")

    # Count months with options vs without
    monthly = bal.resample("MS").first()
    with_opts = (monthly["options capital"] > 0).sum()
    without_opts = (monthly["options capital"] == 0).sum()
    print(f"  Months with options: {with_opts}, without: {without_opts}")

    return bt


def main():
    print("Loading data...")
    options, stocks = load_data()
    schema = options.schema

    # AQR 3.3% — the most extreme case
    run_and_trace(options, stocks, schema, 0.967, 0.033, "AQR 3.3% (96.7% SPY + 3.3% puts)")

    # AQR 0.5% — the standard case
    run_and_trace(options, stocks, schema, 0.995, 0.005, "AQR 0.5% (99.5% SPY + 0.5% puts)")

    # For comparison: Spitznagel 0.5%
    bt_spit = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt_spit.options_budget = lambda date, tc: tc * 0.005
    bt_spit.stocks = [Stock("SPY", 1.0)]
    bt_spit.stocks_data = stocks
    bt_spit.options_data = options
    bt_spit.options_strategy = make_strategy(schema)
    bt_spit.run(rebalance_freq=1, rebalance_unit="BMS")
    bal = bt_spit.balance
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    final = bal["total capital"].iloc[-1]
    tr = (final / bal["total capital"].iloc[0] - 1) * 100
    ar = ((1 + tr / 100) ** (1 / years) - 1) * 100
    print(f"\n  Spitznagel 0.5% for reference: {ar:+.2f}%/yr, Final: ${final:,.0f}")


if __name__ == "__main__":
    main()
