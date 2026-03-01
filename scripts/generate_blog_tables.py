"""Generate the corrected AQR table for the blog post."""

import math
import warnings

import numpy as np

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


def run_aqr(options, stocks, schema, stock_pct, opt_pct, delta_min=-0.10, delta_max=-0.02):
    bt = BacktestEngine(
        {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = make_strategy(schema, delta_min, delta_max)
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
    print("Loading data...")
    options, stocks = load_data()
    schema = options.schema

    stk = stocks._data.sort_values("date")
    years = (stk["date"].iloc[-1] - stk["date"].iloc[0]).days / 365.25
    spy_ret = ((stk["adjClose"].iloc[-1] / stk["adjClose"].iloc[0]) ** (1 / years) - 1) * 100

    # ===================================================================
    # AQR table with deep OTM (blog's delta range)
    # ===================================================================
    print("\n### AQR framing with deep OTM puts (δ -0.10 to -0.02)")
    print()
    print("| Config | Annual Return | Excess vs SPY | Max Drawdown |")
    print("|--------|:------------:|:-------------:|:------------:|")
    print(f"| SPY only | +{spy_ret:.2f}% | | -51.9% |")

    for label, spct, opct in [
        ("99.9% SPY + 0.1% deep OTM", 0.999, 0.001),
        ("99.5% SPY + 0.5% deep OTM", 0.995, 0.005),
        ("99% SPY + 1% deep OTM", 0.99, 0.01),
        ("96.7% SPY + 3.3% deep OTM", 0.967, 0.033),
    ]:
        bt = run_aqr(options, stocks, schema, spct, opct)
        ar, dd, vol, sh = stats(bt.balance)
        excess = ar - spy_ret
        print(f"| {label} | {ar:+.2f}% | {excess:+.2f}% | {dd:.1f}% |")

    # ===================================================================
    # AQR table with near-ATM puts (what AQR actually tested)
    # ===================================================================
    print("\n### AQR framing with near-ATM puts (δ -0.40 to -0.25) — AQR's actual methodology")
    print()
    print("| Config | Annual Return | Excess vs SPY | Max Drawdown |")
    print("|--------|:------------:|:-------------:|:------------:|")
    print(f"| SPY only | +{spy_ret:.2f}% | | -51.9% |")

    for label, spct, opct in [
        ("99.9% SPY + 0.1% near-ATM", 0.999, 0.001),
        ("99.5% SPY + 0.5% near-ATM", 0.995, 0.005),
        ("99% SPY + 1% near-ATM", 0.99, 0.01),
        ("96.7% SPY + 3.3% near-ATM", 0.967, 0.033),
    ]:
        bt = run_aqr(options, stocks, schema, spct, opct, delta_min=-0.40, delta_max=-0.25)
        ar, dd, vol, sh = stats(bt.balance)
        excess = ar - spy_ret
        print(f"| {label} | {ar:+.2f}% | {excess:+.2f}% | {dd:.1f}% |")

    print(f"\nSPY baseline: {spy_ret:+.2f}%/yr")


if __name__ == "__main__":
    main()
