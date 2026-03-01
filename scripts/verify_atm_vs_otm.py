"""Verify: AQR's near-ATM puts lose, deep OTM works in both framings.

AQR tested 5% OTM puts (delta ~0.30-0.40).
Spitznagel uses deep OTM (delta -0.02 to -0.10).

We test both put types in both framings to confirm:
1. Near-ATM AQR: loses (matching AQR's findings)
2. Near-ATM Spitznagel: ???
3. Deep OTM AQR: beats SPY (new finding after fix)
4. Deep OTM Spitznagel: beats SPY (known)
"""

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


def make_strategy(schema, delta_min, delta_max, dte_min=90, dte_max=180, exit_dte=14):
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= dte_min)
        & (schema.dte <= dte_max)
        & (schema.delta >= delta_min)
        & (schema.delta <= delta_max)
    )
    leg.entry_sort = ("delta", False)
    leg.exit_filter = schema.dte <= exit_dte
    s = Strategy(schema)
    s.add_leg(leg)
    s.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)
    return s


def run_aqr(options, stocks, schema, stock_pct, opt_pct, delta_min, delta_max):
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


def run_spitznagel(options, stocks, schema, budget_pct, delta_min, delta_max):
    bt = BacktestEngine(
        {"stocks": 1.0, "options": 0.0, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    _bp = budget_pct
    bt.options_budget = lambda date, tc, bp=_bp: tc * bp
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

    print(f"SPY B&H: {spy_ret:+.2f}%/yr")
    print(f"All use DTE 90-180, exit DTE 14, monthly rebalance\n")

    # Near-ATM puts (AQR's actual parameters): delta -0.40 to -0.25
    # This approximates 5% OTM (95% moneyness)
    print("=" * 95)
    print("  NEAR-ATM PUTS (delta -0.40 to -0.25) — what AQR actually tested")
    print("=" * 95)
    print(f"  {'Strategy':<45} {'Annual%':>8} {'vs SPY':>7} {'MaxDD%':>7} {'Vol%':>6} {'Sharpe':>7}")
    print("-" * 95)

    for label, fn in [
        ("AQR 0.5% (99.5/0.5)",   lambda: run_aqr(options, stocks, schema, 0.995, 0.005, -0.40, -0.25)),
        ("AQR 1.0% (99/1)",       lambda: run_aqr(options, stocks, schema, 0.99, 0.01, -0.40, -0.25)),
        ("AQR 3.3% (96.7/3.3)",   lambda: run_aqr(options, stocks, schema, 0.967, 0.033, -0.40, -0.25)),
        ("Spitznagel 0.5%",        lambda: run_spitznagel(options, stocks, schema, 0.005, -0.40, -0.25)),
        ("Spitznagel 1.0%",        lambda: run_spitznagel(options, stocks, schema, 0.01, -0.40, -0.25)),
        ("Spitznagel 3.3%",        lambda: run_spitznagel(options, stocks, schema, 0.033, -0.40, -0.25)),
    ]:
        bt = fn()
        ar, dd, vol, sh = stats(bt.balance)
        excess = ar - spy_ret
        print(f"  {label:<45} {ar:>+7.2f}% {excess:>+6.2f}% {dd:>6.1f}% {vol:>5.1f}% {sh:>7.3f}")

    # Standard OTM puts (blog's "standard OTM"): delta -0.25 to -0.10
    print(f"\n{'=' * 95}")
    print("  STANDARD OTM PUTS (delta -0.25 to -0.10)")
    print("=" * 95)
    print(f"  {'Strategy':<45} {'Annual%':>8} {'vs SPY':>7} {'MaxDD%':>7} {'Vol%':>6} {'Sharpe':>7}")
    print("-" * 95)

    for label, fn in [
        ("AQR 0.5% (99.5/0.5)",   lambda: run_aqr(options, stocks, schema, 0.995, 0.005, -0.25, -0.10)),
        ("AQR 1.0% (99/1)",       lambda: run_aqr(options, stocks, schema, 0.99, 0.01, -0.25, -0.10)),
        ("AQR 3.3% (96.7/3.3)",   lambda: run_aqr(options, stocks, schema, 0.967, 0.033, -0.25, -0.10)),
        ("Spitznagel 0.5%",        lambda: run_spitznagel(options, stocks, schema, 0.005, -0.25, -0.10)),
        ("Spitznagel 1.0%",        lambda: run_spitznagel(options, stocks, schema, 0.01, -0.25, -0.10)),
        ("Spitznagel 3.3%",        lambda: run_spitznagel(options, stocks, schema, 0.033, -0.25, -0.10)),
    ]:
        bt = fn()
        ar, dd, vol, sh = stats(bt.balance)
        excess = ar - spy_ret
        print(f"  {label:<45} {ar:>+7.2f}% {excess:>+6.2f}% {dd:>6.1f}% {vol:>5.1f}% {sh:>7.3f}")

    # Deep OTM puts (Spitznagel's parameters): delta -0.10 to -0.02
    print(f"\n{'=' * 95}")
    print("  DEEP OTM PUTS (delta -0.10 to -0.02) — what Spitznagel uses")
    print("=" * 95)
    print(f"  {'Strategy':<45} {'Annual%':>8} {'vs SPY':>7} {'MaxDD%':>7} {'Vol%':>6} {'Sharpe':>7}")
    print("-" * 95)

    for label, fn in [
        ("AQR 0.5% (99.5/0.5)",   lambda: run_aqr(options, stocks, schema, 0.995, 0.005, -0.10, -0.02)),
        ("AQR 1.0% (99/1)",       lambda: run_aqr(options, stocks, schema, 0.99, 0.01, -0.10, -0.02)),
        ("AQR 3.3% (96.7/3.3)",   lambda: run_aqr(options, stocks, schema, 0.967, 0.033, -0.10, -0.02)),
        ("Spitznagel 0.5%",        lambda: run_spitznagel(options, stocks, schema, 0.005, -0.10, -0.02)),
        ("Spitznagel 1.0%",        lambda: run_spitznagel(options, stocks, schema, 0.01, -0.10, -0.02)),
        ("Spitznagel 3.3%",        lambda: run_spitznagel(options, stocks, schema, 0.033, -0.10, -0.02)),
    ]:
        bt = fn()
        ar, dd, vol, sh = stats(bt.balance)
        excess = ar - spy_ret
        print(f"  {label:<45} {ar:>+7.2f}% {excess:>+6.2f}% {dd:>6.1f}% {vol:>5.1f}% {sh:>7.3f}")

    print(f"\n  SPY baseline: {spy_ret:+.2f}%/yr")
    print("=" * 95)


if __name__ == "__main__":
    main()
