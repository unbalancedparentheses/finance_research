"""Reproduce blog's AQR numbers with the fixed backtester.

Blog params: DTE 90-180, delta (-0.10, -0.02), exit DTE 14, monthly rebalance.

Blog AQR results (pre-fix, had money-creation bug):
| Config                        | Annual  | Max DD  |
| 99.9% SPY + 0.1% deep OTM    | +10.70% | -51.8%  |
| 99.5% SPY + 0.5% deep OTM    | +9.23%  | -50.3%  |
| 99%   SPY + 1.0% deep OTM    | +7.38%  | -48.4%  |
| 96.7% SPY + 3.3% deep OTM    | -1.28%  | -39.6%  |
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


def make_deep_otm_put_strategy(schema):
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


def run_aqr(options, stocks, schema, stock_pct, opt_pct):
    bt = BacktestEngine(
        {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        initial_capital=INITIAL_CAPITAL,
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks
    bt.options_data = options
    bt.options_strategy = make_deep_otm_put_strategy(schema)
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
    return annual_ret, max_dd, vol, sharpe, bal.iloc[-1]


def main():
    print("Loading data...")
    options, stocks = load_data()
    schema = options.schema

    stk = stocks._data.sort_values("date")
    years = (stk["date"].iloc[-1] - stk["date"].iloc[0]).days / 365.25
    spy_ret = ((stk["adjClose"].iloc[-1] / stk["adjClose"].iloc[0]) ** (1 / years) - 1) * 100
    print(f"SPY B&H: {spy_ret:+.2f}%/yr\n")

    blog_aqr = {
        (0.999, 0.001): {"annual": 10.70, "max_dd": -51.8},
        (0.995, 0.005): {"annual": 9.23,  "max_dd": -50.3},
        (0.99,  0.01):  {"annual": 7.38,  "max_dd": -48.4},
        (0.967, 0.033): {"annual": -1.28, "max_dd": -39.6},
    }

    print(f"{'='*95}")
    print(f"  AQR framing (sell stocks to fund puts) — blog params: DTE 90-180, delta (-0.10,-0.02)")
    print(f"{'='*95}")
    print(f"  {'Config':<30}  {'Blog%':>7} {'Now%':>7} {'Diff':>7}  |  {'BlogDD':>7} {'NowDD':>7} {'Diff':>7}  |  {'Final':>14}")
    print(f"  {'-'*93}")

    for (spct, opct), blog in blog_aqr.items():
        bt = run_aqr(options, stocks, schema, spct, opct)
        ar, dd, vol, sh, final = stats(bt.balance)
        ann_diff = ar - blog["annual"]
        dd_diff = dd - blog["max_dd"]
        label = f"{spct*100:.1f}% SPY + {opct*100:.1f}% puts"
        print(f"  {label:<30}  {blog['annual']:>+6.2f}% {ar:>+6.2f}% {ann_diff:>+6.2f}%  |  "
              f"{blog['max_dd']:>6.1f}% {dd:>6.1f}% {dd_diff:>+6.1f}%  |  ${final:>12,.0f}")

    print(f"{'='*95}")
    print(f"\n  Note: Blog AQR numbers had the money-creation bug.")
    print(f"  The fix should make AQR returns WORSE (lower), since puts")
    print(f"  are now properly funded from stocks instead of created from thin air.")


if __name__ == "__main__":
    main()
