"""Debug Spitznagel cash flow step by step.

Trace exactly what happens to cash/stocks/options at each rebalance
to verify the externally_funded fix is correct.
"""

import math
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


def run_and_trace(budget_pct, label):
    options = HistoricalOptionsData(OPTIONS_PATH)
    stocks = TiingoData(STOCKS_PATH)
    schema = options.schema

    leg = StrategyLeg("leg_1", schema, OptionType.PUT, Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= 14)
        & (schema.dte <= 60)
        & (schema.delta >= -0.10)
        & (schema.delta <= -0.02)
    )
    leg.entry_sort = ("delta", False)
    leg.exit_filter = schema.dte <= 7

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

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

    bal = bt.balance
    tl = bt.trade_log

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  Balance columns: {list(bal.columns)}")
    print(f"  Total rows: {len(bal)}")
    print(f"  Trades: {len(tl) if tl is not None and not tl.empty else 0}")

    # Check capital conservation: cash + stocks + options = total
    component_sum = bal["cash"] + bal["stocks capital"] + bal["options capital"]
    total = bal["total capital"]
    diff = (component_sum - total).abs()
    max_diff = diff.max()
    print(f"\n  Capital conservation check: max |components - total| = ${max_diff:.4f}")
    if max_diff > 1.0:
        print(f"  WARNING: components don't sum to total!")
        bad_rows = bal[diff > 1.0][["cash", "stocks capital", "options capital", "total capital"]].head(5)
        print(f"  First bad rows:\n{bad_rows}")

    # Check for leverage: total_capital should be > sum(stocks + cash) when puts are held
    # Because puts are funded externally, total = stocks + options + cash
    # where options came "from outside"

    # Trace first 12 months
    print(f"\n  Monthly trace (first 12 rebalance dates):")
    print(f"  {'Date':<12} {'Cash':>12} {'Stocks':>12} {'Options':>10} {'Total':>12} {'Sum':>12} {'Diff':>8}")

    # Get rebalance dates (monthly)
    rebal_dates = bal.index[bal.index.is_month_start | (bal.index == bal.index[0])]
    if len(rebal_dates) < 12:
        rebal_dates = bal.index[:12]
    else:
        rebal_dates = rebal_dates[:12]

    for dt in rebal_dates:
        if dt not in bal.index:
            continue
        row = bal.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        c = row["cash"]
        s = row["stocks capital"]
        o = row["options capital"]
        t = row["total capital"]
        sm = c + s + o
        d = sm - t
        print(f"  {str(dt.date()):<12} {c:>12,.0f} {s:>12,.0f} {o:>10,.0f} {t:>12,.0f} {sm:>12,.0f} {d:>+8,.0f}")

    # Final stats
    start_val = bal["total capital"].iloc[0]
    end_val = bal["total capital"].iloc[-1]
    years = (bal.index[-1] - bal.index[0]).days / 365.25
    total_ret = (end_val / start_val - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = bal["total capital"].cummax()
    max_dd = ((bal["total capital"] - cummax) / cummax).min() * 100

    print(f"\n  Start: ${start_val:,.0f}  End: ${end_val:,.0f}")
    print(f"  Annual return: {annual_ret:+.2f}%")
    print(f"  Max drawdown: {max_dd:.1f}%")
    print(f"  Years: {years:.1f}")

    # Check if cash ever goes negative (would mean we're spending more than we have)
    min_cash = bal["cash"].min()
    print(f"  Min cash: ${min_cash:,.2f}")
    if min_cash < -100:
        print(f"  WARNING: cash went significantly negative!")
        neg_rows = bal[bal["cash"] < -100][["cash", "stocks capital", "options capital", "total capital"]].head(5)
        print(f"  Negative cash rows:\n{neg_rows}")

    return bt


def main():
    print("Loading and running backtests...")
    print("(Using Python path — Rust not available in this venv)")

    # Run Spitznagel with 0.5% budget
    bt1 = run_and_trace(0.005, "Spitznagel 0.5% budget (externally funded)")

    # Run Spitznagel with higher budgets
    bt2 = run_and_trace(0.01, "Spitznagel 1.0% budget")
    bt3 = run_and_trace(0.02, "Spitznagel 2.0% budget")

    print(f"\n{'='*80}")
    print("  COMPARISON")
    print(f"{'='*80}")
    for label, bt in [
        ("0.5% budget", bt1),
        ("1.0% budget", bt2),
        ("2.0% budget", bt3),
    ]:
        bal = bt.balance["total capital"]
        years = (bal.index[-1] - bal.index[0]).days / 365.25
        tr = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
        ar = ((1 + tr / 100) ** (1 / years) - 1) * 100
        cummax = bal.cummax()
        dd = ((bal - cummax) / cummax).min() * 100
        dr = bal.pct_change().dropna()
        vol = dr.std() * np.sqrt(252) * 100
        sh = ar / vol if vol > 0 else 0
        print(f"  {label:<20} {ar:>+7.2f}%/yr  DD {dd:>6.1f}%  Vol {vol:>5.1f}%  Sharpe {sh:.3f}  Final ${bal.iloc[-1]:>12,.0f}")


if __name__ == "__main__":
    main()
