"""Compare convexity scanner vs options_portfolio_backtester side by side.

Runs identical data through both engines and logs per-month details
to find where the numbers diverge.
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------
OPTIONS_PATH = "../options_backtester/data/processed/options.csv"
STOCKS_PATH = "../options_backtester/data/processed/stocks.csv"

INITIAL_CAPITAL = 1_000_000.0
BUDGET_PCT = 0.005  # 0.5% — but means different things in each engine!
TARGET_DELTA = -0.10
DTE_MIN = 14
DTE_MAX = 60
TAIL_DROP = 0.20


def load_data():
    from options_portfolio_backtester.data.providers import (
        HistoricalOptionsData,
        TiingoData,
    )

    options_data = HistoricalOptionsData(OPTIONS_PATH)
    stocks_data = TiingoData(STOCKS_PATH)
    return options_data, stocks_data


# ---------------------------------------------------------------------------
# Strategy 1: Our Rust scanner
# ---------------------------------------------------------------------------
def run_our_scanner(options_data, stocks_data):
    from options_portfolio_backtester._ob_rust import run_convexity_backtest as rust_backtest
    from options_portfolio_backtester.convexity.backtest import _to_ns

    opt_df = options_data._data
    puts = opt_df[opt_df["type"] == "put"].sort_values("quotedate")
    stk_df = stocks_data._data.sort_values("date")

    result = rust_backtest(
        put_dates_ns=_to_ns(puts["quotedate"]),
        put_expirations_ns=_to_ns(puts["expiration"]),
        put_strikes=puts["strike"].values.astype(np.float64),
        put_bids=puts["bid"].values.astype(np.float64),
        put_asks=puts["ask"].values.astype(np.float64),
        put_deltas=puts["delta"].values.astype(np.float64),
        put_underlying=puts["underlying_last"].values.astype(np.float64),
        put_dtes=puts["dte"].values.astype(np.int32),
        put_ivs=puts["impliedvol"].values.astype(np.float64),
        stock_dates_ns=_to_ns(stk_df["date"]),
        stock_prices=stk_df["adjClose"].values.astype(np.float64),
        initial_capital=INITIAL_CAPITAL,
        budget_pct=BUDGET_PCT,
        target_delta=TARGET_DELTA,
        dte_min=DTE_MIN,
        dte_max=DTE_MAX,
        tail_drop=TAIL_DROP,
    )

    rec = result["records"]
    records = pd.DataFrame(
        {
            "date": pd.to_datetime(rec["dates_ns"], unit="ns"),
            "shares": rec["shares"],
            "stock_price": rec["stock_prices"],
            "equity_value": rec["equity_values"],
            "put_cost": rec["put_costs"],
            "put_exit_value": rec["put_exit_values"],
            "put_pnl": rec["put_pnls"],
            "portfolio_value": rec["portfolio_values"],
            "strike": rec["strikes"],
            "contracts": rec["contracts"],
        }
    ).set_index("date")

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(result["daily_dates_ns"], unit="ns"),
            "balance": result["daily_balances"],
        }
    ).set_index("date")

    return records, daily


# ---------------------------------------------------------------------------
# Strategy 2: options_portfolio_backtester — AQR framing
# ---------------------------------------------------------------------------
def run_backtester_aqr(options_data, stocks_data, stock_pct, opt_pct):
    from options_portfolio_backtester import (
        BacktestEngine,
        Direction,
        OptionType,
        Stock,
        Strategy,
        StrategyLeg,
    )

    schema = options_data.schema
    leg = StrategyLeg("leg_1", schema, OptionType.PUT, Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= DTE_MIN)
        & (schema.dte <= DTE_MAX)
        & (schema.delta >= TARGET_DELTA)
        & (schema.delta <= -0.02)
    )
    leg.entry_sort = ("delta", False)  # deepest OTM first
    leg.exit_filter = schema.dte <= 7

    strategy = Strategy(schema)
    strategy.add_leg(leg)
    strategy.add_exit_thresholds(profit_pct=math.inf, loss_pct=math.inf)

    bt = BacktestEngine(
        {"stocks": stock_pct, "options": opt_pct, "cash": 0.0},
        initial_capital=int(INITIAL_CAPITAL),
    )
    bt.stocks = [Stock("SPY", 1.0)]
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=1, rebalance_unit="BMS")

    return bt.balance, bt.trade_log


# ---------------------------------------------------------------------------
# Strategy 3: options_portfolio_backtester — Spitznagel framing
# ---------------------------------------------------------------------------
def run_backtester_spitznagel(options_data, stocks_data, budget_pct):
    from options_portfolio_backtester import (
        BacktestEngine,
        Direction,
        OptionType,
        Stock,
        Strategy,
        StrategyLeg,
    )

    schema = options_data.schema
    leg = StrategyLeg("leg_1", schema, OptionType.PUT, Direction.BUY)
    leg.entry_filter = (
        (schema.underlying == "SPY")
        & (schema.dte >= DTE_MIN)
        & (schema.dte <= DTE_MAX)
        & (schema.delta >= TARGET_DELTA)
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
    bt.stocks_data = stocks_data
    bt.options_strategy = strategy
    bt.options_data = options_data
    bt.run(rebalance_freq=1, rebalance_unit="BMS")

    return bt.balance, bt.trade_log


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def compute_stats(balance_series, label, years):
    total_ret = (balance_series.iloc[-1] / balance_series.iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    cummax = balance_series.cummax()
    dd = (balance_series - cummax) / cummax
    max_dd = dd.min() * 100
    daily_ret = balance_series.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = (annual_ret / vol) if vol > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Annual return:  {annual_ret:+.2f}%")
    print(f"  Total return:   {total_ret:+.1f}%")
    print(f"  Max drawdown:   {max_dd:.1f}%")
    print(f"  Volatility:     {vol:.1f}%")
    print(f"  Sharpe (0% rf): {sharpe:.3f}")
    print(f"  Final value:    ${balance_series.iloc[-1]:,.0f}")
    return annual_ret, max_dd, sharpe


def log_trades(trade_log, label, n=20):
    """Print first/last trades from options_portfolio_backtester."""
    if trade_log is None or trade_log.empty:
        print(f"\n  {label}: no trades")
        return

    print(f"\n  {label}: {len(trade_log)} total trades")
    # Try to extract useful columns
    try:
        cols = trade_log.columns
        print(f"  Columns: {list(cols[:10])}...")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    options_data, stocks_data = load_data()

    stk_df = stocks_data._data.sort_values("date")
    start_date = stk_df["date"].iloc[0]
    end_date = stk_df["date"].iloc[-1]
    years = (end_date - start_date).days / 365.25
    print(f"Date range: {start_date.date()} to {end_date.date()} ({years:.1f} years)")

    # SPY buy-and-hold baseline
    spy_prices = stk_df["adjClose"].values
    spy_total = (spy_prices[-1] / spy_prices[0] - 1) * 100
    spy_annual = ((1 + spy_total / 100) ** (1 / years) - 1) * 100
    print(f"SPY B&H: {spy_annual:.2f}%/yr, {spy_total:.1f}% total")

    # ------------------------------------------------------------------
    # 1. Our Rust scanner
    # ------------------------------------------------------------------
    print("\n--- Running: Our Rust scanner (sell equity, reinvest proceeds) ---")
    print(f"    budget = initial_capital * {BUDGET_PCT} = ${INITIAL_CAPITAL * BUDGET_PCT:,.0f}/month")
    our_records, our_daily = run_our_scanner(options_data, stocks_data)

    compute_stats(our_daily["balance"], "OUR SCANNER (Rust)", years)
    print(f"  Months: {len(our_records)}")
    print(f"  Months with puts: {(our_records['contracts'] > 0).sum()}")
    print(f"  Total put cost: ${our_records['put_cost'].sum():,.0f}")
    print(f"  Total put P&L: ${our_records['put_pnl'].sum():,.0f}")

    # Monthly log for our scanner
    print(f"\n  Per-month log (first 24 months):")
    print(f"  {'Date':<12} {'Shares':>10} {'StockPx':>8} {'Equity':>12} "
          f"{'PutCost':>10} {'PutExit':>10} {'PutPnL':>10} {'Contracts':>5}")
    for _, row in our_records.head(24).iterrows():
        print(f"  {str(row.name.date()):<12} {row['shares']:>10.1f} {row['stock_price']:>8.2f} "
              f"{row['equity_value']:>12,.0f} {row['put_cost']:>10,.0f} "
              f"{row['put_exit_value']:>10,.0f} {row['put_pnl']:>10,.0f} {row['contracts']:>5}")

    # ------------------------------------------------------------------
    # 2. options_portfolio_backtester — AQR framing (sell equity to fund puts)
    # ------------------------------------------------------------------
    # AQR at 0.5% allocation (closest to our budget semantics)
    aqr_opt_pct = BUDGET_PCT
    aqr_stk_pct = 1.0 - aqr_opt_pct

    print(f"\n--- Running: Backtester AQR ({aqr_stk_pct:.1%} SPY + {aqr_opt_pct:.1%} puts) ---")
    aqr_balance, aqr_trades = run_backtester_aqr(
        options_data, stocks_data, aqr_stk_pct, aqr_opt_pct
    )
    compute_stats(aqr_balance["total capital"], f"BACKTESTER AQR ({aqr_opt_pct:.1%})", years)
    log_trades(aqr_trades, "AQR trades")

    # AQR at 6% (matching our effective annual spend)
    print(f"\n--- Running: Backtester AQR (94% SPY + 6% puts) ---")
    aqr_balance_6, aqr_trades_6 = run_backtester_aqr(
        options_data, stocks_data, 0.94, 0.06
    )
    compute_stats(aqr_balance_6["total capital"], "BACKTESTER AQR (6%)", years)

    # ------------------------------------------------------------------
    # 3. options_portfolio_backtester — Spitznagel framing
    # ------------------------------------------------------------------
    print(f"\n--- Running: Backtester Spitznagel (100% SPY + {BUDGET_PCT:.1%} budget) ---")
    spit_balance, spit_trades = run_backtester_spitznagel(
        options_data, stocks_data, BUDGET_PCT
    )
    compute_stats(spit_balance["total capital"], f"BACKTESTER SPITZNAGEL ({BUDGET_PCT:.1%})", years)
    log_trades(spit_trades, "Spitznagel trades")

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("  SUMMARY: All strategies side by side")
    print("=" * 80)
    print(f"  {'Strategy':<45} {'Annual%':>8} {'MaxDD%':>8} {'Sharpe':>7}")
    print("-" * 80)
    print(f"  {'SPY Buy & Hold':<45} {spy_annual:>+7.2f}% {'':>8} {'':>7}")

    for label, bal in [
        ("Our scanner (sell equity, reinvest, $5K/mo)", our_daily["balance"]),
        (f"Backtester AQR ({aqr_opt_pct:.1%} allocation)", aqr_balance["total capital"]),
        ("Backtester AQR (6% allocation)", aqr_balance_6["total capital"]),
        (f"Backtester Spitznagel ({BUDGET_PCT:.1%} budget)", spit_balance["total capital"]),
    ]:
        tr = (bal.iloc[-1] / bal.iloc[0] - 1) * 100
        ar = ((1 + tr / 100) ** (1 / years) - 1) * 100
        cummax = bal.cummax()
        dd = ((bal - cummax) / cummax).min() * 100
        dr = bal.pct_change().dropna()
        vol = dr.std() * np.sqrt(252) * 100
        sh = ar / vol if vol > 0 else 0
        print(f"  {label:<45} {ar:>+7.2f}% {dd:>7.1f}% {sh:>7.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
