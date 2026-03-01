"""CLI entry point for running the full backtest."""

import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convexity Scanner Backtest")
    parser.add_argument(
        "--options-file",
        default="../options_backtester/data/processed/options.csv",
        help="Path to options CSV",
    )
    parser.add_argument(
        "--stocks-file",
        default="../options_backtester/data/processed/stocks.csv",
        help="Path to stocks CSV",
    )
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--budget-pct", type=float, default=0.005)
    parser.add_argument("--target-delta", type=float, default=-0.10)
    parser.add_argument("--tail-drop", type=float, default=0.20)
    args = parser.parse_args()

    from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData

    from options_portfolio_backtester.convexity.backtest import BacktestResult, run_backtest, run_unhedged
    from options_portfolio_backtester.convexity.config import BacktestConfig, InstrumentConfig
    from options_portfolio_backtester.convexity.scoring import compute_convexity_scores

    config = BacktestConfig(
        initial_capital=args.capital,
        budget_pct=args.budget_pct,
        target_delta=args.target_delta,
        tail_drop=args.tail_drop,
        instruments=[
            InstrumentConfig(
                symbol="SPY",
                options_file=args.options_file,
                stocks_file=args.stocks_file,
                target_delta=args.target_delta,
                tail_drop=args.tail_drop,
            )
        ],
    )

    # Load data
    log.info("Loading options data from %s", args.options_file)
    options = HistoricalOptionsData(args.options_file)
    log.info("Loaded %d option rows (%s to %s)", len(options), options.start_date, options.end_date)

    log.info("Loading stocks data from %s", args.stocks_file)
    stocks = TiingoData(args.stocks_file)
    log.info("Loaded %d stock rows", len(stocks))

    # Compute scores
    log.info("Computing convexity scores...")
    scores = compute_convexity_scores(options, config)
    log.info("Score summary:")
    log.info("  Mean convexity ratio: %.3f", scores["convexity_ratio"].mean())
    log.info("  Median: %.3f", scores["convexity_ratio"].median())
    log.info("  Min: %.3f  Max: %.3f", scores["convexity_ratio"].min(), scores["convexity_ratio"].max())

    # Run hedged backtest
    log.info("Running hedged backtest...")
    result = run_backtest(options, stocks, config)

    # Run unhedged benchmark
    log.info("Running unhedged benchmark...")
    unhedged = run_unhedged(stocks, config)

    # Compute stats
    try:
        from options_portfolio_backtester.analytics.stats import BacktestStats

        hedged_balance = result.daily_balance.rename(columns={"balance": "total capital", "pct_change": "% change"})
        unhedged_balance = unhedged.rename(columns={"balance": "total capital", "pct_change": "% change"})

        hedged_stats = BacktestStats.from_balance(hedged_balance)
        unhedged_stats = BacktestStats.from_balance(unhedged_balance)

        print("\n" + "=" * 60)
        print("HEDGED (SPY + puts)")
        print("=" * 60)
        print(hedged_stats.summary())

        print("\n" + "=" * 60)
        print("UNHEDGED (SPY only)")
        print("=" * 60)
        print(unhedged_stats.summary())
    except ImportError:
        log.warning("BacktestStats not available, printing basic results")
        daily = result.daily_balance
        total_return = daily["balance"].iloc[-1] / daily["balance"].iloc[0] - 1
        print(f"\nHedged total return: {total_return:.2%}")
        print(f"Final value: ${daily['balance'].iloc[-1]:,.0f}")

    # Monthly P&L summary
    records = result.records
    total_put_pnl = records["put_pnl"].sum()
    total_put_cost = records["put_cost"].sum()
    months_positive = (records["put_pnl"] > 0).sum()
    print(f"\nTotal put P&L: ${total_put_pnl:,.0f}")
    print(f"Total put cost: ${total_put_cost:,.0f}")
    print(f"Months with positive put P&L: {months_positive}/{len(records)}")


if __name__ == "__main__":
    main()
