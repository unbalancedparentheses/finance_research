# Convexity Scanner — Exploration

Score computation, backtest results, and visualization for SPY tail hedge overlay.


```python
import sys
sys.path.insert(0, '../src')

from options_portfolio_backtester.data.providers import HistoricalOptionsData, TiingoData
from options_portfolio_backtester.convexity.config import BacktestConfig
from options_portfolio_backtester.convexity.scoring import compute_convexity_scores
from options_portfolio_backtester.convexity.backtest import run_backtest, run_unhedged
from options_portfolio_backtester.convexity.viz import convexity_scores_chart, monthly_pnl_chart, cumulative_pnl_chart
```


```python
# Load data
options = HistoricalOptionsData('../options_backtester/data/processed/options.csv')
stocks = TiingoData('../options_backtester/data/processed/stocks.csv')
config = BacktestConfig()
print(f'Options: {len(options):,} rows ({options.start_date} to {options.end_date})')
print(f'Stocks: {len(stocks):,} rows')
```


```python
# Compute convexity scores
scores = compute_convexity_scores(options, config)
scores.describe()
```


```python
# Convexity ratio over time
convexity_scores_chart(scores)
```


```python
# Run backtest
result = run_backtest(options, stocks, config)
unhedged = run_unhedged(stocks, config)
print(f'Hedged final: ${result.daily_balance["balance"].iloc[-1]:,.0f}')
print(f'Unhedged final: ${unhedged["balance"].iloc[-1]:,.0f}')
```


```python
# Monthly put P&L
monthly_pnl_chart(result.records)
```


```python
# Cumulative portfolio value: hedged vs unhedged
cumulative_pnl_chart({'Hedged (SPY + puts)': result.daily_balance, 'Unhedged (SPY only)': unhedged})
```


```python
# Stats comparison
from options_portfolio_backtester.analytics.stats import BacktestStats

hedged_bal = result.daily_balance.rename(columns={'balance': 'total capital', 'pct_change': '% change'})
unhedged_bal = unhedged.rename(columns={'balance': 'total capital', 'pct_change': '% change'})

h_stats = BacktestStats.from_balance(hedged_bal)
u_stats = BacktestStats.from_balance(unhedged_bal)

print('HEDGED')
print(h_stats.summary())
print()
print('UNHEDGED')
print(u_stats.summary())
```


```python
# Monthly records summary
r = result.records
print(f'Total months: {len(r)}')
print(f'Months with puts: {(r["contracts"] > 0).sum()}')
print(f'Months with positive put P&L: {(r["put_pnl"] > 0).sum()}')
print(f'Total put cost: ${r["put_cost"].sum():,.0f}')
print(f'Total put P&L: ${r["put_pnl"].sum():,.0f}')
print(f'Mean convexity ratio: {r["convexity_ratio"].mean():.3f}')
```
