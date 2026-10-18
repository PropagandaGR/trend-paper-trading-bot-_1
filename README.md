# Live Fast-Trading Paper Lab

**Paper trading only.** This project never connects to a broker and never sends real orders. Starting simulated cash is **$15**. Long-only, no leverage, no short selling.

## Goal

Research short-horizon strategies against historical data, then forward-test candidates against the latest available market data. The goal is to discover whether any signal has a repeatable **net-of-cost** edge—not to promise or assume profitable trades.

## First prototype

- Symbols: SPY, QQQ, IWM, AAPL, MSFT, NVDA (a small, liquid U.S. watchlist; not a claim these are today's best trades).
- Free Yahoo Finance 1-minute bars, aggregated to 5-minute bars.
- A transparent baseline signal combines short/long EMA trend, recent momentum, and a relative-volume filter.
- One simulated position at a time, fractional shares allowed, no leverage/shorts.
- A simple cost/slippage assumption is applied to every fill.
- Persistent JSON state records cash, position, last processed bar and trade log.
- GitHub Actions can run every five minutes; actual timing is not guaranteed and Yahoo data may be delayed, incomplete, rate-limited, or unavailable.
- The first run only initializes the latest completed bar; it does not invent historical live trades.

## Important limitations

This is an experimental prototype, not an investment recommendation. Yahoo's free feed is not a guaranteed real-time execution feed, and 1-minute bars cannot validate one-second strategies. Simulated fills at a bar open are approximations, not real bid/ask fills. The baseline strategy is a hypothesis, not a proven edge.

The existing rapid bot lost money in its short sample, largely with excessive turnover. Therefore this prototype prioritizes cost-aware testing and honest reporting over increasing trade frequency.

## How to run locally

Python 3.11+:

```bash
pip install -r requirements.txt
python live_paper_trader.py --once
```

The script writes `state/paper_state.json` and `reports/latest.json`. Do not add credentials or broker keys; none are required.

## GitHub Actions

The workflow is manually runnable and scheduled every five minutes on weekdays. Scheduled workflows run only from a repository's default branch, so this branch must be reviewed and merged into the default branch before automatic scheduling starts. The workflow commits updated paper state and a small JSON report back to the repository. GitHub's schedule is best-effort, not a guaranteed clock.

## Research plan before trusting results

1. Collect forward paper results without changing rules.
2. Test candidate horizons and simple baselines on multiple years of historical minute data where available.
3. Use chronological walk-forward splits and keep a final untouched holdout.
4. Include spread/fees/slippage stress cases; compare net performance to buy-and-hold and a no-trade baseline.
5. Report expectancy, profit factor, turnover, drawdown, time in market, and performance by month/regime—not win rate alone.
6. Reject strategies that only work at one narrow parameter setting or disappear after costs.

References: [Backtesting pitfalls and costs](https://academy.binance.com/en/articles/what-is-backtesting), [2026 intraday forecasting study with walk-forward evaluation](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1888767/full).
