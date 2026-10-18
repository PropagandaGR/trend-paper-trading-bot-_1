# Fast-trading research protocol

## Why we do not optimize for win rate or trade count

A strategy can win often and still lose money when average losses, spread, slippage, and repeated turnover exceed average gains. The existing rapid experiment made 190 round trips over a short sample and lost substantially more as assumed costs increased. The first objective is therefore to test whether a signal has positive expectancy **after costs**, not to maximize activity.

## Research questions

1. Which holding horizons (5, 10, 15, 30 minutes) survive realistic costs?
2. Do trend continuation, opening-range breakout, and mean-reversion baselines behave differently by volatility regime?
3. Does a volume-confirmation filter improve net results out of sample?
4. Does performance persist across SPY, QQQ, IWM and a small liquid-stock set, or is it concentrated in one symbol?
5. Does a signal remain useful after a one-bar delay and more adverse fills?

## Required evaluation design

- Use chronological train/validation/test splits; never shuffle time series.
- Tune on past data only. Walk forward: fit or select on an earlier window, then evaluate on the next untouched window.
- Keep a final holdout that is not used to choose the strategy or thresholds.
- Record every experiment, including failed versions, to expose selection bias.
- Avoid selecting the best symbol from today's survivors and then pretending it was known historically.
- Compare against cash/no-trade and a simple buy-and-hold benchmark over the same timestamps.
- Include at least three cost scenarios. Treat 0.10% per side as a placeholder, not a verified cost estimate; test 0.05%, 0.10%, 0.20% and a delayed/adverse-fill case.
- Use completed bars for signals and later bars for fills. Never fill at the same close that generated the signal.
- Exclude sessions with missing/corrupt data and report how much data was excluded.
- Report net return, maximum drawdown, trade count, turnover, average trade, expectancy, profit factor, exposure, performance by month, and stability across symbols/regimes. Win rate alone is insufficient.

## Practical limitations of free data

Yahoo Finance intraday history is limited and can be delayed or incomplete. It is useful for prototyping but does not provide a reliable historical order book, queue position, or true bid/ask execution. Minute bars cannot establish that a strategy works at one-second resolution. If the apparent edge is smaller than plausible spread/slippage, reject it.

## Promotion gates

A candidate is not promoted because it has the highest backtest return. Require:
- positive net expectancy in multiple out-of-sample windows;
- performance not dominated by one asset or a few exceptional trades;
- tolerable drawdown and turnover;
- no severe collapse under adverse cost assumptions or one-bar delay;
- a stable forward paper record over a predeclared observation period.

Passing these gates is evidence for further study, not proof of future profitability.
