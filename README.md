# Crypto Fast-Trading Paper Lab

**Separate research track, paper-only.** Starting simulated capital: **$15 USDT**. Spot only; long/flat only; no leverage, no shorts, no broker connection, no private exchange keys, no real orders.

## What it does

- Uses public Binance Spot market-data endpoints only; no API key is needed for public market data.
- Tracks BTC/USDT, ETH/USDT, SOL/USDT.
- Trades on completed 5-minute bars, targeting 5–30 minute holding periods.
- Compares trend/momentum, range breakout and mean-reversion baselines.
- Runs historical walk-forward evaluation and cost stress tests at 5, 10 and 20 basis points per side.
- Saves a persistent simulated portfolio and JSON report.
- GitHub Actions runs offline unit tests and historical research, and schedules paper updates every five minutes (best-effort; not a guaranteed clock).

## Important data/execution limits

Binance public market data is used only for research; the bot has no trading endpoint and never requests credentials. The historical downloader is rate-conscious but network/data availability can still fail. Historical candles are not order-book history, and a bar-open fill is only a simplified approximation. Costs are assumptions, not measured personal exchange fees or realized spreads.

A five-minute bar strategy cannot prove one-second alpha. Even though crypto trades 24/7, the GitHub runner schedule may be delayed and is not a persistent low-latency process. Never treat a paper result as proof of future profit.

## Quick start

Python 3.11+:

```bash
pip install -r requirements.txt
python crypto_paper_bot.py --once
python crypto_research.py --days 90
python -m unittest discover -s tests -v
```

Portfolio state: `state/crypto_paper_state.json`  
Latest paper report: `reports/crypto_latest.json`  
Historical comparison: `reports/research_summary.json`

## Strategy comparison

1. Trend/momentum: fast EMA above slow EMA, positive momentum and a volatility guard.
2. Breakout: close above the prior rolling high, with a volume confirmation.
3. Mean reversion: RSI-style oversold threshold with a trend guard.
4. Buy-and-hold and cash/no-trade baselines.

Walk-forward testing chooses parameters on earlier training data and evaluates on later untouched test windows. The stitched out-of-sample report is the primary result; in-sample winners are not promoted automatically. All strategies are compared under multiple cost assumptions. A result is interesting only if it persists across windows/assets and survives costs.

## Public data references

- [Binance market-data-only endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md)
- [Binance Spot REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
- [Binance WebSocket streams](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)

This is experimental research software, not financial advice. No real trading is implemented.
