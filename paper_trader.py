#!/usr/bin/env python3
"""Long-only, simulated trend-following backtest. Never places real trades."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

INITIAL_CASH = 15.00
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "VGK", "EWG"]
BENCHMARK = "VT"
START_DATE = "2010-01-01"
SMA_FAST = 50
SMA_SLOW = 150
MOMENTUM_DAYS = 63
COMMISSION_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.001
OUTPUT_DIR = Path("results")


def load_prices(ticker: str) -> pd.DataFrame:
    """Download adjusted OHLC data; yfinance is used for public historical data."""
    frame = yf.download(
        ticker, start=START_DATE, auto_adjust=True, progress=False, actions=False
    )
    if frame.empty:
        raise RuntimeError(f"No historical data returned for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = frame[["Open", "Close"]].copy()
    needed = needed[~needed.index.duplicated(keep="last")].sort_index()
    return needed.rename(columns={"Open": f"{ticker}_Open", "Close": f"{ticker}_Close"})


def load_all_prices() -> pd.DataFrame:
    frames = []
    for ticker in sorted(set(UNIVERSE + [BENCHMARK])):
        try:
            frames.append(load_prices(ticker))
        except Exception as exc:
            print(f"WARNING: skipping {ticker}: {exc}")
    if not frames:
        raise RuntimeError("Could not download any price data.")
    prices = pd.concat(frames, axis=1).sort_index()
    available = {c[:-5] for c in prices.columns if c.endswith("_Open")}
    missing = set(UNIVERSE + [BENCHMARK]) - available
    if missing:
        print("WARNING: missing tickers:", ", ".join(sorted(missing)))
    return prices


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_all_prices()
    available = [t for t in UNIVERSE if f"{t}_Close" in prices.columns and f"{t}_Open" in prices.columns]
    if not available:
        raise RuntimeError("No strategy assets have usable data.")

    close = prices[[f"{t}_Close" for t in available]].copy()
    close.columns = available
    open_prices = prices[[f"{t}_Open" for t in available]].copy()
    open_prices.columns = available

    fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    momentum = close / close.shift(MOMENTUM_DAYS) - 1.0

    dates = prices.index
    cash_cost_per_turnover = COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE
    chosen = None
    rows = []
    equity = INITIAL_CASH
    previous_position = None
    benchmark_equity = INITIAL_CASH

    # At today's open, use only yesterday's close and earlier data.
    # Rebalance on the first trading session of a new calendar month.
    for i in range(1, len(dates) - 1):
        today, tomorrow = dates[i], dates[i + 1]
        yesterday = dates[i - 1]
        is_month_start = today.month != yesterday.month or today.year != yesterday.year

        if is_month_start:
            eligible = []
            for ticker in available:
                c = close.at[yesterday, ticker]
                f = fast.at[yesterday, ticker]
                s = slow.at[yesterday, ticker]
                m = momentum.at[yesterday, ticker]
                if pd.notna(c) and pd.notna(f) and pd.notna(s) and pd.notna(m):
                    if c > s and f > s and m > 0:
                        eligible.append((ticker, float(m)))
            chosen = max(eligible, key=lambda item: item[1])[0] if eligible else None

        turnover = 0.0
        if chosen != previous_position:
            if previous_position is not None:
                turnover += 1.0
            if chosen is not None:
                turnover += 1.0
        trading_cost = cash_cost_per_turnover * turnover

        if chosen is None:
            asset_return = 0.0
        else:
            start_open = open_prices.at[today, chosen]
            next_open = open_prices.at[tomorrow, chosen]
            asset_return = (next_open / start_open - 1.0) if pd.notna(start_open) and pd.notna(next_open) and start_open > 0 else 0.0

        strategy_return = (1.0 + asset_return) * (1.0 - trading_cost) - 1.0
        equity *= 1.0 + strategy_return

        b0 = prices.at[today, f"{BENCHMARK}_Open"] if f"{BENCHMARK}_Open" in prices else np.nan
        b1 = prices.at[tomorrow, f"{BENCHMARK}_Open"] if f"{BENCHMARK}_Open" in prices else np.nan
        benchmark_return = (b1 / b0 - 1.0) if pd.notna(b0) and pd.notna(b1) and b0 > 0 else 0.0
        benchmark_equity *= 1.0 + benchmark_return

        rows.append({
            "date": today.date().isoformat(),
            "position": chosen or "CASH",
            "asset_return": float(asset_return),
            "turnover_units": turnover,
            "estimated_trading_cost": float(trading_cost),
            "strategy_return": float(strategy_return),
            "strategy_equity": float(equity),
            "benchmark_return": float(benchmark_return),
            "benchmark_equity": float(benchmark_equity),
        })
        previous_position = chosen

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Not enough historical data for the selected indicators.")

    daily = results["strategy_return"]
    running_peak = results["strategy_equity"].cummax()
    drawdown = results["strategy_equity"] / running_peak - 1.0
    elapsed_years = max((pd.to_datetime(results["date"].iloc[-1]) - pd.to_datetime(results["date"].iloc[0])).days / 365.25, 1 / 365.25)
    total_return = equity / INITIAL_CASH - 1.0
    cagr = (equity / INITIAL_CASH) ** (1.0 / elapsed_years) - 1.0 if equity > 0 else -1.0
    volatility = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else 0.0
    sharpe = float((daily.mean() * 252) / volatility) if volatility > 0 else None

    summary = {
        "mode": "PAPER_BACKTEST_ONLY",
        "real_orders_placed": False,
        "broker_connected": False,
        "initial_simulated_cash": INITIAL_CASH,
        "final_strategy_equity": round(float(equity), 4),
        "strategy_total_return_pct": round(float(total_return * 100), 3),
        "strategy_cagr_pct": round(float(cagr * 100), 3),
        "benchmark": BENCHMARK,
        "benchmark_final_equity": round(float(benchmark_equity), 4),
        "benchmark_total_return_pct": round(float((benchmark_equity / INITIAL_CASH - 1) * 100), 3),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 3),
        "annualized_volatility_pct": round(volatility * 100, 3),
        "sharpe_ratio_approx": round(sharpe, 3) if sharpe is not None else None,
        "estimated_commission_per_side_pct": COMMISSION_PER_SIDE * 100,
        "estimated_slippage_per_side_pct": SLIPPAGE_PER_SIDE * 100,
        "assets_available": available,
        "period_start": results["date"].iloc[0],
        "period_end": results["date"].iloc[-1],
        "important_limitations": [
            "Historical results do not predict future performance.",
            "Yahoo Finance data may contain gaps, revisions, or adjustment differences.",
            "ETF universe and benchmark choice can introduce selection bias.",
            "Cash earns 0%; taxes and fund-specific costs are not modeled.",
            "Execution is approximated with adjusted open prices and fixed costs.",
            "This is not a live trading system and places no real orders."
        ],
    }

    results.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    results.loc[results["position"].ne(results["position"].shift())].to_csv(OUTPUT_DIR / "trades.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved reports to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
