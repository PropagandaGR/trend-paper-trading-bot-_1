#!/usr/bin/env python3
"""Paper-only rapid simulator using free one-minute OHLC bars.

Signals are calculated from completed closes and executed at the NEXT bar's open.
This is not second-resolution data and never places real orders.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOL = "SPY"
PERIOD = "7d"
INTERVAL = "1m"
INITIAL_CASH = 15.0
COST_SCENARIOS = (0.0005, 0.0010, 0.0020)  # all-in estimated cost per side
BASE_COST = 0.0010
OUT = Path("results")


def get_bars(ticker):
    frame = yf.download(
        ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True,
        progress=False, prepost=False
    )
    if frame.empty:
        raise RuntimeError(f"No free 1-minute bars available for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.loc[:, ["Open", "Close"]].dropna()
    if frame.empty:
        raise RuntimeError(f"No usable OHLC bars available for {ticker}.")
    return frame


def prepare_signals(data):
    """Signal at row t uses only data through the close of row t."""
    close = data["Close"]
    fast = close.ewm(span=5, adjust=False).mean()
    slow = close.ewm(span=20, adjust=False).mean()
    momentum = close.pct_change(5)
    signal = ((fast > slow) & (momentum > 0)).fillna(False)
    signal.iloc[:20] = False
    return signal


def simulate(data, cost_per_side, initial_cash=INITIAL_CASH):
    """Run long/cash simulation; signal from prior close executes at next open."""
    data = data.copy().dropna(subset=["Open", "Close"])
    if len(data) < 30:
        raise RuntimeError(f"Only {len(data)} bars available; need at least 30.")
    desired = prepare_signals(data)
    cash = float(initial_cash)
    units = 0.0
    position = False
    trades = 0
    entries = []
    equity_rows = []

    # Fair benchmark: buy once at the first open and pay the same entry cost.
    first_open = float(data["Open"].iloc[0])
    benchmark_units = initial_cash / (first_open * (1.0 + cost_per_side))

    for i, (stamp, row) in enumerate(data.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        # At bar i open, only the signal formed at bar i-1 close is available.
        target = bool(desired.iloc[i - 1]) if i > 0 else False
        if target != position and i > 0:
            if target:
                units = cash / (open_price * (1.0 + cost_per_side))
                cash = 0.0
                entries.append(stamp)
            else:
                cash = units * open_price * (1.0 - cost_per_side)
                units = 0.0
            position = target
            trades += 1

        equity = cash + units * close_price
        benchmark_equity = benchmark_units * close_price
        equity_rows.append({
            "timestamp": stamp.isoformat(),
            "open": open_price,
            "close": close_price,
            "signal_from_previous_close": target,
            "position": "SPY" if position else "CASH",
            "strategy_equity": equity,
            "benchmark_equity": benchmark_equity,
            "cost_per_side_pct": cost_per_side * 100.0,
        })

    result = pd.DataFrame(equity_rows)
    equity = pd.Series([initial_cash] + result["strategy_equity"].tolist(), dtype=float)
    drawdown = equity / equity.cummax() - 1.0
    changes = result["strategy_equity"].pct_change().dropna()
    final_equity = float(result["strategy_equity"].iloc[-1])
    benchmark_final = float(result["benchmark_equity"].iloc[-1])
    return result, {
        "cost_per_side_pct": round(cost_per_side * 100, 3),
        "initial_simulated_cash": initial_cash,
        "strategy_final_equity": round(final_equity, 4),
        "strategy_total_return_pct": round((final_equity / initial_cash - 1) * 100, 3),
        "benchmark_final_equity": round(benchmark_final, 4),
        "benchmark_total_return_pct": round((benchmark_final / initial_cash - 1) * 100, 3),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 3),
        "position_changes": trades,
        "completed_round_trips": trades // 2,
        "minute_bars": len(data),
        "mean_bar_return_pct": round(float(changes.mean() * 100), 5) if len(changes) else 0.0,
    }


def main():
    OUT.mkdir(exist_ok=True)
    data = get_bars(SYMBOL)
    if len(data) < 30:
        raise RuntimeError(f"Only {len(data)} bars available; need at least 30.")

    summaries = []
    base_equity = None
    for cost in COST_SCENARIOS:
        curve, summary = simulate(data, cost)
        summaries.append(summary)
        if abs(cost - BASE_COST) < 1e-12:
            base_equity = curve

    if base_equity is None:
        raise RuntimeError("BASE_COST must be included in COST_SCENARIOS.")
    base_equity.to_csv(OUT / "rapid_paper_equity.csv", index=False)
    summary = {
        "mode": "PAPER_SIMULATION_ONLY",
        "real_orders_placed": False,
        "broker_connected": False,
        "symbol": SYMBOL,
        "benchmark": "SPY buy-and-hold; same entry cost assumption",
        "free_data_source": "Yahoo Finance via yfinance",
        "source_interval": INTERVAL,
        "history_period_requested": PERIOD,
        "bar_count": len(data),
        "data_start": data.index[0].isoformat(),
        "data_end": data.index[-1].isoformat(),
        "execution_rule": "Signal uses completed bar close; trade executes at the next bar open.",
        "simulated_clock": "Minute-bar replay; optional one-second clock is only a presentation/replay concept.",
        "important_data_limit": "This is NOT a one-second backtest. Free minute bars cannot reveal second-by-second prices or execution.",
        "strategy": "Long SPY or cash; EMA(5) above EMA(20) and positive 5-bar momentum; first 20 bars excluded.",
        "cost_model": "Single all-in estimated cost per side; scenarios are alternatives, not additive line items.",
        "cost_scenarios": summaries,
        "base_cost_per_side_pct": BASE_COST * 100,
        "limitations": [
            "Only a short window of free intraday data is available; not comparable with multi-year daily tests.",
            "The next bar open is a simplified execution proxy; real fills, bid/ask spread and latency differ.",
            "Fixed costs are assumptions, not measured SPY transaction costs; scenarios test sensitivity.",
            "Benchmark pays the same assumed entry cost but does not model a final liquidation fee.",
            "No taxes, market impact or variable spreads modeled.",
            "No leverage, short selling, broker connection, credentials, paid data or real orders."
        ],
    }
    (OUT / "rapid_paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
