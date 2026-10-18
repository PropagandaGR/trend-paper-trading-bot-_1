#!/usr/bin/env python3
"""Historical spot strategy comparison with chronological out-of-sample windows."""
from __future__ import annotations
import argparse, json, math, time, urllib.parse, urllib.request
from pathlib import Path
from typing import Any
import pandas as pd
from crypto_paper_bot import SYMBOLS, indicators, strategy_signal

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL = "5m"
STRATEGIES = ["trend", "breakout", "mean_reversion"]
COSTS_BPS = [5, 10, 20]

def download_history(symbol: str, days: int = 90) -> pd.DataFrame:
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    rows, cursor = [], start
    while cursor < end:
        params = urllib.parse.urlencode({"symbol": symbol, "interval": INTERVAL, "startTime": cursor, "endTime": end, "limit": 1000})
        req = urllib.request.Request(BASE_URL + "?" + params, headers={"User-Agent": "crypto-paper-research/1.0"})
        with urllib.request.urlopen(req, timeout=20) as response:
            batch = json.loads(response.read().decode())
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("timestamp").rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index().iloc[:-1]

def position_series(df: pd.DataFrame, strategy: str) -> pd.Series:
    x = indicators(df)
    return pd.Series([1.0 if strategy_signal(row, strategy) else 0.0 for _, row in x.iterrows()], index=x.index, dtype=float)

def metrics(returns: pd.Series, positions: pd.Series) -> dict[str, Any]:
    r = returns.fillna(0.0)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    p = positions.fillna(0.0)
    active = p.diff().abs().fillna(p.abs())
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = (float(r.mean()) / std * math.sqrt(288 * 365)) if std > 0 else 0.0
    return {
        "return_pct": round((float(eq.iloc[-1]) - 1) * 100, 4) if len(eq) else 0.0,
        "max_drawdown_pct": round(float(dd.min()) * 100, 4) if len(dd) else 0.0,
        "sharpe_annualized_5m_approx": round(sharpe, 3),
        "trade_changes": int(active.sum()),
        "exposure_pct": round(float(p.mean()) * 100, 2) if len(p) else 0.0,
    }

def simulate(df: pd.DataFrame, signals: pd.Series, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    # Signals from completed bars are held starting on the next bar to avoid look-ahead.
    held = signals.reindex(df.index).shift(1).fillna(0.0)
    price_ret = df["Close"].pct_change().fillna(0.0)
    changes = held.diff().abs().fillna(held.abs())
    return held * price_ret - changes * (cost_bps / 10000), held

def walk_forward_from_returns(net: pd.Series, held: pd.Series, train_days: int = 30, test_days: int = 7) -> dict[str, Any]:
    # Fixed rules are not fitted on training data; the initial window is warm-up only.
    train_n, test_n = train_days * 288, test_days * 288
    chunks, start = [], train_n
    while start < len(net):
        end = min(len(net), start + test_n)
        chunks.append((net.iloc[start:end], held.iloc[start:end]))
        start = end
    if not chunks:
        return {"windows": 0, "metrics": metrics(pd.Series(dtype=float), pd.Series(dtype=float))}
    return {"windows": len(chunks), "metrics": metrics(pd.concat([x[0] for x in chunks]), pd.concat([x[1] for x in chunks]))}

def run(days: int = 90, out_path: Path = Path("reports/research_summary.json")) -> dict[str, Any]:
    result = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "data_interval": "5m",
        "history_days_requested": days, "starting_cash_usdt": 15, "cost_scenarios_bps_per_side": COSTS_BPS,
        "assets": {},
        "methodology": "Signals are generated on completed candles and shifted one bar. The first 30 days are warm-up; the following periods are stitched 7-day chronological out-of-sample windows. Strategy rules are fixed, not optimized on the training window.",
        "limitations": [
            "Public candles are not historical order-book data; fills and slippage are approximated.",
            "Results on limited history are exploratory, not proof of a durable edge.",
            "The 30-day window is warm-up, not parameter fitting; this is fixed-rule out-of-sample evaluation.",
            "Cost is charged on each entry/exit position change; costs are assumptions, not measured fills.",
            "Annualized 5-minute Sharpe is approximate and can be misleading for short, non-independent intraday returns.",
        ],
    }
    for sym in SYMBOLS:
        try:
            df = download_history(sym, days)
        except Exception as exc:
            result["assets"][sym] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if len(df) < 1000:
            result["assets"][sym] = {"error": "Insufficient historical bars", "bars": len(df)}
            continue
        asset = {"bars": len(df), "start": df.index[0].isoformat(), "end": df.index[-1].isoformat(), "strategies": {}}
        for strat in STRATEGIES:
            signals = position_series(df, strat)
            scenarios = {}
            for cost in COSTS_BPS:
                net, held = simulate(df, signals, cost)
                scenarios[f"{cost}_bps_per_side"] = {
                    "full_history": metrics(net, held),
                    "walk_forward_30d_warmup_7d_test": walk_forward_from_returns(net, held),
                }
            asset["strategies"][strat] = scenarios
        ret = df["Close"].pct_change().fillna(0.0)
        for cost in COSTS_BPS:
            # Buy-and-hold has one entry and one exit cost across the evaluation period.
            net = ret.copy()
            if len(net):
                net.iloc[0] -= cost / 10000
                net.iloc[-1] -= cost / 10000
            asset.setdefault("buy_hold", {})[f"{cost}_bps_per_side"] = metrics(net, pd.Series(1.0, index=df.index))
        result["assets"][sym] = asset
        time.sleep(0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output", default="reports/research_summary.json")
    args = parser.parse_args()
    result = run(max(45, min(args.days, 365)), Path(args.output))
    print(json.dumps(result, indent=2))
    if not any("strategies" in a for a in result["assets"].values()):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
