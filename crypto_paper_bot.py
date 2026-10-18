#!/usr/bin/env python3
"""Public-data-only crypto paper portfolio. No exchange trading endpoints."""
from __future__ import annotations
import argparse, json, math, os, urllib.parse, urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
STARTING_CASH = 15.0
COST_BPS = float(os.getenv("CRYPTO_PAPER_COST_BPS", "10"))
STATE_PATH = Path(os.getenv("CRYPTO_STATE_PATH", "state/crypto_paper_state.json"))
REPORT_PATH = Path(os.getenv("CRYPTO_REPORT_PATH", "reports/crypto_latest.json"))
BAR_INTERVAL = "5m"

@dataclass
class PaperState:
    cash: float = STARTING_CASH
    position: dict[str, Any] | None = None
    last_processed_bar: str | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    peak_equity: float = STARTING_CASH

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def fetch_klines(symbol: str, limit: int = 300, start_time: int | None = None) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": BAR_INTERVAL, "limit": min(1000, limit)}
    if start_time is not None: params["startTime"] = int(start_time)
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-paper-research/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume",
            "trade_count","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(payload, columns=cols)
    if df.empty: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("timestamp").rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna().sort_index()

def closed_bars(symbol: str, limit: int = 300) -> pd.DataFrame:
    df = fetch_klines(symbol, limit)
    return df.iloc[:-1].copy() if not df.empty else df

def load_state(path: Path = STATE_PATH) -> PaperState:
    if not path.exists(): return PaperState()
    d = json.loads(path.read_text(encoding="utf-8"))
    return PaperState(float(d.get("cash", STARTING_CASH)), d.get("position"),
                      d.get("last_processed_bar"), list(d.get("trades", [])),
                      float(d.get("peak_equity", STARTING_CASH)))

def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema_fast"] = x["Close"].ewm(span=6, adjust=False).mean()
    x["ema_slow"] = x["Close"].ewm(span=18, adjust=False).mean()
    x["mom_3"] = x["Close"].pct_change(3)
    x["rsi"] = rsi(x["Close"], 14)
    x["range_high"] = x["High"].shift(1).rolling(12, min_periods=12).max()
    x["range_low"] = x["Low"].shift(1).rolling(12, min_periods=12).min()
    x["volume_median"] = x["Volume"].rolling(20, min_periods=10).median()
    x["volume_ratio"] = x["Volume"] / x["volume_median"].replace(0, float("nan"))
    x["atr_pct"] = (x["High"] - x["Low"]).rolling(14, min_periods=10).mean() / x["Close"]
    return x

def strategy_signal(row: pd.Series, strategy: str) -> bool:
    required = ["ema_fast","ema_slow","mom_3","rsi","range_high","range_low","volume_ratio","atr_pct"]
    if any(pd.isna(row.get(k)) for k in required): return False
    vol_ok = 0.0002 <= row["atr_pct"] <= 0.025
    if strategy == "trend":
        return bool(row["ema_fast"] > row["ema_slow"] and row["mom_3"] > 0.001 and vol_ok)
    if strategy == "breakout":
        return bool(row["Close"] > row["range_high"] and row["volume_ratio"] >= 1.0 and vol_ok)
    if strategy == "mean_reversion":
        return bool(row["rsi"] < 28 and row["Close"] > row["ema_slow"] * 0.94 and vol_ok)
    raise ValueError(f"Unknown strategy: {strategy}")

def equity(state: PaperState, marks: dict[str, float]) -> float:
    value = state.cash
    if state.position:
        value += float(state.position["quantity"]) * marks.get(state.position["symbol"], state.position["entry_price"])
    return value

def buy(state: PaperState, symbol: str, price: float, ts: str, strategy: str) -> None:
    if state.position or state.cash <= 0 or not math.isfinite(price) or price <= 0: return
    fill = price * (1 + COST_BPS / 10000)
    qty = state.cash / fill
    state.position = {"symbol":symbol,"quantity":qty,"entry_price":fill,"entry_time":ts,"strategy":strategy}
    state.cash = 0.0
    state.trades.append({"side":"BUY","symbol":symbol,"time":ts,"reference_price":price,
        "fill_price":fill,"quantity":qty,"strategy":strategy,"cost_bps_per_side":COST_BPS})

def sell(state: PaperState, price: float, ts: str, reason: str) -> None:
    p = state.position
    if not p or not math.isfinite(price) or price <= 0: return
    fill = price * (1 - COST_BPS / 10000)
    qty = float(p["quantity"])
    proceeds = qty * fill
    pnl = proceeds - qty * float(p["entry_price"])
    state.cash += proceeds
    state.trades.append({"side":"SELL","symbol":p["symbol"],"time":ts,"reference_price":price,
        "fill_price":fill,"quantity":qty,"strategy":p["strategy"],"reason":reason,
        "realized_pnl":pnl,"cost_bps_per_side":COST_BPS})
    state.position = None

def make_report(state: PaperState, marks: dict[str,float], errors: dict[str,str], status: str) -> dict[str,Any]:
    eq = equity(state, marks)
    return {"updated_at":now_utc(),"mode":"paper_only","status":status,
        "starting_cash":STARTING_CASH,"cash":round(state.cash,8),"equity":round(eq,8),
        "pnl_usdt":round(eq-STARTING_CASH,8),"return_pct":round((eq/STARTING_CASH-1)*100,5),
        "peak_equity":round(state.peak_equity,8),
        "drawdown_from_peak_pct":round((eq/state.peak_equity-1)*100,5) if state.peak_equity else 0,
        "position":state.position,"last_processed_bar":state.last_processed_bar,
        "trade_count":len(state.trades),"recent_trades":state.trades[-10:],
        "marks_usdt":marks,"data_errors":errors,"cost_bps_per_side":COST_BPS,
        "symbols":SYMBOLS,"note":"Public market data and simulated fills only. No exchange credentials or real orders."}

def run_once(state_path: Path = STATE_PATH, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    state = load_state(state_path)
    data, errors = {}, {}
    for sym in SYMBOLS:
        try:
            bars = closed_bars(sym, 300)
            if len(bars) >= 40: data[sym] = indicators(bars)
            else: errors[sym] = "insufficient_closed_bars"
        except Exception as exc: errors[sym] = f"{type(exc).__name__}: {exc}"
    if not data:
        report = make_report(state, {}, errors, "no_market_data")
        save_json(report_path, report); save_json(state_path, asdict(state)); return report
    marks = {s:float(df["Close"].iloc[-1]) for s,df in data.items()}
    latest = max(df.index[-1] for df in data.values())
    if state.last_processed_bar is None:
        state.last_processed_bar = latest.isoformat()
        state.peak_equity = max(state.peak_equity, equity(state, marks))
        report = make_report(state, marks, errors, "initialized_no_trade")
        save_json(state_path, asdict(state)); save_json(report_path, report); return report
    last = pd.Timestamp(state.last_processed_bar)
    unseen = [df.index[-1] for df in data.values() if df.index[-1] > last]
    if unseen:
        ts = max(unseen)
        candidates = []
        for sym, df in data.items():
            if ts not in df.index: continue
            loc = int(df.index.get_loc(ts))
            if loc < 1: continue
            prev, cur = df.iloc[loc-1], df.iloc[loc]
            price = float(cur["Open"])
            if state.position and state.position["symbol"] == sym:
                if prev["ema_fast"] < prev["ema_slow"] or prev["mom_3"] < 0 or prev["rsi"] > 72:
                    sell(state, price, ts.isoformat(), "trend_momentum_or_rsi_exit")
            if not state.position:
                for strat in ["trend","breakout","mean_reversion"]:
                    if strategy_signal(prev, strat):
                        score = float(prev["mom_3"]) if strat != "mean_reversion" else float(30-prev["rsi"])/1000
                        candidates.append((score,sym,strat,price))
        if not state.position and candidates:
            _, sym, strat, price = max(candidates, key=lambda x:x[0])
            buy(state, sym, price, ts.isoformat(), strat)
        state.last_processed_bar = ts.isoformat()
    state.peak_equity = max(state.peak_equity, equity(state, marks))
    report = make_report(state, marks, errors, "updated")
    save_json(state_path, asdict(state)); save_json(report_path, report)
    return report

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    parser.parse_args()
    report=run_once()
    print(json.dumps(report,indent=2))
    if report["status"]=="no_market_data": raise SystemExit(1)
if __name__=="__main__": main()
