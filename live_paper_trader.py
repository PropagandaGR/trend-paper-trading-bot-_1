#!/usr/bin/env python3
"""Cost-aware long-only paper trader. No broker integration or real orders."""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"]
STARTING_CASH = 15.0
# Placeholder assumption, not measured spread/fee; evaluate higher-cost scenarios too.
COST_PER_SIDE = float(os.getenv("PAPER_COST_PER_SIDE", "0.001"))
STATE_PATH = Path(os.getenv("PAPER_STATE_PATH", "state/paper_state.json"))
REPORT_PATH = Path(os.getenv("PAPER_REPORT_PATH", "reports/latest.json"))

@dataclass
class PaperState:
    cash: float = STARTING_CASH
    position: dict[str, Any] | None = None
    last_processed_bar: str | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    peak_equity: float = STARTING_CASH

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

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

def fetch_bars(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period="5d", interval="1m", auto_adjust=True,
                      progress=False, prepost=False, threads=False)
    if raw is None or raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(raw.columns): return pd.DataFrame()
    raw = raw[cols].dropna().sort_index()
    bars = raw.resample("5min", label="right", closed="right").agg({
        "Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"
    }).dropna()
    now = pd.Timestamp.now(tz=bars.index.tz) if bars.index.tz else pd.Timestamp.now()
    # Exclude the currently forming 5-minute bucket, even at an exact boundary.
    last_completed_end = now.floor("5min") - pd.Timedelta(minutes=5)
    return bars[bars.index <= last_completed_end]

def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["ema_fast"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["ema_slow"] = x["Close"].ewm(span=21, adjust=False).mean()
    x["mom_3"] = x["Close"].pct_change(3)
    med = x["Volume"].rolling(20, min_periods=10).median().replace(0, float("nan"))
    x["volume_ratio"] = x["Volume"] / med
    x["atr_pct"] = (x["High"] - x["Low"]).rolling(14, min_periods=10).mean() / x["Close"]
    return x

def signal(row: pd.Series) -> tuple[bool, str]:
    keys = ("ema_fast", "ema_slow", "mom_3", "volume_ratio", "atr_pct")
    if any(pd.isna(row.get(k)) for k in keys): return False, "insufficient_history"
    enter = (row["ema_fast"] > row["ema_slow"] and row["mom_3"] > 0.001
             and row["volume_ratio"] >= 1.0 and 0.0005 <= row["atr_pct"] <= 0.012)
    return bool(enter), "trend_momentum_volume" if enter else "no_entry"

def equity(state: PaperState, marks: dict[str, float]) -> float:
    value = state.cash
    if state.position:
        value += float(state.position["quantity"]) * marks.get(
            state.position["symbol"], float(state.position["entry_price"]))
    return value

def close_position(state: PaperState, symbol: str, price: float, timestamp: str, reason: str) -> None:
    p = state.position
    if not p or p["symbol"] != symbol: return
    qty, entry = float(p["quantity"]), float(p["entry_price"])
    fill = price * (1 - COST_PER_SIDE)
    proceeds = qty * fill
    state.cash += proceeds
    state.trades.append({"side":"SELL", "symbol":symbol, "time":timestamp,
        "reference_price":price, "fill_price":fill, "quantity":qty, "reason":reason,
        "realized_pnl":proceeds - qty * entry, "cost_per_side_assumption":COST_PER_SIDE})
    state.position = None

def open_position(state: PaperState, symbol: str, price: float, timestamp: str, reason: str) -> None:
    if state.position or state.cash <= 0 or not math.isfinite(price) or price <= 0: return
    fill = price * (1 + COST_PER_SIDE)
    qty = state.cash / fill
    if not math.isfinite(qty) or qty <= 0: return
    state.position = {"symbol":symbol, "quantity":qty, "entry_price":fill, "entry_time":timestamp}
    state.cash = 0.0
    state.trades.append({"side":"BUY", "symbol":symbol, "time":timestamp,
        "reference_price":price, "fill_price":fill, "quantity":qty, "reason":reason,
        "cost_per_side_assumption":COST_PER_SIDE})

def make_report(state: PaperState, marks: dict[str, float], errors: dict[str, str], status: str) -> dict[str, Any]:
    eq = equity(state, marks)
    return {"updated_at":utc_now(), "mode":"paper_only", "status":status,
        "starting_cash":STARTING_CASH, "cash":round(state.cash, 6), "equity":round(eq, 6),
        "pnl_dollars":round(eq-STARTING_CASH, 6),
        "return_pct":round((eq/STARTING_CASH-1)*100, 4),
        "peak_equity":round(state.peak_equity, 6),
        "drawdown_from_peak_pct":round((eq/state.peak_equity-1)*100, 4) if state.peak_equity else 0,
        "position":state.position, "last_processed_bar":state.last_processed_bar,
        "trade_count":len(state.trades), "recent_trades":state.trades[-10:],
        "marks":marks, "data_errors":errors, "cost_per_side_assumption":COST_PER_SIDE,
        "symbols":SYMBOLS, "note":"Simulated prices/fills only; free data can be delayed/incomplete. No broker or real orders."}

def process_once(state_path: Path = STATE_PATH, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    state = load_state(state_path)
    data, errors = {}, {}
    for sym in SYMBOLS:
        try:
            bars = fetch_bars(sym)
            if not bars.empty and len(bars) >= 30: data[sym] = add_indicators(bars)
            else: errors[sym] = "no_data_or_insufficient_bars"
        except Exception as exc: errors[sym] = f"{type(exc).__name__}: {exc}"
    if not data:
        report = make_report(state, {}, errors, "no_market_data")
        report["note"] = "No trades created; feed may be delayed, unavailable, or rate-limited."
        save_json(report_path, report); save_json(state_path, asdict(state)); return report
    marks = {s:float(df["Close"].iloc[-1]) for s,df in data.items()}
    bars_union = set.union(*(set(df.index) for df in data.values()))
    latest = max(bars_union)
    if state.last_processed_bar is None:
        state.last_processed_bar = latest.isoformat()
        state.peak_equity = max(state.peak_equity, equity(state, marks))
        report = make_report(state, marks, errors, "initialized_no_trade")
        save_json(state_path, asdict(state)); save_json(report_path, report); return report
    try: last = pd.Timestamp(state.last_processed_bar)
    except Exception: last = latest
    # Do not replay every missed bar as if we had traded it live. Process only the
    # newest newly available completed bar; skipped intervals remain skipped.
    timeline = [max(t for t in bars_union if t > last)] if any(t > last for t in bars_union) else []
    # Signal uses the prior completed bar; approximate fill uses the next bar open.
    for ts in timeline:
        candidates = []
        for sym, df in data.items():
            if ts not in df.index: continue
            prior = df.index[df.index < ts]
            if not len(prior): continue
            prev, cur = df.loc[prior[-1]], df.loc[ts]
            price = float(cur["Open"])
            if not math.isfinite(price) or price <= 0: continue
            enter, why = signal(prev)
            if state.position and state.position["symbol"] == sym:
                if prev["ema_fast"] < prev["ema_slow"] or prev["mom_3"] < 0:
                    close_position(state, sym, price, ts.isoformat(), "trend_or_momentum_exit")
            if not state.position and enter: candidates.append((float(prev["mom_3"]), sym, why, price))
        if not state.position and candidates:
            _, sym, why, price = max(candidates, key=lambda z:z[0])
            open_position(state, sym, price, ts.isoformat(), why)
        state.last_processed_bar = ts.isoformat()
    state.peak_equity = max(state.peak_equity, equity(state, marks))
    report = make_report(state, marks, errors, "updated")
    save_json(state_path, asdict(state)); save_json(report_path, report)
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Update the simulated portfolio once")
    parser.parse_args()
    report = process_once()
    print(json.dumps(report, indent=2))
    if report["status"] == "no_market_data": raise SystemExit(1)

if __name__ == "__main__": main()
