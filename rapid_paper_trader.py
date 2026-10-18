#!/usr/bin/env python3
"""One-second-clock paper simulator driven by free 1-minute historical bars.
This is NOT second-resolution market data and does not place orders."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOL = "SPY"
BENCHMARK = "SPY"
PERIOD = "7d"          # Yahoo Finance typically limits free 1-minute history
INTERVAL = "1m"
INITIAL_CASH = 15.0
COST_PER_SIDE = 0.001   # estimated 0.10% per side for spread/slippage/fees
OUT = Path("results")

def get_bars(ticker):
    frame = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False, prepost=False)
    if frame.empty: raise RuntimeError(f"No free 1-minute bars available for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex): frame.columns=frame.columns.get_level_values(0)
    close=frame["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    return close.dropna()

def main():
    OUT.mkdir(exist_ok=True)
    bars=get_bars(SYMBOL)
    bench=get_bars(BENCHMARK)
    # Only use timestamps shared by the symbol and benchmark.
    data=pd.concat([bars.rename("close"),bench.rename("benchmark")],axis=1,join="inner").dropna()
    if len(data)<30: raise RuntimeError(f"Only {len(data)} bars available; need at least 30.")
    # One-second clock replay: each minute bar is held for 60 simulated ticks.
    # New decisions are allowed only when a new historical minute bar arrives.
    cash=INITIAL_CASH
    units=0.0
    position=False
    last_signal=False
    rows=[]
    equity_curve=[]
    start_price=float(data.close.iloc[0])
    benchmark_units=INITIAL_CASH/start_price
    previous_equity=INITIAL_CASH
    for i,(stamp,row) in enumerate(data.iterrows()):
        price=float(row.close)
        # Signal uses only current/previous bar close; EMA trend + 5-bar momentum.
        if i>=20:
            history=data.close.iloc[:i+1]
            ema_fast=float(history.ewm(span=5,adjust=False).mean().iloc[-1])
            ema_slow=float(history.ewm(span=20,adjust=False).mean().iloc[-1])
            mom=float(price/history.iloc[-6]-1) if history.iloc[-6]>0 else 0
            signal=ema_fast>ema_slow and mom>0
        else:
            signal=False
        # Execute at bar close with a fixed adverse cost; this is a simplified model.
        if signal != position:
            if signal and not position:
                effective=price*(1+COST_PER_SIDE)
                units=cash/effective
                cash=0.0
            elif position and not signal:
                cash=units*price*(1-COST_PER_SIDE)
                units=0.0
            position=signal
        equity=cash+units*price
        benchmark_value=benchmark_units*float(row.benchmark)
        equity_curve.append(equity)
        rows.append({"timestamp":stamp.isoformat(),"simulated_second_ticks":60,
                     "last_available_minute_close":price,"signal":"HOLD_SPY" if position else "CASH",
                     "strategy_equity":equity,"benchmark_buy_hold_equity":benchmark_value,
                     "estimated_cost_per_side_pct":COST_PER_SIDE*100})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"rapid_paper_equity.csv",index=False)
    eq=pd.Series([INITIAL_CASH]+equity_curve)
    ret=eq.pct_change().dropna()
    dd=eq/eq.cummax()-1
    summary={
      "mode":"PAPER_SIMULATION_ONLY","real_orders_placed":False,"broker_connected":False,
      "symbol":SYMBOL,"benchmark":"SPY buy-and-hold","free_data_source":"Yahoo Finance via yfinance",
      "source_interval":INTERVAL,"history_period_requested":PERIOD,"bar_count":len(data),
      "simulated_clock":"one-second replay clock; each observed 1-minute bar is repeated conceptually for 60 ticks",
      "important_data_limit":"This is NOT a one-second backtest. Free minute bars cannot reveal second-by-second price changes; signals update only when a new minute bar is available.",
      "initial_simulated_cash":INITIAL_CASH,"strategy_final_equity":round(float(equity_curve[-1]),4),
      "strategy_total_return_pct":round((equity_curve[-1]/INITIAL_CASH-1)*100,3),
      "benchmark_final_equity":round(float(benchmark_units*data.benchmark.iloc[-1]),4),
      "benchmark_total_return_pct":round((float(data.benchmark.iloc[-1])/float(data.benchmark.iloc[0])-1)*100,3),
      "max_drawdown_pct":round(float(dd.min()*100),3),"estimated_cost_per_side_pct":COST_PER_SIDE*100,
      "limitations":["Short free intraday history; results are not comparable with the multi-year daily backtest.",
      "Minute bars replayed on a one-second clock do not constitute tick data.",
      "Execution at bar close with fixed costs is simplified and may overstate performance.",
      "No taxes, market impact or spread variation modeled.","No real orders, broker, paid data or paid infrastructure."]
    }
    (OUT/"rapid_paper_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
