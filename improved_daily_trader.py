#!/usr/bin/env python3
"""Improved daily trend strategy, long-only and historical simulation only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

CASH = 15.0
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "VGK", "EWG"]
BENCHMARK = "VT"
START = "2010-01-01"
COST_PER_SIDE = 0.001  # 0.10% each side, combines commission + slippage estimate
OUT = Path("results")

def close_series(ticker):
    data = yf.download(ticker, start=START, auto_adjust=True, progress=False, actions=False)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna()

def metrics(equity, returns, initial):
    peak = equity.cummax()
    dd = equity / peak - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1/365.25)
    final = float(equity.iloc[-1])
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns)>1 else 0.0
    return {
      "ending_value": round(final, 4),
      "total_return_pct": round((final/initial-1)*100, 3),
      "cagr_pct": round(((final/initial)**(1/years)-1)*100, 3) if final>0 else -100.0,
      "max_drawdown_pct": round(float(dd.min())*100, 3),
      "annualized_volatility_pct": round(vol*100, 3),
      "sharpe_approx": round(float(returns.mean()*252/vol), 3) if vol>0 else None,
    }

def main():
    OUT.mkdir(exist_ok=True)
    series = {}
    for t in sorted(set(UNIVERSE+[BENCHMARK])):
        try: series[t] = close_series(t)
        except Exception as e: print(f"WARNING {t}: {e}")
    if BENCHMARK not in series:
        raise RuntimeError("Benchmark VT unavailable; cannot make a fair comparison.")
    assets = [t for t in UNIVERSE if t in series]
    if not assets: raise RuntimeError("No strategy assets available.")
    close = pd.concat([series[t] for t in assets], axis=1).sort_index()
    bench = series[BENCHMARK]
    # Use dates on which benchmark and at least one strategy asset have prices.
    common = close.join(bench.rename(BENCHMARK), how="inner").dropna(subset=[BENCHMARK])
    close = common[assets]
    bench = common[BENCHMARK]
    fast = close.rolling(50, min_periods=50).mean()
    slow = close.rolling(150, min_periods=150).mean()
    momentum = close / close.shift(63) - 1
    vol20 = close.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    # Decisions from prior close; returns start at next close to avoid same-close look-ahead.
    daily_ret = close.pct_change().fillna(0)
    bench_ret = bench.pct_change().fillna(0)
    positions = pd.Series("CASH", index=close.index, dtype=object)
    current = "CASH"
    for i in range(1, len(close)):
        d = close.index[i-1]
        if close.index[i].month != d.month or close.index[i].year != d.year:
            eligible = []
            for t in assets:
                vals = [close.at[d,t], fast.at[d,t], slow.at[d,t], momentum.at[d,t], vol20.at[d,t]]
                if all(pd.notna(v) for v in vals):
                    # Added regime/risk filter: positive momentum, trend confirmation,
                    # and avoid assets with annualized realized volatility above 45%.
                    if close.at[d,t] > slow.at[d,t] and fast.at[d,t] > slow.at[d,t] and momentum.at[d,t] > 0 and vol20.at[d,t] <= 0.45:
                        eligible.append((t, float(momentum.at[d,t])))
            current = max(eligible, key=lambda x:x[1])[0] if eligible else "CASH"
        positions.iloc[i] = current
    gross = pd.Series(0.0, index=close.index)
    for t in assets:
        gross.loc[positions.eq(t)] = daily_ret.loc[positions.eq(t), t]
    changes = positions.ne(positions.shift()).astype(float)
    changes.iloc[0] = 0
    # A switch out and into a different asset incurs two sides; entry/exit one side.
    prev = positions.shift().fillna("CASH")
    sides = ((prev != positions).astype(int) * ((prev != "CASH").astype(int)+(positions != "CASH").astype(int)))
    costs = sides * COST_PER_SIDE
    strat_ret = (1+gross)*(1-costs)-1
    strat_eq = CASH*(1+strat_ret).cumprod()
    bench_eq = CASH*(1+bench_ret).cumprod()
    frame = pd.DataFrame({"position":positions,"strategy_return":strat_ret,"strategy_equity":strat_eq,
                          "estimated_cost":costs,"benchmark_return":bench_ret,"benchmark_equity":bench_eq})
    frame.to_csv(OUT/"improved_daily_equity.csv", index_label="date")
    frame.loc[positions.ne(positions.shift())].to_csv(OUT/"improved_daily_trades.csv", index_label="date")
    summary = {
      "mode":"PAPER_BACKTEST_ONLY","real_orders_placed":False,"broker_connected":False,
      "initial_simulated_cash":CASH,"benchmark":BENCHMARK,
      "period_start":str(frame.index[0].date()),"period_end":str(frame.index[-1].date()),
      "strategy":metrics(strat_eq,strat_ret,CASH),"benchmark_results":metrics(bench_eq,bench_ret,CASH),
      "assets_available":assets,"rules":{"rebalance":"monthly","trend":"50-day SMA > 150-day SMA and close > 150-day SMA",
      "momentum":"63 trading days > 0","risk_filter":"20-day annualized realized volatility <= 45%",
      "cash_when_no_candidate":True,"estimated_cost_per_side_pct":COST_PER_SIDE*100},
      "limitations":["Public Yahoo Finance adjusted daily prices; may be revised or have gaps.",
      "The volatility filter is a hypothesis, not proven to improve future returns.",
      "No taxes, fund expense ratios, market impact or cash interest modeled.",
      "Historical backtest only; no live data feed or order execution."]
    }
    (OUT/"improved_daily_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
