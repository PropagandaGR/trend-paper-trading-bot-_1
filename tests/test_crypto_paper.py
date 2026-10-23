import tempfile, unittest
from pathlib import Path
import pandas as pd
import crypto_paper_bot as bot
import crypto_research as research

class CryptoPaperTests(unittest.TestCase):
    def test_signal_rejects_missing_history(self):
        row=pd.Series({"ema_fast":float("nan"),"ema_slow":1,"mom_3":1,"rsi":30,
            "range_high":2,"range_low":1,"volume_ratio":1,"atr_pct":0.01})
        self.assertFalse(bot.strategy_signal(row,"trend"))

    def test_spot_position_bounded_by_cash_and_long_only(self):
        state=bot.PaperState()
        old=bot.COST_BPS
        try:
            bot.COST_BPS=10
            bot.buy(state,"BTCUSDT",100.0,"2026-10-18T00:00:00Z","trend")
            self.assertIsNotNone(state.position)
            self.assertGreater(state.position["quantity"],0)
            self.assertLessEqual(state.position["quantity"]*state.position["entry_price"],15.000001)
            self.assertEqual(state.cash,0.0)
            bot.sell(state,99.0,"2026-10-18T00:05:00Z","test")
            self.assertIsNone(state.position)
            self.assertLess(state.cash,15.0)
        finally: bot.COST_BPS=old

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"state.json"
            bot.save_json(p,bot.asdict(bot.PaperState()))
            self.assertEqual(bot.load_state(p).cash,15.0)

    def test_backtest_delays_signal_and_costs_reduce_returns(self):
        idx=pd.date_range("2026-01-01",periods=1200,freq="5min",tz="UTC")
        close=pd.Series([100+i*0.01 for i in range(len(idx))],index=idx)
        df=pd.DataFrame({"Open":close,"High":close*1.001,"Low":close*.999,"Close":close,"Volume":1000.0})
        gross,pos=research.backtest(df,"trend",0)
        costly,_=research.backtest(df,"trend",20)
        self.assertEqual(pos.iloc[0],0)
        self.assertLessEqual(float(costly.sum()),float(gross.sum())+1e-9)


    def test_rsi_nan_does_not_disable_trend_or_breakout(self):
        row=pd.Series({"Close":105.0,"ema_fast":104.0,"ema_slow":103.0,"mom_3":0.01,"rsi":float("nan"),"range_high":104.0,"range_low":100.0,"volume_ratio":1.5,"atr_pct":0.01})
        self.assertTrue(bot.strategy_signal(row,"trend"))
        self.assertTrue(bot.strategy_signal(row,"breakout"))
        self.assertFalse(bot.strategy_signal(row,"mean_reversion"))

    def test_mean_reversion_has_its_own_exit(self):
        position={"strategy":"mean_reversion"}
        row=pd.Series({"Close":101.0,"ema_fast":100.0,"ema_slow":99.0,"mom_3":0.01,"rsi":52.0})
        should_exit,reason=bot.should_exit(position,row)
        self.assertTrue(should_exit)
        self.assertEqual(reason,"mean_reversion_rsi_normalized")


    def test_run_once_processes_unseen_candles_per_symbol(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            idx=pd.date_range("2026-10-18T00:00:00Z",periods=50,freq="5min")
            close=pd.Series([100.0+i*0.1 for i in range(50)],index=idx)
            bars=pd.DataFrame({"Open":close,"High":close*1.002,"Low":close*0.998,"Close":close,"Volume":1000.0},index=idx)
            cursor=idx[-4].isoformat()
            state={"cash":15.0,"position":None,"last_processed_bar":cursor,"last_processed_bars":{s:cursor for s in bot.SYMBOLS},"trades":[],"peak_equity":15.0}
            state_path=root/"state.json"
            report_path=root/"report.json"
            bot.save_json(state_path,state)
            with patch.object(bot,"closed_bars",return_value=bars):
                with patch.object(bot,"strategy_signal",return_value=False) as signal:
                    report=bot.run_once(state_path,report_path)
            self.assertEqual(signal.call_count,27)
            saved=bot.load_state(state_path)
            for symbol in bot.SYMBOLS:
                self.assertEqual(saved.last_processed_bars[symbol],idx[-1].isoformat())
            self.assertEqual(report["equity"],15.0)
            self.assertEqual(report["trade_count"],0)

if __name__=="__main__": unittest.main()
