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

if __name__=="__main__": unittest.main()
