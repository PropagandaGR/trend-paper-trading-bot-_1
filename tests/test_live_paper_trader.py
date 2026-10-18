import tempfile, unittest
from pathlib import Path
import pandas as pd
import live_paper_trader as bot

class PaperTraderTests(unittest.TestCase):
    def test_signal_requires_trend_momentum_and_volume(self):
        row = pd.Series({"ema_fast":101.0,"ema_slow":100.0,"mom_3":0.003,"volume_ratio":1.2,"atr_pct":0.004})
        self.assertTrue(bot.signal(row)[0])
        row["mom_3"] = -0.001
        self.assertFalse(bot.signal(row)[0])

    def test_long_only_position_and_transaction_costs(self):
        state = bot.PaperState()
        old = bot.COST_PER_SIDE
        try:
            bot.COST_PER_SIDE = 0.001
            bot.open_position(state, "SPY", 100.0, "2026-10-18T10:00:00+00:00", "test")
            self.assertIsNotNone(state.position)
            self.assertGreater(state.position["quantity"], 0)
            self.assertLessEqual(state.position["quantity"] * state.position["entry_price"], 15.000001)
            self.assertEqual(state.cash, 0.0)
            bot.close_position(state, "SPY", 99.0, "2026-10-18T10:05:00+00:00", "test_exit")
            self.assertIsNone(state.position)
            self.assertLess(state.cash, 15.0)
        finally: bot.COST_PER_SIDE = old

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            state = bot.PaperState()
            bot.save_json(path, bot.asdict(state))
            loaded = bot.load_state(path)
            self.assertEqual(loaded.cash, 15.0)
            self.assertIsNone(loaded.position)

if __name__ == "__main__": unittest.main()
