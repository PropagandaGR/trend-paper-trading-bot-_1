import unittest

import pandas as pd

from rapid_paper_trader import prepare_signals, simulate


class RapidPaperTraderTests(unittest.TestCase):
    def sample_data(self):
        # Deterministic, non-flat minute OHLC data without network access.
        idx = pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC")
        close = [100 + i * 0.1 for i in range(80)]
        open_ = [x - 0.02 for x in close]
        return pd.DataFrame({"Open": open_, "Close": close}, index=idx)

    def test_signal_is_based_on_close_history(self):
        data = self.sample_data()
        signal = prepare_signals(data)
        altered = data.copy()
        altered.iloc[-1, altered.columns.get_loc("Close")] = 10000
        changed = prepare_signals(altered)
        # A close at t cannot change signals already computed through t-1.
        pd.testing.assert_series_equal(signal.iloc[:-1], changed.iloc[:-1])

    def test_simulation_is_long_only_and_finite(self):
        curve, summary = simulate(self.sample_data(), 0.001)
        self.assertTrue(curve["strategy_equity"].map(lambda x: pd.notna(x) and x >= 0).all())
        self.assertTrue(set(curve["position"]).issubset({"SPY", "CASH"}))
        self.assertEqual(summary["initial_simulated_cash"], 15.0)
        self.assertGreaterEqual(summary["position_changes"], 0)

    def test_higher_cost_does_not_improve_same_strategy_result(self):
        data = self.sample_data()
        _, low = simulate(data, 0.0005)
        _, high = simulate(data, 0.002)
        self.assertLessEqual(high["strategy_final_equity"], low["strategy_final_equity"] + 1e-9)
        self.assertLessEqual(high["benchmark_final_equity"], low["benchmark_final_equity"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
