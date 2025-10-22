import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis  # noqa: E402


class IntradayProfileTests(unittest.TestCase):
    def _make_dataframe(self, start_price: float, increments: np.ndarray) -> pd.DataFrame:
        base_time = datetime(2024, 6, 5, 7, 0, tzinfo=timezone.utc)
        index = pd.date_range(base_time, periods=len(increments), freq="1min", tz=timezone.utc)
        opens = start_price + np.concatenate(([0.0], increments[:-1]))
        closes = start_price + increments
        highs = np.maximum(opens, closes) + 0.0003
        lows = np.minimum(opens, closes) - 0.0003
        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }, index=index)

    def test_intraday_exhaustion_long_flags(self) -> None:
        increments = np.linspace(0.0, 0.0045, 180)
        df = self._make_dataframe(1.1000, increments)
        now = df.index[-1]
        profile = analysis.compute_intraday_profile(
            asset="EURUSD",
            k1m=df,
            price_now=1.1048,
            atr5=0.0025,
            now=now,
        )

        self.assertTrue(profile.get("range_exhaustion_long"))
        self.assertGreaterEqual(profile.get("range_position") or 0.0, analysis.INTRADAY_EXHAUSTION_PCT)
        self.assertEqual(profile.get("range_bias"), "upper")
        guard = profile.get("range_guard", {})
        self.assertTrue(guard.get("long"))
        self.assertIn("felső", " ".join(profile.get("notes", [])))

    def test_intraday_compression_detected(self) -> None:
        increments = np.linspace(0.0, 0.0005, 90)
        df = self._make_dataframe(1.2000, increments)
        now = df.index[-1]
        profile = analysis.compute_intraday_profile(
            asset="EURUSD",
            k1m=df,
            price_now=1.2002,
            atr5=0.0025,
            now=now,
        )

        self.assertTrue(profile.get("range_compression"))
        self.assertEqual(profile.get("range_state"), "compression")
        self.assertLess(profile.get("range_vs_atr") or 1.0, analysis.INTRADAY_COMPRESSION_TH + 1e-6)
        notes = " ".join(profile.get("notes", []))
        self.assertIn("Napi range", notes)


if __name__ == "__main__":
    unittest.main()
