import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis
from news_feed import SentimentSignal


class TimeUtilityTests(unittest.TestCase):
    def test_min_of_day_bounds(self) -> None:
        self.assertEqual(analysis._min_of_day(13, 30), 13 * 60 + 30)
        self.assertEqual(analysis._min_of_day(-2, -15), 0)
        self.assertEqual(analysis._min_of_day(26, 90), 23 * 60 + 59)


class PositionManagementSentimentTests(unittest.TestCase):
    def test_sentiment_triggers_exit_signal(self) -> None:
        session_meta = {"open": True, "entry_open": False, "status": "open"}
        anchor_record = {
            "entry_price": 1.2010,
            "stop_loss": 1.1980,
            "initial_risk_abs": 0.0015,
            "p_score": 70,
        }
        sentiment = SentimentSignal(
            score=-0.8,
            bias="eur_bearish",
            headline="Fed surprise",
            expires_at=None,
            severity=0.9,
        )

        note, exit_signal = analysis.derive_position_management_note(
            asset="EURUSD",
            session_meta=session_meta,
            regime_ok=True,
            effective_bias="long",
            structure_flag="bos_up",
            atr1h=0.0008,
            anchor_bias="long",
            anchor_timestamp="2024-05-25T10:00:00Z",
            anchor_record=anchor_record,
            current_p_score=60.0,
            current_rel_atr=0.0009,
            current_atr5=0.0009,
            current_price=1.1995,
            invalid_level_buy=1.1985,
            invalid_level_sell=1.2050,
            invalid_buffer_abs=0.0005,
            current_signal="buy",
            sentiment_signal=sentiment,
        )

        self.assertIsNotNone(exit_signal)
        assert exit_signal is not None
        self.assertEqual(exit_signal.get("category"), "sentiment_risk")
        self.assertEqual(exit_signal.get("state"), "scale_out")
        actions = exit_signal.get("actions") or []
        self.assertTrue(any(action.get("type") == "scale_out" for action in actions))
        self.assertTrue(any(action.get("type") == "tighten_stop" for action in actions))
        self.assertIn("Sentiment", note or "")
        self.assertAlmostEqual(exit_signal.get("sentiment_score"), -0.8)
        self.assertGreater(exit_signal.get("sentiment_severity", 0.0), 0.8)
        self.assertEqual(exit_signal.get("direction"), "long")


if __name__ == "__main__":
    unittest.main()
