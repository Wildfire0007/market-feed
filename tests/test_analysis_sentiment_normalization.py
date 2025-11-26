import importlib
import os
import unittest
from collections import namedtuple

import analysis
from news_feed import SentimentSignal


class SentimentNormalizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("SENTIMENT_NORMALIZER_ROLLBACK", None)

    def test_accepts_tuple_and_namedtuple_inputs(self) -> None:
        raw_tuple = (-0.4, "bearish")
        named = namedtuple("NamedSignal", ["score", "bias"])(0.7, "btc_bullish")

        normalized_tuple = analysis._normalize_btcusd_sentiment(raw_tuple)  # type: ignore[arg-type]
        normalized_named = analysis._normalize_btcusd_sentiment(named)  # type: ignore[arg-type]

        self.assertAlmostEqual(normalized_tuple, 0.4)
        self.assertAlmostEqual(normalized_named, 0.7)

    def test_invalid_input_logs_and_defaults_to_neutral(self) -> None:
        class Dummy:
            pass

        with self.assertLogs(analysis.LOGGER, level="WARNING") as logs:
            normalized = analysis._normalize_btcusd_sentiment(Dummy())  # type: ignore[arg-type]

        self.assertEqual(normalized, 0.0)
        self.assertTrue(any(getattr(record, "sentiment_type_error", False) for record in logs.records))

    def test_rollback_flag_restores_original_behaviour(self) -> None:
        os.environ["SENTIMENT_NORMALIZER_ROLLBACK"] = "1"
        importlib.reload(analysis)

        raw_tuple = (-0.4, "bearish")
        with self.assertRaises(AttributeError):
            analysis._normalize_btcusd_sentiment(raw_tuple)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
