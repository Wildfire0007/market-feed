import importlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SentimentOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.prev_public_dir = os.environ.get("PUBLIC_DIR")
        os.environ["PUBLIC_DIR"] = self.temp_dir
        self.override_path = Path(self.temp_dir) / "overrides.json"
        self.prev_override = os.environ.get("SENTIMENT_OVERRIDE_FILE")
        os.environ["SENTIMENT_OVERRIDE_FILE"] = str(self.override_path)
        if "news_feed" in sys.modules:
            del sys.modules["news_feed"]
        importlib.invalidate_caches()
        self.news_feed = importlib.import_module("news_feed")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.prev_public_dir is not None:
            os.environ["PUBLIC_DIR"] = self.prev_public_dir
        else:
            os.environ.pop("PUBLIC_DIR", None)
        if self.prev_override is not None:
            os.environ["SENTIMENT_OVERRIDE_FILE"] = self.prev_override
        else:
            os.environ.pop("SENTIMENT_OVERRIDE_FILE", None)
        importlib.invalidate_caches()
        if "news_feed" in sys.modules:
            del sys.modules["news_feed"]
        importlib.import_module("news_feed")

    def test_override_signal_selected(self) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "defaults": {"severity": 0.5, "ttl_minutes": 90},
            "events": [
                {
                    "assets": ["EURUSD"],
                    "score": -0.65,
                    "bias": "eur_bearish",
                    "headline": "Flash PMI miss",
                    "category": "macro",
                    "severity": 0.9,
                    "start": (now - timedelta(minutes=10)).isoformat(),
                    "end": (now + timedelta(minutes=45)).isoformat(),
                }
            ],
        }
        with self.override_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

        self.news_feed.reload_sentiment_overrides()
        asset_dir = Path(self.temp_dir) / "EURUSD"
        asset_dir.mkdir(parents=True, exist_ok=True)

        signal = self.news_feed.load_sentiment("EURUSD", asset_dir)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertAlmostEqual(signal.score, -0.65)
        self.assertEqual(signal.bias, "eur_bearish")
        self.assertEqual(signal.category, "macro")
        self.assertGreater(signal.effective_severity, 0.8)
        self.assertEqual(signal.source, self.override_path)
        self.assertFalse(signal.is_expired())


if __name__ == "__main__":
    unittest.main()
