import importlib
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ProbabilityFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.prev_public_dir = os.environ.get("PUBLIC_DIR")
        os.environ["PUBLIC_DIR"] = self.temp_dir
        if "ml_model" in sys.modules:
            del sys.modules["ml_model"]
        importlib.invalidate_caches()
        self.ml_model = importlib.import_module("ml_model")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.prev_public_dir is not None:
            os.environ["PUBLIC_DIR"] = self.prev_public_dir
        else:
            os.environ.pop("PUBLIC_DIR", None)
        importlib.invalidate_caches()
        if "ml_model" in sys.modules:
            del sys.modules["ml_model"]
        importlib.import_module("ml_model")

    def test_fallback_probability_available(self) -> None:
        features = {name: 0.0 for name in self.ml_model.MODEL_FEATURES}
        features.update(
            {
                "p_score": 62.0,
                "rel_atr": 0.0012,
                "ema21_slope": 0.045,
                "bias_long": 1.0,
                "bias_short": 0.0,
                "momentum_vol_ratio": 1.15,
                "order_flow_imbalance": 0.7,
                "order_flow_pressure": 0.5,
                "order_flow_aggressor": 0.4,
                "news_sentiment": 0.4,
                "news_event_severity": 0.85,
                "realtime_confidence": 0.8,
                "volatility_ratio": 1.1,
                "volatility_regime_flag": 1.0,
                "precision_score": 78.0,
                "precision_trigger_ready": 1.0,
                "precision_trigger_arming": 0.0,
                "precision_trigger_fire": 1.0,
                "precision_trigger_confidence": 0.9,
                "precision_order_flow_ready": 1.0,
                "structure_flip_flag": 1.0,
                "momentum_trail_activation_rr": 1.6,
                "momentum_trail_lock_ratio": 0.55,
            }
        )

        result = self.ml_model.predict_signal_probability("EURUSD", features)

        self.assertIsNotNone(result.probability)
        self.assertGreater(result.probability or 0.0, 0.5)
        self.assertAlmostEqual(result.raw_probability or 0.0, 0.62, delta=0.05)
        self.assertIsNotNone(result.threshold)
        self.assertEqual(result.metadata.get("source"), "fallback")
        self.assertEqual(result.metadata.get("unavailable_reason"), "model_missing")
        fallback_meta = result.metadata.get("fallback") or {}
        self.assertIn("steps", fallback_meta)
        self.assertGreater(len(fallback_meta.get("steps", [])), 0)


if __name__ == "__main__":
    unittest.main()
