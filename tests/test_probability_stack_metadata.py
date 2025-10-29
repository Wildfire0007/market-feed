import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ProbabilityStackMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure analysis reads artefacts from a temporary public directory to
        # avoid interacting with repository fixtures.
        self.temp_dir = tempfile.mkdtemp()
        self.prev_public_dir = os.environ.get("PUBLIC_DIR")
        os.environ["PUBLIC_DIR"] = self.temp_dir
        if "analysis" in sys.modules:
            del sys.modules["analysis"]
        importlib.invalidate_caches()
        self.analysis = importlib.import_module("analysis")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.prev_public_dir is not None:
            os.environ["PUBLIC_DIR"] = self.prev_public_dir
        else:
            os.environ.pop("PUBLIC_DIR", None)
        importlib.invalidate_caches()
        if "analysis" in sys.modules:
            del sys.modules["analysis"]
        importlib.import_module("analysis")

    def test_ensure_probability_metadata_defaults_source(self) -> None:
        helper = self.analysis.ensure_probability_metadata

        empty_payload = helper(None)
        self.assertEqual(empty_payload["source"], "sklearn")
        self.assertEqual(set(empty_payload.keys()), {"source"})

        explicit = helper({"source": "fallback", "status": "placeholder"})
        self.assertEqual(explicit["source"], "fallback")
        self.assertEqual(explicit["status"], "placeholder")

        missing_source = helper({"detail": "foo"})
        self.assertEqual(missing_source["source"], "sklearn")
        self.assertEqual(missing_source["detail"], "foo")

    def test_data_gap_signal_contains_probability_stack(self) -> None:
        result = self.analysis.build_data_gap_signal(
            asset="BTCUSD",
            spot_price=None,
            spot_utc="-",
            spot_retrieved="-",
            leverage=3.0,
            reasons=["Missing spot price"],
            display_spot=None,
            diagnostics={"latency_flags": []},
            session_meta=None,
        )

        stack = result.get("probability_stack") or {}
        self.assertEqual(stack.get("source"), "sklearn")
        self.assertEqual(stack.get("status"), "data_gap")
        self.assertEqual(stack.get("unavailable_reason"), "data_gap")
        self.assertEqual(result.get("probability_model_source"), "sklearn")


if __name__ == "__main__":
    unittest.main()
