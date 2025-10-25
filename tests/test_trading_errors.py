import os
import sys
import unittest
from typing import Any, Dict
from unittest import mock

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import Trading


class DummyResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code and self.status_code >= 400:
            raise requests.HTTPError(response=self)


class TDGetErrorTests(unittest.TestCase):
    def test_td_get_raises_on_error_payload(self):
        response = DummyResponse({"code": 401, "message": "Invalid API key"})
        with mock.patch("Trading.requests.get", return_value=response) as mock_get, \
             mock.patch("Trading.time.sleep"), \
             mock.patch("Trading.TD_MAX_RETRIES", 1), \
             mock.patch.object(Trading.TD_RATE_LIMITER, "wait", return_value=None), \
             mock.patch.object(Trading.TD_RATE_LIMITER, "record_failure") as record_failure, \
             mock.patch.object(Trading.TD_RATE_LIMITER, "record_success") as record_success, \
             mock.patch("Trading.API_KEY", "demo"):
            with self.assertRaises(Trading.TDError) as ctx:
                Trading.td_get("quote", symbol="EUR/USD")

        self.assertIn("Invalid API key", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 401)
        mock_get.assert_called_once()
        record_failure.assert_called()
        record_success.assert_not_called()


class CollectHttpFramesTests(unittest.TestCase):
    def test_collect_http_frames_breaks_after_repeated_failures(self):
        symbol_cycle = [("EUR/USD", "FX")]

        class TimeStub:
            def __init__(self):
                self.current = 0.0

            def time(self):
                value = self.current
                self.current += 0.5
                return value

        time_stub = TimeStub()

        td_error = Trading.TDError("not found", status_code=404)

        with mock.patch("Trading.td_quote", side_effect=td_error) as mock_quote, \
             mock.patch("Trading.time.sleep") as sleep_mock, \
             mock.patch("Trading.time.time", side_effect=time_stub.time):
            frames, abort_reason = Trading._collect_http_frames(
                symbol_cycle,
                deadline=10.0,
                interval=0.5,
                max_samples=3,
                force=True,
            )

        self.assertEqual(frames, [])
        self.assertEqual(abort_reason, "client_error_404")
        self.assertLessEqual(mock_quote.call_count, 2)
        sleep_mock.assert_called_once()


class CollectRealtimeSpotTests(unittest.TestCase):
    def test_force_mode_uses_short_http_window(self):
        calls: Dict[str, Any] = {}

        def fake_http(symbol_cycle, deadline, interval, max_samples, force=False):
            calls.update(
                {
                    "symbol_cycle": symbol_cycle,
                    "deadline": deadline,
                    "interval": interval,
                    "max_samples": max_samples,
                    "force": force,
                }
            )
            frames = [
                {
                    "price": 1.2345,
                    "utc": "2024-01-01T00:00:00Z",
                    "retrieved_at_utc": "2024-01-01T00:00:01Z",
                }
            ]
            return frames, None

        with mock.patch("Trading.REALTIME_FLAG", False), \
             mock.patch("Trading.REALTIME_WS_ENABLED", False), \
             mock.patch("Trading.ensure_dir"), \
             mock.patch("Trading.save_json"), \
             mock.patch("Trading._collect_http_frames", side_effect=fake_http) as http_mock, \
             mock.patch("Trading.time.sleep"), \
             mock.patch("Trading.time.time", return_value=1000.0):
            Trading.collect_realtime_spot(
                "EURUSD",
                [("EUR/USD", "FX")],
                out_dir="/tmp/out",
                force=True,
                reason="spot_fallback",
            )

        http_mock.assert_called_once()
        self.assertLessEqual(calls["max_samples"], 2)
        self.assertLessEqual(calls["interval"], 2.0)
        self.assertTrue(calls["force"])


class TDQuotePriceExtractionTests(unittest.TestCase):
    def test_td_quote_uses_close_when_price_missing(self):
        payload = {
            "close": "1.2345",
            "datetime": "2025-10-22 16:35:00",
        }

        with mock.patch("Trading.td_get", return_value=payload), \
             mock.patch("Trading.now_utc", return_value="2025-10-22T16:40:00Z"):
            result = Trading.td_quote("EUR/USD")

        self.assertTrue(result["ok"])
        self.assertEqual(result["price"], 1.2345)
        self.assertEqual(result["price_usd"], 1.2345)
        self.assertEqual(result["utc"], "2025-10-22T16:35:00+00:00")

    def test_td_quote_rejects_nan_prices(self):
        payload = {
            "price": "NaN",
            "close": None,
            "datetime": "2025-10-22 16:35:00",
        }

        with mock.patch("Trading.td_get", return_value=payload), \
             mock.patch("Trading.now_utc", return_value="2025-10-22T16:40:00Z"):
            result = Trading.td_quote("EUR/USD")

        self.assertFalse(result["ok"])
        self.assertIsNone(result["price"])
        self.assertIsNone(result["price_usd"])
        self.assertEqual(result["utc"], "2025-10-22T16:35:00+00:00")


if __name__ == "__main__":
    unittest.main()
