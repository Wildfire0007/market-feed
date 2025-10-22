import unittest
from unittest import mock

import requests

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
            with self.assertRaises(RuntimeError) as ctx:
                Trading.td_get("quote", symbol="EUR/USD")

        self.assertIn("Invalid API key", str(ctx.exception))
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

        with mock.patch("Trading.td_quote", side_effect=RuntimeError("bad key")) as mock_quote, \
             mock.patch("Trading.time.sleep") as sleep_mock, \
             mock.patch("Trading.time.time", side_effect=time_stub.time):
            frames = Trading._collect_http_frames(symbol_cycle, deadline=10.0, interval=0.5, max_samples=3)

        self.assertEqual(frames, [])
        self.assertLessEqual(mock_quote.call_count, 2)
        sleep_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
