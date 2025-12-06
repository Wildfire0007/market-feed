import os
import sys
import time
from unittest import mock

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts import notify_discord


class FakeResponse:
    def __init__(self, status: int, headers=None):
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_post_batches_retries_on_rate_limit():
    first = FakeResponse(429, headers={"Retry-After": "0.1"})
    second = FakeResponse(200)
    with mock.patch("scripts.notify_discord.time.sleep") as sleep_mock, \
         mock.patch.dict(notify_discord._WEBHOOK_COOLDOWN_UNTIL, {}, clear=True), \
         mock.patch("scripts.notify_discord.requests.post", side_effect=[first, second]) as post_mock:
        notify_discord.post_batches("https://hook.example", "ping", [{}])
        assert post_mock.call_count == 2
        sleep_mock.assert_called()  # backoff applied
        assert notify_discord._WEBHOOK_COOLDOWN_UNTIL


def test_post_batches_respects_cooldown_skip():
    with mock.patch("scripts.notify_discord.time.time", return_value=100.0):
        notify_discord._WEBHOOK_COOLDOWN_UNTIL["h"] = 200.0
        with mock.patch("scripts.notify_discord.requests.post") as post_mock:
            notify_discord.post_batches("h", "ping", [{}])
    post_mock.assert_not_called()
