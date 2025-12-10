from datetime import datetime, timezone
from unittest import mock

from config import analysis_settings as settings


def test_schedule_default_bucket_applied():
    schedule = {"default": {"open": "suppressed", "mid": "baseline", "close": "suppressed"}}
    with mock.patch.object(settings, "_ENTRY_PROFILE_SCHEDULE", schedule):
        with mock.patch("config.analysis_settings._current_tod_bucket", return_value="open"):
            profile = settings.get_entry_threshold_profile_name_for_asset("EURUSD")
    assert profile == "suppressed"


def test_schedule_asset_override_wins():
    schedule = {
        "default": {"open": "suppressed"},
        "assets": {"BTCUSD": {"open": "baseline"}},
    }
    with mock.patch.object(settings, "_ENTRY_PROFILE_SCHEDULE", schedule):
        with mock.patch("config.analysis_settings._current_tod_bucket", return_value="open"):
            profile = settings.get_entry_threshold_profile_name_for_asset("BTCUSD")
    assert profile == "baseline"


def test_session_bucket_proxy_used_for_btc(monkeypatch):
    # NVDA session proxy should suppress BTC buckets before the U.S. open.
    early_us = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)
    bucket = settings._session_bucket("BTCUSD", early_us)  # pylint: disable=protected-access
    assert bucket is None

    us_open = datetime(2025, 1, 1, 14, 45, tzinfo=timezone.utc)
    bucket = settings._session_bucket("BTCUSD", us_open)  # pylint: disable=protected-access
    assert bucket == "open"


def test_btc_intraday_schedule_follows_session_bucket(monkeypatch):
    active_mid = datetime(2025, 1, 1, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(settings, "_now_utc", lambda: active_mid)

    profile = settings.get_entry_threshold_profile_name_for_asset("BTCUSD")

    assert profile == "intraday"
