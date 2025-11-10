import importlib
import sys
from datetime import datetime, timezone

import pytest


def _reload_analysis(monkeypatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


def test_nvda_session_open_after_us_dst_start(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    now = datetime(2024, 3, 11, 13, 45, tzinfo=timezone.utc)
    entry_open, info = analysis.session_state("NVDA", now=now)
    assert entry_open is True
    assert info["weekday_ok"] is True
    assert info["within_entry_window"] is True
    assert info["status"] == "open"


def test_nvda_session_closed_before_open_after_us_dst_end(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    before_open = datetime(2024, 11, 4, 13, 0, tzinfo=timezone.utc)
    entry_open, info = analysis.session_state("NVDA", now=before_open)
    assert entry_open is False
    assert info["status"] in {"closed_out_of_hours", "open_entry_limited"}

    after_open = datetime(2024, 11, 4, 14, 45, tzinfo=timezone.utc)
    entry_open_late, info_late = analysis.session_state("NVDA", now=after_open)
    assert entry_open_late is True
    assert info_late["status"] == "open"


def test_eurusd_sunday_open_around_eu_dst(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    before = datetime(2024, 3, 31, 20, 55, tzinfo=timezone.utc)
    entry_open_before, info_before = analysis.session_state("EURUSD", now=before)
    assert entry_open_before is False
    assert info_before["status"].startswith("closed")

    after = datetime(2024, 3, 31, 21, 10, tzinfo=timezone.utc)
    entry_open_after, info_after = analysis.session_state("EURUSD", now=after)
    assert entry_open_after is True
    assert info_after["status"] == "open"
