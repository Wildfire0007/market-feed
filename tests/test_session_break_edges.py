import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reload_analysis(monkeypatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


def test_empty_session_window_keeps_market_open(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    monkeypatch.setitem(analysis.SESSION_WINDOWS_UTC, "EMPTY_ASSET", {"entry": [], "monitor": []})

    now = datetime(2024, 5, 1, 10, 0, tzinfo=timezone.utc)
    entry_open, info = analysis.session_state("EMPTY_ASSET", now=now)

    assert entry_open is True
    assert info["status"] in {"open", "open_entry_limited"}


def test_daily_break_wraps_from_sixty_to_midnight(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    monkeypatch.setitem(analysis.SESSION_TIME_RULES, "FAKE_DST", {"daily_breaks": [(60, 0)]})
    monkeypatch.setitem(analysis.RULE_TIMEZONES, "FAKE_DST", analysis.BUDAPEST_TIMEZONE)
    monkeypatch.setitem(analysis.SESSION_WINDOWS_UTC, "FAKE_DST", {"entry": [(0, 0, 23, 59)]})

    during_break = datetime(2024, 3, 31, 0, 30, tzinfo=timezone.utc)
    open_during, info_during = analysis.session_state("FAKE_DST", now=during_break)
    assert open_during is False
    assert info_during.get("status") == "maintenance"
    window = info_during.get("daily_break_window_utc")
    assert window is not None
    assert len(window) == 2


def test_zero_spread_is_allowed(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    snapshot = {"bid": 100.0, "ask": 100.0}
    assert analysis._extract_bid_ask_spread(snapshot) == pytest.approx(0.0)


def test_nan_atr_input_is_rejected(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    assert analysis.btc_atr_gate_ok("baseline", "BTCUSD", np.nan, datetime.now(timezone.utc)) is False
