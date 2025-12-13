import importlib
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from freezegun import freeze_time

import Trading
from config import analysis_settings as settings


def test_closes_from_ts_filters_nan():
    payload = {"raw": {"values": [{"close": "nan"}, {"close": "1.23"}]}}
    assert Trading.closes_from_ts(payload) == [None, 1.23]


@freeze_time("2024-03-31 00:30:00", tz_offset=0)
def test_session_status_handles_dst_boundary(monkeypatch):
    settings.reload_config()
    monkeypatch.setattr(settings, "SESSION_STATUS_PROFILE_NAME", "weekend", raising=False)
    monkeypatch.setattr(settings, "SESSION_STATUS_PROFILE", settings.SESSION_STATUS_PROFILES.get("weekend", {}), raising=False)
    name, profile = settings.resolve_session_status_for_asset(
        "EURUSD", when=datetime.now(timezone.utc), weekday_ok=True
    )
    assert name == "default"
    assert profile.get("status") != "closed_weekend"


def test_fib_tolerance_relaxed_metals_profile(monkeypatch):
    monkeypatch.setenv("ACTIVE_ENTRY_PROFILE", "relaxed")
    settings.reload_config()
    refreshed = importlib.reload(settings)
    tol = refreshed.get_fib_tolerance("GOLD_CFD")
    assert tol == pytest.approx(0.014)
    settings.reload_config()


@freeze_time("2024-01-02 22:02:00", tz_offset=0)
def test_daily_break_blocks_with_cooldown_template(monkeypatch):
    from analysis import session_state

    settings.reload_config()
    now_dt = datetime(2024, 1, 2, 22, 2, tzinfo=timezone.utc)
    open_now, meta = session_state("EURUSD", now=now_dt)
    assert open_now is False
    assert meta.get("daily_break_active") is True
    assert meta.get("daily_break_window_utc") == ["22:00", "22:05"]
    template = settings.get_risk_template("EURUSD", settings.ENTRY_THRESHOLD_PROFILE_NAME)
    assert template.get("cooldown_minutes") == 20
