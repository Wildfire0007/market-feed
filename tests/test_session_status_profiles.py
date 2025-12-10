import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _reload_for_status_profile(
    monkeypatch, profile: Optional[str], forced_now: Optional[datetime] = None
):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config.analysis_settings as settings_module

    if profile is None:
        monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("SESSION_STATUS_PROFILE", profile)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)

    settings_module.reload_config()
    settings = importlib.reload(settings_module)

    if forced_now is not None:
        monkeypatch.setattr(settings, "_now_utc", lambda: forced_now)
        settings.refresh_session_status_profile()

    analysis_module = sys.modules.get("analysis")
    if analysis_module is None:
        analysis_module = importlib.import_module("analysis")
    else:
        analysis_module = importlib.reload(analysis_module)

    return settings, analysis_module


def test_weekend_session_status_profile(monkeypatch):
    fake_now = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)
    settings, analysis_module = _reload_for_status_profile(
        monkeypatch, None, forced_now=fake_now
    )
    try:
        assert settings.SESSION_STATUS_PROFILE_NAME == "weekend"
        profile_name, profile = settings.resolve_session_status_for_asset("EURUSD")
        assert profile_name == "weekend"
        assert profile.get("force_session_closed") is True

        entry_open, info = analysis_module.session_state("EURUSD", now=fake_now)

        assert entry_open is False
        assert info["status"] == "closed_weekend"
        assert "Piac zárva" in info["status_note"]
        assert info["status_profile"] == "weekend"
        assert info.get("market_closed_assumed") is True
        assert info.get("market_closed_reason") == "weekend"
        assert "Weekend profil" in " ".join(info.get("notes", []))
        assert "weekend" in info.get("status_profile_tags", [])
    finally:
        _reload_for_status_profile(monkeypatch, None)


def test_weekend_profile_skips_non_target_assets(monkeypatch):
    fake_now = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)
    settings, analysis_module = _reload_for_status_profile(
        monkeypatch, None, forced_now=fake_now
    )
    try:
        assert settings.SESSION_STATUS_PROFILE_NAME == "weekend"
        name, profile = settings.resolve_session_status_for_asset("BTCUSD")
        assert name == "default"
        assert profile.get("force_session_closed") is not True

        entry_open, info = analysis_module.session_state("BTCUSD", now=fake_now)
        assert entry_open is True
        assert info["status_profile"] == "default"
        assert info.get("market_closed_reason") != "weekend"
    finally:
        _reload_for_status_profile(monkeypatch, None)


def test_default_profile_sets_outside_hours_reason(monkeypatch):
    fake_now = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)  # hétfő, nyitás előtt
    settings, analysis_module = _reload_for_status_profile(
        monkeypatch, None, forced_now=fake_now
    )
    try:
        assert settings.SESSION_STATUS_PROFILE_NAME == "default"

        entry_open, info = analysis_module.session_state("NVDA", now=fake_now)

        assert entry_open is False
        assert info["status"] == "closed_out_of_hours"
        assert info.get("market_closed_reason") == "outside_hours"
    finally:
        _reload_for_status_profile(monkeypatch, None)


def test_invalid_session_status_profile_fallback(monkeypatch):
    # Időfüggetlenítés: fagyasszuk hétfőre az "aktuális" időt, hogy a weekend auto-profil ne befolyásolja a fallbacket.
    fake_weekday = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)  # hétfő
    settings, _ = _reload_for_status_profile(
        monkeypatch, "does-not-exist", forced_now=fake_weekday
    )
    try:
        assert settings.SESSION_STATUS_PROFILE_NAME == "default"
    finally:
        _reload_for_status_profile(monkeypatch, None)
