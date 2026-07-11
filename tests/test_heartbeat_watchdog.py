import json
import sys
from pathlib import Path

from freezegun import freeze_time

from scripts import heartbeat_watchdog


def _write_heartbeat(tmp_path: Path, stamp: str) -> Path:
    path = tmp_path / "system_heartbeat.json"
    path.write_text(json.dumps({"last_update_utc": stamp}), encoding="utf-8")
    return path


def _run(monkeypatch, tmp_path: Path, heartbeat: Path) -> tuple[int, list[dict]]:
    calls = []

    class Response:
        status_code = 204
        text = ""

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(heartbeat_watchdog.requests, "post", fake_post)
    monkeypatch.setattr(heartbeat_watchdog._webhook_log_response, "__call__", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL_ACTIONABLE", "https://example.test/actionable")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL_DIAGNOSTIC", "https://example.test/diagnostic")
    monkeypatch.setattr(
        heartbeat_watchdog.settings,
        "WATCHDOG",
        {"quiet_hours_utc": ["22:00", "05:00"], "quiet_hours_max_age_min": 150, "quiet_hours_channel": "diagnostic", "weekend_as_quiet": True},        
    )
    monkeypatch.setattr(sys, "argv", ["heartbeat_watchdog.py", "--heartbeat", str(heartbeat), "--public-dir", str(tmp_path), "--max-age-min", "30"])
    return heartbeat_watchdog.main(), calls


@freeze_time("2026-07-07T01:17:00Z")
def test_quiet_hours_routes_to_diagnostic_with_higher_threshold(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-07T00:19:00Z")

    code, calls = _run(monkeypatch, tmp_path, heartbeat)

    assert code == 0
    assert calls == []


@freeze_time("2026-07-07T01:17:00Z")
def test_position_override_uses_actionable_normal_threshold(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-07T00:46:00Z")
    (tmp_path / "_position_lifecycle_state.json").write_text(
        json.dumps({"positions": {"GOLD_CFD": {"status": "open"}}}), encoding="utf-8"
    )

    code, calls = _run(monkeypatch, tmp_path, heartbeat)

    assert code == 1
    assert calls[0]["url"] == "https://example.test/actionable"
    description = calls[0]["json"]["embeds"][0]["description"]
    assert "Csendes időszak: `éjszaka`" in description    
    assert "Open/pending positions: `1`" in description
    assert "Europe/Budapest" in description


@freeze_time("2026-07-11T09:00:00Z")
def test_weekend_quiet_suppresses_below_quiet_threshold(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-11T08:20:00Z")

    code, calls = _run(monkeypatch, tmp_path, heartbeat)

    assert code == 0
    assert calls == []


@freeze_time("2026-07-11T09:00:00Z")
def test_weekend_quiet_routes_stale_to_diagnostic(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-11T06:20:00Z")

    code, calls = _run(monkeypatch, tmp_path, heartbeat)

    assert code == 1
    assert calls[0]["url"] == "https://example.test/diagnostic"
    description = calls[0]["json"]["embeds"][0]["description"]
    assert "Csendes időszak: `hétvége`" in description


@freeze_time("2026-07-11T09:00:00Z")
def test_weekend_position_override_uses_actionable_normal_threshold(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-11T08:29:00Z")
    (tmp_path / "_position_lifecycle_state.json").write_text(
        json.dumps({"positions": {"GOLD_CFD": {"status": "open"}}}), encoding="utf-8"
    )

    code, calls = _run(monkeypatch, tmp_path, heartbeat)

    assert code == 1
    assert calls[0]["url"] == "https://example.test/actionable"
    description = calls[0]["json"]["embeds"][0]["description"]
    assert "Csendes időszak: `hétvége`" in description
    assert "Open/pending positions: `1`" in description


@freeze_time("2026-07-07T10:00:00Z")
def test_band_dedup_sends_first_and_hourly_edges(monkeypatch, tmp_path):
    heartbeat = _write_heartbeat(tmp_path, "2026-07-07T09:29:00Z")
    code, calls = _run(monkeypatch, tmp_path, heartbeat)
    assert code == 1
    assert len(calls) == 1

    heartbeat = _write_heartbeat(tmp_path, "2026-07-07T09:20:00Z")
    code, calls = _run(monkeypatch, tmp_path, heartbeat)
    assert code == 0
    assert calls == []

    heartbeat = _write_heartbeat(tmp_path, "2026-07-07T08:28:00Z")
    code, calls = _run(monkeypatch, tmp_path, heartbeat)
    assert code == 1
    assert len(calls) == 1
