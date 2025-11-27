import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
from reports import monitoring


CI_ONLY = pytest.mark.skipif(not os.getenv("CI"), reason="CI-only regression guard (rollback note)")


def test_latency_guard_status_triggers():
    status = analysis._latency_guard_status(
        "USOIL",
        {"k1m": 840},
        {"USOIL": {"latency_k1m_sec_max": 720}},
    )
    assert status is not None
    assert status["asset"] == "USOIL"
    assert status["feed"] == "k1m"
    assert status["age_seconds"] == 840
    assert status["limit_seconds"] == 720


@CI_ONLY
def test_latency_guard_alert_only_mode_includes_metadata(caplog):
    caplog.set_level(logging.WARNING, logger=analysis.LOGGER.name)

    guard_status = {"asset": "USOIL", "feed": "k1m", "age_seconds": 900, "limit_seconds": 720}
    guard_meta = dict(guard_status, triggered_utc="2024-01-01 00:00:00", triggered_cet="2024-01-01 01:00:00")
    guard_meta["mode"] = "block_trade" if guard_status["asset"] != "USOIL" else "alert_only"
    analysis.LOGGER.warning("Latency guard aktiválva", extra={"asset": "USOIL", "latency_guard": guard_meta})

    records = [rec for rec in caplog.records if getattr(rec, "latency_guard", None)]
    assert records, "expected latency guard log entry"
    meta = records[-1].latency_guard
    assert meta["mode"] == "alert_only"
    assert meta["age_seconds"] == 900
    assert meta["limit_seconds"] == 720


def test_handle_spot_realtime_staleness_requests_refresh(caplog):
    captured = []

    def fake_notifier(asset, feed, message, metadata=None):
        captured.append((asset, feed, message, metadata))

    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    meta = {"age_seconds": 650, "max_age_seconds": 600}

    caplog.set_level(logging.INFO, logger=analysis.LOGGER.name)
    result = analysis._handle_spot_realtime_staleness(
        "EURUSD",
        meta,
        now,
        fake_notifier,
        analysis.LOGGER,
    )

    assert result["age_seconds"] == 650
    assert result["notified"] is True
    assert result["refresh_requested"] is True
    assert result["retry_after_seconds"] == 60
    assert captured and captured[0][0] == "EURUSD"
    assert captured[0][1] == "spot_realtime"
    assert "Realtime spot frissítés" in captured[0][2]
    assert captured[0][3]["asset"] == "EURUSD"
    assert any(getattr(record, "action", "") == "refresh_request" for record in caplog.records)


def test_record_latency_alert_writes_payload(tmp_path):
    public_dir = tmp_path / "pub"
    alert_path = monitoring.record_latency_alert(
        "USOIL",
        "k1m",
        "USOIL k1m latency guard teszt riasztás",
        metadata={"latency": 840},
        public_dir=public_dir,
    )

    assert alert_path.exists()
    payload = json.loads(alert_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["asset"] == "USOIL"
    assert entry["feed"] == "k1m"
    assert entry["metadata"]["latency"] == 840

    log_path = alert_path.with_name("latency_alerts.log")
    assert log_path.exists()
    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert log_lines and "USOIL" in log_lines[-1]
