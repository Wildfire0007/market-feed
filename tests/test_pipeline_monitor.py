from __future__ import annotations

import json
import logging
import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(Path(__file__).resolve().parent.parent))

from freezegun import freeze_time

from reports import pipeline_monitor


def test_record_ml_model_status_reminder_cadence(tmp_path):
    monitor_path = tmp_path / "pipeline.json"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    payload, remind = pipeline_monitor.record_ml_model_status(
        missing=["eurusd"],
        placeholders=[],
        remind_after_days=7,
        path=monitor_path,
        now=start,
    )
    assert remind is True
    assert payload["ml_models"]["status"]["missing"] == ["EURUSD"]

    _, remind_second = pipeline_monitor.record_ml_model_status(
        missing=["eurusd"],
        placeholders=[],
        remind_after_days=7,
        path=monitor_path,
        now=start + timedelta(days=3),
    )
    assert remind_second is False

    _, remind_third = pipeline_monitor.record_ml_model_status(
        missing=["eurusd"],
        placeholders=[],
        remind_after_days=7,
        path=monitor_path,
        now=start + timedelta(days=8),
    )
    assert remind_third is True


def test_record_ml_model_status_resets_payload(tmp_path):
    monitor_path = tmp_path / "pipeline.json"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    pipeline_monitor.record_ml_model_status(
        missing=["eurusd"],
        placeholders=["xagusd"],
        remind_after_days=3,
        path=monitor_path,
        now=start,
    )

    payload, remind = pipeline_monitor.record_ml_model_status(
        missing=[],
        placeholders=[],
        remind_after_days=3,
        path=monitor_path,
        now=start + timedelta(days=4),
    )
    assert remind is False
    assert payload["ml_models"]["status"]["missing"] == []
    assert payload["ml_models"]["status"]["placeholders"] == []


def test_summarize_pipeline_warnings_detects_exceptions(tmp_path):
    log_path = tmp_path / "pipeline.log"
    log_path.write_text(
        "\n".join(
            [
                "2024-06-01 10:00:00,000Z WARNING Slow response from feed",
                "2024-06-01 10:01:00,000Z WARNING Client error 404 while fetching feed for EURUSD",
                "2024-06-01 10:01:30,000Z WARNING Rate limit 429 from provider for BTCUSD",
                "2024-06-01 10:02:00,000Z ERROR Failed to compute metrics",
                "Traceback (most recent call last):",
                '  File "analysis.py", line 10, in <module>',
                "NameError: name 'foo' is not defined",
                "2024-06-01 10:03:00,000Z WARNING [sentiment_exit] EURUSD state=scale_out score=-0.80 severity=0.90",
            ]
        )
    )

    summary = pipeline_monitor.summarize_pipeline_warnings(log_path)

    assert summary["warning_lines"] == 4
    assert summary["client_error_lines"] == 1
    assert summary["error_lines"] == 1
    assert summary["exception_lines"] >= 3
    assert summary["client_error_ratio"] == 0.25
    assert summary["exception_types"].get("NameError") == 1
    assert summary["last_exception"]["type"] == "NameError"
    assert summary["last_error"]["message"] == "Failed to compute metrics"
    assert summary["sentiment_exit_events"]
    assert "EURUSD" in summary["sentiment_exit_events"][-1]["detail"]
    assert summary["last_timestamp_utc"] == "2024-06-01T10:03:00Z"
    recent_symbols = summary["recent_warning_symbols"]
    assert recent_symbols["total_events"] == 2
    symbol_entries = {entry["symbol"]: entry for entry in recent_symbols["symbols"]}
    assert "EURUSD" in symbol_entries
    assert "BTCUSD" in symbol_entries
    trend_series = summary["warning_trend"]["series"]
    assert trend_series
    latest_bucket = trend_series[-1]
    assert latest_bucket["client_errors"] == 1
    assert latest_bucket["throttling"] == 1


def test_summarize_pipeline_warnings_handles_json_logs(tmp_path):
    log_path = tmp_path / "pipeline.log"
    entries = [
        json.dumps(
            {
                "timestamp": "2024-06-02T10:00:00Z",
                "level": "WARNING",
                "message": "Client error 404 while fetching feed for EURUSD",
                "event": "fetch_completed",
                "asset": "EURUSD",
            }
        ),
        json.dumps(
            {
                "timestamp": "2024-06-02T10:01:00Z",
                "level": "ERROR",
                "message": "Failed to publish update",
                "exc_info": "Traceback (most recent call last):\n  File \"notify.py\", line 10, in <module>\nRuntimeError: publish failed",
            }
        ),
    ]
    log_path.write_text("\n".join(entries))

    summary = pipeline_monitor.summarize_pipeline_warnings(log_path)

    assert summary["warning_lines"] == 1
    assert summary["client_error_lines"] == 1
    assert summary["error_lines"] == 1
    assert summary["exception_types"].get("RuntimeError") == 1
    assert summary["last_exception"]["type"] == "RuntimeError"
    assert summary["last_error"]["message"] == "Failed to publish update"
    recent_symbols = summary["recent_warning_symbols"]
    assert recent_symbols["total_events"] == 1
    assert recent_symbols["symbols"]
    assert recent_symbols["symbols"][0]["symbol"] == "EURUSD"
    trend_series = summary["warning_trend"]["series"]
    assert trend_series
    assert trend_series[-1]["client_errors"] == 1


def test_finalize_analysis_records_artifact_hashes(tmp_path, monkeypatch):
    monitor_path = tmp_path / "pipeline.json"
    artifact_a = tmp_path / "status.json"
    artifact_a.write_text("ok")
    monkeypatch.setenv("PIPELINE_ARTIFACTS", str(artifact_a))

    pipeline_monitor.record_analysis_run(
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        path=monitor_path,
    )
    payload = pipeline_monitor.finalize_analysis_run(
        completed_at=datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc),
        path=monitor_path,
    )

    hashes = payload.get("artifacts", {}).get("hashes", {})
    digest = hashlib.sha256(b"ok").hexdigest()
    assert hashes[str(artifact_a)]["sha256"] == digest
    assert hashes[str(artifact_a)]["size"] == 2


def test_finalize_analysis_rebuilds_missing_status(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    monitoring_dir = public_dir / "monitoring"
    monitor_path = monitoring_dir / "pipeline_timing.json"
    summary_path = public_dir / "analysis_summary.json"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_utc": "2024-01-01T00:00:00Z",
                "assets": {"EURUSD": {"ok": True, "signal": "buy"}},
            }
        ),
        encoding="utf-8",
    )

    status_path = public_dir / "status.json"
    assert not status_path.exists()

    monkeypatch.setattr(pipeline_monitor, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(pipeline_monitor, "MONITOR_DIR", monitoring_dir)
    monkeypatch.setattr(pipeline_monitor, "PIPELINE_MONITOR_PATH", monitor_path)
    monkeypatch.setattr(pipeline_monitor, "PIPELINE_LOG_PATH", monitoring_dir / "pipeline.log")
    monkeypatch.setattr(
        pipeline_monitor,
        "DEFAULT_ARTIFACTS",
        (summary_path, status_path, monitor_path),
    )

    pipeline_monitor.record_analysis_run(
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        path=monitor_path,
    )
    payload = pipeline_monitor.finalize_analysis_run(
        completed_at=datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc),
        path=monitor_path,
    )

    hashes = payload.get("artifacts", {}).get("hashes", {})
    assert status_path.exists()
    assert hashes[str(status_path)] is not None


def test_compute_run_timing_deltas_handles_missing_sections():
    payload = {
        "run": {"started_at_utc": "2024-01-01T00:00:00Z", "captured_at_utc": "2024-01-01T00:01:00Z"},
        "trading": {"started_utc": "2024-01-01T00:00:00Z", "completed_utc": "2024-01-01T00:05:00Z"},
    }

    deltas = pipeline_monitor.compute_run_timing_deltas(
        payload, now=datetime(2024, 1, 1, 0, 6, tzinfo=timezone.utc)
    )

    assert deltas["trading_duration_seconds"] == 300.0
    assert deltas["analysis_duration_seconds"] is None
    assert deltas["analysis_age_seconds"] is None
    assert deltas["trading_to_analysis_gap_seconds"] is None
    assert deltas["run_capture_offset_seconds"] == 60.0


@freeze_time("2024-01-01 00:00:00", tz_offset=0)
def test_pipeline_monitor_logs_invalid_timestamps(tmp_path, caplog):
    monitor_path = tmp_path / "pipeline.json"
    monitor_path.write_text(
        json.dumps(
            {
                "updated_utc": "bad-ts",
                "trading": {"started_utc": "2024-01-01T00:00:00Z", "completed_utc": "n/a"},
            }
        )
    )

    caplog.set_level(logging.ERROR)
    pipeline_monitor.record_trading_run(
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
        path=monitor_path,
    )

    assert any("Érvénytelen időbélyeg" in record.getMessage() for record in caplog.records)
    payload = json.loads(monitor_path.read_text())
    assert payload["trading"]["completed_utc"] == "2024-01-01T00:05:00Z"
