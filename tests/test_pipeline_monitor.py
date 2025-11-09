from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(Path(__file__).resolve().parent.parent))

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
