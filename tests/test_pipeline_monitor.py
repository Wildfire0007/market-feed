from __future__ import annotations

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
