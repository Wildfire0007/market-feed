import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis

CI_ONLY = pytest.mark.skipif(not os.getenv("CI"), reason="CI-only regression guard (rollback note)")


def test_probability_stack_gap_guard_uses_recent_snapshot(tmp_path, caplog):
    base_dir = tmp_path / "public"
    snapshot_dir = base_dir / "BTCUSD"
    snapshot_dir.mkdir(parents=True)
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    snapshot_payload = {
        "source": "sklearn",
        "status": "enabled",
        "detail": "from_snapshot",
        "retrieved_at_utc": fresh_ts.isoformat(),
    }
    snapshot_path = snapshot_dir / analysis.PROB_STACK_SNAPSHOT_FILENAME
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    caplog.set_level("WARNING", logger=analysis.LOGGER.name)
    result = analysis._apply_probability_stack_gap_guard(
        "BTCUSD",
        {},
        now=datetime.now(timezone.utc),
        base_dir=base_dir,
    )

    assert result.get("gap_fallback") is True
    assert result.get("detail") == "from_snapshot"
    assert any(record.message == "prob_stack_gap" for record in caplog.records)


def test_probability_stack_gap_guard_uses_export_file(tmp_path, caplog):
    base_dir = tmp_path / "public"
    export_dir = base_dir / "BTCUSD"
    export_dir.mkdir(parents=True)
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=3)
    export_payload = {
        "source": "sklearn",
        "status": "enabled",
        "detail": "from_export",
        "retrieved_at_utc": fresh_ts.isoformat(),
    }
    export_path = export_dir / analysis.PROB_STACK_EXPORT_FILENAME
    export_path.write_text(json.dumps(export_payload), encoding="utf-8")

    caplog.set_level("WARNING", logger=analysis.LOGGER.name)
    result = analysis._apply_probability_stack_gap_guard(
        "BTCUSD",
        {},
        now=datetime.now(timezone.utc),
        base_dir=base_dir,
    )

    assert result.get("gap_fallback") is True
    assert result.get("detail") == "from_export"
    assert any(record.message == "prob_stack_gap" for record in caplog.records)


def test_probability_stack_gap_guard_respects_disable_env(monkeypatch, tmp_path):
    monkeypatch.setenv(analysis.PROB_STACK_GAP_ENV_DISABLE, "1")
    result = analysis._apply_probability_stack_gap_guard(
        "BTCUSD",
        {},
        now=datetime.now(timezone.utc),
        base_dir=tmp_path,
    )

    assert result == {}


@CI_ONLY
def test_probability_stack_gap_guard_marks_snapshot_fallback(tmp_path):
    base_dir = tmp_path / "public"
    snapshot_dir = base_dir / "BTCUSD"
    snapshot_dir.mkdir(parents=True)
    payload = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail": "stale_gap_snapshot",
    }
    snapshot_path = snapshot_dir / analysis.PROB_STACK_SNAPSHOT_FILENAME
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    result = analysis._apply_probability_stack_gap_guard(
        "BTCUSD",
        {},
        now=datetime.now(timezone.utc),
        base_dir=base_dir,
    )

    assert result.get("status") == "stale_snapshot"
    assert result.get("gap_fallback") is True
    assert result.get("snapshot_path") == str(snapshot_path)
    assert isinstance(result.get("snapshot_age_minutes"), float)
