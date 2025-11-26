import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis


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


def test_probability_stack_gap_guard_respects_disable_env(monkeypatch, tmp_path):
    monkeypatch.setenv(analysis.PROB_STACK_GAP_ENV_DISABLE, "1")
    result = analysis._apply_probability_stack_gap_guard(
        "BTCUSD",
        {},
        now=datetime.now(timezone.utc),
        base_dir=tmp_path,
    )

    assert result == {}
