from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.td_spot_watchdog import collect_spot_statuses, write_state


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_collect_spot_statuses_marks_stale_when_age_exceeds_limit(tmp_path):
    public_dir = tmp_path / "public"
    asset_dir = public_dir / "EURUSD"
    asset_dir.mkdir(parents=True)

    now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    stale_timestamp = now - timedelta(seconds=601)

    spot_payload = {"utc": _iso(stale_timestamp)}
    (asset_dir / "spot.json").write_text(json.dumps(spot_payload), encoding="utf-8")

    statuses = collect_spot_statuses(
        ["EURUSD"],
        public_dir,
        limits={"default": 600},
        now=now,
    )

    assert len(statuses) == 1
    status = statuses[0]
    assert status.stale is True
    assert status.reason == "age_exceeds_limit"
    assert status.age_seconds == pytest.approx(601.0)
    assert status.threshold_seconds == pytest.approx(600.0)


def test_collect_spot_statuses_handles_missing_files(tmp_path):
    public_dir = tmp_path / "public"
    public_dir.mkdir()

    now = datetime(2024, 6, 1, tzinfo=timezone.utc)

    statuses = collect_spot_statuses(
        ["BTCUSD"],
        public_dir,
        limits={"default": 900, "BTCUSD": 300},
        now=now,
    )

    assert len(statuses) == 1
    status = statuses[0]
    assert status.stale is True
    assert status.reason == "missing_file"
    assert status.age_seconds is None
    assert status.limit_seconds == pytest.approx(300.0)


def test_collect_spot_statuses_honours_freshness_violation_flag(tmp_path):
    public_dir = tmp_path / "public"
    asset_dir = public_dir / "NVDA"
    asset_dir.mkdir(parents=True)

    now = datetime(2024, 7, 1, tzinfo=timezone.utc)
    fresh_timestamp = now - timedelta(seconds=5)

    spot_payload = {"utc": _iso(fresh_timestamp), "freshness_violation": True}
    (asset_dir / "spot.json").write_text(json.dumps(spot_payload), encoding="utf-8")

    statuses = collect_spot_statuses(
        ["NVDA"],
        public_dir,
        limits={"default": 900, "NVDA": 600},
        now=now,
    )

    status = statuses[0]
    assert status.stale is True
    assert status.reason == "freshness_violation_flag"
    assert status.age_seconds == pytest.approx(5.0)


def test_write_state_serialises_watchdog_status(tmp_path):
    public_dir = tmp_path / "public"
    asset_dir = public_dir / "XAGUSD"
    asset_dir.mkdir(parents=True)

    now = datetime(2024, 8, 1, tzinfo=timezone.utc)
    recent_timestamp = now - timedelta(seconds=45)

    spot_payload = {"utc": _iso(recent_timestamp)}
    (asset_dir / "spot.json").write_text(json.dumps(spot_payload), encoding="utf-8")

    statuses = collect_spot_statuses(
        ["XAGUSD"],
        public_dir,
        limits={"default": 900},
        now=now,
    )

    state_path = tmp_path / "state.json"
    payload = write_state(statuses, state_path, generated_at=now)

    assert state_path.exists()
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == on_disk
    asset_entry = payload["assets"]["XAGUSD"]
    assert asset_entry["stale"] is False
    assert asset_entry["age_seconds"] == pytest.approx(45.0)
    assert payload["stale_assets"] == []
