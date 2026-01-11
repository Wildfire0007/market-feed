from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import active_anchor
import state_db
from active_anchor import save_anchor_state

from scripts.market_open_watchdog import (
    WatchdogConfig,
    perform_reset,
    should_trigger_reset,
    _load_state,
    _make_backup,
    _reset_health_file,
    _state_path,
)
from scripts.reset_dashboard_state import RESET_STATUS


UTC = timezone.utc


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_should_trigger_reset_skips_weekend() -> None:
    config = WatchdogConfig(
        market_open_time=time(13, 30, tzinfo=UTC),
        allowed_weekdays={0, 1, 2, 3, 4},
        pre_open_leeway=timedelta(minutes=0),
        anchor_max_age_hours=24.0,
    )
    saturday = datetime(2024, 6, 1, 13, 35, tzinfo=UTC)
    assert not should_trigger_reset(
        saturday, config=config, last_reset=None, holidays=set(), force=False
    )


def test_should_trigger_reset_after_open(tmp_path: Path, monkeypatch) -> None:
    config = WatchdogConfig(
        market_open_time=time(13, 30, tzinfo=UTC),
        allowed_weekdays={0, 1, 2, 3, 4},
        pre_open_leeway=timedelta(minutes=0),
        anchor_max_age_hours=24.0,
    )
    monday = datetime(2024, 6, 3, 13, 45, tzinfo=UTC)
    holidays = {date(2024, 6, 4)}
    assert should_trigger_reset(
        monday, config=config, last_reset=None, holidays=holidays, force=False
    )

    public_dir = tmp_path / "public"
    db_path = tmp_path / "trading.db"
    monkeypatch.setattr(state_db, "DEFAULT_DB_PATH", db_path)
    active_anchor._DB_INITIALIZED = False
    _write(public_dir / "status.json", {"ok": True, "assets": {"EURUSD": {}}})
    _write(
        public_dir / "_notify_state.json",
        {
            "_meta": {"last_reset_utc": "2024-05-31T12:00:00Z", "last_reset_reason": "manual"},
            "EURUSD": {"last_notification_utc": "2024-05-31T12:00:00Z"},
        },
    )
    stale_anchor = monday - timedelta(days=3)
    save_anchor_state(
        {
            "EURUSD": {
                "side": "sell",
                "timestamp": stale_anchor.isoformat().replace("+00:00", "Z"),
                "last_update": stale_anchor.isoformat().replace("+00:00", "Z"),
            }
        },
        db_path=str(db_path),
    )
    _write(
        public_dir / "monitoring" / "health.json",
        {
            "generated_utc": "2024-05-31T12:00:00Z",
            "assets": [
                {
                    "asset": "EURUSD",
                    "status": "error",
                    "signal": "market closed",
                }
            ],
            "alerts": ["stale"],
        },
    )

    results = perform_reset(
        public_dir,
        now=monday,
        config=config,
        reason="market open reset",
        dry_run=False,
        backup_dir=public_dir / "monitoring" / "reset_backups",
    )

    anchor_result, notify_result, status_result, health_result = results
    assert anchor_result.changed
    assert notify_result.changed
    assert status_result.changed
    assert health_result.changed

    status_payload = json.loads((public_dir / "status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == RESET_STATUS
    assert status_payload["notes"][0]["message"] == "market open reset"

    health_payload = json.loads(
        (public_dir / "monitoring" / "health.json").read_text(encoding="utf-8")
    )
    assert health_payload["assets"] == []
    assert health_payload["notes"][0]["type"] == "reset"


def test_should_not_trigger_twice(tmp_path: Path, monkeypatch) -> None:
    config = WatchdogConfig(
        market_open_time=time(13, 30, tzinfo=UTC),
        allowed_weekdays={0, 1, 2, 3, 4},
        pre_open_leeway=timedelta(minutes=0),
        anchor_max_age_hours=24.0,
    )

    monday = datetime(2024, 6, 3, 14, 0, tzinfo=UTC)
    public_dir = tmp_path / "public"
    db_path = tmp_path / "trading.db"
    monkeypatch.setattr(state_db, "DEFAULT_DB_PATH", db_path)
    active_anchor._DB_INITIALIZED = False
    _write(public_dir / "status.json", {"ok": False, "assets": {}})
    _write(public_dir / "_notify_state.json", {"_meta": {}})
    save_anchor_state({}, db_path=str(db_path))
    _write(public_dir / "monitoring" / "health.json", {"generated_utc": "x", "assets": [], "alerts": []})

    perform_reset(
        public_dir,
        now=monday,
        config=config,
        reason="market open reset",
        dry_run=False,
        backup_dir=public_dir / "monitoring" / "reset_backups",
    )

    state_file = _state_path(public_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"last_reset_date": "2024-06-03"}), encoding="utf-8")
    state = _load_state(state_file)
    assert not should_trigger_reset(
        monday,
        config=config,
        last_reset=state.last_reset_date,
        holidays=set(),
        force=False,
    )


def test_reset_health_file_dry_run(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir(parents=True)
    now = datetime(2024, 6, 3, 13, 45, tzinfo=UTC)
    _write(public_dir / "monitoring" / "health.json", {"generated_utc": "x", "assets": [1], "alerts": []})
    result = _reset_health_file(
        public_dir,
        now=now,
        reason="market open reset",
        dry_run=True,
        backup_dir=public_dir / "monitoring" / "reset_backups",
    )
    assert not result.changed
    payload = json.loads((public_dir / "monitoring" / "health.json").read_text(encoding="utf-8"))
    assert payload["assets"] == [1]


def test_make_backup_creates_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    backup_dir = tmp_path / "backups"
    source.write_text("{}", encoding="utf-8")
    backup = _make_backup(source, backup_dir)
    assert backup is not None
    assert backup.exists()
