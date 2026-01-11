import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_anchor import load_anchor_state, save_anchor_state
from config.analysis_settings import ASSETS
from scripts.notify_discord import build_default_state, to_utc_iso
from scripts.reset_dashboard_state import (
    RESET_STATUS,
    reset_anchor_state_file,
    reset_notify_state_file,
    reset_status_file,
)


CI_ONLY = pytest.mark.skipif(not os.getenv("CI"), reason="CI-only regression guard (rollback note)")


def _iso(dt: datetime) -> str:
    return to_utc_iso(dt)


def test_reset_anchor_state_file_prunes_old_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "trading.db"
    now = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    stale = now - timedelta(days=3)
    fresh = now - timedelta(hours=6)

    save_anchor_state(
        {
            "BTCUSD": {
                "side": "buy",
                "timestamp": _iso(stale),
                "last_update": _iso(stale),
            },
            "EURUSD": {
                "side": "sell",
                "timestamp": _iso(fresh),
                "last_update": _iso(fresh),
            },
        },
        db_path=str(db_path),
    )

    result = reset_anchor_state_file(
        max_age_hours=24,
        db_path=db_path,
        now=now,
        dry_run=False,
        backup_dir=tmp_path / "backups",
    )

    assert result.before == 2
    assert result.after == 1
    assert result.removed == 1
    assert result.changed is True

    persisted = load_anchor_state(str(db_path))
    assert list(persisted.keys()) == ["EURUSD"]


def test_reset_anchor_state_file_clears_stale_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "trading.db"
    save_anchor_state({"BTCUSD": {"side": "buy"}}, db_path=str(db_path))

    result = reset_anchor_state_file(
        max_age_hours=24,
        db_path=db_path,
        now=datetime(2025, 1, 15, 12, tzinfo=timezone.utc),
        dry_run=False,
        backup_dir=tmp_path / "backups",
    )

    assert result.changed is True
    assert load_anchor_state(str(db_path)) == {}
    

def test_reset_anchor_state_file_dry_run_leaves_file_untouched(tmp_path: Path) -> None:
    db_path = tmp_path / "trading.db"
    now = datetime(2025, 1, 15, tzinfo=timezone.utc)
    timestamps = [now - timedelta(hours=3), now - timedelta(days=2)]

    save_anchor_state(
        {
            "BTCUSD": {
                "side": "buy",
                "timestamp": _iso(timestamps[0]),
                "last_update": _iso(timestamps[0]),
            },
            "EURUSD": {
                "side": "sell",
                "timestamp": _iso(timestamps[1]),
                "last_update": _iso(timestamps[1]),
            },
        },
        db_path=str(db_path),
    )

    original = load_anchor_state(str(db_path))

    result = reset_anchor_state_file(
        max_age_hours=24,
        db_path=db_path,
        now=now,
        dry_run=True,
        backup_dir=tmp_path / "backups",
    )

    assert result.changed is False
    assert load_anchor_state(str(db_path)) == original


def test_build_default_state_sets_reset_metadata() -> None:
    now = datetime(2025, 1, 2, 9, 0, tzinfo=timezone.utc)
    state = build_default_state(now=now, reason="nightly reset")

    meta = state.get("_meta")
    assert meta["last_reset_utc"] == _iso(now)
    assert meta["last_reset_reason"] == "nightly reset"

    asset_keys = {asset for asset in state.keys() if asset != "_meta"}
    assert set(ASSETS) == asset_keys
    for asset in asset_keys:
        entry = state[asset]
        assert entry["last"] == "no entry"
        assert entry["count"] == 0


def test_reset_status_file_writes_reset_snapshot(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_utc": "2025-01-01T00:00:00Z",                
                "assets": {"BTCUSD": {"ok": True}},
                "notes": [],
            }
        ),
        encoding="utf-8",
    )

    now = datetime(2025, 1, 5, 8, 30, tzinfo=timezone.utc)
    result = reset_status_file(
        path=status_path,
        now=now,
        dry_run=False,
        backup_dir=tmp_path / "backups",
        reason="daily cleanup",
    )

    assert result.changed is True
    assert result.backup_path is not None and result.backup_path.exists()

    with status_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["ok"] is False
    assert data["status"] == RESET_STATUS
    assert data["generated_utc"] == _iso(now)
    assert data["assets"] == {}
    assert data["notes"][0]["message"] == "daily cleanup"
    assert data["notes"][0]["reset_utc"] == _iso(now)
    assert "td_base" not in data


def test_reset_notify_state_file_resets_structure(tmp_path: Path) -> None:
    notify_path = tmp_path / "_notify_state.json"
    notify_path.write_text(
        json.dumps(
            {
                "_meta": {"last_heartbeat_key": "2024010101"},
                "BTCUSD": {"last": "buy", "count": 42},
                "EXTRA": {"last": "sell", "count": 10},
            }
        ),
        encoding="utf-8",
    )

    now = datetime(2025, 1, 5, tzinfo=timezone.utc)
    result = reset_notify_state_file(
        path=notify_path,
        now=now,
        dry_run=False,
        backup_dir=tmp_path / "backups",
        reason="nightly reset",
    )

    assert result.changed is True
    assert result.backup_path is not None and result.backup_path.exists()

    with notify_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["_meta"]["last_reset_reason"] == "nightly reset"
    assert data["_meta"]["last_reset_utc"] == _iso(now)

    asset_keys = {key for key in data.keys() if key != "_meta"}
    assert set(ASSETS) == asset_keys
    for asset in ASSETS:
        entry = data[asset]
        assert entry["last"] == "no entry"
        assert entry["count"] == 0


@CI_ONLY
def test_reset_status_file_detects_existing_reset(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    existing = {
        "ok": False,
        "status": RESET_STATUS,
        "generated_utc": _iso(datetime(2025, 1, 1, tzinfo=timezone.utc)),
        "assets": {},
        "notes": [],
    }
    status_path.write_text(json.dumps(existing), encoding="utf-8")

    now = datetime(2025, 1, 5, 8, 30, tzinfo=timezone.utc)
    result = reset_status_file(path=status_path, now=now, dry_run=False, backup_dir=None, reason="daily cleanup")

    assert result.changed is False
    assert result.backup_path is None
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["status"] == RESET_STATUS
