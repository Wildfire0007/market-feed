import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_anchor import load_anchor_state, save_anchor_state
from config.analysis_settings import ASSETS
from scripts.notify_discord import build_default_state, to_utc_iso
from scripts.reset_dashboard_state import (
    reset_anchor_state_file,
    reset_notify_state_file,
    reset_status_file,
)


def _iso(dt: datetime) -> str:
    return to_utc_iso(dt)


def test_reset_anchor_state_file_prunes_old_entries(tmp_path: Path) -> None:
    anchor_path = tmp_path / "_active_anchor.json"
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
        path=str(anchor_path),
    )

    result = reset_anchor_state_file(
        max_age_hours=24,
        path=anchor_path,
        now=now,
        dry_run=False,
        backup_dir=tmp_path / "backups",
    )

    assert result.before == 2
    assert result.after == 1
    assert result.removed == 1
    assert result.changed is True

    persisted = load_anchor_state(str(anchor_path))
    assert list(persisted.keys()) == ["EURUSD"]


def test_reset_anchor_state_file_dry_run_leaves_file_untouched(tmp_path: Path) -> None:
    anchor_path = tmp_path / "_active_anchor.json"
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
        path=str(anchor_path),
    )

    original = anchor_path.read_text(encoding="utf-8")

    result = reset_anchor_state_file(
        max_age_hours=24,
        path=anchor_path,
        now=now,
        dry_run=True,
        backup_dir=tmp_path / "backups",
    )

    assert result.changed is False
    assert anchor_path.read_text(encoding="utf-8") == original


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
                "td_base": "https://api.example.com",
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
    assert data["generated_utc"] == _iso(now)
    assert data["assets"] == {}
    assert data["notes"][0]["message"] == "daily cleanup"
    assert data["notes"][0]["reset_utc"] == _iso(now)


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
