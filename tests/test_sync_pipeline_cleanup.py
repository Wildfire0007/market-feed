from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts import sync_pipeline_env as sync_env


def test_archive_analysis_summaries_moves_existing_files(tmp_path):
    public_dir = tmp_path
    root_summary = public_dir / "analysis_summary.json"
    root_summary.write_text("{}", encoding="utf-8")
    asset_dir = public_dir / "BTCUSD"
    asset_dir.mkdir()
    asset_summary = asset_dir / "analysis_summary.json"
    asset_summary.write_text("{}", encoding="utf-8")

    now = datetime(2025, 11, 2, 0, 0, tzinfo=timezone.utc)
    archived = sync_env._archive_analysis_summaries(public_dir, now=now, retention=2)

    assert archived == 2
    assert not root_summary.exists()
    assert not asset_summary.exists()

    archive_dir = public_dir / "archive" / "analysis_summary" / "20251102T000000Z"
    assert (archive_dir / "analysis_summary.json").exists()
    assert (archive_dir / "BTCUSD" / "analysis_summary.json").exists()


def test_archive_stale_feature_monitors(tmp_path):
    public_dir = tmp_path
    monitor_dir = public_dir / "ml_features" / "monitoring"
    monitor_dir.mkdir(parents=True)
    monitor_json = monitor_dir / "NVDA_monitor.json"
    monitor_csv = monitor_dir / "NVDA_monitor.csv"
    payload = {"generated_utc": "2025-10-31T00:00:00Z"}
    monitor_json.write_text(json.dumps(payload), encoding="utf-8")
    monitor_csv.write_text("asset,feature,generated_utc,psi\n", encoding="utf-8")

    now = datetime(2025, 11, 2, tzinfo=timezone.utc)
    archived = sync_env._archive_stale_feature_monitors(public_dir, now=now, max_age_hours=24, retention=1)

    assert archived == 1
    target_dir = public_dir / "archive" / "feature_monitor" / "20251102T000000Z" / "ml_features" / "monitoring"
    assert target_dir.exists()
    assert (target_dir / "NVDA_monitor.json").exists()
    assert (target_dir / "NVDA_monitor.csv").exists()


def test_maybe_reset_notify_state_resets_when_heartbeat_stale(tmp_path):
    public_dir = tmp_path
    state_path = public_dir / "_notify_state.json"
    payload = {
        "_meta": {"last_heartbeat_utc": "2025-10-20T08:00:00Z"},
        "GOLD_CFD": {"last_sent": "2025-10-20T07:59:00Z"},
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    now = datetime(2025, 11, 2, 0, 0, tzinfo=timezone.utc)
    result = sync_env._maybe_reset_notify_state(public_dir, now=now, stale_hours=24)

    assert result is not None
    assert getattr(result, "changed", False)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["_meta"]["last_reset_reason"] == "pipeline_cleanup"


def test_maybe_reset_status_skips_recent_snapshot(tmp_path):
    public_dir = tmp_path
    status_path = public_dir / "status.json"
    status_payload = {"generated_utc": "2025-11-02T00:00:00Z", "ok": True}
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    now = datetime(2025, 11, 2, 1, 0, tzinfo=timezone.utc)
    result = sync_env._maybe_reset_status(public_dir, now=now, stale_hours=48)

    assert result is None
    reloaded = json.loads(status_path.read_text(encoding="utf-8"))
    assert reloaded == status_payload
