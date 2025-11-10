import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sync_pipeline_env import (
    _sanitize_order_flow_summary_files,
    _sanitize_spot_snapshots,
)


def test_sanitize_spot_snapshot_marks_missing_bid_ask(tmp_path):
    asset_dir = tmp_path / "BTCUSD"
    asset_dir.mkdir()
    timestamp = datetime(2025, 11, 9, 9, 59, tzinfo=timezone.utc).isoformat()
    snapshot = {
        "asset": "BTCUSD",
        "ok": True,
        "transport": "http",
        "retrieved_at_utc": timestamp,
        "utc": timestamp,
        "frames": [{"price": 101000.0, "utc": timestamp}],
        "forced": True,
        "force_reason": "spot_error",
    }
    path = asset_dir / "spot_realtime.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    marked = _sanitize_spot_snapshots(tmp_path, now=datetime(2025, 11, 9, 10, 0, tzinfo=timezone.utc))
    assert marked == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["stale_reason"] == "missing_bid_ask"
    assert "price" not in payload
    assert payload.get("stale_snapshot_utc") == timestamp
    assert payload.get("stale_snapshot_retrieved_at_utc") == timestamp
    assert payload.get("stale_age_seconds") == 60.0


def test_sanitize_spot_snapshot_marks_old_snapshot(tmp_path):
    asset_dir = tmp_path / "EURUSD"
    asset_dir.mkdir()
    timestamp_dt = datetime(2025, 11, 9, 7, 30, tzinfo=timezone.utc)
    timestamp = timestamp_dt.isoformat()
    snapshot = {
        "asset": "EURUSD",
        "ok": True,
        "transport": "http",
        "retrieved_at_utc": timestamp,
        "utc": timestamp,
        "price": 1.0642,
        "frames": [
            {
                "price": 1.0641,
                "utc": timestamp,
                "retrieved_at_utc": timestamp,
                "bid": 1.064,
                "ask": 1.0642,
            }
        ],
    }
    path = asset_dir / "spot_realtime.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    now_dt = datetime(2025, 11, 9, 10, 0, tzinfo=timezone.utc)
    marked = _sanitize_spot_snapshots(tmp_path, now=now_dt)
    assert marked == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["stale_reason"] == "older_than_300s"
    assert payload.get("stale_snapshot_utc") == timestamp
    assert payload.get("stale_age_seconds") == (now_dt - timestamp_dt).total_seconds()


def test_sanitize_order_flow_summary_adds_status(tmp_path):
    summary = {
        "ok": True,
        "generated_utc": "2025-11-09T09:55:37Z",
        "assets": {
            "BTCUSD": {
                "order_flow_metrics": {
                    "imbalance": None,
                    "pressure": 0.0,
                    "delta_volume": 0.0,
                    "aggressor_ratio": None,
                }
            }
        },
    }
    path = tmp_path / "analysis_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    updated = _sanitize_order_flow_summary_files(tmp_path)
    assert updated == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["assets"]["BTCUSD"]["order_flow_metrics"]
    assert metrics["status"] == "volume_unavailable"
    assert metrics["pressure"] is None
    assert metrics["delta_volume"] is None
