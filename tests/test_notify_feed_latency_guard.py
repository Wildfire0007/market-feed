import json
from datetime import datetime, timedelta, timezone

import scripts.notify_discord as notify


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_compute_feed_latency_prefers_newer_kline(tmp_path):
    now = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)
    asset_dir = tmp_path / "BTCUSD"
    asset_dir.mkdir()

    _write(asset_dir / "spot.json", {"utc": (now - timedelta(hours=3)).isoformat()})
    _write(asset_dir / "klines_1m_meta.json", {"latest_utc": (now - timedelta(minutes=10)).isoformat()})

    latency = notify.compute_feed_latency("BTCUSD", public_dir=str(tmp_path), now=now)

    assert latency.seconds == 600
    assert latency.minutes == 10
    assert latency.source == "kline_1m"


def test_compute_feed_latency_handles_missing_payloads(tmp_path):
    now = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)
    asset_dir = tmp_path / "ETHUSD"
    asset_dir.mkdir()

    latency = notify.compute_feed_latency("ETHUSD", public_dir=str(tmp_path), now=now)

    assert latency.seconds is None
    assert latency.source is None
    assert latency.issues  # missing spot + kline snapshot markers
