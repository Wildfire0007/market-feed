import json
from datetime import datetime, timezone

from scripts.reset_notify_state import normalise_notify_state_file


def test_normalise_notify_state_file_clears_counts_and_cooldown(tmp_path):
    state_path = tmp_path / "_notify_state.json"
    payload = {
        "_meta": {"last_reset_utc": "2025-11-10T00:00:00Z"},
        "BTCUSD": {
            "last": "no entry",
            "count": 68,
            "last_sent": None,
            "cooldown_until": "2025-11-10T02:00:00Z",
        },
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    now = datetime(2025, 11, 12, 6, 0, tzinfo=timezone.utc)
    result = normalise_notify_state_file(
        path=state_path,
        now=now,
        reset_counts=True,
        max_cooldown_age_minutes=60.0,
        reason="unit_test",
    )

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["BTCUSD"]["count"] == 0
    assert data["BTCUSD"]["cooldown_until"] is None
    assert data["_meta"]["last_reset_reason"] == "unit_test"
    assert result.changed
    assert "BTCUSD" in result.reset_assets
    assert result.cleared_counts >= 1


def test_normalise_notify_state_file_rebuilds_missing_assets(tmp_path):
    state_path = tmp_path / "_notify_state.json"
    state_path.write_text("{}", encoding="utf-8")

    now = datetime(2025, 11, 12, tzinfo=timezone.utc)
    result = normalise_notify_state_file(
        path=state_path,
        now=now,
        reset_counts=True,
        max_cooldown_age_minutes=0.0,
        reason="unit_test",
    )

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert "_meta" in data
    assert result.changed
    assert data["_meta"]["last_reset_reason"] == "unit_test"
    # At least one configured asset should be present after rebuild
    assert any(asset != "_meta" for asset in data.keys())
