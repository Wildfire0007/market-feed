import datetime

import position_tracker


def test_open_position_sets_has_position_true():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    opened_at = "2025-01-01T11:00:00Z"

    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 120.0, opened_at, positions={}
    )

    state = position_tracker.compute_state(
        "BTCUSD", {"enabled": True}, manual_positions, now_dt
    )

    assert state["has_position"] is True
    assert state["is_flat"] is False
    assert state["side"] == "buy"


def test_close_position_sets_cooldown_state():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    opened_at = "2025-01-01T11:00:00Z"
    closed_at = "2025-01-01T12:05:00Z"
    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 120.0, opened_at, positions={}
    )

    cooled_positions = position_tracker.close_position(
        "BTCUSD",
        reason="hard_exit",
        closed_at_utc=closed_at,
        cooldown_minutes=30,
        positions=manual_positions,
    )

    state = position_tracker.compute_state(
        "BTCUSD", {"enabled": True}, cooled_positions, now_dt
    )

    assert state["cooldown_active"] is True
    assert state["has_position"] is False


def test_compute_state_disabled_never_sets_position_true():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 120.0, "2025-01-01T11:00:00Z", positions={}
    )

    state = position_tracker.compute_state(
        "BTCUSD", {"enabled": False}, manual_positions, now_dt
    )

    assert state["enabled"] is False
    assert state["tracking_enabled"] is False
    assert state["has_position"] is False
    assert state["is_flat"] is True
    assert state["side"] is None
    assert state["opened_at_utc"] is None
    assert state["entry"] is None
    assert state["sl"] is None
    assert state["tp2"] is None
    assert state["position"] is None


def test_cooldown_expiry_returns_to_flat_state():
    opened_at = "2025-01-01T11:00:00Z"
    closed_at = "2025-01-01T12:00:00Z"
    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 120.0, opened_at, positions={}
    )

    cooled_positions = position_tracker.close_position(
        "BTCUSD",
        reason="tp2_hit",
        closed_at_utc=closed_at,
        cooldown_minutes=30,
        positions=manual_positions,
    )

    after_cooldown = datetime.datetime(
        2025, 1, 1, 12, 45, tzinfo=datetime.timezone.utc
    )
    state = position_tracker.compute_state(
        "BTCUSD", {"enabled": True}, cooled_positions, after_cooldown
    )

    assert state["cooldown_active"] is False
    assert state["has_position"] is False
    assert state["is_flat"] is True


def test_compute_state_disabled_never_sets_cooldown_active():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "EURUSD": {
            "side": None,
            "cooldown_until_utc": "2025-01-01T12:10:00Z",
        }
    }

    state = position_tracker.compute_state(
        "EURUSD", {"enabled": False}, positions, now_dt
    )

    assert state["tracking_enabled"] is False
    assert state["has_position"] is False
    assert state["cooldown_active"] is False
    assert state["is_flat"] is True


def test_open_position_sets_side_and_clears_cooldown():
    opened_at = "2025-01-01T00:00:00Z"
    positions = {"BTCUSD": {"side": None, "cooldown_until_utc": "2025-01-02T00:00:00Z"}}

    updated = position_tracker.open_position(
        "BTCUSD", "buy", 100.0, 90.0, 120.0, opened_at, positions
    )

    entry = updated["BTCUSD"]
    assert entry["side"] == "long"
    assert entry["cooldown_until_utc"] is None
    assert entry["entry"] == 100.0
    assert entry["sl"] == 90.0
    assert entry["tp2"] == 120.0
    assert entry["opened_at_utc"] == opened_at


def test_close_position_sets_cooldown_until():
    closed_at = "2025-01-01T12:00:00Z"
    updated = position_tracker.close_position(
        "EURUSD",
        reason="hard_exit",
        closed_at_utc=closed_at,
        cooldown_minutes=30,
        positions={"EURUSD": {"side": "long", "opened_at_utc": "2025-01-01T10:00:00Z"}},
    )

    entry = updated["EURUSD"]
    assert entry["side"] is None
    assert entry["close_reason"] == "hard_exit"
    assert entry["closed_at_utc"] == closed_at
    assert entry["cooldown_until_utc"] == "2025-01-01T12:30:00Z"


def test_compute_state_blocks_entry_during_cooldown():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "XAGUSD": {
            "side": None,
            "cooldown_until_utc": "2025-01-01T12:10:00Z",
        }
    }
    state = position_tracker.compute_state(
        "XAGUSD", {"enabled": True}, positions, now_dt
    )

    assert state["tracking_enabled"] is True
    assert state["cooldown_active"] is True
    assert state["has_position"] is False
    assert state["is_flat"] is False


def test_check_close_by_levels_hits_sl_and_tp2():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {"BTCUSD": {"side": "long", "sl": 95.0, "tp2": 110.0}}

    changed, reason, updated = position_tracker.check_close_by_levels(
        "BTCUSD", positions, 94.5, now_dt, 15
    )
    assert changed is True
    assert reason == "sl_hit"
    assert updated["BTCUSD"]["close_reason"] == "sl_hit"

    positions_short = {"EURUSD": {"side": "short", "sl": 1.1, "tp2": 1.0}}
    changed, reason, updated = position_tracker.check_close_by_levels(
        "EURUSD", positions_short, 0.99, now_dt, 10
    )
    assert changed is True
    assert reason == "tp2_hit"
    assert updated["EURUSD"]["close_reason"] == "tp2_hit"
