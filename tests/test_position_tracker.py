import datetime
from pathlib import Path

import position_tracker


def test_open_position_sets_has_position_true():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    opened_at = "2025-01-01T11:00:00Z"

    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 110.0, 120.0, opened_at, positions={}
    )

    state = position_tracker.compute_state(
        "BTCUSD", {"enabled": True}, manual_positions, now_dt
    )

    assert state["has_position"] is True
    assert state["is_flat"] is False
    assert state["side"] == "buy"


def test_load_positions_reads_from_db(tmp_path: Path) -> None:
    db_path = tmp_path / "trading.db"
    position_tracker.state_db.initialize(db_path)
    connection = position_tracker.state_db.connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO positions (asset, entry_price, size, sl, tp, status, strategy_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSD",
                100.5,
                1.25,
                95.0,
                120.0,
                "OPEN",
                '{"side":"long","opened_at_utc":"2025-01-01T11:00:00Z"}',
            ),
        )
        connection.commit()
    finally:
        connection.close()

    loaded = position_tracker.load_positions(str(db_path), treat_missing_as_flat=False)

    assert "BTCUSD" in loaded
    assert loaded["BTCUSD"]["entry"] == 100.5
    assert loaded["BTCUSD"]["size"] == 1.25
    assert loaded["BTCUSD"]["side"] == "long"


def test_close_position_sets_cooldown_state():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    opened_at = "2025-01-01T11:00:00Z"
    closed_at = "2025-01-01T12:05:00Z"
    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 110.0, 120.0, opened_at, positions={}
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
        "BTCUSD", "buy", 100.5, 95.0, 110.0, 120.0, "2025-01-01T11:00:00Z", positions={}
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
    assert state["tp1"] is None
    assert state["tp2"] is None
    assert state["position"] is None


def test_cooldown_expiry_returns_to_flat_state():
    opened_at = "2025-01-01T11:00:00Z"
    closed_at = "2025-01-01T12:00:00Z"
    manual_positions = position_tracker.open_position(
        "BTCUSD", "buy", 100.5, 95.0, 110.0, 120.0, opened_at, positions={}
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
        "BTCUSD", "buy", 100.0, 90.0, 110.0, 120.0, opened_at, positions
    )

    entry = updated["BTCUSD"]
    assert entry["side"] == "long"
    assert entry["cooldown_until_utc"] is None
    assert entry["entry"] == 100.0
    assert entry["sl"] == 90.0
    assert entry["tp1"] == 110.0
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


def test_check_close_by_levels_tp1_scales_out_and_moves_sl_to_breakeven():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "BTCUSD": {"side": "long", "entry": 100.0, "sl": 95.0, "tp1": 105.0, "tp2": 110.0, "size": 2.0}
    }

    changed, reason, updated = position_tracker.check_close_by_levels(
        "BTCUSD", positions, 105.0, now_dt, 15, tp1_close_fraction=0.5
    )

    assert changed is True
    assert reason == "tp1_hit"
    assert updated["BTCUSD"]["side"] == "long"
    assert updated["BTCUSD"]["size"] == 1.0
    assert updated["BTCUSD"]["sl"] == 100.0
    assert updated["BTCUSD"]["tp1_scaled"] is True
    assert updated["BTCUSD"]["last_management_signal"]["state"] == "scale_out"


def test_check_close_by_levels_tp1_not_repeated_after_scaled_flag():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "BTCUSD": {"side": "long", "entry": 100.0, "sl": 100.0, "tp1": 105.0, "tp2": 110.0, "size": 1.0, "tp1_scaled": True}
    }

    changed, reason, updated = position_tracker.check_close_by_levels(
        "BTCUSD", positions, 105.0, now_dt, 15, tp1_close_fraction=0.5
    )

    assert changed is False
    assert reason is None
    assert updated["BTCUSD"]["tp1_scaled"] is True


def test_pending_exit_record_and_clear(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"

    position_tracker.record_pending_exit(
        str(pending_path),
        "BTCUSD",
        reason="hard_exit",
        closed_at_utc="2025-01-01T00:00:00Z",
        cooldown_minutes=15,
        source="test",
    )

    pending = position_tracker.load_pending_exits(str(pending_path))
    assert pending["BTCUSD"]["cooldown_minutes"] == 15

    position_tracker.clear_pending_exits(str(pending_path), ["BTCUSD"])
    assert not pending_path.exists()


def test_register_precision_pending_position_and_fill_limit_buy():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    payload = {
        "signal": "precision_arming",
        "precision_plan": {
            "direction": "buy",
            "order_type": "LIMIT",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit_1": 105.0,
            "take_profit_2": 110.0,
        },
    }

    positions = position_tracker.register_precision_pending_position(
        "XAGUSD", payload, now_dt, {}
    )
    assert positions["XAGUSD"]["status"] == "pending"

    changed, reason, updated = position_tracker.check_close_by_levels(
        "XAGUSD", positions, 99.9, now_dt, cooldown_minutes=20
    )
    assert changed is True
    assert reason == "pending_filled"
    assert updated["XAGUSD"]["status"] == "open"
    assert updated["XAGUSD"]["side"] == "long"


def test_pending_position_expires_after_30_minutes():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "XAGUSD": {
            "status": "pending",
            "direction": "sell",
            "order_type": "LIMIT",
            "entry": 31.0,
            "sl": 32.0,
            "tp1": 30.0,
            "tp2": 29.0,
            "pending_since_utc": "2025-01-01T11:29:00Z",
            "side": None,
        }
    }

    changed, reason, updated = position_tracker.check_close_by_levels(
        "XAGUSD", positions, 30.5, now_dt, cooldown_minutes=20
    )
    assert changed is True
    assert reason == "pending_expired"
    assert "XAGUSD" not in updated


def test_register_precision_pending_does_not_reset_timestamp_for_same_pending():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    payload = {
        "signal": "precision_arming",
        "precision_plan": {
            "direction": "buy",
            "order_type": "LIMIT",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit_1": 105.0,
            "take_profit_2": 110.0,
        },
    }
    positions = {
        "XAGUSD": {
            "status": "pending",
            "direction": "buy",
            "order_type": "LIMIT",
            "entry": 100.0,
            "sl": 95.0,
            "tp1": 105.0,
            "tp2": 110.0,
            "pending_since_utc": "2025-01-01T11:20:00Z",
        }
    }

    updated = position_tracker.register_precision_pending_position(
        "XAGUSD", payload, now_dt, positions
    )

    assert updated["XAGUSD"]["pending_since_utc"] == "2025-01-01T11:20:00Z"


def test_register_precision_pending_does_not_override_open_position():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    payload = {
        "signal": "precision_arming",
        "precision_plan": {
            "direction": "sell",
            "order_type": "LIMIT",
            "entry": 31.0,
            "stop_loss": 32.0,
            "take_profit_1": 30.0,
            "take_profit_2": 29.0,
        },
    }
    positions = {
        "XAGUSD": {
            "status": "open",
            "side": "long",
            "entry": 30.5,
            "sl": 29.8,
            "tp1": 31.2,
            "tp2": 31.8,
            "opened_at_utc": "2025-01-01T11:50:00Z",
        }
    }

    updated = position_tracker.register_precision_pending_position(
        "XAGUSD", payload, now_dt, positions
    )

    assert updated["XAGUSD"]["status"] == "open"
    assert updated["XAGUSD"]["side"] == "long"
    assert updated["XAGUSD"]["entry"] == 30.5


def test_update_pending_positions_activates_limit_buy_and_expires_old_pending():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "XAGUSD": {
            "status": "pending",
            "direction": "buy",
            "order_type": "LIMIT",
            "entry": 30.5,
            "pending_since_utc": "2025-01-01T11:55:00Z",
        },
        "EURUSD": {
            "status": "pending",
            "direction": "sell",
            "order_type": "LIMIT",
            "entry": 1.1,
            "pending_since_utc": "2025-01-01T11:00:00Z",
        },
    }

    updated, changes = position_tracker.update_pending_positions(
        positions,
        {"XAGUSD": 30.4, "EURUSD": 1.2},
        now_dt,
        pending_expiry_minutes=30,
    )

    assert changes["XAGUSD"] == "pending_filled"
    assert updated["XAGUSD"]["status"] == "open"
    assert updated["XAGUSD"]["side"] == "long"
    assert changes["EURUSD"] == "pending_expired"
    assert "EURUSD" not in updated


def test_update_pending_positions_respects_asset_expiry_config():
    now_dt = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    positions = {
        "XAGUSD": {
            "status": "pending",
            "direction": "buy",
            "order_type": "LIMIT",
            "entry": 30.5,
            "pending_since_utc": "2025-01-01T11:40:00Z",
        }
    }

    updated, changes = position_tracker.update_pending_positions(
        positions,
        {"XAGUSD": 30.8},
        now_dt,
        pending_expiry_minutes=30,
        pending_expiry_by_asset={"XAGUSD": 15},
    )

    assert changes["XAGUSD"] == "pending_expired"
    assert "XAGUSD" not in updated

