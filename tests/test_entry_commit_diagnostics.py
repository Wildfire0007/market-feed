import os
from datetime import datetime, timezone

import pytest

import position_tracker
import scripts.notify_discord as notify_discord


def _base_pending(asset: str, manual_state: dict, signal_payload: dict, entry_record: notify_discord.EntryAuditRecord):
    return {
        "intent": "entry",
        "decision": "buy",
        "setup_grade": "A",
        "entry_side": "buy",
        "send_kind": "normal",
        "display_stable": True,
        "notify_meta": signal_payload.get("notify"),
        "manual_state_pre": manual_state,
        "manual_tracking_enabled": True,
        "can_write_positions": True,
        "state_loaded": True,
        "levels": {"entry": 100.0, "sl": 95.0, "tp2": 110.0},
        "gates_missing": [],
        "signal_payload": signal_payload,
        "audit": entry_record,
        "channel": "live",
    }


def test_entry_commit_blocked_until_dispatch_success(tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    entry_record = notify_discord.EntryAuditRecord(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=True,
        positions_file=str(positions_path),
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )

    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)
    dispatch_result = {"attempted": True, "success": False, "http_status": 500, "error": "boom"}

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    assert manual_state_after.get("has_position") is False
    assert not commit_result.get("committed")
    assert entry_record._commit_reason() == "dispatch_failed"
    assert not os.path.exists(positions_path)
    assert manual_positions == {}


def test_entry_commit_persists_after_dispatch_success(tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    entry_record = notify_discord.EntryAuditRecord(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=True,
        positions_file=str(positions_path),
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )

    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)
    dispatch_result = {"attempted": True, "success": True, "http_status": 204, "error": None}

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    assert commit_result.get("committed") is True
    assert commit_result.get("verified") is True
    assert manual_state_after.get("has_position") is True
    persisted = position_tracker.load_positions(str(positions_path), treat_missing_as_flat=True)
    assert "BTCUSD" in persisted
    assert entry_record._commit_reason() == "commit_ok"


def test_entry_commit_handles_missing_dispatch_result(tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    entry_record = notify_discord.EntryAuditRecord(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=True,
        positions_file=str(positions_path),
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )

    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)
    dispatch_result = {}

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    assert entry_record.dispatch_attempted is False
    assert entry_record.dispatch_success is False
    assert manual_state_after.get("has_position") is False
    assert not commit_result.get("committed")
    assert entry_record._commit_reason() == "dispatch_not_attempted"
    assert not os.path.exists(positions_path)
    assert manual_positions == {}

def test_entry_suppressed_when_state_not_loaded(tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    entry_record = notify_discord.EntryAuditRecord(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=False,
        positions_file=str(positions_path),
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )

    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)
    pending["state_loaded"] = False
    dispatch_result = {"attempted": True, "success": True, "http_status": 204, "error": None}

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    assert manual_state_after.get("has_position") is False
    assert commit_result.get("committed") is False
    assert entry_record._commit_reason() == "state_not_loaded"


def test_entry_commit_verification_failure(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    entry_record = notify_discord.EntryAuditRecord(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=True,
        positions_file=str(positions_path),
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )

    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)
    dispatch_result = {"attempted": True, "success": True, "http_status": 204, "error": None}

    monkeypatch.setattr(position_tracker, "load_positions", lambda path, treat_missing_as_flat: {})

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    assert commit_result.get("committed") is True
    assert commit_result.get("verified") is False
    assert manual_state_after.get("has_position") is False
    assert entry_record._commit_reason() == "commit_verify_failed"
