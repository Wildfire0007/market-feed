import json
import logging
from datetime import datetime, timezone

import position_tracker
from scripts import notify_discord as nd


class _BufferHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple collector
        self.records.append(record.__dict__)


def _collect_events(handler: _BufferHandler, event_name: str):
    return [rec for rec in handler.records if rec.get("event") == event_name]


def _build_entry_record(asset: str, manual_state: dict, positions_file: str, can_write: bool) -> nd.EntryAuditRecord:
    return nd.EntryAuditRecord(
        asset=asset,
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=can_write,
        state_loaded=True,
        positions_file=positions_file,
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )


def test_entry_dispatch_without_commit(monkeypatch, tmp_path):
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "0")
    position_tracker.set_audit_context(source="test", run_id="RUN_NO_COMMIT")
    handler = _BufferHandler()
    position_tracker.LOGGER.addHandler(handler)

    positions_path = tmp_path / "manual_positions.json"
    positions_path.write_text("{}\n", encoding="utf-8")
    manual_positions = position_tracker.load_positions(str(positions_path), True)
    tracking_cfg = {"enabled": True}
    now_dt = datetime.now(timezone.utc)
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now_dt)

    record = _build_entry_record("BTCUSD", manual_state, str(positions_path), can_write=False)
    record.log_candidate()

    signal_payload = {"trade": {"entry": 100.0, "sl": 90.0, "tp2": 120.0}}
    manual_positions, manual_state, positions_changed, entry_opened, commit_result = nd._apply_and_persist_manual_transitions(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        notify_meta={"should_notify": True},
        signal_payload=signal_payload,
        manual_tracking_enabled=True,
        can_write_positions=False,
        manual_state=manual_state,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now_dt,
        now_iso=position_tracker._to_utc_iso(now_dt),
        send_kind="normal",
        display_stable=True,
        missing_list=[],
        cooldown_map={},
        cooldown_default=0,
        positions_path=str(positions_path),
        entry_level=None,
        sl_level=None,
        tp2_level=None,
        open_commits_this_run=set(),
        sig={},
    )
    record.positions_changed = positions_changed
    record.entry_opened = entry_opened
    record.commit_result = commit_result

    record.dispatch_attempted = True
    record.dispatch_success = True
    record.dispatch_status = 204
    record.channel = "live"
    record.log_dispatch_result()
    record.log_commit_decision()

    dispatch_events = _collect_events(handler, "ENTRY_DISPATCH_RESULT")
    assert dispatch_events
    assert dispatch_events[-1]["success"] is True

    decision_events = _collect_events(handler, "ENTRY_COMMIT_DECISION")
    assert decision_events
    assert decision_events[-1]["will_commit"] is False
    assert decision_events[-1]["commit_reason"] == "writer_read_only"

    assert json.loads(positions_path.read_text()) == {}

    position_tracker.LOGGER.removeHandler(handler)


def test_entry_commit_success(monkeypatch, tmp_path):
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "0")
    position_tracker.set_audit_context(source="test", run_id="RUN_COMMIT_OK")
    handler = _BufferHandler()
    position_tracker.LOGGER.addHandler(handler)

    positions_path = tmp_path / "manual_positions.json"
    positions_path.write_text("{}\n", encoding="utf-8")
    manual_positions = position_tracker.load_positions(str(positions_path), True)
    tracking_cfg = {"enabled": True}
    now_dt = datetime.now(timezone.utc)
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now_dt)

    record = _build_entry_record("BTCUSD", manual_state, str(positions_path), can_write=True)
    record.log_candidate()

    signal_payload = {"trade": {"entry": 100.0, "sl": 90.0, "tp2": 120.0}}
    manual_positions, manual_state, positions_changed, entry_opened, commit_result = nd._apply_and_persist_manual_transitions(
        asset="BTCUSD",
        intent="entry",
        decision="buy",
        setup_grade="A",
        notify_meta={"should_notify": True},
        signal_payload=signal_payload,
        manual_tracking_enabled=True,
        can_write_positions=True,
        manual_state=manual_state,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now_dt,
        now_iso=position_tracker._to_utc_iso(now_dt),
        send_kind="normal",
        display_stable=True,
        missing_list=[],
        cooldown_map={},
        cooldown_default=0,
        positions_path=str(positions_path),
        entry_level=None,
        sl_level=None,
        tp2_level=None,
        open_commits_this_run=set(),
        sig={},
    )
    record.positions_changed = positions_changed
    record.entry_opened = entry_opened
    record.commit_result = commit_result

    record.dispatch_attempted = True
    record.dispatch_success = True
    record.dispatch_status = 204
    record.channel = "live"
    record.log_dispatch_result()
    record.log_commit_decision()

    decision_events = _collect_events(handler, "ENTRY_COMMIT_DECISION")
    assert decision_events
    assert decision_events[-1]["will_commit"] is True
    assert decision_events[-1]["commit_reason"] == "commit_ok"

    commit_results = _collect_events(handler, "ENTRY_COMMIT_RESULT")
    assert commit_results
    assert commit_results[-1]["committed"] is True

    content = json.loads(positions_path.read_text())
    assert "BTCUSD" in content
    assert content["BTCUSD"].get("side") == "long"

    position_tracker.LOGGER.removeHandler(handler)
