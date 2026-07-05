import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import position_tracker
import state_db


class _ListHandler(logging.Handler):
    def __init__(self, buffer: List[Dict[str, Any]]):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.buffer.append(record.__dict__)


def _parse_lines(out: str):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _reset_audit_file_handler():
    for handler in list(position_tracker.LOGGER.handlers):
        if getattr(handler, "_manual_positions_audit_file", False):
            position_tracker.LOGGER.removeHandler(handler)
            handler.close()
    position_tracker._FILE_LOGGER_ATTACHED = False


def test_open_and_save_emit_audit_fields(capfd, tmp_path, monkeypatch):
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "0")
    position_tracker.set_audit_context(source="test", run_id="R1")
    buffer: List[Dict[str, Any]] = []
    handler = _ListHandler(buffer)
    position_tracker.LOGGER.addHandler(handler)
    positions = position_tracker.open_position(
        "XAUUSD",
        side="buy",
        entry=25.0,
        sl=24.5,
        tp1=25.5,
        tp2=26.0,
        opened_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    path = tmp_path / "trading.db"
    position_tracker.save_positions_atomic(str(path), positions)

    position_tracker.LOGGER.removeHandler(handler)

    captured = capfd.readouterr()
    merged = captured.out + captured.err
    events = [record.get("event") for record in _parse_lines(merged)]
    events.extend(record.get("event") for record in buffer)
    assert "OPEN_APPLIED" in events
    assert "SAVE_COMMIT" in events


def test_entry_suppressed_logging(capfd, monkeypatch):
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "0")
    position_tracker.set_audit_context(source="test", run_id="R2")
    buffer: List[Dict[str, Any]] = []
    handler = _ListHandler(buffer)
    position_tracker.LOGGER.addHandler(handler)
    position_tracker.log_audit_event(
        "entry suppressed",
        event="ENTRY_SUPPRESSED",
        asset="XAGUSD",
        intent="entry",
        decision="buy",
        suppression_reason="cooldown_active",
    )

    position_tracker.LOGGER.removeHandler(handler)

    captured = capfd.readouterr()
    merged = captured.out + captured.err
    entries = _parse_lines(merged)
    entries.extend(buffer)
    suppressed = [entry for entry in entries if entry.get("event") == "ENTRY_SUPPRESSED"]
    assert suppressed
    last = suppressed[-1]
    assert last.get("suppression_reason") == "cooldown_active"
    assert last.get("source") == "test"
    assert last.get("run_id") == "R2"


def test_audit_includes_github_run_id(capfd, monkeypatch):
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "0")
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    position_tracker.set_audit_context(source="test", run_id="R3")
    buffer: List[Dict[str, Any]] = []
    handler = _ListHandler(buffer)
    position_tracker.LOGGER.addHandler(handler)

    position_tracker.log_audit_event(
        "entry suppressed",
        event="ENTRY_SUPPRESSED",
        asset="XAGUSD",
        intent="entry",
        decision="buy",
        suppression_reason="cooldown_active",
    )

    position_tracker.LOGGER.removeHandler(handler)

    captured = capfd.readouterr()
    merged = captured.out + captured.err
    entries = _parse_lines(merged)
    entries.extend(buffer)
    assert any(entry.get("gh_run_id") == "123456" for entry in entries)


def test_audit_file_rotation_keeps_configured_count(tmp_path, monkeypatch):
    _reset_audit_file_handler()
    audit_path = tmp_path / "_manual_positions_audit.jsonl"
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "1")
    monkeypatch.setenv("MANUAL_POS_AUDIT_FILE", str(audit_path))
    monkeypatch.setattr(position_tracker, "_audit_rotation_limits", lambda: (80, 2))

    position_tracker.set_audit_context(source="test", run_id="ROTATE")
    for index in range(6):
        position_tracker.log_audit_event("rotation probe", event="ROTATE_PROBE", index=index)

    _reset_audit_file_handler()
    assert audit_path.exists()
    assert (tmp_path / "_manual_positions_audit.1.jsonl").exists()
    assert (tmp_path / "_manual_positions_audit.2.jsonl").exists()
    assert not (tmp_path / "_manual_positions_audit.3.jsonl").exists()


def test_manual_positions_logger_setup_is_idempotent():
    position_tracker._LOGGER_CONFIGURED = False
    before = len(position_tracker.LOGGER.handlers)
    position_tracker._configure_logger()
    position_tracker._configure_logger()
    assert len(position_tracker.LOGGER.handlers) == before


def test_load_positions_emits_one_db_load_audit_record(tmp_path, monkeypatch):
    _reset_audit_file_handler()
    audit_path = tmp_path / "_manual_positions_audit.jsonl"
    db_path = tmp_path / "trading.db"
    state_db.initialize(db_path)
    monkeypatch.setenv("MANUAL_POS_AUDIT_TO_FILE", "1")
    monkeypatch.setenv("MANUAL_POS_AUDIT_FILE", str(audit_path))

    position_tracker.set_audit_context(source="test", run_id="LOAD")
    position_tracker._maybe_attach_file_handler()
    position_tracker.load_positions(str(db_path), treat_missing_as_flat=False)

    _reset_audit_file_handler()
    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    assert [record.get("event") for record in records].count("LOAD_POSITIONS_DB") == 1
