import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import json
import logging
from datetime import datetime, timezone

import position_tracker


class _ListHandler(logging.Handler):
    def __init__(self, buffer: List[Dict[str, Any]]):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.buffer.append(record.__dict__)


def _parse_lines(out: str):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


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
        tp2=26.0,
        opened_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    path = tmp_path / "positions.json"
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
