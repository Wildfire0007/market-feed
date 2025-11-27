import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis


CI_ONLY = pytest.mark.skipif(not os.getenv("CI"), reason="CI-only regression guard (rollback note)")


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_precision_gate_log_captures_missing_ofi():
    handler = ListHandler()
    logger = logging.getLogger("analysis.precision_test")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        analysis._emit_precision_gate_log(
            "NVDA",
            "can_enter_core",
            False,
            "ofi_missing",
            order_flow_metrics={},
            tick_order_flow=None,
            latency_seconds={"k1m": "720"},
            precision_plan={
                "order_flow_strength": None,
                "trigger_state": "idle",
                "trigger_ready": False,
                "score": 41.2,
            },
            logger=logger,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            extra={"missing": ["ofi"]},
        )
    finally:
        logger.removeHandler(handler)

    assert handler.records, "A precision lognak legalább egy rekordot kellett írnia"
    record = handler.records[-1]
    assert record.asset == "NVDA"
    assert record.gate == "can_enter_core"
    assert record.decision is False
    assert record.reason_code == "ofi_missing"
    assert record.ofi_present is False
    assert record.ofi_age_seconds == 720
    assert record.ofi_window_minutes == analysis.OFI_Z_LOOKBACK
    assert record.precision_score == 41.2
    assert record.missing == ["ofi"]
    assert record.timestamp_utc.startswith("2024-01-01 12:00")
    assert record.timestamp_cet.startswith("2024-01-01")


def test_precision_flow_missing_components_logged_for_special_assets():
    handler = ListHandler()
    logger = logging.getLogger("analysis.precision_flow_test")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    plan = {
        "order_flow_strength": 0.12,
        "order_flow_signals": 0,
        "order_flow_ready": False,
        "order_flow_blockers": ["imbalance -0.4"],
        "order_flow_settings": {"min_signals": 2, "strength_floor": 0.5},
        "order_flow_status": "stale",
        "order_flow_stalled": True,
        "trigger_state": "standby",
        "trigger_ready": False,
        "score": 48.5,
    }
    try:
        analysis._emit_precision_gate_log(
            "XAGUSD",
            "precision_flow",
            False,
            "flow_blocked",
            order_flow_metrics={},
            tick_order_flow={},
            latency_seconds={},
            precision_plan=plan,
            logger=logger,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    finally:
        logger.removeHandler(handler)

    record = handler.records[-1]
    details = record.missing_precision_components
    assert details["blockers"] == ["imbalance -0.4"]
    assert details["signals"] == {"actual": 0, "required": 2}
    assert details["strength"] == {"actual": 0.12, "required": 0.5}
    assert details["status"] == "stale"
    assert details["stalled"] is True


def test_gate_summary_writes_hungarian_context(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ENTRY_GATE_STATS_PATH", tmp_path / "entry_gate_stats.json")
    decision = {
        "gates": {"mode": "ok", "missing": []},
        "entry_thresholds": {"profile": "baseline", "atr_ratio_ok": True, "spread_gate_ok": True},
        "probability_raw": 55,
        "retrieved_at_utc": "2024-01-01T00:00:00Z",
        "entry_gate_context_hu": {
            "bos_visszatekintes": 20,
            "fib_zona": {"ok": True},
            "session_ablak_utc": {"session": []},
            "p_score_profil": "baseline",
        },
    }

    analysis._log_gate_summary("BTCUSD", decision)

    payload = json.loads((tmp_path / "entry_gate_stats.json").read_text(encoding="utf-8"))
    entry = payload["BTCUSD"][0]
    assert entry["bos_visszatekintes"] == 20
    assert entry["p_score_profil"] == "baseline"


@CI_ONLY
def test_gate_summary_persists_core_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ENTRY_GATE_STATS_PATH", tmp_path / "entry_gate_stats.json")
    decision = {
        "gates": {"mode": "session_closed", "missing": ["session"]},
        "entry_thresholds": {
            "profile": "usoil_baseline",
            "p_score_min": 50,
            "atr_threshold_effective": 2.5,
            "atr_ratio_ok": False,
            "spread_gate_ok": True,
        },
        "active_position_meta": {"atr5_rel": 1.1},
        "probability_raw": 44.2,
        "retrieved_at_utc": "2024-01-02T03:04:05Z",
    }

    analysis._log_gate_summary("USOIL", decision)

    payload = json.loads((tmp_path / "entry_gate_stats.json").read_text(encoding="utf-8"))
    entry = payload["USOIL"][0]
    assert entry["asset"] == "USOIL"
    assert entry["mode"] == "session_closed"
    assert entry["atr_rel"] == 1.1
    assert entry["atr_threshold"] == 2.5
    assert entry["spread_ok"] is True
    assert entry["p_score"] == 44.2
    assert entry["p_score_min"] == 50
