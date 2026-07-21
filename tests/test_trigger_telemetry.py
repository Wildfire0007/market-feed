import json
from datetime import datetime, timezone

import numpy as np


def test_armed_trigger_writes_all_subconditions(analysis_module, tmp_path):
    path = tmp_path / "debug" / "trigger_telemetry.jsonl"
    analysis_module.TRIGGER_TELEMETRY_PATH = path
    subconditions = {
        "score_ready": {"value": 72.0, "threshold": 70.0, "passed": True},
        "retest_touch": True,
        "bos_confirm": True,
        "stabilization": {"required": False, "passed": True},
        "order_flow": {"ready": True, "strength": 1.2},
        "price_trigger": {"inside_window": False, "hit_entry": False},
        "spread_guard": True,
        "other_gates": {"passed": True, "missing": []},
    }

    analysis_module._append_trigger_telemetry(
        "XAGUSD",
        73.5,
        {"trigger_state": "arming", "direction": "sell", "entry": 58.9, "stop_loss": 59.25},    
        subconditions,
        datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["asset"] == "XAGUSD"
    assert row["p_score"] == 73.5
    assert row["trigger_state"] == "arming"
    assert row["direction"] == "sell"
    assert row["entry"] == 58.9
    assert row["stop_loss"] == 59.25    
    assert set(row["subconditions"]) == set(subconditions)


def test_armed_trigger_serializes_numpy_subconditions(analysis_module, tmp_path):
    path = tmp_path / "debug" / "trigger_telemetry.jsonl"
    analysis_module.TRIGGER_TELEMETRY_PATH = path

    analysis_module._append_trigger_telemetry(
        "GOLD_CFD",
        73.5,
        {"trigger_state": "arming"},
        {"passed": np.bool_(False), "strength": np.float64(1.5)},
        datetime(2026, 7, 17, 15, 8, tzinfo=timezone.utc),
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["subconditions"] == {
        "passed": False,
        "strength": 1.5,
    }


def test_trigger_telemetry_uses_size_cap_rotation(analysis_module, tmp_path):
    path = tmp_path / "debug" / "trigger_telemetry.jsonl"
    analysis_module.TRIGGER_TELEMETRY_PATH = path
    analysis_module.TRIGGER_TELEMETRY_MAX_BYTES = 1
    analysis_module._append_trigger_telemetry(
        "XAGUSD", 73.5, {"trigger_state": "arming"}, {"spread_guard": True},
        datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )
    analysis_module._append_trigger_telemetry(
        "XAGUSD", 73.5, {"trigger_state": "arming"}, {"spread_guard": True},
        datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
    )

    assert path.with_name("trigger_telemetry.1.jsonl").exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
