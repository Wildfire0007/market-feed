def test_gate_payload_includes_plan_geometry(analysis_module):
    payload = analysis_module._entry_gate_log_payload(
        "XAGUSD", None, ["P_score>=30"],
        plan={"direction": "sell", "entry": 59.1, "stop_loss": 59.4, "score": 51.0},
    )
    assert payload["result"] == "rejected"
    assert payload["direction"] == "sell"
    assert payload["entry"] == 59.1
    assert payload["stop_loss"] == 59.4
    assert payload["plan_score"] == 51.0


def test_gate_payload_without_plan_unchanged(analysis_module):
    payload = analysis_module._entry_gate_log_payload("GOLD_CFD", None, [])
    assert payload["result"] == "accepted"
    assert "entry" not in payload and "direction" not in payload
