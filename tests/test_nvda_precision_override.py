from analysis import _nvda_precision_override_ready


def test_nvda_precision_override_ready_only_when_guards_pass():
    base_plan = {"trigger_state": "fire", "score": 52.0}
    assert _nvda_precision_override_ready(
        "NVDA",
        precision_plan=base_plan,
        precision_threshold=50.0,
        spread_gate_ok=True,
        session_entry_open=True,
        risk_guard_allowed=True,
        base_core_ok=True,
    )

    assert not _nvda_precision_override_ready(
        "NVDA",
        precision_plan={**base_plan, "score": 45.0},
        precision_threshold=50.0,
        spread_gate_ok=True,
        session_entry_open=True,
        risk_guard_allowed=True,
        base_core_ok=True,
    )

    assert not _nvda_precision_override_ready(
        "NVDA",
        precision_plan=base_plan,
        precision_threshold=50.0,
        spread_gate_ok=False,
        session_entry_open=True,
        risk_guard_allowed=True,
        base_core_ok=True,
    )

    assert not _nvda_precision_override_ready(
        "EURUSD",
        precision_plan=base_plan,
        precision_threshold=50.0,
        spread_gate_ok=True,
        session_entry_open=True,
        risk_guard_allowed=True,
        base_core_ok=True,
    )
