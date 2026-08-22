import pytest

from risk_limits import evaluate_plan_feasibility

CFG = {"margin_usd": 100, "net_tp1_usd_min": 10, "max_required_move_atr1h_mult": 2.4}


def test_micro_plan_blocked_by_floor():
    st = evaluate_plan_feasibility(
        "GOLD_CFD", 4400.0, 4395.60, 40.0,
        min_stoploss_pct=0.008, profit_target_config=CFG,
        leverage=20.0, round_trip_cost=0.0006,
    )
    assert st["feasible"] is False
    assert st["reason"] == "min_stoploss_floor"
    assert st["r_pct"] == pytest.approx(0.001, rel=1e-6)


def test_live_2026_07_17_gold_plan_would_be_blocked():
    # Ledger replay: GOLD_CFD entry 3987.84 / SL 3975.94 -> R% ~0.2984% < 0.8% floor
    st = evaluate_plan_feasibility(
        "GOLD_CFD", 3987.84, 3975.94, 40.0,
        min_stoploss_pct=0.008, profit_target_config=CFG,
        leverage=20.0, round_trip_cost=0.0006,
    )
    assert st["feasible"] is False
    assert st["reason"] == "min_stoploss_floor"
    assert st["r_pct"] == pytest.approx(0.0029841, rel=1e-3)


def test_compliant_wide_plan_passes():
    st = evaluate_plan_feasibility(
        "GOLD_CFD", 4000.0, 3960.0, 12.0,
        min_stoploss_pct=0.008, profit_target_config=CFG,
        leverage=20.0, round_trip_cost=0.0006,
    )
    assert st["feasible"] is True
    assert st["reason"] is None
    assert st["required_pct"] == pytest.approx(0.0056, rel=1e-6)


def test_missing_atr1h_fails_closed():
    for atr in (None, 0.0):
        st = evaluate_plan_feasibility(
            "GOLD_CFD", 4000.0, 3960.0, atr,
            min_stoploss_pct=0.008, profit_target_config=CFG,
            leverage=20.0, round_trip_cost=0.0006,
        )
        assert st["feasible"] is False
        assert st["reason"] == "atr1h_missing"


def test_atr1h_ceiling_reject():
    st = evaluate_plan_feasibility(
        "GOLD_CFD", 4000.0, 3960.0, 6.0,
        min_stoploss_pct=0.008, profit_target_config=CFG,
        leverage=20.0, round_trip_cost=0.0006,
    )
    assert st["feasible"] is False
    assert st["reason"] == "atr1h_ceiling"
    assert st["ceiling_pct"] == pytest.approx(0.0036, rel=1e-6)


def test_tp1_net_min_reject():
    st = evaluate_plan_feasibility(
        "XAGUSD", 100.0, 99.8, 1.0,
        min_stoploss_pct=0.001, profit_target_config=CFG,
        leverage=10.0, round_trip_cost=0.0012,
    )
    assert st["feasible"] is False
    assert st["reason"] == "tp1_net_min"
    assert st["required_pct"] == pytest.approx(0.0112, rel=1e-6)
