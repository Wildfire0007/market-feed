import pytest

from profit_target import build_profit_target_levels
import analysis

COSTS = {"GOLD_CFD": {"type": "pct", "round_trip_pct": 0.0006}}
CFG = {"margin_usd": 100, "net_tp1_usd_min": 10, "tp2_rr_multiple": 2.0, "sl_rr_min": 1.5, "max_required_move_atr1h_mult": 1.2}


def test_profit_target_20x_cost_adjustment():
    r = build_profit_target_levels(asset="GOLD_CFD", side="buy", entry=100.0, leverage=20, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=1.0)
    assert r.feasible
    assert r.meta["required_move_atr1h_ceiling"] == pytest.approx(0.012)    
    assert r.meta["required_net_move"] == pytest.approx(0.005)
    assert r.meta["required_gross_move"] == pytest.approx(0.0056)
    assert r.tp1 == pytest.approx(100.56)


def test_profit_target_10x_requires_one_percent_before_cost():
    r = build_profit_target_levels(asset="GOLD_CFD", side="sell", entry=100.0, leverage=10, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=2.0)
    assert r.feasible
    assert r.meta["required_net_move"] == pytest.approx(0.01)
    assert r.tp1 == pytest.approx(98.94)


def test_profit_target_infeasible_when_atr1h_target_unrealistic():
    r = build_profit_target_levels(asset="GOLD_CFD", side="buy", entry=100.0, leverage=20, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=0.1)
    assert not r.feasible
    assert r.reason == "profit_target_infeasible"
    assert r.meta["required_move_atr1h_ceiling"] == pytest.approx(0.0012)
    assert r.meta["required_move_over_ceiling"] == pytest.approx(0.0044)    


def test_soft_penalty_cap():
    assert analysis._cap_soft_penalty(50, 20, 12) == pytest.approx(38)


def test_bos_direction_consistency():
    assert analysis._bos_direction_to_bias("bos_up") == "long"
    assert analysis._bos_direction_to_bias("bos_down") == "short"


def test_profit_target_feasibility_gap_record_pass_fields():
    cfg = dict(CFG, max_required_move_atr1h_mult=2.4)
    r = build_profit_target_levels(
        asset="GOLD_CFD",
        side="buy",
        entry=100.0,
        leverage=20,
        config=cfg,
        asset_cost_model=COSTS,
        min_stoploss_pct=0.001,
        atr5=0.01,
        atr1h=0.3,
    )

    record = analysis._profit_target_feasibility_gap_record(
        "GOLD_CFD", r.meta, r.feasible, analysis.parse_utc_timestamp("2026-07-07T08:00:00Z")
    )

    assert r.feasible
    assert record == {
        "ts_utc": "2026-07-07T08:00:00Z",
        "asset": "GOLD_CFD",
        "gate": "profit_target_feasibility",
        "result": "pass",
        "required_gross_move_pct": pytest.approx(0.56),
        "atr1h_pct": pytest.approx(0.3),
        "ceiling_pct": pytest.approx(0.72),
        "mult": 2.4,
    }    
