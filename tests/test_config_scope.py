import json
from pathlib import Path


def test_active_assets_are_traded_scope():
    cfg = json.loads(Path("config/analysis_settings.json").read_text(encoding="utf-8"))
    assert cfg["assets"] == ["GOLD_CFD", "XAGUSD", "USOIL"]
    assert cfg["spot_safe_mode_seconds"]["default"] == 300
    assert cfg["td_calls_per_minute_max"] == 45


def test_measured_broker_cost_model_calibration():
    cfg = json.loads(Path("config/analysis_settings.json").read_text(encoding="utf-8"))
    costs = cfg["asset_cost_model"]

    assert costs["GOLD_CFD"]["round_trip_pct"] == 0.0006
    assert costs["XAGUSD"]["round_trip_pct"] == 0.0012
    assert costs["USOIL"]["round_trip_pct"] == 0.0011

    leverage = cfg["leverage"]
    model = cfg["profit_target"]
    margin = model["margin_usd"]
    net_min = model["net_tp1_usd_min"]

    assert net_min / (margin * leverage["XAGUSD"]) + costs["XAGUSD"]["round_trip_pct"] == 0.0112
    assert net_min / (margin * leverage["USOIL"]) + costs["USOIL"]["round_trip_pct"] == 0.0111    
