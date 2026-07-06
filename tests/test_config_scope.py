import json
from pathlib import Path


def test_active_assets_are_traded_scope():
    cfg = json.loads(Path("config/analysis_settings.json").read_text(encoding="utf-8"))
    assert cfg["assets"] == ["GOLD_CFD", "XAGUSD", "USOIL"]
    assert cfg["spot_safe_mode_seconds"]["default"] == 300
    assert cfg["td_calls_per_minute_max"] == 45
