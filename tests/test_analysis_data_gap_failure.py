import json
import logging
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import apply_common_analysis_stubs, make_raw_klines


def test_main_raises_on_probability_data_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog, analysis_module, fixed_now, asset_registry
):
    analysis = analysis_module    
    monkeypatch.setattr(analysis, "ASSETS", ["BTCUSD"], raising=False)

    apply_common_analysis_stubs(
        analysis,
        monkeypatch,
        missing_models={},
        record_run_result=(None, None, None, False),
        record_status_result=(None, False),
    )
    
    asset_map = {}
    k1m_final = fixed_now - timedelta(minutes=48)
    asset_map["klines_1m.json"] = make_raw_klines(k1m_final, 60, timedelta(minutes=1), 30000.0)
    asset_map["klines_5m.json"] = make_raw_klines(fixed_now - timedelta(minutes=5), 60, timedelta(minutes=5), 30010.0)
    asset_map["klines_1h.json"] = make_raw_klines(fixed_now - timedelta(hours=1), 60, timedelta(hours=1), 30050.0)
    asset_map["klines_4h.json"] = make_raw_klines(fixed_now - timedelta(hours=4), 60, timedelta(hours=4), 30100.0)
    asset_map["klines_1m_meta.json"] = {}
    asset_map["klines_5m_meta.json"] = {}
    asset_map["klines_1h_meta.json"] = {}
    asset_map["klines_4h_meta.json"] = {}
    spot_time = fixed_now - timedelta(minutes=2)
    asset_map["spot.json"] = {
        "price": 30025.0,
        "utc": spot_time.isoformat(),
        "retrieved_at_utc": spot_time.isoformat(),
    }
    asset_map["spot_realtime.json"] = {}
    asset_map["latency_profile.json"] = {"ema_delay": 90}
    asset_map["order_flow_ticks.json"] = {}
    asset_registry["BTCUSD"] = asset_map
    monkeypatch.setattr(analysis, "load_latency_profile", lambda outdir: {"ema_delay": 90})
    monkeypatch.setattr(analysis, "update_latency_profile", lambda outdir, latency_seconds: None)

    def fake_analyze(asset: str):
        return {
            "asset": asset,
            "probability_stack": {"status": "data_gap", "source": "sklearn"},
            "diagnostics": {},
        }

    monkeypatch.setattr(analysis, "analyze", fake_analyze)

    caplog.set_level("WARNING", logger=analysis.LOGGER.name)

    original_handlers = list(analysis.LOGGER.handlers)
    original_level = analysis.LOGGER.level
    original_propagate = analysis.LOGGER.propagate
    try:
        analysis.main()
    finally:
        current_handlers = list(analysis.LOGGER.handlers)
        for handler in current_handlers:
            if handler not in original_handlers:
                try:
                    handler.close()
                finally:
                    analysis.LOGGER.removeHandler(handler)
        for handler in original_handlers:
            if handler not in analysis.LOGGER.handlers:
                analysis.LOGGER.addHandler(handler)
        analysis.LOGGER.setLevel(original_level)
        analysis.LOGGER.propagate = original_propagate

    error_records = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert not error_records

    summary_path = Path(tmp_path) / "analysis_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = summary.get("assets", {}).get("BTCUSD")
    assert isinstance(payload, dict)
    stack = payload.get("probability_stack")
    assert isinstance(stack, dict)
    assert stack.get("status") == "data_gap"
