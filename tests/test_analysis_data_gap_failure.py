import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def _reload_analysis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


def _make_raw_klines(final_time: datetime, periods: int, step: timedelta, base: float):
    rows = []
    start = final_time - step * (periods - 1)
    for idx in range(periods):
        ts = start + step * idx
        rows.append(
            {
                "datetime": ts.isoformat(),
                "open": base + idx * 0.1,
                "high": base + idx * 0.1 + 0.2,
                "low": base + idx * 0.1 - 0.2,
                "close": base + idx * 0.1 + 0.05,
                "volume": 100 + idx,
            }
        )
    return rows


def test_main_raises_on_probability_data_gap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog):
    analysis = _reload_analysis(monkeypatch)

    fixed_now = datetime(2024, 1, 10, 15, 0, tzinfo=timezone.utc)
    real_datetime = analysis.datetime

    class FixedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(analysis, "datetime", FixedDateTime)
    monkeypatch.setattr(analysis, "PUBLIC_DIR", str(tmp_path))
    monkeypatch.setattr(analysis, "ASSETS", ["BTCUSD"], raising=False)

    monkeypatch.setattr(analysis, "evaluate_news_lockout", lambda asset, now: (False, None))
    monkeypatch.setattr(analysis, "load_funding_snapshot", lambda asset: {})
    monkeypatch.setattr(analysis, "load_tick_order_flow", lambda asset, outdir: {})
    monkeypatch.setattr(analysis, "compute_order_flow_metrics", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "current_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "log_feature_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "inspect_model_artifact", lambda asset: {})
    monkeypatch.setattr(analysis, "missing_model_artifacts", lambda assets=None: {})
    monkeypatch.setattr(analysis, "predict_signal_probability", lambda *a, **k: (0.42, {"model": "stub"}))
    monkeypatch.setattr(analysis, "runtime_dependency_issues", lambda: [])
    monkeypatch.setattr(analysis, "load_sentiment", lambda asset, now: ([], None))
    monkeypatch.setattr(analysis, "load_volatility_overlay", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "update_precision_gate_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_signal_health_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_data_latency_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_live_validation", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_signal_event", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_analysis_run", lambda *a, **k: (None, None, None, False))
    monkeypatch.setattr(analysis, "record_ml_model_status", lambda *a, **k: (None, False))
    monkeypatch.setattr(analysis, "load_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "record_anchor", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_anchor_metrics", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "ensure_closed_candles", lambda df, now, tolerance_seconds=0: df)
    monkeypatch.setattr(analysis, "file_mtime", lambda path: None)

    data_registry = {}
    asset_map = {}
    k1m_final = fixed_now - timedelta(minutes=48)
    asset_map["klines_1m.json"] = _make_raw_klines(k1m_final, 60, timedelta(minutes=1), 30000.0)
    asset_map["klines_5m.json"] = _make_raw_klines(fixed_now - timedelta(minutes=5), 60, timedelta(minutes=5), 30010.0)
    asset_map["klines_1h.json"] = _make_raw_klines(fixed_now - timedelta(hours=1), 60, timedelta(hours=1), 30050.0)
    asset_map["klines_4h.json"] = _make_raw_klines(fixed_now - timedelta(hours=4), 60, timedelta(hours=4), 30100.0)
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
    data_registry["BTCUSD"] = asset_map

    def fake_load_json(path: str):
        p = Path(path)
        asset = p.parent.name
        asset_map = data_registry.get(asset)
        if asset_map and p.name in asset_map:
            return asset_map[p.name]
        return {}

    monkeypatch.setattr(analysis, "load_json", fake_load_json)
    monkeypatch.setattr(analysis, "load_latency_profile", lambda outdir: {"ema_delay": 90})
    monkeypatch.setattr(analysis, "update_latency_profile", lambda outdir, latency_seconds: None)

    def fake_analyze(asset: str):
        return {
            "asset": asset,
            "probability_stack": {"status": "data_gap", "source": "sklearn"},
            "diagnostics": {},
        }

    monkeypatch.setattr(analysis, "analyze", fake_analyze)

    caplog.set_level("INFO", logger=analysis.LOGGER.name)

    original_handlers = list(analysis.LOGGER.handlers)
    original_level = analysis.LOGGER.level
    original_propagate = analysis.LOGGER.propagate
    try:
        with pytest.raises(SystemExit) as excinfo:
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

    assert "BTCUSD" in str(excinfo.value)

    summary_path = Path(tmp_path) / "analysis_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = summary.get("assets", {}).get("BTCUSD")
    assert isinstance(payload, dict)
    stack = payload.get("probability_stack")
    assert isinstance(stack, dict)
    assert stack.get("status") == "data_gap"
