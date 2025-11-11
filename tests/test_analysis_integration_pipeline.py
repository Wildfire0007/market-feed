import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def _reload_analysis(monkeypatch):
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


def test_integration_latency_guard_and_precision(monkeypatch, tmp_path, caplog):
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

    alerts = []

    def fake_record_latency_alert(asset, feed, message, metadata=None):
        alerts.append((asset, feed, message, metadata))
        alert_path = tmp_path / f"alert_{asset}_{feed}.json"
        alert_path.write_text(json.dumps(metadata or {}, ensure_ascii=False), encoding="utf-8")
        return alert_path

    monkeypatch.setattr(analysis, "record_latency_alert", fake_record_latency_alert)

    def fake_session_state(asset, now=None):
        return True, {
            "open": True,
            "entry_open": True,
            "within_window": True,
            "within_entry_window": True,
            "within_monitor_window": True,
            "weekday_ok": True,
            "status": "open",
            "status_note": "Teszt ablak nyitva",
        }

    monkeypatch.setattr(analysis, "session_state", fake_session_state)
    monkeypatch.setattr(analysis, "evaluate_news_lockout", lambda asset, now: (False, None))
    monkeypatch.setattr(analysis, "load_funding_snapshot", lambda asset: {})
    monkeypatch.setattr(analysis, "load_tick_order_flow", lambda asset, outdir: {})
    monkeypatch.setattr(analysis, "compute_order_flow_metrics", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "current_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "log_feature_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "inspect_model_artifact", lambda asset: {})
    monkeypatch.setattr(analysis, "missing_model_artifacts", lambda asset: [])
    monkeypatch.setattr(analysis, "predict_signal_probability", lambda *a, **k: (0.42, {"model": "stub"}))
    monkeypatch.setattr(analysis, "runtime_dependency_issues", lambda: [])
    monkeypatch.setattr(analysis, "load_sentiment", lambda asset, now: ([], None))
    monkeypatch.setattr(analysis, "load_volatility_overlay", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "update_precision_gate_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_signal_health_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_data_latency_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_live_validation", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_signal_event", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_analysis_run", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_ml_model_status", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "load_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "record_anchor", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_anchor_metrics", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "ensure_closed_candles", lambda df, now, tolerance_seconds=0: df)
    monkeypatch.setattr(analysis, "file_mtime", lambda path: None)

    data_registry = {}
    for asset, stale_offset in {"USOIL": timedelta(minutes=8), "NVDA": timedelta(minutes=1)}.items():
        asset_map = {}
        final_k1m = fixed_now - stale_offset
        asset_map["klines_1m.json"] = _make_raw_klines(final_k1m, 30, timedelta(minutes=1), 70.0 if asset == "USOIL" else 450.0)
        asset_map["klines_5m.json"] = _make_raw_klines(fixed_now - timedelta(minutes=5), 30, timedelta(minutes=5), 71.0 if asset == "USOIL" else 452.0)
        asset_map["klines_1h.json"] = _make_raw_klines(fixed_now - timedelta(hours=1), 30, timedelta(hours=1), 72.0 if asset == "USOIL" else 455.0)
        asset_map["klines_4h.json"] = _make_raw_klines(fixed_now - timedelta(hours=4), 30, timedelta(hours=4), 73.0 if asset == "USOIL" else 458.0)
        asset_map["klines_1m_meta.json"] = {}
        asset_map["klines_5m_meta.json"] = {}
        asset_map["klines_1h_meta.json"] = {}
        asset_map["klines_4h_meta.json"] = {}
        spot_time = fixed_now - timedelta(minutes=2)
        asset_map["spot.json"] = {
            "price": 70.5 if asset == "USOIL" else 455.5,
            "utc": spot_time.isoformat(),
            "retrieved_at_utc": spot_time.isoformat(),
        }
        asset_map["spot_realtime.json"] = {}
        asset_map["latency_profile.json"] = {"ema_delay": 120}
        asset_map["order_flow_ticks.json"] = {}
        data_registry[asset] = asset_map

    def fake_load_json(path):
        p = Path(path)
        asset = p.parent.name
        asset_map = data_registry.get(asset)
        if asset_map and p.name in asset_map:
            return asset_map[p.name]
        return {}

    monkeypatch.setattr(analysis, "load_json", fake_load_json)
    monkeypatch.setattr(analysis, "load_latency_profile", lambda outdir: {"ema_delay": 90})
    monkeypatch.setattr(analysis, "update_latency_profile", lambda outdir, latency_seconds: None)

    saved_payloads = {}

    def fake_save_json(path, payload):
        saved_payloads[str(path)] = payload

    monkeypatch.setattr(analysis, "save_json", fake_save_json)

    caplog.set_level("INFO", logger=analysis.LOGGER.name)

    guard_result = analysis.analyze("USOIL")
    assert guard_result["signal"] in {"no entry", "market closed"}
    assert any("Critical data latency" in reason for reason in guard_result.get("reasons", []))
    assert alerts and alerts[0][0] == "USOIL"
    assert alerts[0][1] == "k1m"

    monkeypatch.setattr(analysis, "ENTRY_THRESHOLD_PROFILE_NAME", "suppressed", raising=False)
    precision_result = analysis.analyze("NVDA")
    nvda_meta: dict = {}
    if isinstance(precision_result, dict):
        nvda_meta = precision_result.get("entry_thresholds_meta") or precision_result.get("entry_thresholds", {})
    fib_tol = nvda_meta.get("fib_tolerance_fraction")
    if fib_tol is None:
        nvda_signal_path = next(
            (key for key in saved_payloads if key.endswith("NVDA/signal.json")),
            None,
        )
        if nvda_signal_path is not None:
            payload = saved_payloads[nvda_signal_path]
            saved_meta = payload.get("entry_thresholds_meta") or payload.get("entry_thresholds", {})
            fib_tol = saved_meta.get("fib_tolerance_fraction")
    assert fib_tol is not None
    assert pytest.approx(fib_tol, rel=1e-4) == 0.004
    assert any(
        getattr(record, "gate", "") == "can_enter_core" and record.asset == "NVDA"
        for record in caplog.records
    )
