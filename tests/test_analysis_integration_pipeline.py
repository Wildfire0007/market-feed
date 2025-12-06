import json
from datetime import timedelta

import pytest

from conftest import apply_common_analysis_stubs, make_raw_klines


def test_integration_latency_guard_and_precision(
    monkeypatch, tmp_path, caplog, analysis_module, fixed_now, asset_registry
):
    analysis = analysis_module
    
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
    apply_common_analysis_stubs(analysis, monkeypatch, missing_models={})
    
    for asset, stale_offset in {"USOIL": timedelta(minutes=16), "NVDA": timedelta(minutes=1)}.items():
        asset_map = {}
        final_k1m = fixed_now - stale_offset
        asset_map["klines_1m.json"] = make_raw_klines(final_k1m, 30, timedelta(minutes=1), 70.0 if asset == "USOIL" else 450.0)
        asset_map["klines_5m.json"] = make_raw_klines(fixed_now - timedelta(minutes=5), 30, timedelta(minutes=5), 71.0 if asset == "USOIL" else 452.0)
        asset_map["klines_1h.json"] = make_raw_klines(fixed_now - timedelta(hours=1), 30, timedelta(hours=1), 72.0 if asset == "USOIL" else 455.0)
        asset_map["klines_4h.json"] = make_raw_klines(fixed_now - timedelta(hours=4), 30, timedelta(hours=4), 73.0 if asset == "USOIL" else 458.0)
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
        asset_registry[asset] = asset_map
   
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
    expected_fib_tol = analysis.get_fib_tolerance("NVDA", profile="suppressed")
    assert pytest.approx(fib_tol, rel=1e-4) == expected_fib_tol
    assert any(
        getattr(record, "gate", "") == "can_enter_core" and record.asset == "NVDA"
        for record in caplog.records
    )
