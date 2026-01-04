from datetime import timedelta

from conftest import apply_common_analysis_stubs, make_raw_klines


def test_micro_bias_locals_are_initialized(monkeypatch, analysis_module, fixed_now, asset_registry):
    analysis = analysis_module
    apply_common_analysis_stubs(analysis, monkeypatch, missing_models={})

    monkeypatch.setattr(analysis, "session_state", lambda asset, now=None: (True, {
        "open": True,
        "entry_open": True,
        "within_window": True,
        "within_entry_window": True,
        "within_monitor_window": True,
        "weekday_ok": True,
        "status": "open",
        "status_note": "test window",
    }))
    monkeypatch.setattr(analysis, "save_json", lambda path, payload: None)
    monkeypatch.setattr(analysis, "record_latency_alert", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "load_latency_profile", lambda outdir: {"ema_delay": 0})
    monkeypatch.setattr(analysis, "update_latency_profile", lambda outdir, latency_seconds: None)
    monkeypatch.setattr(analysis, "load_latency_guard_state", lambda outdir: {})

    # Force neutral HTF bias while micro-bias is long.
    monkeypatch.setattr(analysis, "bias_from_emas", lambda *a, **k: "neutral")
    monkeypatch.setattr(analysis, "structure_break_with_retest", lambda *a, **k: False)
    monkeypatch.setattr(analysis, "broke_structure", lambda *a, **k: False)
    monkeypatch.setattr(analysis, "ema_cross_recent", lambda *a, **k: False)

    micro_bos_calls = []

    def fake_micro_bos(k1m, k5m, direction, lookback=None):
        micro_bos_calls.append(direction)
        return direction == "long"

    monkeypatch.setattr(analysis, "micro_bos_with_retest", fake_micro_bos)

    asset_map = {}
    asset_map["klines_1m.json"] = make_raw_klines(fixed_now, 30, timedelta(minutes=1), 1.08)
    asset_map["klines_5m.json"] = make_raw_klines(fixed_now, 30, timedelta(minutes=5), 1.08)
    asset_map["klines_1h.json"] = make_raw_klines(fixed_now, 30, timedelta(hours=1), 1.08)
    asset_map["klines_4h.json"] = make_raw_klines(fixed_now, 30, timedelta(hours=4), 1.08)
    asset_map["klines_1m_meta.json"] = {}
    asset_map["klines_5m_meta.json"] = {}
    asset_map["klines_1h_meta.json"] = {}
    asset_map["klines_4h_meta.json"] = {}
    asset_map["spot.json"] = {
        "price": 1.08,
        "utc": fixed_now.isoformat(),
        "retrieved_at_utc": fixed_now.isoformat(),
    }
    asset_map["spot_realtime.json"] = {}
    asset_map["latency_profile.json"] = {"ema_delay": 0}
    asset_registry["EURUSD"] = asset_map

    result = analysis.analyze("EURUSD")
    assert isinstance(result, dict)
    # Regression guard: micro-bias evaluation runs without uninitialized locals.
    assert {"long", "short"}.issubset(set(micro_bos_calls))
 
EOF
)
