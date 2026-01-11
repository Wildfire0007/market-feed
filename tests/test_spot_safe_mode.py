from datetime import timedelta

from conftest import apply_common_analysis_stubs, make_raw_klines


def test_safe_mode_triggers_on_stale_spot_timestamp(
    monkeypatch, analysis_module, fixed_now, asset_registry
):
    analysis = analysis_module

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
    monkeypatch.setattr(
        analysis,
        "SPOT_SAFE_MODE_SECONDS",
        {"default": 60, "BTCUSD": 60},
        raising=False,
    )
    monkeypatch.setattr(analysis, "load_latency_profile", lambda outdir: {"ema_delay": 90})
    monkeypatch.setattr(analysis, "update_latency_profile", lambda outdir, latency_seconds: None)
    apply_common_analysis_stubs(analysis, monkeypatch, missing_models={})

    asset_map = {}
    asset_map["klines_1m.json"] = make_raw_klines(
        fixed_now, 30, timedelta(minutes=1), 40000.0
    )
    asset_map["klines_5m.json"] = make_raw_klines(
        fixed_now, 30, timedelta(minutes=5), 40100.0
    )
    asset_map["klines_1h.json"] = make_raw_klines(
        fixed_now, 30, timedelta(hours=1), 40200.0
    )
    asset_map["klines_4h.json"] = make_raw_klines(
        fixed_now, 30, timedelta(hours=4), 40300.0
    )
    asset_map["klines_1m_meta.json"] = {}
    asset_map["klines_5m_meta.json"] = {}
    asset_map["klines_1h_meta.json"] = {}
    asset_map["klines_4h_meta.json"] = {}
    stale_spot_time = fixed_now - timedelta(seconds=120)
    asset_map["spot.json"] = {
        "price": 40500.0,
        "utc": stale_spot_time.isoformat(),
        "retrieved_at_utc": stale_spot_time.isoformat(),
    }
    asset_map["spot_realtime.json"] = {}
    asset_map["latency_profile.json"] = {"ema_delay": 120}
    asset_map["order_flow_ticks.json"] = {}
    asset_registry["BTCUSD"] = asset_map

    result = analysis.analyze("BTCUSD")

    assert result.get("signal") == "no entry"
    reasons = result.get("reasons", [])
    assert any("SAFE MODE" in str(reason) for reason in reasons)
    meta = result.get("entry_thresholds_meta") or result.get("entry_thresholds", {})
    assert meta.get("safe_mode", {}).get("active") is True
