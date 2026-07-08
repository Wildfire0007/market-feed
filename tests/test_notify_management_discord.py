import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NOTIFY_PUBLIC_DIR", str(tmp_path))
    module_name = "scripts.notify_management_discord"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_management_spot_stale_guard_blocks_tp_sl(monkeypatch, tmp_path: Path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_exit_cards_enabled", lambda: True)    

    now = datetime.now(timezone.utc)
    stale_ts = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(module, "_load_open_positions", lambda: {
        "BTCUSD": {
            "status": "open",
            "side": "long",
            "entry": 100.0,
            "tp1": 101.0,
            "tp2": 102.0,
            "sl": 99.0,
            "opened_at_utc": "2026-01-01T00:00:00Z",
        }
    })

    def fake_load_json(path: Path):
        if path == module.STATE_PATH:
            return {}
        if path.name == "signal.json":
            return {
                "spot": {"price": 103.0, "utc": stale_ts},
                "exit_signal": {"state": "hold"},
            }
        return {}

    sent = []
    monkeypatch.setattr(module, "load_json", fake_load_json)
    monkeypatch.setattr(module, "_fetch_live_quote", lambda asset: {})
    monkeypatch.setattr(module, "_send_embed", lambda embed: sent.append(embed))

    module.process()

    assert sent == []
    diag = (tmp_path / "debug" / "management_notify_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(diag[-1])
    assert payload["event"] == "management_eval"
    assert payload["skip_reason"] == "spot_stale_or_missing"
    assert payload["sent_events"] == []


def test_management_live_quote_fallback_unblocks_tp_sl(monkeypatch, tmp_path: Path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_exit_cards_enabled", lambda: True)    

    now = datetime.now(timezone.utc)
    stale_ts = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fresh_ts = now.isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(module, "_load_open_positions", lambda: {
        "BTCUSD": {
            "status": "open",
            "side": "long",
            "entry": 100.0,
            "tp1": 101.0,
            "tp2": 102.0,
            "sl": 99.0,
            "opened_at_utc": "2026-01-01T00:00:00Z",
        }
    })

    def fake_load_json(path: Path):
        if path == module.STATE_PATH:
            return {}
        if path.name == "signal.json":
            return {
                "spot": {"price": 103.0, "utc": stale_ts},
                "exit_signal": {"state": "hold"},
            }
        return {}

    sent = []
    monkeypatch.setattr(module, "load_json", fake_load_json)
    monkeypatch.setattr(module, "_fetch_live_quote", lambda asset: {"price": 103.0, "utc": fresh_ts, "source": "test"})
    monkeypatch.setattr(module, "_send_embed", lambda embed: sent.append(embed))

    module.process()

    assert len(sent) == 1
    diag = (tmp_path / "debug" / "management_notify_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    fallback = json.loads(diag[-2])
    summary = json.loads(diag[-1])
    assert fallback["event"] == "spot_fallback_used"
    assert summary["event"] == "management_eval"
    assert summary["spot_fresh"] is True
    assert summary["sent_events"] == ["tp2_hit"]


def test_management_exit_cards_disabled_by_default(monkeypatch, tmp_path: Path):
    module = _load_module(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(module, "_load_open_positions", lambda: {"BTCUSD": {"status": "open", "side": "long", "entry": 100.0, "tp1": 101.0, "tp2": 102.0, "sl": 99.0, "opened_at_utc": "2026-01-01T00:00:00Z"}})
    monkeypatch.setattr(module, "load_json", lambda path: {"spot": {"price": 103.0, "utc": now}, "exit_signal": {"state": "hard_exit", "reason": "trend_reversal"}} if path.name == "signal.json" else {})
    sent = []
    monkeypatch.setattr(module, "_send_embed", lambda embed: sent.append(embed))
    module.process()
    assert sent == []


def test_tp1_embed_full_close_and_partial_numeric(monkeypatch, tmp_path: Path):
    module = _load_module(monkeypatch, tmp_path)
    full = module._build_embed("tp1_hit", "XAGUSD", "long", {"entry": 30, "tp1_closes_position": True, "size_units": 10}, 31, "")
    assert full["title"] == "🟢 TP1 ELÉRVE – ZÁRD A TELJES POZÍCIÓT"
    assert "Zárd a teljes XAGUSD LONG pozíciót" in full["description"]
    partial = module._build_embed("tp1_hit", "XAGUSD", "long", {"entry": 30, "tp1_closes_position": False, "partial_close_pct": 0.5, "size_units": 10, "breakeven_sl": 30.02, "tp2": 32}, 31, "")
    assert partial["title"] == "🟠 TP1 ELÉRVE – RÉSZZÁRÁS + BE"
    assert "50%-át (5.00000 egység)" in partial["description"]
    assert "30.02" in partial["description"]


def test_close_embed_estimated_pnl_math(monkeypatch, tmp_path: Path):
    module = _load_module(monkeypatch, tmp_path)
    embed = module._build_embed("tp2_hit", "BTCUSD", "long", {"entry": 100, "size_units": 2}, 105, "")
    assert "Becsült realizált PnL: $10.00" in embed["description"]
    sl = module._build_embed("sl_hit", "BTCUSD", "short", {"entry": 100, "size_units": 3}, 102, "")
    assert "Becsült PnL: $-6.00" in sl["description"]    
