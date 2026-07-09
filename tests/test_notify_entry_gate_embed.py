import json
from pathlib import Path

from datetime import datetime, timezone

import scripts.notify_discord as notify


def write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "entry_gate_stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_entry_gate_summary_includes_assets(tmp_path, monkeypatch):
    payload = {
        "BTCUSD": [
            {"missing": ["session", "atr"], "precision_hiany": []},
            {"missing": [], "precision_hiany": []},
        ],
        "EURUSD": [
            {"missing": [], "precision_hiany": ["precision"]},
        ],
    }
    stats_path = write_payload(tmp_path, payload)
    monkeypatch.setattr(notify, "ENTRY_GATE_STATS_PATH", stats_path)

    embed = notify.build_entry_gate_summary_embed()

    assert embed is not None
    assert embed["title"] == "Entry gate toplista (24h)"
    assert "session" in embed["description"]
    fields = embed.get("fields") or []
    assert fields, "Asset field is required for disambiguation"
    field_value = fields[0]["value"]
    assert "BTCUSD" in field_value and "EURUSD" in field_value
    assert "blokkolva" in field_value



def test_entry_gate_summary_sorts_by_rejections(tmp_path, monkeypatch):
    payload = {
        "USOIL": [
            {"missing": ["liquidity"], "precision_hiany": []},
            {"missing": ["session"], "precision_hiany": []},
        ],
        "NVDA": [
            {"missing": ["precision"], "precision_hiany": ["precision"]},
        ],
    }
    stats_path = write_payload(tmp_path, payload)
    monkeypatch.setattr(notify, "ENTRY_GATE_STATS_PATH", stats_path)

    embed = notify.build_entry_gate_summary_embed()

    field_value = embed.get("fields")[0]["value"]
    lines = field_value.splitlines()
    assert lines[0].startswith("• USOIL: 2x")
    assert lines[1].startswith("• NVDA: 1x")


def test_entry_gate_summary_falls_back_to_jsonl(tmp_path, monkeypatch):
    stats_path = tmp_path / "entry_gate_stats.json"
    log_dir = tmp_path / "debug" / "entry_gates"
    log_dir.mkdir(parents=True)

    log_path = log_dir / "entry_gates_2025-01-01.jsonl"
    lines = [
        {
            "asset": "BTCUSD",
            "timestamp": "2025-01-01T11:00:00Z",
            "reasons": ["session", "precision_gate"],
        },
        {
            "asset": "EURUSD",
            "ts_utc": "2024-12-30T10:00:00Z",  # cutoff miatt ignore            
            "missing": ["atr"],
        },
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    monkeypatch.setattr(notify, "ENTRY_GATE_STATS_PATH", stats_path)
    monkeypatch.setattr(notify, "ENTRY_GATE_LOG_DIR", log_dir)

    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    embed = notify.build_entry_gate_summary_embed(now=now)

    assert embed is not None
    assert "session" in embed.get("description", "")
    field_value = (embed.get("fields") or [{}])[0].get("value", "")
    assert "BTCUSD" in field_value
    assert "EURUSD" not in field_value



def test_entry_embed_shows_broker_ready_sizing_lines(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    asset_dir = public_dir / "XAGUSD"
    asset_dir.mkdir(parents=True)
    (asset_dir / "signal.json").write_text(json.dumps({
        "signal": "sell",
        "spot": {"price": 58.75034},
        "entry": 58.75034,
        "sl": 60.69388779,
        "tp1": 57.97291657,
        "tp2": 57.58420486,
        "probability_raw": 40,
        "reasons": ["fixture"],
    }), encoding="utf-8")
    captured = []

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", True)
    monkeypatch.setattr(notify.settings, "LEVERAGE", {"XAGUSD": 10.0})
    monkeypatch.setattr(notify.settings, "ASSET_COST_MODEL", {"XAGUSD": {"round_trip_pct": 0.0}})
    monkeypatch.setattr(notify.settings, "MANUAL_TRADE_MODEL", {
        "sl_risk_usd": 50.0,
        "equity_usd": 100.0,
        "leverage": 10.0,
        "tp1_close_fraction": 1.0,
        "tp1_min_net_usd": 1.0,
        "eta_min_minutes": 1,
        "eta_max_minutes": 999,
        "max_chase_r": 999,
    })
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *args, **kwargs: {})
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: captured.append(embed) or True)

    notify.check_and_notify()

    embed = captured[0]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert "Profit-cél számítási alap:" in fields["🎯 Profit cél"]
    assert "Tőkeáttételes méret:" not in fields["🎯 Profit cél"]
    assert "Notional: `$1511.42`" in fields["⚙️ Paraméterek az eToro-hoz"]
    assert "eToro Amount (X10): `$151.14`" in fields["⚙️ Paraméterek az eToro-hoz"]
    assert "Minimál-tétes protokoll: Amount × 0.1–0.2 az első 10 trade-re." in fields["🧭 Kezelési terv"]    
