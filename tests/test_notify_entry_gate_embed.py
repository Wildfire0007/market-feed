import json
from pathlib import Path

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
