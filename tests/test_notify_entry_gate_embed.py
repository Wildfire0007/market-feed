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
            "utc_ts": "2024-12-30T10:00:00Z",  # cutoff miatt ignore
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
