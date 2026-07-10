import json
from pathlib import Path
import scripts.notify_discord as notify


def test_gate_rejected_armed_plan_suppresses_card_and_writes_shadow(tmp_path: Path, monkeypatch):
    public = tmp_path / "public"
    asset_dir = public / "XAGUSD"; asset_dir.mkdir(parents=True)
    (asset_dir / "signal.json").write_text(json.dumps({
        "signal":"precision_arming",
        "probability_raw":22,
        "entry_thresholds":{"p_score_min":32},
        "precision_plan":{"direction":"buy","order_type":"LIMIT","entry":25,"stop_loss":24.8,"take_profit_1":25.2,"take_profit_2":25.4},
        "spot":{"price":25},
        "gates":{"missing":["choppy_hard_block"]},
        "reasons":["teszt"],
    }), encoding="utf-8")
    monkeypatch.setattr(notify, "PUBLIC_DIR", public)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *_: {})
    monkeypatch.setattr(notify.position_tracker, "compute_state", lambda *_a, **_k: {"has_position": False, "pending_active": False, "cooldown_active": False})
    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    notify.check_and_notify()

    assert sent == []
    journal = public / "journal" / "trade_journal.csv"
    assert "suppressed_gate_mismatch" in journal.read_text(encoding="utf-8")
