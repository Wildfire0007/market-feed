import json
from pathlib import Path

from freezegun import freeze_time

import scripts.notify_discord as notify


def _write_signal(public: Path) -> None:
    asset_dir = public / "GOLD_CFD"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "signal.json").write_text(json.dumps({
        "signal": "buy",
        "order_type": "LIMIT",
        "entry": 2300,
        "sl": 2290,
        "tp1": 2320,
        "tp2": 2340,
        "probability_raw": 55,
        "spot": {"price": 2301},
        "reasons": ["fixture"],
    }), encoding="utf-8")


def test_second_identical_entry_within_cooldown_dispatches_nothing(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    _write_signal(public)
    monkeypatch.setattr(notify, "PUBLIC_DIR", public)
    monkeypatch.setattr(notify, "LIFECYCLE_INBOX_PATH", public / "_position_lifecycle_inbox.jsonl")
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"GOLD_CFD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "build_expected_trade_outcome", lambda *a, **k: {"passes": True, "tp1_net_usd": 12, "valid_for_minutes": 120, "notional_usd": 1000, "current_chase_r": 0, "max_entry_price": 2310})
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *a, **k: {})
    monkeypatch.setattr(notify.position_tracker, "compute_state", lambda *a, **k: {"has_position": False, "pending_active": False})
    monkeypatch.setattr(notify.position_tracker, "open_position", lambda *a, **k: {"GOLD_CFD": {"status": "open"}})
    monkeypatch.setattr(notify.position_tracker, "save_positions_atomic", lambda *a, **k: None)
    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed) or True)

    with freeze_time("2026-07-08T09:00:00Z"):
        notify.check_and_notify()
    with freeze_time("2026-07-08T09:20:00Z"):
        notify.check_and_notify()

    assert len(sent) == 1
    state = json.loads((public / "_notify_state.json").read_text(encoding="utf-8"))
    assert state["GOLD_CFD"]["last_entry_sent_utc"] == "2026-07-08T09:00:00Z"
