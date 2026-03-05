from __future__ import annotations

import json
from pathlib import Path

import scripts.notify_discord as notify


def _write_signal(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_notify_blocks_duplicate_entry_when_pending_position_exists(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "precision_plan": {
                "direction": "buy",
                "order_type": "LIMIT",
                "entry": 25.0,
                "stop_loss": 24.8,
                "take_profit_1": 25.2,
                "take_profit_2": 25.4,
            },
            "spot": {"price": 25.0},
            "gates": {"missing": []},
            "biases": {},
            "reasons": ["teszt"],
            "alignment_state": "TREND",
        },
    )

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", True)
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *_: {"XAGUSD": {"status": "pending"}})
    monkeypatch.setattr(
        notify.position_tracker,
        "compute_state",
        lambda *_args, **_kwargs: {"has_position": False, "pending_active": True, "cooldown_active": False},
    )

    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    notify.check_and_notify()

    assert sent == []


def test_notify_persists_pending_after_precision_entry_card(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "precision_plan": {
                "direction": "buy",
                "order_type": "LIMIT",
                "entry": 25.0,
                "stop_loss": 24.8,
                "take_profit_1": 25.2,
                "take_profit_2": 25.4,
            },
            "spot": {"price": 25.0},
            "gates": {"missing": []},
            "biases": {},
            "reasons": ["teszt"],
            "alignment_state": "TREND",
        },
    )

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *_: {})
    monkeypatch.setattr(
        notify.position_tracker,
        "compute_state",
        lambda *_args, **_kwargs: {"has_position": False, "pending_active": False, "cooldown_active": False},
    )

    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    persisted = []

    def _register(asset, signal_payload, now_dt, positions):
        next_positions = dict(positions)
        next_positions[asset] = {"status": "pending", "entry": signal_payload["precision_plan"]["entry"]}
        return next_positions

    monkeypatch.setattr(notify.position_tracker, "register_precision_pending_position", _register)
    monkeypatch.setattr(
        notify.position_tracker,
        "save_positions_atomic",
        lambda path, data: persisted.append((path, data.copy())),
    )

    notify.check_and_notify()

    assert len(sent) == 1
    assert len(persisted) == 1
    assert persisted[0][1]["XAGUSD"]["status"] == "pending"
