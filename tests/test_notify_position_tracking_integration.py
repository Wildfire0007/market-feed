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


def test_notify_persists_open_after_market_precision_entry_card(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "precision_plan": {
                "direction": "sell",
                "order_type": "MARKET",
                "entry": 25.0,
                "stop_loss": 25.2,
                "take_profit_1": 24.8,
                "take_profit_2": 24.6,
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

    def _open_position(asset, side, entry, sl, tp1, tp2, opened_at_utc, order_type=None, positions=None):
        next_positions = dict(positions)
        next_positions[asset] = {
            "status": "open",
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "opened_at_utc": opened_at_utc,
        }
        return next_positions

    monkeypatch.setattr(notify.position_tracker, "open_position", _open_position)
    monkeypatch.setattr(
       notify.position_tracker,
        "save_positions_atomic",
        lambda path, data: persisted.append((path, data.copy())),
    )

    notify.check_and_notify()

    assert len(sent) == 1
    assert len(persisted) == 1
    assert persisted[0][1]["XAGUSD"]["status"] == "open"
    assert persisted[0][1]["XAGUSD"]["side"] == "short"


def test_notify_drops_entry_when_levels_do_not_match_direction(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "precision_plan": {
                "direction": "sell",
                "order_type": "MARKET",
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
    monkeypatch.setattr(
        notify.position_tracker,
        "save_positions_atomic",
        lambda path, data: persisted.append((path, data.copy())),
    )

    notify.check_and_notify()

    assert sent == []
    assert persisted == []


def test_notify_precision_arming_not_blocked_by_notify_should_notify_false(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "notify": {"should_notify": False, "reason": "non_actionable"},
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
    monkeypatch.setattr(notify.position_tracker, "register_precision_pending_position", lambda *args: args[-1])
    monkeypatch.setattr(
        notify.position_tracker,
        "save_positions_atomic",
        lambda path, data: persisted.append((path, data.copy())),
    )

    notify.check_and_notify()

    assert len(sent) == 1


def test_notify_precision_arming_autocorrects_direction_before_entry_gate(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "precision_arming",
            "precision_plan": {
                "direction": "buy",
                "order_type": "MARKET",
                "entry": 25.0,
                "stop_loss": 25.2,
                "take_profit_1": 24.8,
                "take_profit_2": 24.6,
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
    monkeypatch.setattr(notify.position_tracker, "open_position", lambda *args, **kwargs: kwargs.get("positions") or args[-1])
    monkeypatch.setattr(notify.position_tracker, "save_positions_atomic", lambda *_: None)

    notify.check_and_notify()
    assert len(sent) == 1


def test_notify_persists_open_for_plain_market_buy_signal(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "buy",
            "entry_order_type": "MARKET",
            "entry": 25.0,
            "sl": 24.8,
            "tp1": 25.2,
            "tp2": 25.4,
            "spot": {"price": 25.0},
            "gates": {"missing": []},
            "biases": {},
            "reasons": ["teszt"],
            "alignment_state": "TREND",
            "notify": {"should_notify": True},
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

    def _open_position(asset, side, entry, sl, tp1, tp2, opened_at_utc, order_type=None, positions=None):
        next_positions = dict(positions)
        next_positions[asset] = {"status": "open", "side": side, "entry": entry}
        return next_positions

    monkeypatch.setattr(notify.position_tracker, "open_position", _open_position)
    monkeypatch.setattr(
        notify.position_tracker,
        "save_positions_atomic",
        lambda path, data: persisted.append((path, data.copy())),
    )

    notify.check_and_notify()

    assert len(sent) == 1
    assert len(persisted) == 1
    assert persisted[0][1]["XAGUSD"]["status"] == "open"
    assert persisted[0][1]["XAGUSD"]["side"] == "long"



def test_notify_sends_single_activation_card_for_open_position(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "no entry",
            "spot": {"price": 25.0},
            "gates": {"missing": []},
            "biases": {},
            "alignment_state": "TREND",
        },
    )

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(
        notify.position_tracker,
        "load_positions",
        lambda *_: {
            "XAGUSD": {
                "status": "open",
                "side": "long",
                "opened_at_utc": "2026-01-01T00:00:00Z",
                "entry": 25.0,
                "order_type": "LIMIT",
                "sl": 24.8,
                "tp1": 25.2,
                "tp2": 25.4,
            }
        },
    )
    monkeypatch.setattr(
        notify.position_tracker,
        "compute_state",
        lambda *_args, **_kwargs: {"has_position": True, "pending_active": False, "cooldown_active": False},
    )

    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    notify.check_and_notify()
    notify.check_and_notify()

    activation = [
        item
        for item in sent
        if "XAGUSD" in item.get("title", "")
        and "Állapot: `Nyitott`" in (item.get("description") or "")
    ]
    assert len(activation) == 1
    assert "Belépő típus: `LIMIT`" in (activation[0].get("description") or "")
    assert "Aktiválva: `2026-01-01 01:00:00`" in (activation[0].get("description") or "")


def test_notify_sends_close_card_once_for_tp2_close(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "no entry",
            "spot": {"price": 25.4},
            "gates": {"missing": []},
            "biases": {},
            "alignment_state": "TREND",
        },
    )

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(
        notify.position_tracker,
        "load_positions",
        lambda *_: {
            "XAGUSD": {
                "status": "closed",
                "close_reason": "tp2_hit",
                "closed_at_utc": "2026-01-01T00:05:00Z",
            }
        },
    )
    monkeypatch.setattr(
        notify.position_tracker,
        "compute_state",
        lambda *_args, **_kwargs: {"has_position": False, "pending_active": False, "cooldown_active": True},
    )

    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    notify.check_and_notify()
    notify.check_and_notify()

    closed = [
        item
        for item in sent
        if "XAGUSD" in item.get("title", "")
        and "Állapot: `Lezárt`" in (item.get("description") or "")
    ]
    assert len(closed) == 1


def test_notify_lifecycle_not_blocked_by_alignment_or_gates(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_path = public_dir / "XAGUSD" / "signal.json"
    _write_signal(
        signal_path,
        {
            "signal": "no entry",
            "spot": {"price": 25.0},
            "gates": {"missing": ["spread_guard"]},
            "biases": {},
            "alignment_state": "MIXED",
        },
    )

    monkeypatch.setattr(notify, "PUBLIC_DIR", public_dir)
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(
        notify.position_tracker,
        "load_positions",
        lambda *_: {
            "XAGUSD": {
                "status": "open",
                "side": "long",
                "opened_at_utc": "2026-01-01T00:00:00Z",
                "entry": 25.0,
                "sl": 24.8,
                "tp1": 25.2,
                "tp2": 25.4,
            }
        },
    )
    monkeypatch.setattr(
        notify.position_tracker,
        "compute_state",
        lambda *_args, **_kwargs: {"has_position": True, "pending_active": False, "cooldown_active": False},
    )

    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed))

    notify.check_and_notify()

    assert any(
        "XAGUSD" in item.get("title", "") and "Állapot: `Nyitott`" in (item.get("description") or "")
        for item in sent        
    )
