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


def _write_asset_signal(public: Path, asset: str = "XAGUSD") -> None:
    asset_dir = public / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "signal.json").write_text(json.dumps({
        "signal": "buy",
        "order_type": "LIMIT",
        "entry": 25.0,
        "sl": 24.5,
        "tp1": 26.0,
        "tp2": 27.0,
        "probability_raw": 57,
        "probability": 57,
        "spot": {"price": 25.1, "utc": "2026-07-09T09:00:00Z"},
        "reasons": ["fixture"],
        "retrieved_at_utc": "2026-07-09T09:00:00Z",
    }), encoding="utf-8")


def _setup_concurrency_notify(tmp_path, monkeypatch, *, cap: int):
    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr(notify, "PUBLIC_DIR", public)
    monkeypatch.setattr(notify, "LIFECYCLE_INBOX_PATH", public / "_position_lifecycle_inbox.jsonl")
    monkeypatch.setattr(notify, "LIFECYCLE_STATE_PATH", public / "_position_lifecycle_state.json")
    monkeypatch.setattr(notify, "DISCORD_NOTIFY_ASSETS", {"XAGUSD"})
    monkeypatch.setattr(notify, "DRY_RUN", False)
    monkeypatch.setattr(notify, "_max_concurrent_positions", lambda: cap)
    monkeypatch.setattr(notify, "build_expected_trade_outcome", lambda *a, **k: {"passes": True, "tp1_net_usd": 12, "valid_for_minutes": 120, "notional_usd": 1000, "current_chase_r": 0, "max_entry_price": 26})
    monkeypatch.setattr(notify.position_tracker, "load_positions", lambda *a, **k: {})
    monkeypatch.setattr(notify.position_tracker, "compute_state", lambda *a, **k: {"has_position": False, "pending_active": False})
    monkeypatch.setattr(notify.position_tracker, "open_position", lambda *a, **k: {"XAGUSD": {"status": "open"}})
    monkeypatch.setattr(notify.position_tracker, "save_positions_atomic", lambda *a, **k: None)
    sent = []
    monkeypatch.setattr(notify, "send_discord_embed", lambda embed: sent.append(embed) or True)
    _write_asset_signal(public)
    return public, sent


def test_concurrency_cap_suppresses_second_asset_without_cooldown(tmp_path, monkeypatch):
    public, sent = _setup_concurrency_notify(tmp_path, monkeypatch, cap=1)
    (public / "_position_lifecycle_state.json").write_text(json.dumps({"positions": {"GOLD_CFD": {"status": "open"}}}), encoding="utf-8")
    before_state = {"XAGUSD": {"last_entry_signature": "old", "last_entry_sent_utc": "2026-07-09T08:00:00Z"}}
    (public / "_notify_state.json").write_text(json.dumps(before_state), encoding="utf-8")

    with freeze_time("2026-07-09T09:00:00Z"):
        notify.check_and_notify()

    assert sent == []
    assert not (public / "_position_lifecycle_inbox.jsonl").exists()
    assert json.loads((public / "_notify_state.json").read_text(encoding="utf-8")) == before_state
    rows = (public / "journal" / "trade_journal.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert "suppressed_concurrency" in rows[1]
    assert "25.0" in rows[1] and "24.5" in rows[1] and "26.0" in rows[1] and "27.0" in rows[1] and "57" in rows[1]


def test_concurrency_cap_all_closed_dispatches_normally(tmp_path, monkeypatch):
    public, sent = _setup_concurrency_notify(tmp_path, monkeypatch, cap=1)
    (public / "_position_lifecycle_state.json").write_text(json.dumps({"positions": {"GOLD_CFD": {"status": "closed"}}}), encoding="utf-8")

    with freeze_time("2026-07-09T09:00:00Z"):
        notify.check_and_notify()

    assert len(sent) == 1
    assert (public / "_position_lifecycle_inbox.jsonl").exists()


def test_concurrency_cap_zero_is_inert(tmp_path, monkeypatch):
    public, sent = _setup_concurrency_notify(tmp_path, monkeypatch, cap=0)
    (public / "_position_lifecycle_state.json").write_text(json.dumps({"positions": {"GOLD_CFD": {"status": "open"}}}), encoding="utf-8")

    with freeze_time("2026-07-09T09:00:00Z"):
        notify.check_and_notify()

    assert len(sent) == 1
    assert (public / "_position_lifecycle_inbox.jsonl").exists()
