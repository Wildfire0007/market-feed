import json
from datetime import datetime, timezone
from pathlib import Path

from freezegun import freeze_time

import scripts.position_lifecycle as lc


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pending_state_blocks_duplicate_entry_and_expires_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "_position_lifecycle_inbox.jsonl")
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"entry_validity_minutes": 120, "entry_validity_atr_adaptive": False})
    lc.INBOX_PATH.write_text('\n'.join([
        json.dumps({"event":"entry_signal","ts_utc":"2026-07-05T10:00:00Z","asset":"XAGUSD","direction":"buy","order_type":"LIMIT","entry":30,"sl":29,"tp1":31}),
        json.dumps({"event":"entry_signal","ts_utc":"2026-07-05T10:05:00Z","asset":"XAGUSD","direction":"buy","order_type":"LIMIT","entry":30.1,"sl":29,"tp1":31}),
    ])+'\n', encoding='utf-8')
    _write(tmp_path / "XAGUSD" / "signal.json", {"spot":{"price":30.5}})
    with freeze_time("2026-07-05T12:01:00Z"):
        lc.process()
    pos = json.loads(lc.STATE_PATH.read_text())["positions"]["XAGUSD"]
    assert pos["status"] == "closed"
    assert pos["close_reason"] == "expired"
    assert pos["entry"] == 30


def test_open_tp1_closes_position(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "_cfg", lambda: {"tp1_closes_position": True, "ambiguous_bar_counts_as": "sl"})
    _write(lc.STATE_PATH, {"positions":{"GOLD_CFD":{"status":"open","side":"long","entry":2300,"sl":2290,"tp1":2310}}})
    _write(tmp_path / "GOLD_CFD" / "signal.json", {})
    _write(tmp_path / "GOLD_CFD" / "klines_5m.json", {"values":[{"open":2301,"high":2311,"low":2300}]})
    lc.process()
    pos=json.loads(lc.STATE_PATH.read_text())["positions"]["GOLD_CFD"]
    assert pos["status"] == "closed"
    assert pos["outcome"] == "tp1_closed"


def test_sl_gap_detail_and_ambiguous_counts_as_sl(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "_cfg", lambda: {"ambiguous_bar_counts_as": "sl"})
    _write(lc.STATE_PATH, {"positions":{"USOIL":{"status":"open","side":"long","entry":80,"sl":79,"tp1":81}}})
    _write(tmp_path / "USOIL" / "signal.json", {})
    _write(tmp_path / "USOIL" / "klines_5m.json", {"values":[{"open":78.5,"high":81.2,"low":78.4}]})
    lc.process()
    pos=json.loads(lc.STATE_PATH.read_text())["positions"]["USOIL"]
    assert pos["outcome"] == "stopped"
    assert "gap az SL" in pos["close_detail"]


def test_hard_exit_hysteresis_and_volatility_shock(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "CLOSE_NOTIFY_STATE_PATH", tmp_path / "_position_close_notify_state.json")    
    monkeypatch.setattr(lc, "_cfg", lambda: {"hard_exit":{"immediate_on":["volatility_shock"],"volatility_shock_atr5m_median_mult":3,"trend_reversal_requires":{"consecutive_runs":2}}})
    sent = []
    monkeypatch.setattr(lc, "_send_close_alert", lambda asset, pos, reason, now: sent.append((asset, reason, pos.copy())) or True)

    _write(lc.STATE_PATH, {"positions":{"XAGUSD":{"status":"open","side":"long","entry":30,"sl":29,"tp1":31,"opened_at_utc":"2026-07-08T09:00:00Z","size_units":10}}})
    _write(tmp_path / "XAGUSD" / "signal.json", {"exit_signal":{"state":"hard_exit","reason":"trend_reversal"}, "spot":{"price":29.5}})
    lc.process(); assert json.loads(lc.STATE_PATH.read_text())["positions"]["XAGUSD"]["status"] == "open"
    assert sent == []    
    lc.process(); assert json.loads(lc.STATE_PATH.read_text())["positions"]["XAGUSD"]["status"] == "closed"
    assert [(asset, reason, pos["close_detail"]) for asset, reason, pos in sent] == [("XAGUSD", "hard_exit", "trend_reversal")]

    sent.clear()
    _write(lc.STATE_PATH, {"positions":{"GOLD_CFD":{"status":"open","side":"long","entry":2300,"sl":2290,"tp1":2310,"opened_at_utc":"2026-07-08T10:00:00Z","size_units":2}}})
    _write(tmp_path / "GOLD_CFD" / "signal.json", {"atr":{"atr5m":4,"atr5m_median_20d":1}, "spot":{"price":2295}})
    lc.process()
    pos = json.loads(lc.STATE_PATH.read_text())["positions"]["GOLD_CFD"]
    assert pos["close_detail"] == "volatility_shock"
    assert [(asset, reason, payload["close_detail"]) for asset, reason, payload in sent] == [("GOLD_CFD", "hard_exit", "volatility_shock")]


def test_expired_pending_dispatches_cancel_alert_once(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "_position_lifecycle_inbox.jsonl")
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "EXPIRY_NOTIFY_STATE_PATH", tmp_path / "_position_expiry_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"entry_validity_minutes": 1, "entry_validity_atr_adaptive": False})
    sent=[]
    monkeypatch.setattr(lc, "_send_expiry_cancel_alert", lambda asset, pos, now: sent.append((asset,pos.copy())) or True)
    lc.INBOX_PATH.write_text(json.dumps({"event":"entry_signal","ts_utc":"2026-07-05T10:00:00Z","asset":"XAGUSD","direction":"buy","order_type":"LIMIT","entry":30,"sl":29,"tp1":31})+'\n', encoding='utf-8')
    _write(tmp_path / "XAGUSD" / "signal.json", {"spot":{"price":30.5}})
    with freeze_time("2026-07-05T10:02:00Z"):
        lc.process(); lc.process()
    assert len(sent) == 1
    assert sent[0][0] == "XAGUSD"


def test_close_transition_dispatches_actionable_card_once(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "CLOSE_NOTIFY_STATE_PATH", tmp_path / "_position_close_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"tp1_closes_position": True, "ambiguous_bar_counts_as": "sl"})
    sent = []
    monkeypatch.setattr(lc, "_send_close_alert", lambda asset, pos, reason, now: sent.append((asset, reason, pos.copy())) or True)
    _write(lc.STATE_PATH, {"positions":{"GOLD_CFD":{"status":"open","side":"long","entry":2300,"sl":2290,"tp1":2310,"opened_at_utc":"2026-07-08T09:00:00Z","size_units":5,"tp1_closes_position":True}}})
    _write(tmp_path / "GOLD_CFD" / "signal.json", {})
    _write(tmp_path / "GOLD_CFD" / "klines_5m.json", {"values":[{"open":2301,"high":2311,"low":2300}]})
    lc.process(); lc.process()
    assert [(asset, reason) for asset, reason, _ in sent] == [("GOLD_CFD", "take_profit_hit")]


def test_each_expired_pending_dispatches_own_cancel_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "_position_lifecycle_inbox.jsonl")
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "EXPIRY_NOTIFY_STATE_PATH", tmp_path / "_position_expiry_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"entry_validity_minutes": 1, "entry_validity_atr_adaptive": False})
    sent=[]
    monkeypatch.setattr(lc, "_send_expiry_cancel_alert", lambda asset, pos, now: sent.append((asset, pos.get("entry"))) or True)
    lc.INBOX_PATH.write_text("\n".join([
        json.dumps({"event":"entry_signal","ts_utc":"2026-07-05T10:00:00Z","asset":"XAGUSD","direction":"buy","order_type":"LIMIT","entry":30,"sl":29,"tp1":31}),
        json.dumps({"event":"entry_signal","ts_utc":"2026-07-05T10:00:00Z","asset":"GOLD_CFD","direction":"buy","order_type":"LIMIT","entry":2300,"sl":2290,"tp1":2310}),
    ])+"\n", encoding="utf-8")
    _write(tmp_path / "XAGUSD" / "signal.json", {"spot":{"price":30.5}})
    _write(tmp_path / "GOLD_CFD" / "signal.json", {"spot":{"price":2305}})
    with freeze_time("2026-07-05T10:02:00Z"):
        lc.process(); lc.process()
    assert sorted(sent) == [("GOLD_CFD", 2300), ("XAGUSD", 30)]


def test_session_force_close_at_cutoff_dispatches_once_and_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "CLOSE_NOTIFY_STATE_PATH", tmp_path / "_position_close_notify_state.json")
    monkeypatch.setattr(lc, "STALE_OPEN_POSITION_NOTIFY_STATE_PATH", tmp_path / "_position_stale_open_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"session_force_close_utc": "16:25", "max_hold_minutes": 300, "open_position_data_stale_minutes": 10})
    sent = []
    monkeypatch.setattr(lc, "_send_close_alert", lambda asset, pos, reason, now: sent.append((asset, reason, pos.copy())) or True)
    _write(lc.STATE_PATH, {"positions":{"SPY":{"status":"open","side":"long","entry":500,"opened_at_utc":"2026-07-08T15:00:00Z","size_units":2}}})
    _write(tmp_path / "SPY" / "spot.json", {"price":501,"utc":"2026-07-08T16:25:00Z"})
    _write(tmp_path / "SPY" / "signal.json", {"spot":{"price":501}})
    with freeze_time("2026-07-08T16:25:00Z"):
        lc.process(); lc.process()
    pos = json.loads(lc.STATE_PATH.read_text())["positions"]["SPY"]
    assert pos["status"] == "closed"
    assert pos["close_reason"] == "session_force_close"
    assert pos["outcome"] == "force_closed"
    assert [(asset, reason) for asset, reason, _ in sent] == [("SPY", "session_force_close")]


def test_max_hold_force_close_dispatches(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "CLOSE_NOTIFY_STATE_PATH", tmp_path / "_position_close_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"max_hold_minutes": 300})
    sent = []
    monkeypatch.setattr(lc, "_send_close_alert", lambda asset, pos, reason, now: sent.append(reason) or True)
    _write(lc.STATE_PATH, {"positions":{"SPY":{"status":"open","side":"long","entry":500,"opened_at_utc":"2026-07-08T10:00:00Z"}}})
    _write(tmp_path / "SPY" / "signal.json", {"spot":{"price":502}})
    with freeze_time("2026-07-08T15:00:00Z"):
        lc.process()
    assert json.loads(lc.STATE_PATH.read_text())["positions"]["SPY"]["outcome"] == "force_closed"
    assert sent == ["session_force_close"]


def test_stale_data_open_position_alert_once_and_clears_on_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "STALE_OPEN_POSITION_NOTIFY_STATE_PATH", tmp_path / "_position_stale_open_notify_state.json")
    monkeypatch.setattr(lc, "_cfg", lambda: {"open_position_data_stale_minutes": 10})
    sent = []
    monkeypatch.setattr(lc, "_send_stale_open_position_alert", lambda asset, age, now: sent.append((asset, round(age, 1))) or True)
    _write(lc.STATE_PATH, {"positions":{"SPY":{"status":"open","side":"long","entry":500,"opened_at_utc":"2026-07-08T10:00:00Z","sl":490,"tp1":510}}})
    _write(tmp_path / "SPY" / "spot.json", {"price":501,"utc":"2026-07-08T11:49:00Z"})
    _write(tmp_path / "SPY" / "signal.json", {})
    with freeze_time("2026-07-08T12:00:00Z"):
        lc.process(); lc.process()
    assert sent == [("SPY", 11.0)]
    _write(tmp_path / "SPY" / "spot.json", {"price":501,"utc":"2026-07-08T12:00:00Z"})
    with freeze_time("2026-07-08T12:00:00Z"):
        lc.process()
    _write(tmp_path / "SPY" / "spot.json", {"price":501,"utc":"2026-07-08T12:10:00Z"})
    with freeze_time("2026-07-08T12:21:00Z"):
        lc.process()
    assert sent == [("SPY", 11.0), ("SPY", 11.0)]


def test_stale_data_no_card_without_position(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(lc, "STATE_PATH", tmp_path / "_position_lifecycle_state.json")
    monkeypatch.setattr(lc, "INBOX_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(lc, "_cfg", lambda: {"open_position_data_stale_minutes": 10})
    sent = []
    monkeypatch.setattr(lc, "_send_stale_open_position_alert", lambda asset, age, now: sent.append(asset) or True)
    _write(lc.STATE_PATH, {"positions":{}})
    _write(tmp_path / "SPY" / "spot.json", {"price":501,"utc":"2026-07-08T11:00:00Z"})
    _write(tmp_path / "SPY" / "signal.json", {})
    with freeze_time("2026-07-08T12:00:00Z"):
        lc.process()
    assert sent == []    
