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
    monkeypatch.setattr(lc, "_cfg", lambda: {"hard_exit":{"immediate_on":["volatility_shock"],"volatility_shock_atr5m_median_mult":3,"trend_reversal_requires":{"consecutive_runs":2}}})
    _write(lc.STATE_PATH, {"positions":{"XAGUSD":{"status":"open","side":"long","entry":30,"sl":29,"tp1":31}}})
    _write(tmp_path / "XAGUSD" / "signal.json", {"exit_signal":{"state":"hard_exit","reason":"trend_reversal"}})
    lc.process(); assert json.loads(lc.STATE_PATH.read_text())["positions"]["XAGUSD"]["status"] == "open"
    lc.process(); assert json.loads(lc.STATE_PATH.read_text())["positions"]["XAGUSD"]["status"] == "closed"
    _write(lc.STATE_PATH, {"positions":{"GOLD_CFD":{"status":"open","side":"long","entry":2300,"sl":2290,"tp1":2310}}})
    _write(tmp_path / "GOLD_CFD" / "signal.json", {"atr":{"atr5m":4,"atr5m_median_20d":1}})
    lc.process(); assert json.loads(lc.STATE_PATH.read_text())["positions"]["GOLD_CFD"]["close_detail"] == "volatility_shock"


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
