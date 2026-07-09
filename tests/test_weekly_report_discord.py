import json
from pathlib import Path
from unittest import mock

from freezegun import freeze_time

from scripts import weekly_report_discord as weekly
from scripts.label_trades import _wilson_interval


class Resp:
    status_code = 204
    text = ""


def setup_public(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "PUBLIC", tmp_path)
    monkeypatch.setattr(weekly, "STATE", tmp_path / "monitoring" / "weekly_report_state.json")
    monkeypatch.setattr(weekly, "urls", lambda: ["https://hook"])
    return tmp_path


def test_resolve_log_ts_prefers_ts_utc_with_bud_ts_fallback():
    assert weekly.resolve_log_ts({"ts_utc": "2026-07-05T20:30:00Z"}).isoformat().endswith("+00:00")
    assert weekly.resolve_log_ts({"bud_ts": "2026-07-05T22:30:00+02:00"}).hour == 20


def test_weekly_boundary_and_dedupe(tmp_path, monkeypatch):
    setup_public(tmp_path, monkeypatch)
    with freeze_time("2026-07-05T20:29:00Z"):
        assert weekly.main() == 0
    assert not weekly.STATE.exists()
    with freeze_time("2026-07-05T20:31:00Z"), mock.patch("scripts.weekly_report_discord.requests.post", return_value=Resp()) as post:
        assert weekly.main() == 0
        assert post.call_count == 1
    with freeze_time("2026-07-06T05:00:00Z"), mock.patch("scripts.weekly_report_discord.requests.post", return_value=Resp()) as post:
        assert weekly.main() == 0
        assert post.call_count == 0
    weekly.STATE.unlink()
    with freeze_time("2026-07-06T05:00:00Z"), mock.patch("scripts.weekly_report_discord.requests.post", return_value=Resp()) as post:
        assert weekly.main() == 0
        assert post.call_count == 1


def test_weekly_wilson_line_fixture(tmp_path, monkeypatch):
    setup_public(tmp_path, monkeypatch)
    ledger = tmp_path / "journal" / "trade_ledger.csv"
    ledger.parent.mkdir(parents=True)
    lines = ["ledger_id,asset,side,order_type,entry,sl,tp1,tp2,size_units,opened_at_utc,closed_at_utc,close_reason,outcome,est_pnl_usd,source_signal,entry_signature"]
    lines += [f"w{i},GOLD_CFD,long,LIMIT,1,0,2,,10,2026-06-30T11:00:00Z,2026-06-30T12:0{i}:00Z,take_profit_hit,tp1_closed,10,," for i in range(7)]
    lines += [f"l{i},XAGUSD,long,LIMIT,1,0,2,,10,2026-06-30T12:00:00Z,2026-06-30T13:0{i}:00Z,stop_loss_hit,stopped,-10,," for i in range(3)]
    ledger.write_text("\n".join(lines), encoding="utf-8")    
    journal = tmp_path / "journal" / "trade_journal.csv"
    journal.write_text(
        "journal_id,asset,analysis_timestamp,signal,mode,validation_outcome\n"
        + "\n".join(
            [f"s{i},GOLD_CFD,2026-06-30T11:0{i}:00Z,buy,suppressed_momentum,tp_hit" for i in range(7)]
            + [f"x{i},GOLD_CFD,2026-06-30T12:0{i}:00Z,buy,suppressed_momentum,stopped" for i in range(3)]
        ),
        encoding="utf-8",
    )    
    embed = weekly.build_embed(weekly.datetime.fromisoformat("2026-07-05T20:31:00+00:00"))
    lo, hi = _wilson_interval(7, 10)
    assert f"Precision (élő): 70% — 95% Wilson CI: [{lo*100:.0f}%, {hi*100:.0f}%] (N=10)" in embed["description"]
    assert f"Momentum (árnyék): 70% — 95% Wilson CI: [{lo*100:.0f}%, {hi*100:.0f}%] (N=10)" in embed["description"]

def test_weekly_zero_data_variants(tmp_path, monkeypatch):
    setup_public(tmp_path, monkeypatch)
    embed = weekly.build_embed(weekly.datetime.fromisoformat("2026-07-05T20:31:00+00:00"))
    assert "nincs adat" in embed["description"]
    assert "Precision (élő): nincs értékelhető minta (N=0)" in embed["description"]
    assert "Momentum (árnyék): nincs értékelhető minta (N=0)" in embed["description"]   
    assert "Feasibility pass-arány: nincs adat" in embed["description"]
