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
    journal = tmp_path / "journal" / "trade_journal.csv"
    journal.parent.mkdir(parents=True)
    lines = ["analysis_timestamp,asset,validation_outcome,validation_rr"]
    lines += [f"2026-06-30T12:0{i}:00Z,GOLD_CFD,tp1,1" for i in range(7)]
    lines += [f"2026-06-30T13:0{i}:00Z,XAGUSD,stopped,-1" for i in range(3)]
    journal.write_text("\n".join(lines), encoding="utf-8")
    embed = weekly.build_embed(weekly.datetime.fromisoformat("2026-07-05T20:31:00+00:00"))
    lo, hi = _wilson_interval(7, 10)
    assert "Címkézett ügyletek: 10" in embed["description"]
    assert f"TP1-találat SL előtt: 70% — 95% Wilson CI: [{lo*100:.0f}%, {hi*100:.0f}%]" in embed["description"]


def test_weekly_zero_data_variants(tmp_path, monkeypatch):
    setup_public(tmp_path, monkeypatch)
    embed = weekly.build_embed(weekly.datetime.fromisoformat("2026-07-05T20:31:00+00:00"))
    assert "nincs adat" in embed["description"]
    assert "még nincs értékelhető ügylet" in embed["description"]
    assert "Feasibility pass-arány: nincs adat" in embed["description"]
