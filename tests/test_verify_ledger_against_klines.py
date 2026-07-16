import csv, json
from pathlib import Path
from scripts.verify_ledger_against_klines import verify


def test_verify_ledger_flags_phantom_and_keeps_honest(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"; asset.mkdir(parents=True)
    (asset / "klines_5m.json").write_text(json.dumps({"values":[{"datetime":"2026-07-10T10:05:00Z","open":100,"high":111,"low":99,"close":110}]}), encoding="utf-8")
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","side","order_type","entry","sl","tp1","tp2","size_units","opened_at_utc","closed_at_utc","trigger_bar_utc","close_reason","outcome","est_pnl_usd","source_signal","entry_signature","voided","void_reason"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"ok","asset":"GOLD_CFD","entry":"100","tp1":"110","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
        w.writerow({"ledger_id":"bad","asset":"GOLD_CFD","entry":"100","tp1":"120","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})       
    assert verify(public, ledger) == ["bad GOLD_CFD take_profit_hit level=120.0"]


def test_verify_ledger_flags_exit_touch_without_entry_touch(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"; asset.mkdir(parents=True)
    (asset / "klines_5m.json").write_text(json.dumps({"values":[{"datetime":"2026-07-10T10:05:00Z","open":109,"high":111,"low":108,"close":110}]}), encoding="utf-8")
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","side","order_type","entry","sl","tp1","tp2","size_units","opened_at_utc","closed_at_utc","trigger_bar_utc","close_reason","outcome","est_pnl_usd","source_signal","entry_signature","voided","void_reason"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"entry_bad","asset":"GOLD_CFD","entry":"100","tp1":"110","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
    assert verify(public, ledger) == ["entry_bad GOLD_CFD entry_never_touched level=100.0"]


def test_verify_ledger_flags_unparseable_non_voided_closed_rows(tmp_path: Path):
    public = tmp_path / "public"
    (public / "GOLD_CFD").mkdir(parents=True)
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","side","order_type","entry","sl","tp1","tp2","size_units","opened_at_utc","closed_at_utc","trigger_bar_utc","close_reason","outcome","est_pnl_usd","source_signal","entry_signature","voided","void_reason"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"bad_ts","asset":"GOLD_CFD","entry":"100","tp1":"110","opened_at_utc":"not-a-date","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
        w.writerow({"ledger_id":"bad_level","asset":"GOLD_CFD","entry":"100","tp1":"","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
        w.writerow({"ledger_id":"void_bad","asset":"GOLD_CFD","entry":"100","tp1":"","opened_at_utc":"not-a-date","closed_at_utc":"","close_reason":"take_profit_hit","voided":"true"})
    assert verify(public, ledger) == ["bad_ts GOLD_CFD unparseable_row", "bad_level GOLD_CFD unparseable_row"]


def test_verify_ledger_skips_truth_verified_rows(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"; asset.mkdir(parents=True)
    (asset / "klines_5m.json").write_text(json.dumps({"values":[{"datetime":"2026-07-10T10:05:00Z","open":100,"high":111,"low":99,"close":110}]}), encoding="utf-8")
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","entry","tp1","opened_at_utc","closed_at_utc","close_reason","voided","truth_verified_utc"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"old_ok","asset":"GOLD_CFD","entry":"100","tp1":"120","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false","truth_verified_utc":"2026-07-10T10:15:00Z"})
    assert verify(public, ledger) == []


def test_verify_ledger_stamps_fresh_verified_rows(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"; asset.mkdir(parents=True)
    (asset / "klines_5m.json").write_text(json.dumps({"values":[{"datetime":"2026-07-10T10:05:00Z","open":100,"high":111,"low":99,"close":110}]}), encoding="utf-8")
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","entry","tp1","opened_at_utc","closed_at_utc","close_reason","voided"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"fresh","asset":"GOLD_CFD","entry":"100","tp1":"110","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
    assert verify(public, ledger, now_stamp="2026-07-10T10:11:00Z") == []
    row = next(csv.DictReader(ledger.open(newline="", encoding="utf-8")))
    assert row["truth_verified_utc"] == "2026-07-10T10:11:00Z"
    assert "verify_note" in row


def test_verify_ledger_warns_for_aged_unverified_rows(tmp_path: Path, capsys):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"; asset.mkdir(parents=True)
    (asset / "klines_5m.json").write_text(json.dumps({"values":[{"datetime":"2026-07-11T10:05:00Z","open":100,"high":111,"low":99,"close":110}]}), encoding="utf-8")
    ledger = public / "journal" / "trade_ledger.csv"; ledger.parent.mkdir()
    fields = ["ledger_id","asset","entry","tp1","opened_at_utc","closed_at_utc","close_reason","voided"]
    with ledger.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        w.writerow({"ledger_id":"aged","asset":"GOLD_CFD","entry":"100","tp1":"120","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
    assert verify(public, ledger) == []
    assert "WARNING aged GOLD_CFD outside_retention" in capsys.readouterr().out
