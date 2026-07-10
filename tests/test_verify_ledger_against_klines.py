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
        w.writerow({"ledger_id":"ok","asset":"GOLD_CFD","tp1":"110","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
        w.writerow({"ledger_id":"bad","asset":"GOLD_CFD","tp1":"120","opened_at_utc":"2026-07-10T10:00:00Z","closed_at_utc":"2026-07-10T10:10:00Z","close_reason":"take_profit_hit","voided":"false"})
    assert verify(public, ledger) == ["bad GOLD_CFD take_profit_hit level=120.0"]
