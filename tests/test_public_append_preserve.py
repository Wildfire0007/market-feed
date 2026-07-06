import json
from pathlib import Path

from scripts.preserve_public_append_logs import restore, save


def test_append_preserve_survives_second_run_artifact_refresh(tmp_path):
    public = tmp_path / "public"
    state = tmp_path / "state" / "public"
    journal = public / "journal" / "trade_journal.csv"
    gate = public / "debug" / "entry_gates" / "entry_gates_2026-07-06.jsonl"
    gap = public / "debug" / "entry_gate_gap_log.jsonl"
    webhook = public / "monitoring" / "webhook_delivery.jsonl"

    journal.parent.mkdir(parents=True)
    gate.parent.mkdir(parents=True)
    webhook.parent.mkdir(parents=True)
    journal.write_text("analysis_timestamp,asset,side\n2026-07-06T16:55:00Z,GOLD_CFD,buy\n", encoding="utf-8")
    gate.write_text(json.dumps({"ts": "run1", "asset": "GOLD_CFD"}) + "\n", encoding="utf-8")
    gap.write_text(json.dumps({"ts": "run1", "kapu": "atr"}) + "\n", encoding="utf-8")
    webhook.write_text(json.dumps({"ts_utc": "2026-07-06T16:55:00Z", "script": "notify_discord"}) + "\n", encoding="utf-8")

    assert save(public, state) == 4

    # Simulate rm -rf public + download-artifact on the second run.
    for child in public.iterdir():
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    journal.parent.mkdir(parents=True)
    gate.parent.mkdir(parents=True)
    webhook.parent.mkdir(parents=True)
    journal.write_text("analysis_timestamp,asset,side\n2026-07-06T20:32:00Z,XAGUSD,sell\n", encoding="utf-8")
    gate.write_text(json.dumps({"ts": "run2", "asset": "XAGUSD"}) + "\n", encoding="utf-8")
    gap.write_text(json.dumps({"ts": "run2", "kapu": "spread"}) + "\n", encoding="utf-8")
    webhook.write_text(json.dumps({"ts_utc": "2026-07-06T20:32:00Z", "script": "daily_actionable_digest"}) + "\n", encoding="utf-8")

    assert restore(public, state) == 4

    assert "2026-07-06T16:55:00Z" in journal.read_text(encoding="utf-8")
    assert "2026-07-06T20:32:00Z" in journal.read_text(encoding="utf-8")
    assert "run1" in gate.read_text(encoding="utf-8")
    assert "run2" in gate.read_text(encoding="utf-8")
    assert "run1" in gap.read_text(encoding="utf-8")
    assert "run2" in gap.read_text(encoding="utf-8")
    assert "16:55" in webhook.read_text(encoding="utf-8")
    assert "20:32" in webhook.read_text(encoding="utf-8")
