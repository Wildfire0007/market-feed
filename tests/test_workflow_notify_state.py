from pathlib import Path


def test_notify_job_preserves_state_files():
    workflow = Path(".github/workflows/td-pipeline.yml").read_text(encoding="utf-8")
    assert "Preserve notify state before public refresh" in workflow
    assert "Restore notify state after public refresh" in workflow
    assert "public/monitoring/daily_digest_state.json" in workflow
    assert "scripts/preserve_public_append_logs.py save" in workflow
    assert "scripts/preserve_public_append_logs.py restore" in workflow
    helper = Path("scripts/preserve_public_append_logs.py").read_text(encoding="utf-8")
    assert "journal/trade_journal.csv" in helper
    assert "journal/trade_ledger.csv" in helper

