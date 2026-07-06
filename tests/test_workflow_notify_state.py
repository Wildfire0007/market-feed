from pathlib import Path


def test_notify_job_preserves_state_files():
    workflow = Path(".github/workflows/td-pipeline.yml").read_text(encoding="utf-8")
    assert "Preserve notify state before public refresh" in workflow
    assert "Restore notify state after public refresh" in workflow
    assert "public/monitoring/daily_digest_state.json" in workflow
    assert "cp -a /tmp/public-notify-state/public/. public/" in workflow
