import sys
from datetime import datetime, timezone

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from scripts.notify_discord import build_pipeline_diag_embed  # noqa: E402


def test_build_pipeline_diag_embed_includes_hashes_and_deltas():
    payload = {
        "run": {"run_id": "123", "started_at_utc": "2024-01-01T00:00:00Z", "captured_at_utc": "2024-01-01T00:01:00Z"},
        "trading": {
            "started_utc": "2024-01-01T00:00:00Z",
            "completed_utc": "2024-01-01T00:05:00Z",
            "duration_seconds": 300,
        },
        "analysis": {
            "started_utc": "2024-01-01T00:05:00Z",
            "completed_utc": "2024-01-01T00:10:00Z",
            "duration_seconds": 300,
        },
        "artifacts": {
            "hashes": {
                "status.json": {"sha256": "a" * 64, "size": 2},
                "analysis_summary.json": None,
            }
        },
    }

    embed = build_pipeline_diag_embed(
        payload, now=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc)
    )

    assert embed["title"] == "Pipeline diagnosztika"
    assert "Trading→analysis" in embed["description"]
    assert embed["fields"]
    assert any("Artefakt-hash" == field.get("name") for field in embed["fields"])
