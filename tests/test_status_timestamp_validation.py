import json
import logging
from pathlib import Path

from freezegun import freeze_time

from analysis import build_status_snapshot


@freeze_time("2024-01-02 12:00:00", tz_offset=0)
def test_build_status_snapshot_normalizes_generated_utc(tmp_path: Path, caplog):
    summary = {
        "ok": True,
        "generated_utc": "not-a-timestamp",
        "assets": {"BTCUSD": {"ok": True, "signal": "hold"}},
        "notes": [
            {
                "type": "reset",
                "message": "scheduled monitoring reset",
                "reset_utc": "invalid",  # deliberately malformed
            }
        ],
    }

    caplog.set_level(logging.ERROR)
    payload = build_status_snapshot(summary, tmp_path)
    status_path = tmp_path / "status.json"
    on_disk = json.loads(status_path.read_text())

    generated = payload["generated_utc"]
    assert generated.endswith("Z")
    assert generated == on_disk["generated_utc"]
    assert on_disk["notes"][0]["reset_utc"] == generated
    assert any("Érvénytelen időbélyeg" in record.getMessage() for record in caplog.records)
