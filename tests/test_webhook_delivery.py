import json
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from scripts import daily_actionable_digest
from scripts.webhook_delivery import record


class Resp:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = "body"


def test_webhook_delivery_jsonl_populated(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTIFY_PUBLIC_DIR", str(tmp_path))
    record("unit", "actionable", 204, True)
    path = tmp_path / "monitoring" / "webhook_delivery.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["script"] == "unit"
    assert row["channel_kind"] == "actionable"
    assert row["status"] == 204
    assert row["ok"] is True


def test_daily_digest_state_written_only_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_actionable_digest, "PUBLIC", tmp_path)
    monkeypatch.setattr(daily_actionable_digest, "STATE", tmp_path / "monitoring" / "daily_digest_state.json")
    monkeypatch.setattr(daily_actionable_digest, "urls", lambda: ["https://hook"])
    monkeypatch.setattr(daily_actionable_digest.settings, "POSITION_LIFECYCLE", {"daily_digest_utc": "00:00"}, raising=False)
    monkeypatch.setattr(daily_actionable_digest, "datetime", mock.Mock(now=lambda tz=None: datetime(2026, 7, 6, 21, 0, tzinfo=timezone.utc)))
    with mock.patch("scripts.daily_actionable_digest.requests.post", return_value=Resp(500)):
        assert daily_actionable_digest.main() == 0
    assert not daily_actionable_digest.STATE.exists()
    with mock.patch("scripts.daily_actionable_digest.requests.post", return_value=Resp(200)):
        assert daily_actionable_digest.main() == 0
    assert daily_actionable_digest.STATE.exists()
