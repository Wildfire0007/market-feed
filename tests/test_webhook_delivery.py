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
    (tmp_path / "journal").mkdir()
    (tmp_path / "journal" / "trade_journal.csv").write_text("analysis_timestamp,asset\n2026-07-06T20:00:00Z,GOLD_CFD\n", encoding="utf-8")
    (tmp_path / "monitoring").mkdir()
    (tmp_path / "monitoring" / "webhook_delivery.jsonl").write_text(json.dumps({"ts_utc":"2026-07-06T20:10:00Z","script":"notify_discord","ok":True}) + "\n", encoding="utf-8")
    payloads = []    
    monkeypatch.setattr(daily_actionable_digest, "PUBLIC", tmp_path)
    monkeypatch.setattr(daily_actionable_digest, "STATE", tmp_path / "monitoring" / "daily_digest_state.json")
    monkeypatch.setattr(daily_actionable_digest, "urls", lambda: ["https://hook"])
    monkeypatch.setattr(daily_actionable_digest.settings, "POSITION_LIFECYCLE", {"daily_digest_utc": "00:00"}, raising=False)
    monkeypatch.setattr(daily_actionable_digest, "datetime", mock.Mock(
        now=lambda tz=None: datetime(2026, 7, 6, 21, 0, tzinfo=timezone.utc),
        fromisoformat=datetime.fromisoformat,
    ))    
    with mock.patch("scripts.daily_actionable_digest.requests.post", return_value=Resp(500)):
        assert daily_actionable_digest.main() == 0
    assert not daily_actionable_digest.STATE.exists()
    def post(_url, json, timeout):
        payloads.append(json)
        return Resp(200)
    with mock.patch("scripts.daily_actionable_digest.requests.post", side_effect=post):
        assert daily_actionable_digest.main() == 0
    assert daily_actionable_digest.STATE.exists()
    desc = payloads[-1]["embeds"][0]["description"]
    assert "Mai jelzés-jelöltek: 1" in desc
    assert "Kiküldött ENTRY kártyák: 1" in desc    


def test_webhook_delivery_rotates_by_configured_size(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    settings = tmp_path / "analysis_settings.json"
    settings.write_text(json.dumps({"webhook_delivery_log": {"max_mb": 0.0001, "keep_files": 1}}), encoding="utf-8")
    path = public_dir / "monitoring" / "webhook_delivery.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("x" * 256, encoding="utf-8")
    monkeypatch.setenv("NOTIFY_PUBLIC_DIR", str(public_dir))
    monkeypatch.setenv("ANALYSIS_SETTINGS_PATH", str(settings))

    record("unit", "diagnostic", 200, True)

    rotated = public_dir / "monitoring" / "webhook_delivery.1.jsonl"
    assert rotated.exists()
    assert rotated.read_text(encoding="utf-8") == "x" * 256
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["script"] == "unit"    
