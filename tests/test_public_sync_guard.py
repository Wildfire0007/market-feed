import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import Trading


def _write_refresh(root: Path, completed_at: datetime) -> Path:
    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    payload = {
        "trading_started_at_utc": (completed_at - timedelta(seconds=30)).isoformat(),
        "trading_completed_at_utc": completed_at.isoformat(),
        "duration_seconds": 30.0,
        "out_dir": str(root),
    }
    refresh_path = pipeline / "public_refresh.json"
    refresh_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = Trading.hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (pipeline / "public_refresh.sha256").write_text(digest, encoding="utf-8")
    return refresh_path


def test_validate_public_refresh_sync_state_accepts_fresh_marker(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    _write_refresh(tmp_path, now - timedelta(seconds=10))
    monkeypatch.setattr(Trading, "PUBLIC_SYNC_MAX_AGE_SECONDS", 120.0)

    Trading._validate_public_refresh_sync_state(str(tmp_path), Trading.logging.getLogger("test"))


def test_validate_public_refresh_sync_state_rejects_stale_marker(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    _write_refresh(tmp_path, now - timedelta(seconds=500))
    monkeypatch.setattr(Trading, "PUBLIC_SYNC_MAX_AGE_SECONDS", 120.0)

    with pytest.raises(RuntimeError, match="stale"):
        Trading._validate_public_refresh_sync_state(str(tmp_path), Trading.logging.getLogger("test"))


def test_validate_public_refresh_sync_state_rejects_checksum_mismatch(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    refresh_path = _write_refresh(tmp_path, now - timedelta(seconds=10))
    checksum_path = refresh_path.with_suffix(".sha256")
    checksum_path.write_text("0" * 64, encoding="utf-8")
    monkeypatch.setattr(Trading, "PUBLIC_SYNC_MAX_AGE_SECONDS", 120.0)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        Trading._validate_public_refresh_sync_state(str(tmp_path), Trading.logging.getLogger("test"))
