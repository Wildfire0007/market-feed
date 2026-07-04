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


def test_copy_sources_excludes_python_sources_and_caches(tmp_path):
    from scripts.update_public import copy_sources

    source_root = tmp_path / "source"
    reports = source_root / "reports"
    cache = reports / "__pycache__"
    cache.mkdir(parents=True)
    (reports / "dummy.py").write_text("print('do not publish')\n", encoding="utf-8")
    (reports / "dummy.json").write_text('{"publish": true}\n', encoding="utf-8")
    (cache / "dummy.pyc").write_bytes(b"cache")

    target = tmp_path / "public"

    copy_sources([str(reports)], target)

    copied_reports = target / "reports"
    assert (copied_reports / "dummy.json").is_file()
    assert not (copied_reports / "dummy.py").exists()
    assert not (copied_reports / "__pycache__").exists()
