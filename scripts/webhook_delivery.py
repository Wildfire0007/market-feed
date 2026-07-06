"""Shared Discord webhook delivery audit helpers."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_MAX_MB = 5
DEFAULT_KEEP_FILES = 1


def public_dir() -> Path:
    return Path(os.getenv("NOTIFY_PUBLIC_DIR", "public"))


def audit_path() -> Path:
    return public_dir() / "monitoring" / "webhook_delivery.jsonl"


def _settings_path() -> Path:
    return Path(os.getenv("ANALYSIS_SETTINGS_PATH", "config/analysis_settings.json"))


def _rotation_limits() -> tuple[int, int]:
    try:
        cfg = json.loads(_settings_path().read_text(encoding="utf-8"))
        raw = cfg.get("webhook_delivery_log", {}) if isinstance(cfg, dict) else {}
        log_cfg = raw if isinstance(raw, dict) else {}
    except Exception:
        log_cfg = {}
    try:
        max_bytes = max(0, int(float(log_cfg.get("max_mb", DEFAULT_MAX_MB)) * 1024 * 1024))
    except (TypeError, ValueError):
        max_bytes = DEFAULT_MAX_MB * 1024 * 1024
    try:
        keep_files = max(0, int(log_cfg.get("keep_files", DEFAULT_KEEP_FILES)))
    except (TypeError, ValueError):
        keep_files = DEFAULT_KEEP_FILES
    return max_bytes, keep_files


def _rotated_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}.{index}{path.suffix}")


def _rotate_if_needed(path: Path, max_bytes: Optional[int] = None, keep_files: Optional[int] = None) -> None:    
    try:
        if max_bytes is None or keep_files is None:
            max_bytes, keep_files = _rotation_limits()
        if max_bytes <= 0 or not path.exists() or path.stat().st_size <= max_bytes:
            return
        if keep_files <= 0:
            path.unlink(missing_ok=True)
            return
        _rotated_path(path, keep_files).unlink(missing_ok=True)
        for index in range(keep_files - 1, 0, -1):
            src = _rotated_path(path, index)
            if src.exists():
                os.replace(src, _rotated_path(path, index + 1))
        os.replace(path, _rotated_path(path, 1))
    except Exception:
        pass


def record(script: str, channel_kind: str, status: Optional[int], ok: bool) -> None:
    path = audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "script": script,
            "channel_kind": channel_kind,
            "status": status,
            "ok": bool(ok),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


def log_response(logger: logging.Logger, script: str, channel_kind: str, response: Any) -> bool:
    status = getattr(response, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except Exception:
        status_int = None
    ok = bool(status_int is not None and 200 <= status_int < 300)
    body = ""
    if not ok:
        body = str(getattr(response, "text", "") or "")[:200]
        logger.warning("Discord webhook POST %s status=%s ok=%s body=%s", channel_kind, status_int, ok, body)
    else:
        logger.info("Discord webhook POST %s status=%s ok=%s", channel_kind, status_int, ok)
    record(script, channel_kind, status_int, ok)
    return ok


def log_exception(logger: logging.Logger, script: str, channel_kind: str, exc: BaseException) -> None:
    logger.warning("Discord webhook POST %s failed: %s", channel_kind, exc)
    record(script, channel_kind, None, False)
