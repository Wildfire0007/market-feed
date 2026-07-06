"""Shared Discord webhook delivery audit helpers."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MAX_BYTES = 1_000_000


def public_dir() -> Path:
    import os
    return Path(os.getenv("NOTIFY_PUBLIC_DIR", "public"))


def audit_path() -> Path:
    return public_dir() / "monitoring" / "webhook_delivery.jsonl"


def _rotate_if_needed(path: Path, max_bytes: int = MAX_BYTES) -> None:
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            rotated = path.with_suffix(path.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            path.replace(rotated)
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
