#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduled cleanup helpers for monitoring JSON snapshots.

The cron entry for this script is intended exclusively for monitoring hygiene
(resetting dashboard caches) and must not be treated as a production readiness
signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_anchor import load_anchor_state, reset_anchor_state
import state_db
from scripts.notify_discord import (
    STATE_PATH as NOTIFY_STATE_PATH,
    build_default_state as build_notify_default_state,
    to_utc_iso,
)

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
STATUS_PATH = PUBLIC_DIR / "status.json"
DEFAULT_STATUS_MESSAGE = "monitoring cron reset (non-production)"
DEFAULT_BACKUP_DIR = PUBLIC_DIR / "monitoring" / "reset_backups"
RESET_STATUS = "reset"


@dataclass
class ResetResult:
    """Structured result of a cleanup action."""

    path: Path
    before: int = 0
    after: int = 0
    removed: int = 0
    changed: bool = False
    backup_path: Optional[Path] = None
    message: str = ""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _make_backup(path: Path, backup_dir: Optional[Path]) -> Optional[Path]:
    if not path.exists():
        return None

    target_dir = backup_dir or DEFAULT_BACKUP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = target_dir / f"{path.name}.{timestamp}.bak"
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def reset_anchor_state_file(
    *,
    max_age_hours: Optional[float] = 24.0,
    db_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
) -> ResetResult:
    """Prune or clear anchor entries stored in SQLite."""

    anchor_db = Path(db_path or state_db.DEFAULT_DB_PATH)
    existing = load_anchor_state(str(anchor_db))
    before = len(existing)

    candidate = reset_anchor_state(
        max_age_hours,
        str(anchor_db),
        now=now,
        dry_run=True,
    )

    after = len(candidate)
    removed = max(before - after, 0)    
    changed = candidate != existing
    backup_path = None

    if not dry_run and changed:
        backup_path = _make_backup(anchor_db, backup_dir)
        candidate = reset_anchor_state(
            max_age_hours,
            str(anchor_db),
            now=now,
            dry_run=False,
        )
        after = len(candidate)

    message = (
        f"removed {removed} stale anchor entries" if removed else "anchor state already fresh"
    )

    return ResetResult(
        path=anchor_db,
        before=before,
        after=after,
        removed=removed,
        changed=changed and not dry_run,
        backup_path=backup_path,
        message=message,
    )


def reset_notify_state_file(
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
    reason: str = DEFAULT_STATUS_MESSAGE,
) -> ResetResult:
    """Replace ``_notify_state.json`` with a fresh default snapshot."""

    notify_path = Path(path or NOTIFY_STATE_PATH)
    existing_raw = _load_json(notify_path)
    existing = existing_raw if isinstance(existing_raw, dict) else {}
    existing_assets = {key for key in existing.keys() if key != "_meta"}

    refreshed = build_notify_default_state(now=now, reason=reason)
    refreshed_assets = {key for key in refreshed.keys() if key != "_meta"}

    removed = len(existing_assets - refreshed_assets)
    before = len(existing_assets)
    after = len(refreshed_assets)
    changed = existing != refreshed
    backup_path = None

    if not dry_run and changed:
        backup_path = _make_backup(notify_path, backup_dir)
        _ensure_dir(notify_path)
        with notify_path.open("w", encoding="utf-8") as fh:
            json.dump(refreshed, fh, ensure_ascii=False, indent=2)

    message = "reset notify state" if changed else "notify state already default"

    return ResetResult(
        path=notify_path,
        before=before,
        after=after,
        removed=removed,
        changed=changed and not dry_run,
        backup_path=backup_path,
        message=message,
    )


def reset_status_file(
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
    reason: str = DEFAULT_STATUS_MESSAGE,
) -> ResetResult:
    """Write a minimal status snapshot highlighting the reset."""

    status_path = Path(path or STATUS_PATH)
    existing_raw = _load_json(status_path)
    existing = existing_raw if isinstance(existing_raw, dict) else {}
    before = len(existing.get("assets", {})) if isinstance(existing.get("assets"), dict) else 0

    timestamp = to_utc_iso(now or datetime.now(timezone.utc))
    payload: Dict[str, Any] = {
        "ok": False,
        "status": RESET_STATUS,
        "generated_utc": timestamp,        
        "assets": {},
        "notes": [
            {
                "type": "reset",
                "message": reason,
                "reset_utc": timestamp,
            }
        ],
    }

    already_reset = (
        existing.get("status") == RESET_STATUS
        and existing.get("ok") is False
        and isinstance(existing.get("assets"), dict)
        and len(existing.get("assets", {})) == 0
    )

    changed = not already_reset and existing != payload
    backup_path = None

    if not dry_run and changed:
        backup_path = _make_backup(status_path, backup_dir)
        _ensure_dir(status_path)
        with status_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    return ResetResult(
        path=status_path,
        before=before,
        after=0,
        removed=before,
        changed=changed and not dry_run,
        backup_path=backup_path,
        message="status snapshot reset" if changed else "status already in reset state",
    )


def _result_to_str(label: str, result: ResetResult) -> str:
    status = "changed" if result.changed else "unchanged"
    parts = [
        f"{label}: {status}",
        f"before={result.before}",
        f"after={result.after}",
        f"removed={result.removed}",
    ]
    if result.backup_path:
        parts.append(f"backup={result.backup_path}")
    if result.message:
        parts.append(result.message)
    return " | ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-max-age-hours",
        type=float,
        default=24.0,
        help="Entries older than this will be removed from the anchor table (<=0 clears all).",
    )
    parser.add_argument("--anchor-db", type=str, default=None, help="Override anchor DB path.")
    parser.add_argument("--notify-path", type=str, default=None, help="Override notify state path.")
    parser.add_argument("--status-path", type=str, default=None, help="Override status.json path.")
    parser.add_argument(
        "--status-message",
        type=str,
        default=DEFAULT_STATUS_MESSAGE,
        help="Human readable note stored in status.json and notify state metadata.",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Directory for JSON backups (defaults to public/monitoring/reset_backups).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files.")

    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    anchor_result = reset_anchor_state_file(
        max_age_hours=args.anchor_max_age_hours,
        db_path=Path(args.anchor_db) if args.anchor_db else None,
        now=now,
        dry_run=args.dry_run,
        backup_dir=backup_dir,
    )
    notify_result = reset_notify_state_file(
        path=Path(args.notify_path) if args.notify_path else None,
        now=now,
        dry_run=args.dry_run,
        backup_dir=backup_dir,
        reason=args.status_message,
    )
    status_result = reset_status_file(
        path=Path(args.status_path) if args.status_path else None,
        now=now,
        dry_run=args.dry_run,
        backup_dir=backup_dir,
        reason=args.status_message,
    )

    print(_result_to_str("anchor", anchor_result))
    print(_result_to_str("notify", notify_result))
    print(_result_to_str("status", status_result))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
