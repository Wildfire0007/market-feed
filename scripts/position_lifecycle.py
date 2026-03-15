#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
position_lifecycle.py — külön futó pozícióéletciklus worker
A notify_discord entry-only esemény-inboxából nyit/pending állapotot kezel,
és az asset signal.json exit mezőiből zárási állapotot frissít.
"""

from __future__ import annotations

import json
import os
import sys
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else BASE_DIR / "public"
if not PUBLIC_DIR.exists() and (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"

LOCK_PATH = PUBLIC_DIR / ".position_lifecycle.lock"
INBOX_PATH = PUBLIC_DIR / "_position_lifecycle_inbox.jsonl"
STATE_PATH = PUBLIC_DIR / "_position_lifecycle_state.json"

CLOSE_STATES = {"hard_exit", "stop_loss_hit", "take_profit_hit", "take_profit_2_hit", "closed"}


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as h:
            data = json.load(h)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as h:
            json.dump(payload, h, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _read_inbox_new_lines(path: Path, last_line: int) -> tuple[list[Dict[str, Any]], int]:
    if not path.exists():
        return [], last_line
    events: list[Dict[str, Any]] = []
    current_line = 0
    with path.open("r", encoding="utf-8") as h:
        for raw in h:
            current_line += 1
            if current_line <= last_line:
                continue
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            if isinstance(evt, dict):
                events.append(evt)
    return events, current_line


def process() -> None:
    if not PUBLIC_DIR.exists():
        return

    state = load_json(STATE_PATH)
    meta = state.get("_meta") if isinstance(state.get("_meta"), dict) else {}
    positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}

    last_line = int(meta.get("last_inbox_line") or 0)
    events, new_last_line = _read_inbox_new_lines(INBOX_PATH, last_line)

    for evt in events:
        if str(evt.get("event") or "") != "entry_signal":
            continue
        asset = str(evt.get("asset") or "").strip()
        if not asset:
            continue
        order_type = str(evt.get("order_type") or "MARKET").upper()
        direction = str(evt.get("direction") or "buy").lower()
        if direction not in {"buy", "sell"}:
            continue
        now = str(evt.get("ts_utc") or to_utc_iso(datetime.now(timezone.utc)))
        positions[asset] = {
            "status": "open" if order_type == "MARKET" else "pending",
            "side": "long" if direction == "buy" else "short",
            "entry": safe_float(evt.get("entry")),
            "sl": safe_float(evt.get("sl")),
            "tp1": safe_float(evt.get("tp1")),
            "tp2": safe_float(evt.get("tp2")),
            "order_type": order_type,
            "source_signal": str(evt.get("signal") or ""),
            "entry_signature": str(evt.get("entry_signature") or ""),
            "updated_at_utc": now,
            "opened_at_utc": now if order_type == "MARKET" else None,
        }

    for asset_dir in sorted([d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")], key=lambda p: p.name):
        asset = asset_dir.name
        pos = positions.get(asset)
        if not isinstance(pos, dict) or str(pos.get("status") or "") in {"", "closed"}:
            continue
        signal_data = load_json(asset_dir / "signal.json")
        exit_signal = signal_data.get("position_exit_signal") if isinstance(signal_data, dict) else None
        if not isinstance(exit_signal, dict):
            continue
        exit_state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()
        if exit_state not in CLOSE_STATES:
            continue
        now = to_utc_iso(datetime.now(timezone.utc))
        pos["status"] = "closed"
        pos["close_reason"] = exit_state
        pos["closed_at_utc"] = str(exit_signal.get("triggered_at") or exit_signal.get("timestamp") or now)
        pos["updated_at_utc"] = now
        positions[asset] = pos

    state["_meta"] = {"last_inbox_line": new_last_line}
    state["positions"] = positions
    save_json(STATE_PATH, state)


if __name__ == "__main__":
    if not PUBLIC_DIR.exists():
        sys.exit(0)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        process()
