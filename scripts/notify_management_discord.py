#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discord értesítések nyitott pozíciók menedzsment eseményeihez."""

from __future__ import annotations

import fcntl
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None
BUDAPEST_TZ = ZoneInfo("Europe/Budapest")

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else BASE_DIR / "public"
if not PUBLIC_DIR.exists() and (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"

DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
DISCORD_WEBHOOK_URLS = [url.strip() for url in os.getenv("DISCORD_WEBHOOK_URL", "").replace("\\n", ",").split(",") if url.strip()]
LOCK_PATH = PUBLIC_DIR / ".notify_management_discord.lock"
STATE_PATH = PUBLIC_DIR / "_management_notify_state.json"
LIFECYCLE_STATE_PATH = PUBLIC_DIR / "_position_lifecycle_state.json"

COLOR_GREEN, COLOR_RED, COLOR_ORANGE = 0x2ECC71, 0xE74C3C, 0xE67E22


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


def safe_float(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def format_price(price: Any) -> str:
    val = safe_float(price)
    if val is None:
        return "N/A"
    return f"{val:,.1f}" if abs(val) >= 1000 else f"{val:.2f}" if abs(val) >= 10 else f"{val:.5f}"


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def format_budapest_time(dt: datetime) -> str:
    return dt.astimezone(BUDAPEST_TZ).strftime("%Y-%m-%d %H:%M:%S CET/CEST")


def _position_key(asset: str, position: Dict[str, Any]) -> str:
    opened = str(position.get("opened_at_utc") or "")
    if opened:
        return f"{asset}|{opened}"
    side = str(position.get("side") or "").lower()
    entry = format_price(position.get("entry"))
    return f"{asset}|{side}|{entry}"


def _event_triggered(event: str, side: str, spot: Optional[float], pos: Dict[str, Any], hard_exit_state: str) -> bool:
    if event == "hard_exit":
        return hard_exit_state == "hard_exit"
    if spot is None:
        return False
    tp1, tp2, sl = safe_float(pos.get("tp1")), safe_float(pos.get("tp2")), safe_float(pos.get("sl"))
    if side == "long":
        return {
            "tp1_hit": tp1 is not None and spot >= tp1,
            "tp2_hit": tp2 is not None and spot >= tp2,
            "sl_hit": sl is not None and spot <= sl,
        }.get(event, False)
    if side == "short":
        return {
            "tp1_hit": tp1 is not None and spot <= tp1,
            "tp2_hit": tp2 is not None and spot <= tp2,
            "sl_hit": sl is not None and spot >= sl,
        }.get(event, False)
    return False


def _build_embed(event: str, asset: str, side: str, entry: Any, spot: Any, reason: str) -> Dict[str, Any]:
    now_dt = datetime.now(timezone.utc)    
    side_label = "LONG" if side == "long" else "SHORT"
    presets = {
        "tp1_hit": ("🟠 RÉSZZÁRÁS (TP1) ELÉRVE!", "Húzd a Stop-Loss-t a belépési árra (Nullába) az eTorón!", COLOR_ORANGE),
        "tp2_hit": ("🟢 CÉLÁR ELÉRVE!", "Pozíció nyereségben lezárva.", COLOR_GREEN),
        "sl_hit": ("🔴 STOP LOSS KIÜTVE!", "Pozíció veszteségben lezárva.", COLOR_RED),
        "hard_exit": ("🔴 AZONNAL ZÁRD A POZÍCIÓT! (Hard Exit)", f"Ok: {reason or 'Hard exit jelzés'}", COLOR_RED),
    }
    title, description, color = presets[event]
    return {
        "title": title,
        "description": description,
        "color": color,
        "fields": [
            {"name": "Eszköz", "value": f"`{asset}`", "inline": True},
            {"name": "Irány", "value": f"`{side_label}`", "inline": True},
            {"name": "Eredeti Belépő", "value": f"`{format_price(entry)}`", "inline": True},
            {"name": "Aktuális Spot Ár", "value": f"`{format_price(spot)}`", "inline": True},
            {"name": "🕒 Időbélyeg", "value": f"`{format_budapest_time(now_dt)}` (Budapest)", "inline": False},
        ],
        "footer": {"text": f"Management Notify • Budapest: {format_budapest_time(now_dt)} • Anti-spam aktív"},
    }


def _send_embed(embed: Dict[str, Any]) -> None:
    if DRY_RUN or not requests or not DISCORD_WEBHOOK_URLS:
        return
    for url in DISCORD_WEBHOOK_URLS:
        try:
            requests.post(url, json={"embeds": [embed]}, timeout=5)
        except Exception:
            pass


def _load_open_positions() -> Dict[str, Dict[str, Any]]:
    state = load_json(LIFECYCLE_STATE_PATH)
    positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}
    open_positions = {
        asset: pos
        for asset, pos in positions.items()
        if isinstance(pos, dict) and str(pos.get("status") or "").lower() == "open"
    }
    if open_positions:
        return open_positions

    try:
        import position_tracker

        tracked = position_tracker.load_positions("public/trading.db", treat_missing_as_flat=True)
        return {
            asset: pos
            for asset, pos in tracked.items()
            if isinstance(pos, dict) and str(pos.get("status") or "").lower() == "open"
        }
    except Exception:
        return {}


def process() -> None:
    if not PUBLIC_DIR.exists():
        return

    notify_state = load_json(STATE_PATH)
    notify_changed = False
    now_iso = to_utc_iso(datetime.now(timezone.utc))

    for asset, pos in _load_open_positions().items():
        side = str(pos.get("side") or "").lower()
        if side not in {"long", "short"}:
            continue

        signal = load_json(PUBLIC_DIR / asset / "signal.json")
        spot = safe_float((signal.get("spot") or {}).get("price"))
        exit_signal = signal.get("exit_signal") if isinstance(signal.get("exit_signal"), dict) else signal.get("position_exit_signal")
        exit_state = str((exit_signal or {}).get("state") or (exit_signal or {}).get("action") or "").lower()
        exit_reason = str((exit_signal or {}).get("reason") or (exit_signal or {}).get("comment") or "")

        key = _position_key(asset, pos)
        sent_events = notify_state.get(key) if isinstance(notify_state.get(key), dict) else {}

        for event in ("hard_exit", "sl_hit", "tp2_hit", "tp1_hit"):
            if sent_events.get(event):
                continue
            if not _event_triggered(event, side, spot, pos, exit_state):
                continue
            _send_embed(_build_embed(event, asset, side, pos.get("entry"), spot, exit_reason))
            sent_events[event] = now_iso
            notify_changed = True
            if event in {"sl_hit", "tp2_hit", "hard_exit"}:
                break

        notify_state[key] = sent_events

    if notify_changed and not DRY_RUN:
        save_json(STATE_PATH, notify_state)


if __name__ == "__main__":
    if not PUBLIC_DIR.exists():
        sys.exit(0)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        process()
