# -*- coding: utf-8 -*-
"""Közös állapottároló az utolsó akcióképes BUY/SELL ("anchor") jelzéshez.

Az analysis.py a végleges jel generálásakor frissíti, a Discord értesítő és
egyéb komponensek pedig olvashatják. A formátum egyszerű JSON:
{
  "USOIL": {"side": "sell", "price": 82.15, "timestamp": "2024-03-12T14:05:00Z"},
  ...
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

PUBLIC_DIR = os.getenv("PUBLIC_DIR", "public")
ANCHOR_STATE_PATH = os.path.join(PUBLIC_DIR, "_active_anchor.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_anchor_state(path: str = ANCHOR_STATE_PATH) -> Dict[str, Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    norm: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        record = {
            "side": str(value.get("side") or "").lower(),
            "price": value.get("price"),
            "timestamp": value.get("timestamp"),
        }
        if not record["side"]:
            continue
        norm[key.upper()] = record
    return norm


def save_anchor_state(state: Dict[str, Dict[str, Any]], path: str = ANCHOR_STATE_PATH) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def record_anchor(
    asset: str,
    side: str,
    price: Optional[float] = None,
    timestamp: Optional[str] = None,
    path: str = ANCHOR_STATE_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Frissíti az anchor állapotot (ha újabb jel érkezett)."""

    if not asset or not side:
        return load_anchor_state(path)

    asset_key = asset.upper()
    side_norm = side.lower()
    state = load_anchor_state(path)
    current = state.get(asset_key)

    ts = timestamp or _utc_now_iso()
    new_ts = _parse_iso(ts)
    current_ts = _parse_iso((current or {}).get("timestamp"))

    should_update = False
    if current is None:
        should_update = True
    elif current.get("side") != side_norm:
        should_update = True
    elif new_ts and current_ts and new_ts > current_ts:
        should_update = True
    elif new_ts and not current_ts:
        should_update = True

    if should_update:
        state[asset_key] = {
            "side": side_norm,
            "price": price,
            "timestamp": ts,
        }
        save_anchor_state(state, path)

    return state


def touch_anchor(
    asset: str,
    side: str,
    price: Optional[float] = None,
    timestamp: Optional[str] = None,
    path: str = ANCHOR_STATE_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Garantálja, hogy legalább egy alap rekord létezzen (oldja, ha hiányzik)."""

    state = load_anchor_state(path)
    asset_key = asset.upper()
    if asset_key in state and state[asset_key].get("side"):
        return state
    return record_anchor(asset, side, price=price, timestamp=timestamp, path=path)

__all__ = [
    "ANCHOR_STATE_PATH",
    "load_anchor_state",
    "save_anchor_state",
    "record_anchor",
    "touch_anchor",
]
