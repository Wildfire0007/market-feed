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

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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
        record: Dict[str, Any] = {
            "side": str(value.get("side") or "").lower(),
            "price": value.get("price"),
            "timestamp": value.get("timestamp"),
        }
        for extra_key, extra_value in value.items():
            if extra_key not in record:
                record[extra_key] = extra_value
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
    extras: Optional[Dict[str, Any]] = None,
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
        payload: Dict[str, Any] = {
            "side": side_norm,
            "price": price,
            "timestamp": ts,
        }
        if extras:
            payload.update(extras)
        payload.setdefault("entry_price", payload.get("price"))
        payload.setdefault("trail_log", [])
        payload.setdefault("partial_exits", [])
        payload.setdefault("max_favorable_excursion", 0.0)
        payload.setdefault("max_adverse_excursion", 0.0)
        if payload.get("initial_risk_abs") is None and extras and extras.get("initial_risk_abs") is not None:
            payload["initial_risk_abs"] = extras.get("initial_risk_abs")
        payload["last_update"] = ts
        state[asset_key] = payload
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


def update_anchor_metrics(
    asset: str,
    extras: Optional[Dict[str, Any]] = None,
    path: str = ANCHOR_STATE_PATH,
) -> Dict[str, Dict[str, Any]]:
    if not extras:
        return load_anchor_state(path)
    state = load_anchor_state(path)
    asset_key = asset.upper()
    if asset_key not in state:
        return state
    record = state[asset_key]
    record.update(extras)

    entry_price = _safe_float(record.get("entry_price") or record.get("price"))
    current_price = _safe_float(extras.get("current_price") or extras.get("spot_price"))
    side = (record.get("side") or "").lower()
    risk_abs = _safe_float(record.get("initial_risk_abs"))
    timestamp = extras.get("analysis_timestamp") or _utc_now_iso()

    if entry_price is not None and current_price is not None and side in {"buy", "sell"}:
        diff = current_price - entry_price if side == "buy" else entry_price - current_price
        record["current_pnl_abs"] = diff
        if risk_abs and risk_abs > 0:
            record["current_pnl_r"] = diff / risk_abs
        max_fav = _safe_float(record.get("max_favorable_excursion")) or 0.0
        max_adv = _safe_float(record.get("max_adverse_excursion")) or 0.0
        if diff > max_fav:
            record["max_favorable_excursion"] = diff
        if diff < -max_adv:
            record["max_adverse_excursion"] = -diff
        trail_entry = {
            "timestamp": timestamp,
            "price": current_price,
            "pnl_abs": diff,
            "pnl_r": record.get("current_pnl_r"),
            "p_score": extras.get("p_score", record.get("p_score")),
        }
        trail_log = record.get("trail_log")
        if not isinstance(trail_log, list):
            trail_log = []
        trail_log.append(trail_entry)
        if len(trail_log) > 100:
            trail_log = trail_log[-100:]
        record["trail_log"] = trail_log

    record["last_update"] = timestamp
    save_anchor_state(state, path)
    return state

__all__ = [
    "ANCHOR_STATE_PATH",
    "load_anchor_state",
    "save_anchor_state",
    "record_anchor",
    "touch_anchor",
    "update_anchor_metrics",
]
