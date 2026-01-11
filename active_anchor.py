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
import sqlite3
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Dict, Optional

import state_db

PUBLIC_DIR = os.getenv("PUBLIC_DIR", "public")
ANCHOR_STATE_PATH = os.path.join(PUBLIC_DIR, "_active_anchor.json")
_DB_INITIALIZED = False


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


def _ensure_db_initialized() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    state_db.initialize()
    _DB_INITIALIZED = True


def _load_anchor_state_from_db() -> Dict[str, Dict[str, Any]]:
    _ensure_db_initialized()
    connection = state_db.connect()
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT asset, anchor_type, price, timestamp, meta_json FROM anchors"
        ).fetchall()
    finally:
        connection.close()

    state: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        asset = str(row["asset"] or "").upper()
        if not asset:
            continue
        payload: Dict[str, Any] = {}
        meta_json = row["meta_json"]
        if meta_json:
            try:
                parsed = json.loads(meta_json)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                payload.update(parsed)
        side = str(row["anchor_type"] or payload.get("side") or "").lower()
        price = row["price"] if row["price"] is not None else payload.get("price")
        timestamp = row["timestamp"] or payload.get("timestamp")
        if side:
            payload["side"] = side
        if price is not None:
            payload["price"] = price
        if timestamp:
            payload["timestamp"] = timestamp
        if not payload.get("side"):
            continue
        existing = state.get(asset)
        if existing:
            new_ts = _parse_iso(payload.get("timestamp"))
            current_ts = _parse_iso(existing.get("timestamp"))
            if current_ts and (not new_ts or new_ts <= current_ts):
                continue
        state[asset] = payload

    return state


def _load_anchor_state_from_json(path: str) -> Dict[str, Dict[str, Any]]:
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


def load_anchor_state(path: str = ANCHOR_STATE_PATH) -> Dict[str, Dict[str, Any]]:
    try:
        return _load_anchor_state_from_db()
    except Exception:
        return _load_anchor_state_from_json(path)


def save_anchor_state(state: Dict[str, Dict[str, Any]], path: str = ANCHOR_STATE_PATH) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    try:
        _ensure_db_initialized()
        connection = state_db.connect()
    except sqlite3.Error:
        return
    try:
        with connection:
            connection.execute("DELETE FROM anchors")
            for asset, payload in state.items():
                if not isinstance(payload, dict):
                    continue
                side = str(payload.get("side") or "").lower()
                if not side:
                    continue
                price = payload.get("price")
                timestamp = payload.get("timestamp") or _utc_now_iso()
                meta_payload = dict(payload)
                meta_payload.setdefault("side", side)
                meta_payload.setdefault("price", price)
                meta_payload.setdefault("timestamp", timestamp)
                meta_json = json.dumps(meta_payload, ensure_ascii=False)
                connection.execute(
                    """
                    INSERT INTO anchors (asset, anchor_type, price, timestamp, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        asset.upper(),
                        side,
                        price,
                        timestamp,
                        meta_json,
                    ),
                )
    except sqlite3.Error:
        return
    finally:
        connection.close()
      

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

        pnl_series = [
            _safe_float(entry.get("pnl_r"))
            for entry in trail_log
            if _safe_float(entry.get("pnl_r")) is not None
        ]
        pnl_series = [float(x) for x in pnl_series if x is not None]
        drift_score = None
        drift_state = None
        if len(pnl_series) >= 6:
            mid = len(pnl_series) // 2
            first_mean = mean(pnl_series[:mid]) if mid else None
            last_mean = mean(pnl_series[mid:]) if mid else None
            if first_mean is not None and last_mean is not None:
                drift_score = float(last_mean - first_mean)
                if drift_score < -0.25:
                    drift_state = "deteriorating"
                elif drift_score > 0.25:
                    drift_state = "improving"
                else:
                    drift_state = "stable"
        if drift_state:
            record["drift_state"] = drift_state
        if drift_score is not None:
            record["drift_score"] = drift_score
        record["drift_last_check"] = timestamp

    record["last_update"] = timestamp
    state[asset_key] = record
    save_anchor_state(state, path)
    return state

def reset_anchor_state(
    max_age_hours: Optional[int] = None,
    path: str = ANCHOR_STATE_PATH,
    *,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Clear or prune the anchor state file.

    Parameters
    ----------
    max_age_hours:
        If ``None`` or ``<= 0`` every entry is removed.  Otherwise records with a
        ``last_update`` (or ``timestamp``) older than ``now - max_age_hours`` are
        dropped.
    path:
        Location of the anchor state JSON file.
    now:
        Reference timestamp used for age comparisons.  Defaults to ``utcnow`` and
        is primarily exposed for deterministic unit tests.

    dry_run:
        When ``True`` the function performs calculations without touching the
        underlying JSON file.

    Returns
    -------
    dict
        The updated anchor state.
    """

    state = load_anchor_state(path)
    if not state:
        if not dry_run:
            save_anchor_state({}, path)
        return {}

    if max_age_hours is None or max_age_hours <= 0:
        if not dry_run:
            save_anchor_state({}, path)
        return {}

    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(hours=float(max_age_hours))

    pruned: Dict[str, Dict[str, Any]] = {}
    for asset, payload in state.items():
        timestamp_raw = payload.get("last_update") or payload.get("timestamp")
        timestamp = _parse_iso(timestamp_raw)
        if timestamp is None or timestamp >= cutoff:
            pruned[asset] = payload

    if pruned != state and not dry_run:
        save_anchor_state(pruned, path)

    return pruned


__all__ = [
    "ANCHOR_STATE_PATH",
    "load_anchor_state",
    "save_anchor_state",
    "record_anchor",
    "touch_anchor",
    "update_anchor_metrics",
    "reset_anchor_state",
]
