#!/usr/bin/env python3
"""Append-only authoritative lifecycle trade ledger."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

LEDGER_HEADER = [
    "ledger_id", "asset", "side", "order_type", "entry", "sl", "tp1", "tp2", "size_units",
    "opened_at_utc", "closed_at_utc", "close_reason", "outcome", "est_pnl_usd",
    "source_signal", "entry_signature", "trigger_bar_utc", "voided", "void_reason",
]
WIN_OUTCOMES = {"tp1_closed", "take_profit_2_hit"}
TERMINAL_EXCLUDE = {"expired"}


def safe_float(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def parse_utc(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except Exception:
        return None


def fmt(value: Any) -> str:
    n = safe_float(value)
    if n is None:
        return "" if value is None else str(value)
    return f"{n:.8f}".rstrip("0").rstrip(".")


def exit_level_for_reason(pos: Dict[str, Any], reason: str) -> Any:
    if reason == "take_profit_hit":
        return pos.get("tp1")
    if reason == "take_profit_2_hit":
        return pos.get("tp2")
    if reason == "stop_loss_hit":
        return pos.get("sl")
    if reason in {"hard_exit", "session_force_close"}:
        return pos.get("close_spot") or pos.get("last_spot") or pos.get("entry")
    return pos.get("entry")


def pnl_usd(side: str, entry: Any, exit_level: Any, units: Any) -> Optional[float]:
    entry_f, exit_f, units_f = safe_float(entry), safe_float(exit_level), safe_float(units)
    if entry_f is None or exit_f is None or units_f is None:
        return None
    return (exit_f - entry_f) * units_f * (1 if str(side).lower() == "long" else -1)


def ledger_id(asset: str, pos: Dict[str, Any]) -> str:
    raw = "|".join(str(x or "") for x in (
        asset.upper(), pos.get("opened_at_utc") or pos.get("pending_since_utc"), pos.get("entry"),
        pos.get("side"), pos.get("order_type"), pos.get("entry_signature"),
    ))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return {str(r.get("ledger_id") or "") for r in csv.DictReader(handle) if r.get("ledger_id")}
    except Exception:
        return set()


def append_position(path: Path, asset: str, pos: Dict[str, Any], state_meta: Dict[str, Any]) -> bool:
    lid = ledger_id(asset, pos)
    done = state_meta.setdefault("ledger_ids", [])
    existing_ids = _existing_ids(path)
    if lid in existing_ids:
        if lid not in done:
            done.append(lid)
        return False
    reason = str(pos.get("close_reason") or "")
    pnl = pnl_usd(str(pos.get("side") or ""), pos.get("entry"), exit_level_for_reason(pos, reason), pos.get("size_units"))
    row = {
        "ledger_id": lid,
        "asset": asset.upper(),
        "side": str(pos.get("side") or ""),
        "order_type": str(pos.get("order_type") or ""),
        "entry": fmt(pos.get("entry")),
        "sl": fmt(pos.get("sl")),
        "tp1": fmt(pos.get("tp1")),
        "tp2": fmt(pos.get("tp2")),
        "size_units": fmt(pos.get("size_units")),
        "opened_at_utc": str(pos.get("opened_at_utc") or pos.get("pending_since_utc") or ""),
        "closed_at_utc": str(pos.get("closed_at_utc") or ""),
        "trigger_bar_utc": str(pos.get("trigger_bar_utc") or ""),        
        "close_reason": reason,
        "outcome": str(pos.get("outcome") or reason),
        "est_pnl_usd": "" if pnl is None else f"{pnl:.2f}",
        "source_signal": str(pos.get("source_signal") or ""),
        "entry_signature": str(pos.get("entry_signature") or ""),
        "voided": "false",
        "void_reason": "",        
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    if lid not in done:
        done.append(lid)    
    return True


def rows_between(path: Path, start: datetime, end: datetime) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if str(r.get("voided") or "").strip().lower() != "true" and (ts := parse_utc(r.get("closed_at_utc"))) and start <= ts <= end]


def stats(rows: Iterable[dict[str, str]]) -> Dict[str, Any]:
    closed = [r for r in rows if str(r.get("voided") or "").strip().lower() != "true" and str(r.get("outcome") or "").lower() not in TERMINAL_EXCLUDE]    
    wins = sum(1 for r in closed if str(r.get("outcome") or "").lower() in WIN_OUTCOMES)
    losses = sum(1 for r in closed if (safe_float(r.get("est_pnl_usd")) or 0.0) < 0)
    pnl = sum(safe_float(r.get("est_pnl_usd")) or 0.0 for r in closed)
    return {"total": len(closed), "wins": wins, "losses": losses, "pnl": pnl}
