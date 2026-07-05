"""Daily manual-trading risk lockout helpers."""
from __future__ import annotations

import csv
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOSS_OUTCOMES = {"stopped"}


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _day_start(now: datetime, boundary: str) -> datetime:
    hour, minute = [int(part) for part in str(boundary or "00:00").split(":")[:2]]
    start = datetime.combine(now.astimezone(timezone.utc).date(), time(hour, minute, tzinfo=timezone.utc))
    return start if now >= start else start - timedelta(days=1)


def evaluate_daily_lockout(config: Dict[str, Any], journal_path: Path, *, now: datetime) -> Dict[str, Any]:
    """Compute today's labeled loss count/PnL and return a hard-lockout decision."""
    cfg = config.get("risk_limits") or {}
    if not cfg.get("enabled", False):
        return {"locked": False, "enabled": False}
    scope = {str(a).upper() for a in cfg.get("lockout_scope", [])}
    start = _day_start(now.astimezone(timezone.utc), str(cfg.get("day_boundary_utc", "00:00")))
    losses = 0
    pnl = 0.0
    count = 0
    loss_outcomes = set(LOSS_OUTCOMES)
    if cfg.get("count_ambiguous_as_loss", True):
        loss_outcomes.add("ambiguous")
    if journal_path.exists():
        with journal_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                asset = str(row.get("asset") or "").upper()
                if scope and asset not in scope:
                    continue
                ts = _parse_utc(row.get("analysis_timestamp"))
                if ts is None or ts < start or ts > now:
                    continue
                outcome = str(row.get("validation_outcome") or "").strip().lower()
                if not outcome:
                    continue
                count += 1
                rr = float(row.get("validation_rr") or 0.0)
                pnl += rr * 10.0
                if outcome in loss_outcomes or rr < 0:
                    losses += 1
    locked = pnl <= -float(cfg.get("daily_loss_limit_usd", 15) or 15) or losses >= int(cfg.get("daily_max_losing_trades", 2) or 2)
    return {"enabled": True, "locked": locked, "day_start_utc": start.isoformat().replace("+00:00", "Z"), "realized_pnl_usd": round(pnl, 4), "losing_trades": losses, "labeled_trades": count}
