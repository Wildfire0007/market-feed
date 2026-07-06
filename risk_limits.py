"""Daily manual-trading risk lockout helpers."""
from __future__ import annotations

import csv
import json
import os
import sys
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover - optional runtime dependency
    requests = None

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



def _notify_daily_lockout_once(state: Dict[str, Any], *, now: datetime) -> None:
    raw = os.getenv("DISCORD_WEBHOOK_URL_ACTIONABLE") or os.getenv("DISCORD_WEBHOOK_URL", "")
    urls = [u.strip() for u in raw.replace("\n", ",").split(",") if u.strip()]    
    if not urls or requests is None:
        return
    day = now.astimezone(timezone.utc).date().isoformat()
    state_path = Path(os.getenv("RISK_LOCKOUT_NOTIFY_STATE", "public/monitoring/risk_lockout_notify_state.json"))
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        payload = {}
    if payload.get("last_notified_utc_day") == day:
        return
    msg = (
        f"Napi kockázati lockout aktív ({day} UTC).\n"
        f"Realizált PnL: {state.get('realized_pnl_usd')} USD\n"
        f"Vesztes ügyletek: {state.get('losing_trades')} / címkézett: {state.get('labeled_trades')}\n"
        f"Napi ablak kezdete: {state.get('day_start_utc')}"
    )
    embed = {"title": "⛔ Daily risk lockout", "description": msg, "color": 0xE74C3C}
    sent = False
    for url in urls:
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            sent = _webhook_log_response(LOGGER, "risk_limits", "actionable", resp) or sent
        except Exception as exc:
            _webhook_log_exception(LOGGER, "risk_limits", "actionable", exc)            
            print(f"Discord risk lockout alert failed: {exc}", file=sys.stderr)
    if sent:
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"last_notified_utc_day": day, "last_notified_at_utc": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

def compute_daily_labeled_pnl(config: Dict[str, Any], journal_path: Path, *, now: datetime) -> Dict[str, Any]:
    """Compute today's labeled loss count/PnL using the risk-lockout rules."""
    cfg = config.get("risk_limits") or {}    
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
    return {"day_start_utc": start.isoformat().replace("+00:00", "Z"), "realized_pnl_usd": round(pnl, 4), "losing_trades": losses, "labeled_trades": count}


def evaluate_daily_lockout(config: Dict[str, Any], journal_path: Path, *, now: datetime) -> Dict[str, Any]:
    """Compute today's labeled loss count/PnL and return a hard-lockout decision."""
    cfg = config.get("risk_limits") or {}
    if not cfg.get("enabled", False):
        return {"locked": False, "enabled": False}
    metrics = compute_daily_labeled_pnl(config, journal_path, now=now)
    locked = metrics["realized_pnl_usd"] <= -float(cfg.get("daily_loss_limit_usd", 15) or 15) or metrics["losing_trades"] >= int(cfg.get("daily_max_losing_trades", 2) or 2)
    result = {"enabled": True, "locked": locked, **metrics}    
    if locked:
        _notify_daily_lockout_once(result, now=now)
    return result   
