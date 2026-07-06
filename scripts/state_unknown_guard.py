#!/usr/bin/env python3
"""Send one actionable alert per stale-data episode while a position is open."""
from __future__ import annotations
import json, os, sys, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
try:
    import requests
except Exception:
    requests = None
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config import analysis_settings as settings
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER=logging.getLogger(__name__)
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "public"))
STATE = PUBLIC_DIR / "_position_lifecycle_state.json"
DEDUP = PUBLIC_DIR / "monitoring" / "state_unknown_guard.json"

def load(path: Path) -> Dict[str, Any]:
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception: return {}

def parse(value: Any) -> Optional[datetime]:
    try: return datetime.fromisoformat(str(value).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception: return None

def sf(v: Any) -> Optional[float]:
    try:
        f=float(v); return f if f==f else None
    except Exception: return None

def urls() -> list[str]:
    raw=os.getenv("DISCORD_WEBHOOK_URL_ACTIONABLE") or os.getenv("DISCORD_WEBHOOK_URL", "")
    return [u.strip() for u in raw.replace("\n", ",").split(",") if u.strip()]

def fmt(v: Any) -> str:
    x=sf(v); return "N/A" if x is None else (f"{x:,.1f}" if abs(x)>=1000 else f"{x:.5f}")

def main() -> int:
    cfg = getattr(settings, "POSITION_LIFECYCLE", {}) or {}
    max_age = float(cfg.get("state_unknown_max_age_minutes", 15) or 15)
    include_pending = bool(cfg.get("state_unknown_include_pending", False))
    hb = load(PUBLIC_DIR / "system_heartbeat.json")
    stamp = parse(hb.get("last_update_utc") or hb.get("generated_at_utc"))
    stale = stamp is None or (datetime.now(timezone.utc)-stamp).total_seconds()/60.0 > max_age
    lifecycle = load(STATE).get("positions") or {}
    dedup = load(DEDUP)
    changed = False
    for asset,pos in lifecycle.items():
        if not isinstance(pos, dict): continue
        status=str(pos.get("status") or "").lower()
        if status != "open" and not (include_pending and status == "pending"): continue
        sig = load(PUBLIC_DIR / asset / "signal.json")
        spot = (sig.get("spot") or {}).get("price")
        condition = "heartbeat_stale" if stale else ""
        if not condition:
            dedup.pop(str(asset), None); changed=True; continue
        key=f"{asset}|{pos.get('opened_at_utc') or pos.get('pending_since_utc')}|{condition}"
        if dedup.get(str(asset)) == key: continue
        embed={"title":"⚠️ Nyitott pozíció, elavult adat — kezeld manuálisan","description":f"Eszköz: `{asset}`\nOk: `{condition}`\nSpot: `{fmt(spot)}`\nSL: `{fmt(pos.get('sl'))}`\nTP1: `{fmt(pos.get('tp1'))}`","color":0xE74C3C}
        for url in urls():
            if requests:
                try:
                    resp=requests.post(url,json={"embeds":[embed]},timeout=8)
                    _webhook_log_response(LOGGER,'state_unknown_guard','actionable',resp)
                except Exception as exc:
                    _webhook_log_exception(LOGGER,'state_unknown_guard','actionable',exc)            
        dedup[str(asset)] = key; changed=True
    if changed:
        DEDUP.parent.mkdir(parents=True, exist_ok=True)
        DEDUP.write_text(json.dumps(dedup, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0
if __name__ == "__main__": raise SystemExit(main())
