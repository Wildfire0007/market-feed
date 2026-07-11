"""Alert when the analysis heartbeat is stale."""
from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import analysis_settings as settings
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER = logging.getLogger(__name__)
BUDAPEST = ZoneInfo("Europe/Budapest")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _parse_hhmm(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(int(hour), int(minute), tzinfo=timezone.utc)


def _quiet_context(now: datetime, window: Any, weekend_as_quiet: bool = False) -> str:
    utc_now = now.astimezone(timezone.utc)
    if weekend_as_quiet and utc_now.weekday() >= 5:
        return "hétvége"    
    if not isinstance(window, list) or len(window) != 2:
        return "nem"
    start = _parse_hhmm(window[0])
    end = _parse_hhmm(window[1])
    current = utc_now.timetz().replace(second=0, microsecond=0)
    in_window = start <= current < end if start <= end else current >= start or current < end
    return "éjszaka" if in_window else "nem"


def _position_count(public_dir: Path) -> int:
    positions = (_load(public_dir / "_position_lifecycle_state.json").get("positions") or {})
    count = 0
    for pos in positions.values():
        if isinstance(pos, dict) and str(pos.get("status") or "").lower() in {"open", "pending"}:
            count += 1
    return count


def _is_detection_band(age_min: float, threshold_min: float, *, first_band_min: float = 5.0) -> bool:    
    """Return true only in stateless alert bands for a stale episode.

    A job with no persisted state can still deduplicate by alerting only in the
    first 15-minute detection band and then in 15-minute hourly escalation
    bands, all measured from the active threshold. Examples with threshold 30:
    age 31 is in [30,45) and sends at the first sampled edge, age 40 is
    suppressed inside the same band, age 92 is in [90,105) and sends at the
    hourly sampled edge, while age 50 and 80 suppress. In production this bounds
    repeats to about one notification per hour per stale episode without state.
    """
    if age_min < threshold_min:
        return False
    elapsed = age_min - threshold_min
    if elapsed < first_band_min:    
        return True
    return elapsed >= 60 and (elapsed % 60) < 5


def _urls(channel: str) -> list[str]:
    env_name = "DISCORD_WEBHOOK_URL_ACTIONABLE" if channel == "actionable" else "DISCORD_WEBHOOK_URL_DIAGNOSTIC"
    raw = os.getenv(env_name) or os.getenv("DISCORD_WEBHOOK_URL", "")
    return [u.strip() for u in raw.replace("\n", ",").split(",") if u.strip()]


def _alert(msg: str, title: str = "⚠️ TD pipeline heartbeat stale", *, channel: str = "diagnostic") -> None:
    embed = {"title": title, "description": msg, "color": 0xE74C3C}
    for url in _urls(channel):   
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            _webhook_log_response(LOGGER, "heartbeat_watchdog", channel, resp)
        except Exception as exc:
            _webhook_log_exception(LOGGER, "heartbeat_watchdog", channel, exc)
            print(f"Discord alert failed: {exc}", file=sys.stderr)


def _format_stamp(stamp: Optional[datetime]) -> str:
    if stamp is None:
        return "N/A"
    utc = stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    budapest = stamp.astimezone(BUDAPEST).isoformat()
    return f"{utc} / {budapest}"


def _message(text: str, *, stamp: Optional[datetime], quiet_context: str, position_count: int) -> str:    
    return "\n".join([
        text,
        f"Stale since: `{_format_stamp(stamp)}` (UTC / Europe/Budapest)",
        f"Csendes időszak: `{quiet_context}`",        
        f"Open/pending positions: `{position_count}`",
    ])


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", default="public/system_heartbeat.json")
    parser.add_argument("--public-dir", default=os.getenv("NOTIFY_PUBLIC_DIR", "public"))
    parser.add_argument("--max-age-min", type=float, default=30.0)
    parser.add_argument("--force-alert", action="store_true", help="Send a test alert regardless of heartbeat age")
    args = parser.parse_args()
    path = Path(args.heartbeat)
    public_dir = Path(args.public_dir)
    cfg = getattr(settings, "WATCHDOG", {}) or {}
    now = datetime.now(timezone.utc)
    quiet_context = _quiet_context(now, cfg.get("quiet_hours_utc"), bool(cfg.get("weekend_as_quiet")))
    quiet = quiet_context != "nem"    
    position_count = _position_count(public_dir)
    threshold = float(args.max_age_min)
    channel = "actionable"
    if quiet and position_count <= 0:
        threshold = float(cfg.get("quiet_hours_max_age_min", threshold) or threshold)
        channel = str(cfg.get("quiet_hours_channel") or "diagnostic").lower()    
    if not path.exists():
        msg = _message(f"Heartbeat hiányzik: {path}", stamp=None, quiet_context=quiet_context, position_count=position_count)        
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp_raw = payload.get("last_update_utc") or payload.get("generated_at_utc")
        stamp = parse_utc(str(stamp_raw)) if stamp_raw else None
        if stamp is None:
            msg = _message(f"Heartbeat timestamp hiányzik: {path}", stamp=None, quiet_context=quiet_context, position_count=position_count)            
        else:
            age_min = (now - stamp).total_seconds() / 60.0
            if args.force_alert:
                msg = _message(f"TD pipeline heartbeat test alert: heartbeat age {age_min:.1f} min", stamp=stamp, quiet_context=quiet_context, position_count=position_count)                
                print(f"WATCHDOG ALERT — heartbeat age {age_min:.1f} min")
                _alert(msg, "⚠️ TD watchdog test alert", channel=channel)
                return 0
            if age_min <= threshold:
                print(f"WATCHDOG OK — heartbeat age {age_min:.1f} min")
                return 0
            first_band_min = 15.0 if quiet and position_count <= 0 else 5.0
            if not _is_detection_band(age_min, threshold, first_band_min=first_band_min):                
                print(f"WATCHDOG SUPPRESSED — heartbeat age {age_min:.1f} min")
                return 0
            msg = _message(f"TD pipeline heartbeat stale: {age_min:.1f} min > {threshold:.1f} min", stamp=stamp, quiet_context=quiet_context, position_count=position_count)                
    print(msg, file=sys.stderr)
    _alert(msg, channel=channel)
    return 1


def main() -> int:
    try:
        return _main()
    except SystemExit:
        raise
    except Exception as exc:
        msg = f"⚠️ Watchdog internal error: {exc}"
        print(msg, file=sys.stderr)
        _alert(msg, "⚠️ Watchdog internal error", channel="actionable")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
