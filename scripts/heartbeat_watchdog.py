"""Alert when the analysis heartbeat is stale."""
from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response

import requests
LOGGER = logging.getLogger(__name__)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _alert(msg: str, title: str = "⚠️ TD pipeline heartbeat stale") -> None:
    urls = [u.strip() for u in os.getenv("DISCORD_WEBHOOK_URL", "").replace("\n", ",").split(",") if u.strip()]
    embed = {"title": title, "description": msg, "color": 0xE74C3C}
    for url in urls:
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            _webhook_log_response(LOGGER, "heartbeat_watchdog", "diagnostic", resp)
        except Exception as exc:
            _webhook_log_exception(LOGGER, "heartbeat_watchdog", "diagnostic", exc)
            print(f"Discord alert failed: {exc}", file=sys.stderr)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", default="public/system_heartbeat.json")
    parser.add_argument("--max-age-min", type=float, default=30.0)
    parser.add_argument("--force-alert", action="store_true", help="Send a test alert regardless of heartbeat age")    
    args = parser.parse_args()
    path = Path(args.heartbeat)
    if not path.exists():
        msg = f"Heartbeat hiányzik: {path}"
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = payload.get("last_update_utc") or payload.get("generated_at_utc")
        if not stamp:
            msg = f"Heartbeat timestamp hiányzik: {path}"
        else:
            age_min = (datetime.now(timezone.utc) - parse_utc(str(stamp))).total_seconds() / 60.0
            if args.force_alert:
                msg = f"TD pipeline heartbeat test alert: heartbeat age {age_min:.1f} min"
                print(f"WATCHDOG ALERT — heartbeat age {age_min:.1f} min")
                _alert(msg, "⚠️ TD watchdog test alert")
                return 0            
            if age_min <= args.max_age_min:
                print(f"WATCHDOG OK — heartbeat age {age_min:.1f} min")
                return 0
            msg = f"TD pipeline heartbeat stale: {age_min:.1f} min > {args.max_age_min:.1f} min"
    print(msg, file=sys.stderr)
    _alert(msg)
    return 1


def main() -> int:
    try:
        return _main()
    except SystemExit:
        raise
    except Exception as exc:
        msg = f"⚠️ Watchdog internal error: {exc}"
        print(msg, file=sys.stderr)
        _alert(msg, "⚠️ Watchdog internal error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
