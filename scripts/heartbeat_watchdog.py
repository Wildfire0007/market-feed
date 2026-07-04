"""Alert when the analysis heartbeat is stale."""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from datetime import datetime, timezone
from pathlib import Path


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", default="public/system_heartbeat.json")
    parser.add_argument("--max-age-min", type=float, default=30.0)
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
            if age_min <= args.max_age_min:
                print(f"Heartbeat OK: {age_min:.1f} min")
                return 0
            msg = f"TD pipeline heartbeat stale: {age_min:.1f} min > {args.max_age_min:.1f} min"
    print(msg, file=sys.stderr)
    urls = [u.strip() for u in os.getenv("DISCORD_WEBHOOK_URL", "").replace("\n", ",").split(",") if u.strip()]
    embed = {"title": "⚠️ TD pipeline heartbeat stale", "description": msg, "color": 0xE74C3C}
    for url in urls:
        try:
            requests.post(url, json={"embeds": [embed]}, timeout=8)
        except Exception as exc:
            print(f"Discord alert failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
