#!/usr/bin/env python3
"""Quick sanity checks for EXIT de-duplication timing.

Scenarios
---------
a) last EXIT sent 5 minutes ago → expect suppression (None)
b) last EXIT sent 40 minutes ago → expect embed allowed + timestamp refresh
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.notify_discord import _dedup_minutes_for_exit, _now_utc_iso, iso_to_epoch


def apply_exit_dedup(prev_entry, asset, now):
    state_record = {
        "state": "EXIT",
        "key": "EXIT|buy|event=0",
        "anchor": "buy",
        "updated": _now_utc_iso(now),
    }
    exit_dedup_min = _dedup_minutes_for_exit(asset)
    suppressed = False

    if exit_dedup_min > 0:
        last_exit_sent_iso = prev_entry.get("last_exit_sent_utc")
        if last_exit_sent_iso:
            age_sec = iso_to_epoch(_now_utc_iso(now)) - iso_to_epoch(last_exit_sent_iso)
            if age_sec >= 0 and age_sec < exit_dedup_min * 60:
                state_record["last_exit_sent_utc"] = last_exit_sent_iso
                suppressed = True
                return suppressed, state_record
        state_record["last_exit_sent_utc"] = _now_utc_iso(now)

    return suppressed, state_record


def scenario(label: str, last_exit_minutes_ago: int) -> None:
    now = datetime.now(timezone.utc)
    prev_entry = {
        "last_exit_sent_utc": _now_utc_iso(now - timedelta(minutes=last_exit_minutes_ago)),
        "key": "EXIT|buy|event=0",
    }
    suppressed, state_record = apply_exit_dedup(prev_entry, "EURUSD", now)
    status = "SUPPRESSED" if suppressed else "SENT"
    print(f"[{label}] last_exit={last_exit_minutes_ago}m ago → {status}")
    print(f"    last_exit_sent_utc: {state_record.get('last_exit_sent_utc')}")
    print(f"    updated: {state_record.get('updated')}")


def main():
    scenario("A", 5)
    scenario("B", 40)


if __name__ == "__main__":
    main()
