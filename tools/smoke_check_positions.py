"""Lightweight smoke check to guard against missing manual position state."""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ENTRY_LOOKBACK = int(os.environ.get("ENTRY_LOOKBACK", "25"))
ENTRY_SIGNALS = {"buy", "sell", "entry", "long", "short"}
PUBLIC_DIR = Path("public")


def _load_signal_states() -> List[Dict[str, object]]:
    signal_files = sorted(PUBLIC_DIR.glob("*/signal.json"))
    states = []
    for path in signal_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        position_state = data.get("position_state") or {}
        states.append(
            {
                "asset": data.get("asset") or path.parent.name,
                "has_position": bool(position_state.get("has_position")),
            }
        )
    return states


def _recent_entries(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    recent: List[Dict[str, str]] = []
    for row in rows:
        signal = (row.get("signal") or "").strip().lower()
        if signal in ENTRY_SIGNALS:
            recent.append(row)
    return recent


def _load_recent_journal_entries() -> List[Dict[str, str]]:
    journal_path = PUBLIC_DIR / "journal" / "trade_journal.csv"
    if not journal_path.exists():
        return []

    with journal_path.open(newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))

    if ENTRY_LOOKBACK > 0:
        reader = reader[-ENTRY_LOOKBACK:]
    return _recent_entries(reader)


def main() -> int:
    signal_states = _load_signal_states()
    if not signal_states:
        print("[smoke-check] no signal.json files found; skipping", flush=True)
        return 0

    all_flat = all(not state.get("has_position") for state in signal_states)
    recent_entries = _load_recent_journal_entries()

    if all_flat and recent_entries:
        assets = {row.get("asset") for row in recent_entries if row.get("asset")}
        print(
            "[smoke-check] detected %d recent entries across assets %s but all signals report flat" % (
                len(recent_entries),
                sorted(assets),
            ),
            flush=True,
        )
        return 1

    print(
        "[smoke-check] signals_flat=%s recent_entries=%d" % (all_flat, len(recent_entries)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
