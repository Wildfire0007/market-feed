"""Lightweight smoke check to guard against missing manual position state.

Goal:
- Fail only when there is evidence that we *should* have an open position tracked
  (manual_positions indicates an open position), but all signal.json files still
  report flat.
- If there are recent ENTRY journal rows but manual_positions is empty/missing,
  treat as WARN (do not fail). This avoids false failures when ENTRY commit is
  performed later (e.g., by notify job) and state has not yet persisted.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ENTRY_LOOKBACK = int(os.environ.get("ENTRY_LOOKBACK", "25"))
ENTRY_SIGNALS = {"buy", "sell", "entry", "long", "short"}
PUBLIC_DIR = Path("public")

# Where manual state may live (support both current and mirrored paths)
MANUAL_POS_PATHS = [
    Path("public/_manual_positions.json"),
    Path("config/manual_positions.json"),
]


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


def _load_manual_positions_state() -> Dict[str, object]:
    """
    Returns a dict that is either:
    - { "ASSET": { ...position fields... }, ... }  (legacy/simple)
    - or { "positions": { "ASSET": { ... } } }    (wrapped)
    If missing/invalid, returns {}.
    """
    for path in MANUAL_POS_PATHS:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
            data = json.loads(raw or "{}")
        except Exception:
            return {}

        if isinstance(data, dict) and isinstance(data.get("positions"), dict):
            return data["positions"]
        if isinstance(data, dict):
            return data
        return {}
    return {}


def _manual_has_open_position(asset: str, manual_positions: Dict[str, object]) -> bool:
    """
    Conservative: treat as open if there is a non-empty dict for asset and it does not explicitly say has_position=false.
    """
    v = manual_positions.get(asset)
    if not v:
        return False
    if isinstance(v, dict):
        if v.get("has_position") is False:
            return False
        return True
    return bool(v)


def main() -> int:
    signal_states = _load_signal_states()
    if not signal_states:
        print("[smoke-check] no signal.json files found; skipping", flush=True)
        return 0

    all_flat = all(not state.get("has_position") for state in signal_states)
    recent_entries = _load_recent_journal_entries()

    # New: load manual state to decide whether "flat + recent entry" is a real error or just pending commit.
    manual_positions = _load_manual_positions_state()

    if all_flat and recent_entries:
        assets = {row.get("asset") for row in recent_entries if row.get("asset")}
        tracked_assets = sorted(a for a in assets if _manual_has_open_position(a, manual_positions))

        if tracked_assets:
            # Real failure: manual state claims open position(s), yet all signal.json says flat
            print(
                "[smoke-check] FAIL: detected %d recent entries; manual_positions has open=%s but all signals report flat"
                % (len(recent_entries), tracked_assets),
                flush=True,
            )
            return 1

        # Warn-only: entry journal exists, but we have no persisted/manual open state yet
        print(
            "[smoke-check] WARN: detected %d recent entries across assets %s but all signals report flat; "
            "manual_positions is empty/missing -> not failing"
            % (len(recent_entries), sorted(assets)),
            flush=True,
        )
        return 0

    print(
        "[smoke-check] signals_flat=%s recent_entries=%d manual_positions_assets=%d"
        % (all_flat, len(recent_entries), len(manual_positions)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
