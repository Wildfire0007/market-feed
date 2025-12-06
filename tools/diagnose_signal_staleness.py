"""Diagnose why public/<ASSET>/signal.json files are stale.

The script inspects the trading artefact (public/pipeline/trading_status.json)
that analysis.py requires and compares its timestamp to the latest signal files.
If the trading artefact is missing or older than the analysis staleness budget
(Pipeline default: PIPELINE_MAX_LAG_SECONDS=240), analysis will exit before
writing new signals, leaving the previous values in place.

Usage:
    python tools/diagnose_signal_staleness.py [--public-dir public]

Outputs a human readable report with actionable remediation steps.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

DEFAULT_PUBLIC_DIR = Path("public")
DEFAULT_LAG_SECONDS = 240  # mirrors reports.pipeline_monitor.DEFAULT_MAX_LAG_SECONDS


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _load_json(path: Path) -> Optional[Dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _iter_asset_dirs(base: Path) -> Iterable[Path]:
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in {"pipeline", "monitoring", "debug", "debug-entry-gates"}:
            continue
        yield child


def diagnose(public_dir: Path) -> Tuple[Optional[datetime], Dict[str, Tuple[Optional[datetime], Path]]]:
    trading_path = public_dir / "pipeline" / "trading_status.json"
    trading_payload = _load_json(trading_path)
    trading_ts = None
    if isinstance(trading_payload, dict):
        trading_ts = _parse_ts(trading_payload.get("generated_at_utc"))
    assets: Dict[str, Tuple[Optional[datetime], Path]] = {}
    for asset_dir in _iter_asset_dirs(public_dir):
        signal_path = asset_dir / "signal.json"
        signal_ts = None
        payload = _load_json(signal_path)
        if isinstance(payload, dict):
            signal_ts = _parse_ts(payload.get("retrieved_at_utc") or payload.get("updated_at_utc"))
        assets[asset_dir.name] = (signal_ts, signal_path)
    return trading_ts, assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--lag-seconds", type=float, default=DEFAULT_LAG_SECONDS)
    args = parser.parse_args()

    public_dir: Path = args.public_dir
    lag_budget = args.lag_seconds
    now = datetime.now(timezone.utc)

    trading_ts, assets = diagnose(public_dir)
    print(f"Public dir        : {public_dir.resolve()}")
    if trading_ts is None:
        print("Trading artefact  : MISSING (public/pipeline/trading_status.json)")
        print("Action            : Futtasd a Trading.py lépést, majd az analysis.py-t.")
    else:
        age = (now - trading_ts).total_seconds()
        print(f"Trading artefact  : {trading_ts.isoformat()} (késés: {age/60:.1f} perc)")
        if age > lag_budget:
            print(f"Gate status       : BLOKKOLJA az analysis-t (>{lag_budget:.0f} mp).")
            print("Action            : Futtasd újra a Trading.py lépést, hogy friss adat legyen.")
        else:
            print(f"Gate status       : Rendben ({age:.0f} mp <= {lag_budget:.0f} mp)")

    print("\nSignal állapotok:")
    for asset, (ts, path) in assets.items():
        if ts is None:
            print(f"- {asset:<6}: hiányzik vagy hibás a signal.json ({path})")
            continue
        age = (now - ts).total_seconds()
        freshness = "FRISS" if age <= lag_budget else "ELAVULT"
        print(f"- {asset:<6}: {ts.isoformat()} (késés: {age/3600:.1f} óra) → {freshness}")
    if trading_ts is None:
        print("\nÖsszegzés: az analysis.py nem tud új signal.json-t írni, amíg nincs friss trading artefakt.")
    else:
        age = (now - trading_ts).total_seconds()
        if age > lag_budget:
            print("\nÖsszegzés: a Trading artefakt a megengedett késleltetésen túl van, ezért az analysis.py kilép.")
            print("Javítás: futtasd a Trading.py-t (friss adatgyűjtés), majd az analysis.py-t.")
        else:
            print("\nÖsszegzés: a Trading artefakt friss; ellenőrizd az analysis logokat egyéb hibákért.")


if __name__ == "__main__":
    main()
