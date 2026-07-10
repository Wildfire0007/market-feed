#!/usr/bin/env python3
"""Verify non-voided closed ledger exits against stored OHLC candles."""
from __future__ import annotations
import argparse, csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def parse_utc(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def row_ts(row: dict[str, Any]) -> Optional[datetime]:
    for key in ("datetime", "timestamp", "time", "ts", "utc"):
        if key in row:
            ts = parse_utc(row.get(key))
            if ts: return ts
    return None


def candles(asset_dir: Path):
    for name in ("klines_5m.json", "k1m.json", "klines_1m.json"):
        path = asset_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("values") or data.get("data") or data.get("candles") or []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row


def exit_level(row: dict[str, str]) -> Optional[float]:
    reason = str(row.get("close_reason") or "")
    if reason == "take_profit_hit": return safe_float(row.get("tp1"))
    if reason == "take_profit_2_hit": return safe_float(row.get("tp2"))
    if reason == "stop_loss_hit": return safe_float(row.get("sl"))
    return None


def touched(asset_dir: Path, opened: datetime, closed: datetime, level: float) -> bool:
    for candle in candles(asset_dir):
        ts = row_ts(candle)
        if not ts or ts < opened or ts > closed:
            continue
        low = safe_float(candle.get("low") or candle.get("l"))
        high = safe_float(candle.get("high") or candle.get("h"))
        if low is not None and high is not None and low <= level <= high:
            return True
    return False


def verify(public_dir: Path, ledger_path: Path) -> list[str]:
    violations = []
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("voided") or "").strip().lower() == "true":
                continue
            level = exit_level(row)
            if level is None:
                continue
            opened, closed = parse_utc(row.get("opened_at_utc")), parse_utc(row.get("closed_at_utc"))
            asset = str(row.get("asset") or "").strip()
            if not opened or not closed or not asset or not touched(public_dir / asset, opened, closed, level):
                violations.append(f"{row.get('ledger_id') or '?'} {asset} {row.get('close_reason')} level={level}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", default="public")
    parser.add_argument("--ledger", default=None)
    args = parser.parse_args(argv)
    public_dir = Path(args.public_dir)
    ledger = Path(args.ledger) if args.ledger else public_dir / "journal" / "trade_ledger.csv"
    violations = verify(public_dir, ledger)
    for item in violations:
        print(item)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
