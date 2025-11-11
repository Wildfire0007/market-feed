"""Quick diagnostics for 5 minute intraday coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "CoverageSummary",
    "load_klines",
    "compute_coverage",
    "main",
]


@dataclass
class CoverageSummary:
    asset: str
    date: str
    bar_count: int
    expected_bars: int
    missing_bars: int
    session_coverage_pct: float
    first_bar_utc: Optional[str]
    last_bar_utc: Optional[str]
    gaps: List[str]


FIVE_MINUTES = timedelta(minutes=5)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def load_klines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        values: Sequence[Any] = data.get("values") or ()
    elif isinstance(data, list):
        values = data
    else:
        values = ()
    rows: List[Dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, Sequence) and len(item) >= 6:
            ts = item[0]
            rows.append({"datetime": ts, "open": item[1], "high": item[2], "low": item[3], "close": item[4], "volume": item[5]})
    return rows


def compute_coverage(
    asset: str,
    date: str,
    *,
    public_dir: Path = Path("public"),
) -> CoverageSummary:
    day_path = public_dir / asset / "klines_5m.json"
    rows = load_klines(day_path)
    day = datetime.fromisoformat(f"{date}T00:00:00+00:00")
    next_day = day + timedelta(days=1)
    stamps: List[datetime] = []
    for row in rows:
        ts = _parse_timestamp(row.get("datetime") or row.get("t"))
        if ts is None:
            continue
        if day <= ts < next_day:
            stamps.append(ts)
    stamps.sort()
    expected_bars = int((next_day - day) / FIVE_MINUTES)
    gaps: List[str] = []
    gap_bars_total = 0
    if stamps:
        prev = stamps[0]
        for current in stamps[1:]:
            delta = current - prev
            if delta > FIVE_MINUTES:
                gap_bars = int(delta / FIVE_MINUTES) - 1
                gap_bars_total += max(gap_bars, 0)
                gap_start = prev + FIVE_MINUTES
                gap_end = current - FIVE_MINUTES
                gaps.append(f"{gap_start.isoformat().replace('+00:00','Z')}→{gap_end.isoformat().replace('+00:00','Z')}")
            prev = current
    bar_count = len(stamps)
    missing_bars = max(expected_bars - bar_count, 0)
    coverage_pct = 0.0
    if expected_bars:
        coverage_pct = round((bar_count / expected_bars) * 100.0, 2)
    first_bar = stamps[0].isoformat().replace("+00:00", "Z") if stamps else None
    last_bar = stamps[-1].isoformat().replace("+00:00", "Z") if stamps else None
    return CoverageSummary(
        asset=asset,
        date=date,
        bar_count=bar_count,
        expected_bars=expected_bars,
        missing_bars=missing_bars,
        session_coverage_pct=coverage_pct,
        first_bar_utc=first_bar,
        last_bar_utc=last_bar,
        gaps=gaps,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="Instrument ticker (e.g. NVDA)")
    parser.add_argument("--date", required=True, help="UTC trading date YYYY-MM-DD")
    parser.add_argument(
        "--public-dir",
        default="public",
        help="Directory containing asset outputs (default: public)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    summary = compute_coverage(
        args.asset,
        args.date,
        public_dir=Path(args.public_dir),
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
