#!/usr/bin/env python3
"""CLI helper that aggregates entry gate rejection data across recent pipeline runs.

The script is intentionally defensive because TD pipeline debug logs may change
slightly over time and different tasks may emit distinct JSON shapes.  It walks
through recent JSON files, extracts candidate trade/signal evaluations, and
summarises which entry gates act as bottlenecks per instrument.

Assumptions
-----------
* Timestamps provided by the pipeline are interpreted as UTC.  If a timestamp
  is naïve (i.e. lacks time-zone information) we treat it as UTC for the
  purpose of the time-of-day bucketing.
* When we encounter an entry-check structure but no failed reasons we treat the
  candidate as accepted.  In other words, an empty reason list means "not
  rejected" rather than creating a synthetic "no_reasons" bucket.  This mirrors
  the typical pipeline semantics where missing reasons implies a successful
  gate evaluation.
* Time-of-day buckets follow the naming used by
  ``analysis_settings["atr_percentile_min_by_tod"]["buckets"]`` and are
  approximated here as:
    - ``open``  → [00:00, 02:00)
    - ``mid``   → [02:00, 18:00)
    - ``close`` → [18:00, 24:00)
  Adjust the :data:`TOD_BUCKETS` constant below if a different split is needed.

Only the Python standard library is used so the tool can run in minimal
execution environments.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Time-of-day handling
# ---------------------------------------------------------------------------

TOD_BUCKETS: Tuple[Tuple[str, range], ...] = (
    ("open", range(0, 120)),     # 00:00 - 01:59 UTC
    ("mid", range(120, 1080)),   # 02:00 - 17:59 UTC
    ("close", range(1080, 1440)) # 18:00 - 23:59 UTC
)

SYMBOL_KEYS = ("symbol", "asset", "instrument", "ticker", "pair")
TIMESTAMP_KEYS = ("timestamp", "ts", "time", "bar_time", "bar_timestamp")
REASON_KEYS_DIRECT = (
    "entry_gate_rejections",
    "entry_gate_missing",
    "entry_rejections",
    "entry_reasons",
    "rejection_reasons",
    "rejections",
    "missing",
    "reasons",
    "gate_failures",
    "failures",
)

# Keys whose values are mappings containing gate evaluation details.
REASON_KEYS_NESTED = (
    "entry_gate",
    "entry_gate_summary",
    "entry_check_summary",
    "entry_checks",
    "gate_checks",
)

JSON_EXTENSIONS = {".json", ".jsonl"}
ENTRY_GATE_LOG_RELATIVE_DIR = Path("debug/entry_gates")
ENTRY_GATE_STATS_RELATIVE_PATH = Path("debug/entry_gate_stats.json")
ENTRY_GATE_DAILY_RELATIVE_PATH = Path("monitoring/entry_gate_daily.json")


@dataclass
class SymbolStats:
    """Aggregated statistics for a single instrument."""

    total_candidates: int = 0
    total_rejected: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)
    by_time_of_day: MutableMapping[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    total_candidates_by_tod: Counter[str] = field(default_factory=Counter)
    daily_candidates: Counter[Tuple[str, str]] = field(default_factory=Counter)
    daily_rejections: Counter[Tuple[str, str]] = field(default_factory=Counter)

    def register(self, reasons: Sequence[str], timestamp: Optional[datetime]) -> None:
        self.total_candidates += 1
        unique_reasons = sorted(set(reasons))
        if unique_reasons:
            self.total_rejected += 1
            for reason in unique_reasons:
                self.by_reason[reason] += 1
        bucket = determine_time_of_day_bucket(timestamp)
        if bucket is not None:
            self.total_candidates_by_tod[bucket] += 1
            if unique_reasons:
                tod_counter = self.by_time_of_day[bucket]
                for reason in unique_reasons:
                    tod_counter[reason] += 1
        if timestamp is not None:
            date_key = timestamp.astimezone(timezone.utc).date().isoformat()
            if bucket is not None:
                self.daily_candidates[(date_key, bucket)] += 1
                if unique_reasons:
                    self.daily_rejections[(date_key, bucket)] += 1


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def resolve_public_root(root: Path) -> Path:
    """Return the directory that contains public outputs."""

    resolved_root = root.resolve()
    candidate = resolved_root / "public"
    if candidate.exists() and candidate.is_dir():
        return candidate
    if resolved_root.name == "public":
        return resolved_root
    if resolved_root.name == "debug" and (resolved_root / "entry_gates").is_dir():
        return resolved_root.parent.resolve()
    if resolved_root.name == "entry_gates":
        return resolved_root.parent.parent.resolve()
    if (resolved_root / "debug").is_dir():
        return resolved_root
    return resolved_root


def discover_entry_gate_logs(root: Path, limit: int) -> List[Path]:
    """Return recent entry gate log files if the dedicated directory exists."""

    public_root = resolve_public_root(root)
    logs_dir = (public_root / ENTRY_GATE_LOG_RELATIVE_DIR).resolve()
    candidates: List[Tuple[float, Path]] = []
    try:
        if logs_dir.exists():
            for path in logs_dir.iterdir():
                if path.suffix.lower() != ".jsonl":
                    continue
                if not path.name.startswith("entry_gates_"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                candidates.append((stat.st_mtime, path))
    except OSError:
        return []

    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return [path for _, path in candidates[:limit]]

  
def discover_recent_json_files(root: Path, limit: int) -> List[Path]:
    """Return up to ``limit`` JSON/JSONL files ordered by descending mtime.

    The discovery prefers ``public`` / ``public/debug`` if those directories
    exist under *root*, but will fall back to the entire tree when necessary.
    """

    candidates: Dict[Path, float] = {}

    search_roots: List[Path] = []
    public_root = resolve_public_root(root)
    for candidate in (
        public_root / "debug",
        public_root,
        root.resolve(),
    ):
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir() and candidate not in search_roots:
            search_roots.append(candidate)

    for search_root in search_roots:
        for path in iterate_json_files(search_root):
            try:
                stat = path.stat()
            except OSError:
                continue
            recorded_mtime = candidates.get(path)
            if recorded_mtime is None or stat.st_mtime > recorded_mtime:
                candidates[path] = stat.st_mtime

    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in sorted_candidates[:limit]]


def iterate_json_files(root: Path) -> Iterator[Path]:
    """Yield JSON files under *root*.

    We avoid ``Path.rglob`` so callers can decide how many directories to
    traverse before enforcing limits.  ``os.walk`` keeps the implementation in
    the standard library and provides predictable ordering.
    """

    for current_root, _, files in os_walk(root):
        current_path = Path(current_root)
        for name in files:
            suffix = Path(name).suffix.lower()
            if suffix in JSON_EXTENSIONS:
                yield current_path / name


def os_walk(root: Path) -> Iterator[Tuple[str, List[str], List[str]]]:
    """Wrapper around ``os.walk`` so type-checkers remain happy."""

    from os import walk

    for current_root, dirnames, filenames in walk(root):
        dirnames.sort()
        filenames.sort()
        yield current_root, dirnames, filenames


# ---------------------------------------------------------------------------
# JSON loading and normalisation
# ---------------------------------------------------------------------------

def load_json_documents(path: Path) -> Iterable[Any]:
    """Load JSON payloads from *path*.

    If the file contains JSON Lines we fall back to parsing it line by line.
    Malformed lines are skipped silently so that one broken entry does not
    prevent processing the rest of the file.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return []

    if not content.strip():
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        documents: List[Any] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return documents
    else:
        return [data]


def extract_candidate_records(document: Any) -> Iterable[Tuple[str, Optional[datetime], List[str]]]:
    """Yield ``(symbol, timestamp, reasons)`` tuples from *document*.

    The traversal keeps track of the most recent symbol/timestamp context while
    walking the structure.  A mapping qualifies as a candidate if it exposes a
    recognisable entry-check or rejection payload.
    """

    def walk(node: Any, context: Dict[str, Any]) -> Iterator[Tuple[str, Optional[datetime], List[str]]]:
        if isinstance(node, Mapping):
            next_context = dict(context)

            symbol = extract_symbol(node)
            if symbol:
                next_context["symbol"] = symbol

            ts = extract_timestamp(node)
            if ts is not None:
                next_context["timestamp"] = ts

            reasons, candidate_detected = collect_reasons(node)
            if candidate_detected:
                yield (
                    next_context.get("symbol", "UNKNOWN"),
                    next_context.get("timestamp"),
                    reasons,
                )

            for value in node.values():
                yield from walk(value, next_context)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                yield from walk(item, context)

    yield from walk(document, {})


def extract_symbol(candidate: Mapping[str, Any]) -> Optional[str]:
    for key in SYMBOL_KEYS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def extract_timestamp(candidate: Mapping[str, Any]) -> Optional[datetime]:
    for key in TIMESTAMP_KEYS:
        if key not in candidate:
            continue
        value = candidate.get(key)
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try ISO-8601 (supporting trailing "Z").
        try:
            if text.endswith("Z") and "+" not in text:
                return datetime.fromisoformat(text[:-1]).replace(tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            # Fallback: treat as integer epoch seconds if possible.
            try:
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                return None
    return None


def collect_reasons(candidate: Mapping[str, Any]) -> Tuple[List[str], bool]:
    """Return a list of rejection reasons and whether the mapping is a candidate."""

    reasons: List[str] = []
    candidate_detected = False

    for key in REASON_KEYS_DIRECT:
        if key in candidate:
            candidate_detected = True
            reasons.extend(normalize_reason_container(candidate.get(key)))

    for key in REASON_KEYS_NESTED:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            nested_reasons = []
            # The nested mapping may itself contain "missing" or similar fields.
            nested_reasons.extend(normalize_reason_container(value.get("missing")))
            nested_reasons.extend(normalize_reason_container(value.get("rejections")))
            nested_reasons.extend(normalize_reason_container(value.get("reasons")))

            # Some pipelines store an "entry_checks" style mapping under this key.
            checks = value.get("checks") if isinstance(value, Mapping) else None
            if isinstance(checks, Mapping):
                nested_reasons.extend(reasons_from_checks_mapping(checks))

            if nested_reasons:
                candidate_detected = True
                reasons.extend(nested_reasons)
            elif key == "entry_checks":
                # Even if every check passed we still consider it a candidate.
                candidate_detected = True

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidate_detected = True
            for item in value:
                reasons.extend(normalize_reason_container(item))

    checks_mapping = candidate.get("entry_checks")
    if isinstance(checks_mapping, Mapping):
        candidate_detected = True
        reasons.extend(reasons_from_checks_mapping(checks_mapping))

    return reasons, candidate_detected


def reasons_from_checks_mapping(checks: Mapping[str, Any]) -> List[str]:
    failures: List[str] = []
    for name, result in checks.items():
        reason_name = normalize_reason_name(name)
        if reason_name is None:
            continue
        if is_failure(result):
            failures.append(reason_name)
    return failures


def is_failure(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, Mapping):
        for key in ("passed", "ok", "success", "result"):
            if key in value:
                nested_value = value.get(key)
                if isinstance(nested_value, bool):
                    return not nested_value
        status = value.get("status")
        if isinstance(status, str):
            lowered = status.lower()
            if lowered in {"fail", "failed", "blocked", "error"}:
                return True
    if isinstance(value, str):
        lowered = value.lower()
        return lowered in {"fail", "failed", "blocked", "rejected", "false"}
    return False


def normalize_reason_container(container: Any) -> List[str]:
    if container is None:
        return []
    if isinstance(container, Mapping):
        results: List[str] = []
        for key, value in container.items():
            reason_name = normalize_reason_name(key)
            if reason_name is None:
                continue
            if isinstance(value, bool):
                if value:
                    results.append(reason_name)
            elif isinstance(value, Mapping):
                if is_failure(value):
                    results.append(reason_name)
            else:
                # Any truthy value counts as the reason being active.
                if value:
                    results.append(reason_name)
        return results
    if isinstance(container, (list, tuple, set)):
        results: List[str] = []
        for item in container:
            if isinstance(item, Mapping):
                # Common shapes: {"reason": "p_score_too_low"}
                potential = item.get("reason") or item.get("name") or item.get("id")
                if isinstance(potential, str) and potential:
                    results.append(normalize_reason_name(potential) or potential)
                    continue
            results.extend(normalize_reason_container(item))
        return results
    if isinstance(container, str):
        reason = normalize_reason_name(container)
        return [reason] if reason else []
    return []


def normalize_reason_name(name: Any) -> Optional[str]:
    if isinstance(name, str):
        cleaned = name.strip()
        if cleaned:
            return cleaned.lower()
    return None


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------

def determine_time_of_day_bucket(timestamp: Optional[datetime]) -> Optional[str]:
    if timestamp is None:
        return None
    ts_utc = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    minute_of_day = ts_utc.hour * 60 + ts_utc.minute
    for bucket_name, minute_range in TOD_BUCKETS:
        if minute_of_day in minute_range:
            return bucket_name
    return None


def accumulate_statistics(files: Sequence[Path]) -> Dict[str, SymbolStats]:
    stats: Dict[str, SymbolStats] = defaultdict(SymbolStats)
    for path in files:
        for document in load_json_documents(path):
            for symbol, timestamp, reasons in extract_candidate_records(document):
                stats[symbol].register(reasons, timestamp)
    return stats


def accumulate_statistics_from_entry_gate_logs(files: Sequence[Path]) -> Dict[str, SymbolStats]:
    stats: Dict[str, SymbolStats] = defaultdict(SymbolStats)
    for path in files:
        for document in load_json_documents(path):
            if not isinstance(document, Mapping):
                continue
            symbol_value = document.get("symbol") or "UNKNOWN"
            symbol = str(symbol_value).strip().upper() or "UNKNOWN"
            timestamp = parse_timestamp(document.get("timestamp"))
            reasons = normalize_reason_container(document.get("reasons"))
            stats[symbol].register(reasons, timestamp)
    return stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_human_readable_summary(stats: Mapping[str, SymbolStats]) -> None:
    if not stats:
        print("No candidate data found.")
        return

    print("=== Entry gate statistics ===\n")
    for symbol in sorted(stats):
        record = stats[symbol]
        print(f"Symbol: {symbol}")
        print(f"  total_candidates: {record.total_candidates}")
        print(f"  total_rejected: {record.total_rejected}")
        print("  by_reason:")
        if record.by_reason:
            for reason, count in sorted(record.by_reason.items(), key=lambda item: (-item[1], item[0])):
                print(f"    {reason}: {count}")
        else:
            print("    (none)")
        if record.by_time_of_day:
            print("  by_time_of_day:")
            for bucket in (name for name, _ in TOD_BUCKETS):
                if bucket not in record.by_time_of_day and bucket not in record.total_candidates_by_tod:
                    continue
                bucket_counter = record.by_time_of_day.get(bucket)
                total_in_bucket = record.total_candidates_by_tod.get(bucket, 0)
                if bucket_counter or total_in_bucket:
                    print(f"    {bucket}:")
                    if total_in_bucket:
                        print(f"      total_candidates: {total_in_bucket}")
                    if bucket_counter:
                        for reason, count in sorted(bucket_counter.items(), key=lambda item: (-item[1], item[0])):
                            print(f"      {reason}: {count}")
                    else:
                        print("      (no rejections)")
        print()


def build_json_summary(stats: Mapping[str, SymbolStats]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for symbol in sorted(stats):
        record = stats[symbol]
        symbol_payload: Dict[str, Any] = {
            "total_candidates": record.total_candidates,
            "total_rejected": record.total_rejected,
            "by_reason": {
                reason: record.by_reason[reason]
                for reason in sorted(record.by_reason)
            },
        }
        by_tod: Dict[str, Any] = {}
        for bucket in (name for name, _ in TOD_BUCKETS):
            reason_counts = record.by_time_of_day.get(bucket)
            total_candidates_bucket = record.total_candidates_by_tod.get(bucket, 0)
            if not reason_counts and total_candidates_bucket == 0:
                continue
            bucket_payload: Dict[str, Any] = {}
            if reason_counts:
                bucket_payload = {
                    reason: reason_counts[reason]
                    for reason in sorted(reason_counts)
                }
            if total_candidates_bucket:
                bucket_payload = {
                    **bucket_payload,
                    "total_candidates": total_candidates_bucket,
                }
            by_tod[bucket] = bucket_payload
        if by_tod:
            symbol_payload["by_time_of_day"] = by_tod
        summary[symbol] = symbol_payload
    return summary


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_summary_deltas(
    summary: Dict[str, Any], previous: Mapping[str, Any]
) -> Dict[str, Any]:
    for symbol, payload in summary.items():
        if not isinstance(payload, dict):
            continue
        prev_payload = previous.get(symbol, {}) if isinstance(previous, Mapping) else {}
        prev_candidates = _coerce_int(
            prev_payload.get("total_candidates") if isinstance(prev_payload, Mapping) else None
        )
        prev_rejected = _coerce_int(
            prev_payload.get("total_rejected") if isinstance(prev_payload, Mapping) else None
        )
        current_candidates = _coerce_int(payload.get("total_candidates"))
        current_rejected = _coerce_int(payload.get("total_rejected"))
        payload["total_candidates_delta"] = (
            None
            if prev_candidates is None or current_candidates is None
            else current_candidates - prev_candidates
        )
        payload["total_rejected_delta"] = (
            None
            if prev_rejected is None or current_rejected is None
            else current_rejected - prev_rejected
        )
    return summary


def build_daily_visualization(stats: Mapping[str, SymbolStats]) -> Dict[str, Any]:
    daily: Dict[str, Dict[str, Dict[str, int]]] = {}
    for record in stats.values():
        for (date_key, bucket), count in record.daily_candidates.items():
            bucket_payload = daily.setdefault(date_key, {}).setdefault(
                bucket, {"total_candidates": 0, "total_rejected": 0}
            )
            bucket_payload["total_candidates"] += count
        for (date_key, bucket), count in record.daily_rejections.items():
            bucket_payload = daily.setdefault(date_key, {}).setdefault(
                bucket, {"total_candidates": 0, "total_rejected": 0}
            )
            bucket_payload["total_rejected"] += count

    normalized: Dict[str, Any] = {}
    for date_key in sorted(daily):
        buckets = daily[date_key]
        normalized[date_key] = {
            bucket: buckets[bucket] for bucket in sorted(buckets)
        }
    return normalized


def write_json_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_existing_summary(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, Mapping) else {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse entry gate rejections across recent pipeline outputs.",
    )
    parser.add_argument(
        "--root",
        default=".",
        type=Path,
        help="Root directory to scan for JSON files (default: current directory).",
    )
    parser.add_argument(
        "--limit-runs",
        "--max-files",
        dest="limit_runs",
        default=20,
        type=int,
        help="Maximum number of recent JSON files to analyse (default: 20).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="If provided, write a machine-readable summary to this path.",
    )
    parser.add_argument(
        "--output-daily-json",
        type=Path,
        help="Optional path for daily bucketised stats (default: public/monitoring/entry_gate_daily.json)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root: Path = args.root.expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    limit_runs: int = max(int(args.limit_runs), 1)

    public_root = resolve_public_root(root)
    logs_dir = (public_root / ENTRY_GATE_LOG_RELATIVE_DIR).resolve()
    entry_gate_files = discover_entry_gate_logs(root, limit_runs)
    fallback_files: List[Path] = []
    if entry_gate_files:
        print(
            f"Processing {len(entry_gate_files)} entry gate log file(s) from {logs_dir}",
            file=sys.stderr,
        )
    else:
        print(
            f"No dedicated entry gate log files found under {logs_dir}; searching recent JSON files instead.",
            file=sys.stderr,
        )
    if entry_gate_files:
        stats = accumulate_statistics_from_entry_gate_logs(entry_gate_files)
    else:
        fallback_files = discover_recent_json_files(root, limit_runs)
        if fallback_files:
            print(
                f"Analysing {len(fallback_files)} JSON file(s) discovered during fallback scan.",
                file=sys.stderr,
            )
            stats = accumulate_statistics(fallback_files)
        else:
            stats = {}

    if not entry_gate_files and not fallback_files:
        print("No JSON files found for analysis.", file=sys.stderr)

    print_human_readable_summary(stats)

    output_json: Path
    if args.output_json is not None:
        output_json = args.output_json.expanduser()
        if not output_json.is_absolute():
            output_json = (Path.cwd() / output_json).resolve()
    else:
        output_json = (public_root / ENTRY_GATE_STATS_RELATIVE_PATH).resolve()

    if args.output_daily_json is not None:
        output_daily = args.output_daily_json.expanduser()
        if not output_daily.is_absolute():
            output_daily = (Path.cwd() / output_daily).resolve()
    else:
        output_daily = (public_root / ENTRY_GATE_DAILY_RELATIVE_PATH).resolve()

    previous_summary = load_existing_summary(output_json)  
    summary_payload = build_json_summary(stats)
    summary_payload = apply_summary_deltas(summary_payload, previous_summary)
    write_json_summary(output_json, summary_payload)
    daily_payload = build_daily_visualization(stats)
    write_json_summary(output_daily, daily_payload)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
