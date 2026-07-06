#!/usr/bin/env python3
"""Save and append-restore public append-only logs across artifact refreshes."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

APPEND_PRESERVE_PATTERNS = (
    "journal/trade_journal.csv",
    "debug/entry_gates/*.jsonl",
    "debug/entry_gate_gap_log.jsonl",
    "monitoring/webhook_delivery.jsonl",
)


def _iter_paths(public_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in APPEND_PRESERVE_PATTERNS:
        paths.extend(path for path in public_dir.glob(pattern) if path.is_file())
    return sorted(set(paths))


def save(public_dir: Path, state_dir: Path) -> int:
    count = 0
    for path in _iter_paths(public_dir):
        rel = path.relative_to(public_dir)
        target = state_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _read_jsonl_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_jsonl(saved: Path, current: Path) -> None:
    rows: list[str] = []
    seen: set[str] = set()
    for row in _read_jsonl_rows(current) + _read_jsonl_rows(saved):
        if row not in seen:
            seen.add(row)
            rows.append(row)
    if not rows:
        return
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _merge_csv(saved: Path, current: Path) -> None:
    current_fields, current_rows = _read_csv(current)
    saved_fields, saved_rows = _read_csv(saved)
    fields = current_fields or saved_fields
    for field in saved_fields:
        if field not in fields:
            fields.append(field)
    if not fields:
        return
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in current_rows + saved_rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    current.parent.mkdir(parents=True, exist_ok=True)
    with current.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def restore(public_dir: Path, state_dir: Path) -> int:
    count = 0
    for saved in _iter_paths(state_dir):
        rel = saved.relative_to(state_dir)
        current = public_dir / rel
        if saved.suffix == ".csv":
            _merge_csv(saved, current)
        else:
            _merge_jsonl(saved, current)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("save", "restore"))
    parser.add_argument("--public-dir", type=Path, default=Path("public"))
    parser.add_argument("--state-dir", type=Path, default=Path("/tmp/public-notify-state/public"))
    args = parser.parse_args()
    count = save(args.public_dir, args.state_dir) if args.mode == "save" else restore(args.public_dir, args.state_dir)
    print(f"{args.mode}d {count} append-preserved public files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
