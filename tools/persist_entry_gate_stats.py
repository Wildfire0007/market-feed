#!/usr/bin/env python3
"""Persist entry gate statistics for the current pipeline run.

This utility copies the aggregated ``entry_gate_stats.json`` content into a
run-specific file under ``public/debug-entry-gates`` so every workflow
execution leaves behind its own snapshot.  The script is intentionally minimal
so it can run inside GitHub Actions without extra dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-file",
        required=True,
        type=Path,
        help="Path to the source entry_gate_stats.json file",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID used to create a unique filename",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("public/debug-entry-gates"),
        type=Path,
        help="Directory where the per-run stats file will be written",
    )
    return parser.parse_args()


def ensure_within_repo(root: Path, target: Path) -> None:
    """Raise an error if *target* is outside the repository workspace."""

    try:
        target.resolve().relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(
            f"Refusing to write outside the repository workspace: {target}"
        ) from exc


def _fallback_payload(reason: str, stats_path: Path) -> Mapping[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "source": str(stats_path),
    }


def load_stats(stats_path: Path) -> Mapping[str, Any]:
    if not stats_path.is_file():
        return _fallback_payload("stats file not found", stats_path)

    try:
        data = json.loads(stats_path.read_text())
    except json.JSONDecodeError:
        return _fallback_payload("stats file is not valid JSON", stats_path)

    if not isinstance(data, Mapping):
        return _fallback_payload("stats content must be a JSON object/dictionary", stats_path)

    return data


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()

    stats = load_stats(args.stats_file)

    run_id = args.run_id.strip()
    if not run_id:
        raise SystemExit("--run-id must not be empty")

    output_dir = (repo_root / args.output_dir).resolve()
    ensure_within_repo(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"entry_gate_stats_{run_id}.json"
    ensure_within_repo(repo_root, output_path)

    output_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
