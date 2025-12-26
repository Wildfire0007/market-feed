"""Synchronize pipeline outputs into the public directory.

This script copies selected source directories into the configured public
output folder and generates a small metadata file containing the run
identifier, commit, and a checksum of the synchronized contents. The
metadata is later used in CI to assert the public folder was refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["data", "reports"],
        help="Source directories to copy into the public output",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("OUT_DIR", "public"),
        help="Destination public directory (default: OUT_DIR or ./public)",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", "manual-run"),
        help="Identifier to record for the synchronization",
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("GITHUB_SHA", "unknown"),
        help="Commit hash associated with the sync",
    )
    return parser.parse_args()


def copy_sources(sources: Iterable[str], target_root: Path) -> List[str]:
    """Copy source directories into the target root.

    Returns a list of sources that were actually copied.
    """

    copied: List[str] = []
    target_root.mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_path = Path(source).resolve()
        if not source_path.exists() or not source_path.is_dir():
            continue

        destination = target_root / source_path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        copied.append(str(source_path))

    return copied


def compute_sha256(data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def preserve_files(target_root: Path, filenames: Iterable[str]) -> Dict[str, bytes]:
    preserved: Dict[str, bytes] = {}
    for filename in filenames:
        path = target_root / filename
        if not path.exists() or not path.is_file():
            continue

        data = path.read_bytes()
        preserved[filename] = data
        print(
            "Preserve",
            filename,
            f"size={len(data)}",
            f"sha256={compute_sha256(data)}",
        )

    return preserved


def restore_files(target_root: Path, preserved: Dict[str, bytes]) -> None:
    for filename, data in preserved.items():
        path = target_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(
            "Restore",
            filename,
            f"size={len(data)}",
            f"sha256={compute_sha256(data)}",
        )


def directory_checksum(directory: Path) -> str:
    """Compute a deterministic checksum of all files within a directory."""

    hasher = hashlib.sha256()
    if not directory.exists():
        return hasher.hexdigest()

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative_path = path.relative_to(directory).as_posix()
            hasher.update(relative_path.encode())
            hasher.update(str(path.stat().st_size).encode())
            with path.open("rb") as handle:
                while chunk := handle.read(8192):
                    hasher.update(chunk)
    return hasher.hexdigest()


def write_metadata(target_root: Path, run_id: str, commit: str, checksum: str, copied: List[str]) -> None:
    metadata = {
        "run_id": run_id,
        "commit": commit,
        "checksum": checksum,
        "copied_sources": copied,
    }
    metadata_path = target_root / ".sync-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    stamp_path = target_root / ".last_sync"
    stamp_path.write_text(run_id)


def main() -> int:
    args = parse_args()
    target_root = Path(args.target).resolve()

    preserved = preserve_files(
        target_root,
        (
            "_manual_positions.json",
            "_manual_positions_audit.jsonl",
            "_active_position_state.json",
        ),
    )

    copied = copy_sources(args.sources, target_root)
    if preserved:
        restore_files(target_root, preserved)

    checksum = directory_checksum(target_root)
    write_metadata(target_root, args.run_id, args.commit, checksum, copied)

    print(f"Synced sources: {copied if copied else 'none'}")
    print(f"Target: {target_root}")
    print(f"Checksum: {checksum}")
    return 0 if copied else 1


if __name__ == "__main__":
    sys.exit(main())
