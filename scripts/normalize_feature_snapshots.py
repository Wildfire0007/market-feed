"""Utility to normalise ML feature snapshots to the canonical schema."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "public" / "ml_features" / "schema.json"


def _load_schema() -> List[str]:
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError("Schema JSON must contain a list under 'features'.")
    return [str(feature) for feature in features]


def _normalise_feature_file(path: Path, schema: List[str]) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature snapshot not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        header_map = {name: idx for idx, name in enumerate(header)}

        rows: List[List[str]] = []
        for raw_row in reader:
            if not raw_row:
                continue
            normalised_row: List[str] = []
            for feature in schema:
                idx = header_map.get(feature)
                value = raw_row[idx] if idx is not None and idx < len(raw_row) else ""
                normalised_row.append(value if value not in {"", None} else "0.0")
            rows.append(normalised_row)

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(schema)
        writer.writerows(rows)

    return rows


def _align_metadata(meta_path: Path, rows: List[List[str]], schema: List[str]) -> None:
    if not meta_path.exists():
        return

    meta_lines = [line for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in meta_lines]

    if len(records) > len(rows):
        records = records[: len(rows)]
    elif len(records) < len(rows):
        padding = len(rows) - len(records)
        records.extend({"asset": meta_path.stem.split("_")[0], "metadata": {}} for _ in range(padding))

    schema_hash = hashlib.sha256(",".join(schema).encode("utf-8")).hexdigest()[:12]

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            record = {"metadata": {}}
        metadata = record.setdefault("metadata", {})
        metadata.setdefault("analysis_timestamp", None)
        metadata["feature_row_index"] = idx
        metadata["feature_schema_columns"] = len(schema)
        metadata["feature_schema_hash"] = schema_hash
        record["asset"] = record.get("asset") or meta_path.stem.split("_")[0]

        records[idx] = record

    with meta_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def normalise_asset(asset: str, features_dir: Path) -> None:
    schema = _load_schema()
    features_path = features_dir / f"{asset.upper()}_features.csv"
    rows = _normalise_feature_file(features_path, schema)
    meta_path = features_path.with_suffix(features_path.suffix + ".meta.jsonl")
    _align_metadata(meta_path, rows, schema)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", help="Asset symbol to normalise (e.g. USOIL)")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "public" / "ml_features",
        help="Directory containing <ASSET>_features.csv exports",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    normalise_asset(args.asset, args.features_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
