#!/usr/bin/env python3
"""Aggregate tick data into order flow metrics for analysis.py.

The script expects a CSV file with at least ``timestamp``, ``price`` and
``volume`` columns.  If an additional ``side`` column is available it should
contain values like ``buy``/``sell`` or ``B``/``S`` to indicate the aggressor.
The output JSON matches the structure that ``analysis.py`` consumes via
``order_flow_ticks.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from order_flow import aggregate_ticks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. EURUSD)")
    parser.add_argument("--input", required=True, help="Path to CSV tick file")
    parser.add_argument(
        "--output",
        help="Target JSON file (defaults to public/<ASSET>/order_flow_ticks.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if not {"timestamp", "price", "volume"}.issubset(df.columns):
        raise SystemExit("input CSV must contain timestamp, price and volume columns")

    metrics = aggregate_ticks(df)
    asset_dir = Path("public") / args.asset.upper()
    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else asset_dir / "order_flow_ticks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"wrote {output_path} ({metrics['ticks']} ticks → window {metrics['window_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
