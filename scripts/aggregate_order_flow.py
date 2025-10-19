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


def _detect_side(row: pd.Series, prev_price: float) -> float:
    side_raw = str(row.get("side") or "").lower()
    if side_raw in {"b", "buy", "bid", "buyer"}:
        return 1.0
    if side_raw in {"s", "sell", "ask", "seller"}:
        return -1.0
    price = row.get("price")
    if price is None or pd.isna(price) or pd.isna(prev_price):
        return 0.0
    if price > prev_price:
        return 1.0
    if price < prev_price:
        return -1.0
    return 0.0


def aggregate_ticks(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise ValueError("input dataframe is empty")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "volume"])
    if df.empty:
        raise ValueError("no valid tick rows after cleaning")

    side = []
    prev_price = df["price"].iloc[0]
    for _, row in df.iterrows():
        s = _detect_side(row, prev_price)
        side.append(s)
        prev_price = row["price"]
    df["aggressor"] = side
    df["signed_volume"] = df["volume"] * df["aggressor"]

    buy_volume = float(df.loc[df["aggressor"] > 0, "volume"].sum())
    sell_volume = float(df.loc[df["aggressor"] < 0, "volume"].sum())
    total_volume = buy_volume + sell_volume

    imbalance = (buy_volume - sell_volume) / total_volume if total_volume else 0.0
    delta_volume = float(df["signed_volume"].sum())
    aggressor_ratio = imbalance

    first_ts = df["timestamp"].iloc[0]
    last_ts = df["timestamp"].iloc[-1]
    window_minutes = max(1.0, (last_ts - first_ts).total_seconds() / 60.0)
    price_change = float(df["price"].iloc[-1] - df["price"].iloc[0])
    pressure = price_change * total_volume / window_minutes if window_minutes else 0.0

    return {
        "window_minutes": window_minutes,
        "volume_buy": buy_volume,
        "volume_sell": sell_volume,
        "delta": delta_volume,
        "imbalance": imbalance,
        "aggressor_ratio": aggressor_ratio,
        "pressure": pressure,
        "price_change": price_change,
        "ticks": int(len(df)),
    }


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
