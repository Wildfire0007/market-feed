"""Order-flow utilities for tick aggregation."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def _detect_side(row: pd.Series, prev_price: float) -> float:
    side_raw = str(row.get("side") or "").lower()
    if side_raw in {"b", "buy", "bid", "buyer", "1", "+1", "long"}:
        return 1.0
    if side_raw in {"s", "sell", "ask", "seller", "-1", "short"}:
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


def aggregate_tick_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    row_list: List[Dict[str, Any]] = list(rows)
    if not row_list:
        raise ValueError("no tick rows provided")
    df = pd.DataFrame(row_list)
    return aggregate_ticks(df)
