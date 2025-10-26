"""Generate binary trade labels for machine learning datasets.

This utility expands on the initial proof-of-concept implementation by sourcing
OHLC data directly from the cached klines that ``analysis.py`` already
maintains.  The goal is to remove the dependency on manually curated
``public/data`` CSV files, making the workflow usable out-of-the-box on any
asset that has been analysed by the pipeline.

Supported labelling methods:

* ``realized`` – compare entry/exit prices and optional fees to realise PnL.
* ``fixed`` – forward return over a fixed horizon (bars).
* ``tbm`` – the triple barrier method using take-profit, stop-loss and a
  vertical time limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def realized_pnl_label(
    df: pd.DataFrame,
    entry_col: str,
    exit_col: str,
    side_col: str | None = None,
    qty_col: str | None = None,
    fees_cols: Iterable[str] = (),
) -> pd.Series:
    """Return 1 when the realised PnL is positive, otherwise 0."""

    entry = pd.to_numeric(df[entry_col], errors="coerce")
    exit_ = pd.to_numeric(df[exit_col], errors="coerce")

    qty: pd.Series | float
    if qty_col and qty_col in df:
        qty = pd.to_numeric(df[qty_col], errors="coerce")
    else:
        qty = 1.0
    if isinstance(qty, (int, float)):
        qty = pd.Series(qty, index=df.index)

    if side_col and side_col in df:
        side = df[side_col].astype(str).str.lower().map(
            lambda token: -1 if token in {"short", "sell", "-1"} else 1
        )
    else:
        side = 1

    if isinstance(side, pd.Series):
        is_long = side.eq(1) | side.eq(True)
    else:
        is_long = (side == 1) or (side is True)

    gross = np.where(is_long, (exit_ - entry) * qty, (entry - exit_) * qty)

    fees = sum(
        pd.to_numeric(df[column], errors="coerce").fillna(0) for column in fees_cols if column in df
    )
    pnl = pd.Series(gross, index=df.index) - (fees if isinstance(fees, pd.Series) else 0.0)

    return (pnl > 0).astype(int)


def fixed_horizon_label(price: pd.Series, horizon: int = 12, threshold: float = 0.0) -> pd.Series:
    """Label rows based on a forward return over ``horizon`` steps."""

    ret_h = price.shift(-horizon) / price - 1.0
    lab = (ret_h > threshold).astype(int)
    if horizon > 0:
        lab.iloc[-horizon:] = 0
    return lab


def triple_barrier_labels(
    price: pd.Series, events_idx: Iterable[int], pt: float = 0.01, sl: float = 0.01, max_h: int = 48
) -> pd.Series:
    """Return labels following the triple-barrier method."""

    labels = pd.Series(0, index=list(events_idx), dtype=int)

    for idx in labels.index:
        if idx >= len(price):
            continue

        entry = price.iloc[idx]
        up = entry * (1 + pt)
        down = entry * (1 - sl)
        end = min(idx + max_h, len(price) - 1)

        if idx + 1 > end:
            ret = price.iloc[end] / entry - 1.0 if entry else 0.0
            labels.loc[idx] = 1 if ret > 0 else 0
            continue

        window = price.iloc[idx + 1 : end + 1]
        hit_up = window[window >= up]
        hit_down = window[window <= down]

        if len(hit_up) and (not len(hit_down) or hit_up.index[0] < hit_down.index[0]):
            labels.loc[idx] = 1
        elif len(hit_down) and (not len(hit_up) or hit_down.index[0] < hit_up.index[0]):
            labels.loc[idx] = 0
        else:
            ret = price.iloc[end] / entry - 1.0 if entry else 0.0
            labels.loc[idx] = 1 if ret > 0 else 0

    return labels


def _load_feature_timestamps(feature_path: Path) -> List[Optional[pd.Timestamp]]:
    """Infer timestamps from the ``*.meta.jsonl`` sidecar next to the features."""

    meta_path = feature_path.with_suffix(feature_path.suffix + ".meta.jsonl")
    if not meta_path.exists():
        return []

    timestamps: List[Optional[pd.Timestamp]] = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                timestamps.append(None)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                timestamps.append(None)
                continue
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            stamp = metadata.get("analysis_timestamp") if isinstance(metadata, dict) else None
            timestamps.append(pd.to_datetime(stamp, utc=True, errors="coerce") if stamp else None)
    return timestamps


def _ensure_timestamp_column(
    frame: pd.DataFrame, timestamp_col: str, feature_path: Path
) -> tuple[pd.DataFrame, bool]:
    """Attach a timestamp column.

    Returns the updated frame and a boolean that is ``True`` when the timestamp
    had to be synthesised (i.e. no column or metadata was available).
    """

    if timestamp_col in frame.columns:
        updated = frame.copy()
        if not pd.api.types.is_datetime64_any_dtype(updated[timestamp_col]):
            updated[timestamp_col] = pd.to_datetime(updated[timestamp_col], utc=True, errors="coerce")
        return updated, False

    inferred = _load_feature_timestamps(feature_path)
    if inferred and len(inferred) >= len(frame):
        updated = frame.copy()
        updated[timestamp_col] = inferred[: len(frame)]
        return updated, False

    updated = frame.copy()
    updated[timestamp_col] = pd.RangeIndex(len(frame), name=timestamp_col)
    return updated, True


def _load_price_from_cache(asset: str, price_root: Path) -> pd.DataFrame:
    """Load OHLC data from ``public/<ASSET>/klines_1m.json``."""

    cache_path = price_root / asset / "klines_1m.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing price cache for asset '{asset}'. Expected file: {cache_path}.\n"
            "Run analysis.py or Trading.py to refresh the klines caches."
        )

    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"Invalid klines payload in {cache_path}: expected a list under 'values'.")

    if not values:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    frame = pd.DataFrame(values)
    if "datetime" not in frame.columns:
        raise ValueError(f"klines cache missing 'datetime' column: {cache_path}")

    frame["timestamp"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for field in ("open", "high", "low", "close"):
        if field not in frame.columns:
            raise ValueError(f"klines cache missing '{field}' column: {cache_path}")
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    return frame[["timestamp", "open", "high", "low", "close"]]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate binary trade labels for ML datasets")
    parser.add_argument("--features", required=True, help="Path to the feature CSV export")
    parser.add_argument("--method", choices=["realized", "fixed", "tbm"], required=True)
    parser.add_argument(
        "--timestamp",
        default="analysis_timestamp",
        help="Column containing feature timestamps (falls back to metadata if missing)",
    )
    parser.add_argument("--price-col", default="close", help="Price column to use from the OHLC source")
    parser.add_argument("--ohlc", help="Optional explicit OHLC CSV/Parquet path")
    parser.add_argument(
        "--asset",
        help=(
            "Asset symbol used to locate cached OHLC data under --price-root/<ASSET>/klines_1m.json. "
            "Required when using fixed/tbm without --ohlc."
        ),
    )
    parser.add_argument(
        "--price-root",
        default=str(Path("public")),
        help="Root directory containing per-asset klines caches (default: public)",
    )

    # realised PnL options
    parser.add_argument("--entry-col")
    parser.add_argument("--exit-col")
    parser.add_argument("--side-col")
    parser.add_argument("--qty-col")
    parser.add_argument("--fees-cols", nargs="*", default=[])

    # fixed / TBM parameters
    parser.add_argument("--horizon", type=int, default=12, help="Lookahead bars or max holding period")
    parser.add_argument("--threshold", type=float, default=0.0, help="Return threshold for fixed horizon")
    parser.add_argument("--pt", type=float, default=0.01, help="Take profit multiplier for TBM")
    parser.add_argument("--sl", type=float, default=0.01, help="Stop loss multiplier for TBM")
    parser.add_argument("--trigger-col", default="precision_trigger_fire", help="Column marking TBM events")

    parser.add_argument(
        "--output",
        help="Optional explicit output path. Defaults to <features> with '_labeled' suffix.",
    )

    args = parser.parse_args(argv)

    feature_path = Path(args.features)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_path}")

    features = pd.read_csv(feature_path)
    features, synthetic_timestamp = _ensure_timestamp_column(features, args.timestamp, feature_path)

    working = features.copy()
    price_series: Optional[pd.Series] = None

    if args.method in {"fixed", "tbm"}:
        if args.ohlc:
            price_frame = pd.read_csv(args.ohlc)
        else:
            if not args.asset:
                raise ValueError("Provide --asset or --ohlc when using fixed/tbm labelling methods")
            price_frame = _load_price_from_cache(args.asset, Path(args.price_root))

        if args.timestamp not in price_frame.columns and "timestamp" in price_frame.columns:
            price_frame = price_frame.rename(columns={"timestamp": args.timestamp})

        if args.timestamp not in price_frame.columns:
            raise ValueError(
                "Price data is missing the timestamp column. Use --timestamp to specify the expected column name."
            )

        if not pd.api.types.is_datetime64_any_dtype(price_frame[args.timestamp]):
            price_frame[args.timestamp] = pd.to_datetime(
                price_frame[args.timestamp], utc=True, errors="coerce"
            )

        if synthetic_timestamp and pd.api.types.is_datetime64_any_dtype(price_frame[args.timestamp]):
            raise ValueError(
                "A feature CSV nem tartalmaz időbélyegeket. Adj meg egy timestamp oszlopot "
                "(--timestamp) vagy biztosíts meta.jsonl fájlt az elemzési exporttal."
            )

        working = working.sort_values(args.timestamp)
        price_frame = price_frame[[args.timestamp, args.price_col]].dropna().sort_values(args.timestamp)

        merged = pd.merge_asof(
            working,
            price_frame,
            on=args.timestamp,
            direction="nearest",
            tolerance=pd.Timedelta("5min"),
        )
        price_series = pd.to_numeric(merged[args.price_col], errors="coerce").ffill()
        working = merged

    if args.method == "realized":
        if not args.entry_col or not args.exit_col:
            raise ValueError("realized method requires --entry-col and --exit-col")
        labels = realized_pnl_label(
            working,
            args.entry_col,
            args.exit_col,
            side_col=args.side_col,
            qty_col=args.qty_col,
            fees_cols=args.fees_cols,
        )
    elif args.method == "fixed":
        if price_series is None:
            raise RuntimeError("Price series not initialised for fixed method")
        labels = fixed_horizon_label(price_series, horizon=args.horizon, threshold=args.threshold)
    else:  # tbm
        if price_series is None:
            raise RuntimeError("Price series not initialised for tbm method")
        if args.trigger_col in working:
            events = working.index[working[args.trigger_col].fillna(0).astype(int) == 1]
        else:
            events = working.index
        labels = triple_barrier_labels(price_series, events, pt=args.pt, sl=args.sl, max_h=args.horizon)
        labels = labels.reindex(working.index).fillna(0).astype(int)

    working["label"] = labels.values

    output_path = (
        Path(args.output)
        if args.output
        else feature_path.with_name(f"{feature_path.stem}_labeled{feature_path.suffix}")
    )
    working.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
