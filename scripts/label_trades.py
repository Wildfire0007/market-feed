#!/usr/bin/env python3
"""Label trade outcomes based on stored OHLC data.

This utility inspects the trade journal exported by ``analysis.py`` and uses the
per-asset one minute OHLC caches to determine whether each trade would have hit
its take profit or stop loss first.  The resulting outcome can be written back
into the journal for auditing purposes and an optional labelled dataset (using
``ml_model.MODEL_FEATURES``) can be produced for supervised learning.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from ml_model import FEATURE_LOG_DIR, MODEL_FEATURES, PUBLIC_DIR  # noqa: E402

DEFAULT_HORIZON_MINUTES = 12 * 60  # 12 hours
OUTCOME_PROFIT = "tp_hit"
OUTCOME_STOP = "stopped"
OUTCOME_NO_EXIT = "no_exit"
OUTCOME_NO_FILL = "no_fill"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_MISSING = "missing_levels"
OUTCOME_DATA_GAP = "insufficient_data"


@dataclass
class TradeResult:
    """Container with the derived outcome for a single trade."""

    asset: str
    journal_id: str
    analysis_timestamp: str
    outcome: str
    label: Optional[int]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_r_multiple: Optional[float]
    max_favorable_rr: Optional[float]
    max_adverse_rr: Optional[float]
    fill_timestamp: Optional[pd.Timestamp]
    exit_timestamp: Optional[pd.Timestamp]
    time_to_outcome_minutes: Optional[float]
    feature_row_index: Optional[int] = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate trade labelling")
    parser.add_argument(
        "--asset",
        action="append",
        help="Limit processing to specific assets (repeatable)",
    )
    parser.add_argument(
        "--journal",
        default=str(PUBLIC_DIR / "journal" / "trade_journal.csv"),
        help="Path to the trade journal CSV (default: public/journal/trade_journal.csv)",
    )
    parser.add_argument(
        "--price-root",
        default=str(PUBLIC_DIR),
        help="Directory containing per-asset price caches (default: public)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FEATURE_LOG_DIR),
        help="Where to store the labelled datasets (default: public/ml_features)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_MINUTES,
        help="Evaluation window in minutes (default: 720)",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column in the exported dataset",
    )
    parser.add_argument(
        "--update-journal",
        action="store_true",
        help="Persist validation_* columns back into the journal CSV",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum number of trades required to emit a labelled dataset",
    )
    return parser.parse_args(argv)


def _load_price_history(asset: str, price_root: Path) -> pd.DataFrame:
    price_path = price_root / asset / "klines_1m.json"
    if not price_path.exists():
        raise FileNotFoundError(f"Missing minute OHLC cache for {asset}: {price_path}")
    try:
        with price_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {price_path}") from exc
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    frame = pd.DataFrame(values)
    if "datetime" not in frame.columns:
        raise ValueError(f"Minute cache missing 'datetime' column: {price_path}")
    frame["timestamp"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for field in ("open", "high", "low", "close"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "open", "high", "low", "close"]]


def _first_valid(values: Iterable[float | int | str | None]) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _evaluate_trade(
    trade_row: pd.Series,
    price_frame: pd.DataFrame,
    horizon_minutes: int,
) -> TradeResult:
    asset = str(trade_row.get("asset", "")).upper()
    journal_id = str(trade_row.get("journal_id", ""))
    analysis_timestamp_raw = trade_row.get("analysis_timestamp")
    try:
        analysis_timestamp = pd.to_datetime(analysis_timestamp_raw, utc=True)
    except (TypeError, ValueError):
        analysis_timestamp = None
    if analysis_timestamp is None or analysis_timestamp.tzinfo is None:
        raise ValueError(f"Invalid analysis timestamp for journal entry {journal_id}")

    entry_price = _first_valid([trade_row.get("entry_price"), trade_row.get("spot_price")])
    stop_loss = _first_valid([trade_row.get("stop_loss")])
    take_profit = _first_valid([trade_row.get("take_profit_1"), trade_row.get("take_profit_2")])

    mode = str(trade_row.get("signal", "")).lower()
    if mode not in {"buy", "sell"}:
        raise ValueError("evaluate_trade called on a non-executable signal")

    if entry_price is None or stop_loss is None or take_profit is None:
        return TradeResult(
            asset=asset,
            journal_id=journal_id,
            analysis_timestamp=str(analysis_timestamp_raw),
            outcome=OUTCOME_MISSING,
            label=None,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_r_multiple=None,
            max_favorable_rr=None,
            max_adverse_rr=None,
            fill_timestamp=None,
            exit_timestamp=None,
            time_to_outcome_minutes=None,
        )

    horizon = analysis_timestamp + pd.Timedelta(minutes=int(max(horizon_minutes, 1)))
    window_start = analysis_timestamp.floor("min")
    price_slice = price_frame[(price_frame["timestamp"] >= window_start) & (price_frame["timestamp"] <= horizon)]

    if price_slice.empty:
        return TradeResult(
            asset=asset,
            journal_id=journal_id,
            analysis_timestamp=str(analysis_timestamp_raw),
            outcome=OUTCOME_DATA_GAP,
            label=None,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_r_multiple=None,
            max_favorable_rr=None,
            max_adverse_rr=None,
            fill_timestamp=None,
            exit_timestamp=None,
            time_to_outcome_minutes=None,
        )

    direction = 1 if mode == "buy" else -1
    risk = entry_price - stop_loss if direction == 1 else stop_loss - entry_price
    if not np.isfinite(risk) or risk <= 0:
        return TradeResult(
            asset=asset,
            journal_id=journal_id,
            analysis_timestamp=str(analysis_timestamp_raw),
            outcome=OUTCOME_MISSING,
            label=None,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_r_multiple=None,
            max_favorable_rr=None,
            max_adverse_rr=None,
            fill_timestamp=None,
            exit_timestamp=None,
            time_to_outcome_minutes=None,
        )

    filled = False
    fill_timestamp: Optional[pd.Timestamp] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    outcome = OUTCOME_NO_EXIT
    label: Optional[int] = None
    exit_price: Optional[float] = None
    max_favorable_rr = 0.0
    max_adverse_rr = 0.0
    best_price = entry_price
    worst_price = entry_price

    for _, row in price_slice.iterrows():
        bar_time = row["timestamp"]
        high = float(row["high"])
        low = float(row["low"])

        if not filled:
            filled = True
            fill_timestamp = bar_time

        if direction == 1:
            best_price = max(best_price, high)
            worst_price = min(worst_price, low)
            hit_tp = high >= take_profit
            hit_sl = low <= stop_loss
            max_favorable_rr = max(max_favorable_rr, (best_price - entry_price) / risk)
            max_adverse_rr = max(max_adverse_rr, (entry_price - worst_price) / risk)
        else:
            best_price = min(best_price, low)
            worst_price = max(worst_price, high)
            hit_tp = low <= take_profit
            hit_sl = high >= stop_loss
            max_favorable_rr = max(max_favorable_rr, (entry_price - best_price) / risk)
            max_adverse_rr = max(max_adverse_rr, (worst_price - entry_price) / risk)

        if hit_tp and hit_sl:
            outcome = OUTCOME_AMBIGUOUS
            exit_timestamp = bar_time
            label = None
            exit_price = None
            break
        if hit_tp:
            outcome = OUTCOME_PROFIT
            label = 1
            exit_timestamp = bar_time
            exit_price = take_profit
            break
        if hit_sl:
            outcome = OUTCOME_STOP
            label = 0
            exit_timestamp = bar_time
            exit_price = stop_loss
            break

    time_to_outcome: Optional[float]
    if fill_timestamp is not None and exit_timestamp is not None:
        delta = exit_timestamp - fill_timestamp
        time_to_outcome = delta.total_seconds() / 60.0
    else:
        time_to_outcome = None

    risk_r_multiple: Optional[float] = None
    if exit_price is not None:
        if direction == 1:
            risk_r_multiple = (exit_price - entry_price) / risk
        else:
            risk_r_multiple = (entry_price - exit_price) / risk

    return TradeResult(
        asset=asset,
        journal_id=journal_id,
        analysis_timestamp=str(analysis_timestamp_raw),
        outcome=outcome,
        label=label,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_r_multiple=risk_r_multiple,
        max_favorable_rr=max_favorable_rr if filled else None,
        max_adverse_rr=max_adverse_rr if filled else None,
        fill_timestamp=fill_timestamp,
        exit_timestamp=exit_timestamp,
        time_to_outcome_minutes=time_to_outcome,
    )


def _load_feature_snapshots(asset: str) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    feature_path = FEATURE_LOG_DIR / f"{asset}_features.csv"
    if not feature_path.exists():
        return pd.DataFrame(columns=MODEL_FEATURES + ["analysis_timestamp"]), []

    rows: List[List[float]] = []
    with feature_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # header (may be stale but skip it regardless)
        except StopIteration:
            return pd.DataFrame(columns=MODEL_FEATURES + ["analysis_timestamp"]), []
        for raw_row in reader:
            if not raw_row:
                continue
            vector: List[float] = []
            for idx in range(len(MODEL_FEATURES)):
                token = raw_row[idx] if idx < len(raw_row) else ""
                try:
                    vector.append(float(token))
                except (TypeError, ValueError):
                    vector.append(0.0)
            rows.append(vector)
    feature_df = pd.DataFrame(rows, columns=MODEL_FEATURES)

    metadata: List[Dict[str, object]] = []
    meta_path = feature_path.with_suffix(feature_path.suffix + ".meta.jsonl")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    metadata.append({})
                    continue
                meta = entry.get("metadata")
                metadata.append(meta if isinstance(meta, dict) else {})
    if metadata and len(metadata) != len(feature_df):
        if len(metadata) > len(feature_df):
            metadata = metadata[: len(feature_df)]
        else:
            metadata.extend({} for _ in range(len(feature_df) - len(metadata)))
    elif not metadata:
        metadata = [{} for _ in range(len(feature_df))]

    feature_df["analysis_timestamp"] = [
        meta.get("analysis_timestamp") if isinstance(meta, dict) else None for meta in metadata
    ]
    return feature_df, metadata


def _attach_feature_indices(
    trade_results: List[TradeResult],
    feature_df: pd.DataFrame,
) -> None:
    if feature_df.empty:
        return
    timestamp_map: Dict[str, int] = {}
    for idx, ts in feature_df["analysis_timestamp"].items():
        if isinstance(ts, str):
            timestamp_map.setdefault(ts, idx)
    for result in trade_results:
        if result.analysis_timestamp in timestamp_map:
            result.feature_row_index = timestamp_map[result.analysis_timestamp]


def _build_labelled_dataset(
    trade_results: List[TradeResult],
    feature_df: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for result in trade_results:
        if result.label is None:
            continue
        if result.feature_row_index is None:
            continue
        feature_row = feature_df.loc[result.feature_row_index, MODEL_FEATURES]
        payload: Dict[str, object] = dict(feature_row)
        payload[label_column] = int(result.label)
        payload["asset"] = result.asset
        payload["journal_id"] = result.journal_id
        payload["analysis_timestamp"] = result.analysis_timestamp
        payload["outcome"] = result.outcome
        rows.append(payload)
    return pd.DataFrame(rows)


def _update_journal(
    journal_path: Path,
    journal_df: pd.DataFrame,
    trade_results: List[TradeResult],
) -> None:
    updated = False
    for result in trade_results:
        mask = journal_df["journal_id"] == result.journal_id
        if not mask.any():
            continue
        idx = mask.idxmax()
        journal_df.at[idx, "validation_outcome"] = result.outcome
        journal_df.at[idx, "validation_rr"] = result.risk_r_multiple
        journal_df.at[idx, "max_favorable_excursion"] = result.max_favorable_rr
        journal_df.at[idx, "max_adverse_excursion"] = result.max_adverse_rr
        journal_df.at[idx, "time_to_outcome_minutes"] = result.time_to_outcome_minutes
        updated = True
    if updated:
        journal_df.to_csv(journal_path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    journal_path = Path(args.journal)
    if not journal_path.exists():
        raise FileNotFoundError(f"Trade journal not found: {journal_path}")
    journal_df = pd.read_csv(journal_path)

    executable = journal_df[journal_df["signal"].isin(["buy", "sell"])]
    if executable.empty:
        print("No executable trades found in journal")
        return 0

    assets = (
        [asset.upper() for asset in args.asset]
        if args.asset
        else sorted(set(executable["asset"].dropna().str.upper()))
    )
    price_root = Path(args.price_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    labelled_datasets: Dict[str, pd.DataFrame] = {}

    for asset in assets:
        asset_trades = executable[executable["asset"].str.upper() == asset]
        if asset_trades.empty:
            continue
        try:
            price_frame = _load_price_history(asset, price_root)
        except FileNotFoundError as exc:
            print(exc)
            continue
        except ValueError as exc:
            print(exc)
            continue

        trade_results: List[TradeResult] = []
        for _, row in asset_trades.iterrows():
            try:
                result = _evaluate_trade(row, price_frame, args.horizon)
            except ValueError as exc:
                print(f"Skipping journal entry due to error: {exc}")
                continue
            trade_results.append(result)
            summary.setdefault(asset, {}).setdefault(result.outcome, 0)
            summary[asset][result.outcome] += 1

        feature_df, _ = _load_feature_snapshots(asset)
        _attach_feature_indices(trade_results, feature_df)
        labelled_df = _build_labelled_dataset(trade_results, feature_df, args.label_column)
        if not labelled_df.empty and len(labelled_df) >= args.min_trades:
            labelled_path = output_dir / f"{asset}_labelled.csv"
            labelled_df.to_csv(labelled_path, index=False)
            labelled_datasets[asset] = labelled_df
            print(f"Wrote {len(labelled_df)} labelled rows to {labelled_path}")
        else:
            print(f"No sufficient labelled trades for {asset}; skipping dataset export")

        if args.update_journal:
            _update_journal(journal_path, journal_df, trade_results)

    if summary:
        print("\nOutcome summary:")
        for asset, counts in summary.items():
            breakdown = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            print(f"  {asset}: {breakdown}")
    else:
        print("No trades processed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
