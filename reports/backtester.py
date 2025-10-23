"""Lightweight live validation utilities driven by public JSON exports.

The backtester previously relied on a coarse five minute OHLC feed and a
simplified fill model.  This version reuses the richer validation logic from
``scripts/label_trades.py`` so that precision workflow states, limit order fill
windows and ambiguous exits are handled consistently with the labelling
pipeline.  Whenever higher resolution price caches (for example 15 second
klines) are present they are preferred to minute data, reducing the number of
ambiguous outcomes.

If a manual fill journal is present (``public/journal/manual_fills.csv``) its
entries override the automatic evaluation, allowing a discretionary audit to be
recorded alongside the automatic labels.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from scripts import label_trades as lt

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

PRICE_CANDIDATES = (
    "klines_15s.json",
    "klines_30s.json",
    "klines_1m.json",
    "klines_5m.json",
)

MANUAL_FILL_FILENAME = "manual_fills.csv"

DEFAULT_ENTRY_GRACE_MINUTES = 60
DEFAULT_ENTRY_TOLERANCE = 0.0005

RESULT_COLUMNS = [
    "validation_outcome",
    "validation_rr",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "time_to_outcome_minutes",
    "fill_timestamp",
    "exit_timestamp",
    "entry_kind",
    "validation_source",
]

NUMERIC_RESULT_COLUMNS = {
    "validation_rr",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "time_to_outcome_minutes",
}

WIN_OUTCOMES = {lt.OUTCOME_PROFIT, "tp_hit", "tp1"}
LOSS_OUTCOMES = {lt.OUTCOME_STOP, "stopped"}
AMBIGUOUS_OUTCOMES = {lt.OUTCOME_AMBIGUOUS, "ambiguous"}
RESOLVED_OUTCOMES = WIN_OUTCOMES | LOSS_OUTCOMES | AMBIGUOUS_OUTCOMES | {"tp2"}

WIN_OUTCOMES_LOWER = {status.lower() for status in WIN_OUTCOMES}
LOSS_OUTCOMES_LOWER = {status.lower() for status in LOSS_OUTCOMES}
AMBIGUOUS_OUTCOMES_LOWER = {status.lower() for status in AMBIGUOUS_OUTCOMES}
RESOLVED_OUTCOMES_LOWER = {status.lower() for status in RESOLVED_OUTCOMES}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_price_payload(path: Path) -> Optional[pd.DataFrame]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        return None
    frame = pd.DataFrame(values)
    if "datetime" not in frame.columns:
        return None
    try:
        frame["timestamp"] = pd.to_datetime(frame["datetime"], utc=True)
    except Exception:
        return None
    numeric_cols = ["open", "high", "low", "close"]
    for column in numeric_cols:
        if column not in frame.columns:
            return None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if frame.empty:
        return None
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "open", "high", "low", "close"]]

def _load_price_history(asset: str, public_dir: Path) -> Optional[pd.DataFrame]:
    asset_dir = public_dir / asset
    for candidate in PRICE_CANDIDATES:
        path = asset_dir / candidate
        if not path.exists():
            continue
        frame = _read_price_payload(path)
        if frame is not None:
            return frame
    return None

    
def _serialise_timestamp(value: Optional[pd.Timestamp]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.isoformat()
    return str(value)

    
def _load_manual_fill_log(public_dir: Path) -> Optional[pd.DataFrame]:
    path = public_dir / "journal" / MANUAL_FILL_FILENAME
    if not path.exists():
        return None
    try:
        overrides = pd.read_csv(path)
    except Exception:
        return None
    if overrides.empty or "journal_id" not in overrides.columns:
        return None
    overrides["journal_id"] = overrides["journal_id"].astype(str)
    return overrides


def _apply_manual_overrides(results_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    if manual_df.empty:
        return results_df
    merged = results_df.merge(manual_df, on="journal_id", how="left", suffixes=("", "_manual"))
    if merged.empty:
        return merged
    override_columns: Iterable[str] = RESULT_COLUMNS
    for column in override_columns:
        manual_col = f"{column}_manual"
        if manual_col not in merged.columns:
            continue
        merged[column] = merged[manual_col].combine_first(merged[column])
        merged.drop(columns=[manual_col], inplace=True)
    merged.loc[merged["validation_outcome"].notna(), "validation_source"] = merged.loc[
        merged["validation_outcome"].notna(),
        "validation_source",
    ].where(merged["validation_source"].notna(), "auto")
    if "validation_source_manual" in manual_df.columns or "validation_source" in manual_df.columns:
        manual_source_column = (
            "validation_source_manual"
            if "validation_source_manual" in merged.columns
            else "validation_source"
        )
        if manual_source_column in merged.columns:
            merged["validation_source"] = merged[manual_source_column].combine_first(
                merged["validation_source"]
            )
            if manual_source_column != "validation_source":
                merged.drop(columns=[manual_source_column], inplace=True)
    affected = merged["journal_id"].isin(manual_df["journal_id"])
    merged.loc[affected, "validation_source"] = merged.loc[affected, "validation_source"].fillna(
        "manual"
    )
    return merged


def _evaluate_trade_row(
    trade_row: pd.Series,
    price_frame: pd.DataFrame,
    horizon_minutes: int,
    entry_grace_minutes: int,
    entry_tolerance: float,
    precision_state_column: Optional[str],
    executed_precision_states: Optional[set[str]],
) -> Optional[Dict[str, Any]]:
    signal = str(trade_row.get("signal", "")).lower()
    if signal not in {"buy", "sell"}:
        return None
    try:
        result = lt._evaluate_trade(
            trade_row,
            price_frame,
            horizon_minutes=horizon_minutes,
            entry_grace_minutes=entry_grace_minutes,
            entry_tolerance=entry_tolerance,
            precision_state_column=precision_state_column,
            executed_precision_states=executed_precision_states,
        )
    except ValueError:
        return None
        
    payload = asdict(result)
    return {
        "journal_id": trade_row.get("journal_id"),
        "validation_outcome": payload.get("outcome"),
        "validation_rr": payload.get("risk_r_multiple"),
        "max_favorable_excursion": payload.get("max_favorable_rr"),
        "max_adverse_excursion": payload.get("max_adverse_rr"),
        "time_to_outcome_minutes": payload.get("time_to_outcome_minutes"),
        "fill_timestamp": _serialise_timestamp(payload.get("fill_timestamp")),
        "exit_timestamp": _serialise_timestamp(payload.get("exit_timestamp")),
        "entry_kind": payload.get("entry_kind"),
        "validation_source": "auto",
    }


def update_live_validation(
    public_dir: Path = PUBLIC_DIR,
    reports_dir: Path = REPORTS_DIR,
    lookahead_hours: float = 48.0,
) -> Optional[Path]:
    journal_path = public_dir / "journal" / "trade_journal.csv"
    if not journal_path.exists():
        return None
    try:
        journal_df = pd.read_csv(journal_path)
    except Exception:
        return None
    if journal_df.empty:
        return None

    executable_mask = journal_df["signal"].astype(str).str.lower().isin({"buy", "sell"})
    if not executable_mask.any():
        return None

    horizon_minutes = int(max(1, lookahead_hours * 60))
    entry_grace_minutes = DEFAULT_ENTRY_GRACE_MINUTES
    entry_tolerance = DEFAULT_ENTRY_TOLERANCE

    precision_column = "precision_state" if "precision_state" in journal_df.columns else None
    executed_states: Optional[set[str]] = {"fire", "executed"} if precision_column else None

    results: List[Dict[str, Any]] = []
    for asset, asset_df in journal_df[executable_mask].groupby("asset"):
        price_history = _load_price_history(str(asset), public_dir)
        if price_history is None or price_history.empty:
            continue
        for _, row in asset_df.iterrows():
            evaluation = _evaluate_trade_row(
                row,
                price_history,
                horizon_minutes=horizon_minutes,
                entry_grace_minutes=entry_grace_minutes,
                entry_tolerance=entry_tolerance,
                precision_state_column=precision_column,
                executed_precision_states=executed_states,
            )
            if not evaluation:
                continue
            evaluation["asset"] = asset
            results.append(evaluation)
    if not results:
        return None

    results_df = pd.DataFrame(results)
    if "journal_id" in results_df.columns:
        results_df["journal_id"] = results_df["journal_id"].astype(str)
    results_df = results_df.dropna(subset=["journal_id", "validation_outcome"], how="any")

    manual_overrides = _load_manual_fill_log(public_dir)
    if manual_overrides is not None:
        results_df = _apply_manual_overrides(results_df, manual_overrides)

    if results_df.empty:
        return None

    for column in RESULT_COLUMNS:
        if column not in results_df.columns:
            if column in NUMERIC_RESULT_COLUMNS:
                results_df[column] = np.nan
            else:
                results_df[column] = None

    if "asset" not in results_df.columns:
        results_df["asset"] = None

    for column in RESULT_COLUMNS:
        if column not in journal_df.columns:
            if column in NUMERIC_RESULT_COLUMNS:
                journal_df[column] = np.nan
            else:
                journal_df[column] = pd.Series([None] * len(journal_df), dtype=object)
        elif column not in NUMERIC_RESULT_COLUMNS:
            journal_df[column] = journal_df[column].astype(object, copy=False)

    for _, result_row in results_df.iterrows():
        journal_mask = journal_df["journal_id"].astype(str) == str(result_row.get("journal_id"))
        if not journal_mask.any():
            continue
        idx = journal_mask.idxmax()
        for column in RESULT_COLUMNS:
            journal_df.at[idx, column] = result_row.get(column)
    journal_df.to_csv(journal_path, index=False)

    trade_df = journal_df[executable_mask]
    resolved = trade_df[
        trade_df["validation_outcome"].astype(str).str.lower().isin(RESOLVED_OUTCOMES_LOWER)
    ]

    resolved_sorted = resolved.sort_values("analysis_timestamp")
    equity_curve: List[float] = []
    equity = 0.0
    for _, trade in resolved_sorted.iterrows():
        outcome = str(trade.get("validation_outcome", "")).lower()
        rr_value = trade.get("validation_rr")
        try:
            rr_numeric = float(rr_value)
        except (TypeError, ValueError):
            rr_numeric = None
        if outcome in WIN_OUTCOMES_LOWER and rr_numeric is not None and np.isfinite(rr_numeric):
            equity += float(rr_numeric)
        elif outcome in LOSS_OUTCOMES_LOWER:
            if rr_numeric is not None and np.isfinite(rr_numeric):
                equity += float(rr_numeric)
            else:
                equity -= 1.0
        elif outcome in AMBIGUOUS_OUTCOMES_LOWER:
            equity += 0.0
        equity_curve.append(equity)

    max_drawdown = 0.0
    peak = -math.inf
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)

    lower_outcomes = resolved["validation_outcome"].astype(str).str.lower()
    win_mask = lower_outcomes.isin(WIN_OUTCOMES_LOWER)
    loss_mask = lower_outcomes.isin(LOSS_OUTCOMES_LOWER)
    ambiguous_mask = lower_outcomes.isin(AMBIGUOUS_OUTCOMES_LOWER)

    avg_rr_series = pd.to_numeric(resolved["validation_rr"], errors="coerce")

    summary = {
        "generated_utc": _now().isoformat(),
        "evaluated_trades": int(len(trade_df)),
        "resolved_trades": int(len(resolved)),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "ambiguous": int(ambiguous_mask.sum()),
        "win_rate": float(win_mask.sum() / len(resolved)) if len(resolved) else None,
        "avg_validation_rr": float(avg_rr_series.mean()) if not avg_rr_series.dropna().empty else None,
        "max_drawdown_rr": max_drawdown,
    }

    asset_breakdown: Dict[str, Dict[str, Any]] = {}
    for asset, asset_df in trade_df.groupby("asset"):
        asset_resolved = asset_df[
            asset_df["validation_outcome"].astype(str).str.lower().isin(RESOLVED_OUTCOMES_LOWER)
        ]
        asset_lower = asset_resolved["validation_outcome"].astype(str).str.lower()
        asset_win_mask = asset_lower.isin(WIN_OUTCOMES_LOWER)
        asset_rr = pd.to_numeric(asset_resolved["validation_rr"], errors="coerce")
        asset_breakdown[asset] = {
            "trades": int(len(asset_df)),
            "resolved": int(len(asset_resolved)),
            "win_rate": float(asset_win_mask.sum() / len(asset_resolved))
            if len(asset_resolved)
            else None,
            "avg_validation_rr": float(asset_rr.mean()) if not asset_rr.dropna().empty else None,
        }
    summary["assets"] = asset_breakdown

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "live_validation.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    details_path = reports_dir / "live_validation.csv"
    results_df.to_csv(details_path, index=False)
    return summary_path


__all__ = ["update_live_validation"]
