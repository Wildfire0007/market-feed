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

import io
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
_DEFAULT_REPORTS_DIR = PUBLIC_DIR / "reports"
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(_DEFAULT_REPORTS_DIR)))

PRICE_CANDIDATES = (
    "klines_15s.json",
    "klines_30s.json",
    "klines_1m.json",
    "klines_15s.csv",
    "klines_30s.csv",
    "klines_1m.csv",
    "klines_5m.json",
    "klines_5m.csv",
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

SIMULATED_TP = "simulated_tp"
SIMULATED_SL = "simulated_sl"
SIMULATED_EXIT = "simulated_exit"

WIN_OUTCOMES = {lt.OUTCOME_PROFIT, "tp_hit", "tp1", SIMULATED_TP}
LOSS_OUTCOMES = {lt.OUTCOME_STOP, "stopped", SIMULATED_SL}
AMBIGUOUS_OUTCOMES = {lt.OUTCOME_AMBIGUOUS, "ambiguous", SIMULATED_EXIT}
RESOLVED_OUTCOMES = WIN_OUTCOMES | LOSS_OUTCOMES | AMBIGUOUS_OUTCOMES | {"tp2"}

WIN_OUTCOMES_LOWER = {status.lower() for status in WIN_OUTCOMES}
LOSS_OUTCOMES_LOWER = {status.lower() for status in LOSS_OUTCOMES}
AMBIGUOUS_OUTCOMES_LOWER = {status.lower() for status in AMBIGUOUS_OUTCOMES}
RESOLVED_OUTCOMES_LOWER = {status.lower() for status in RESOLVED_OUTCOMES}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_price_json(path: Path) -> Optional[pd.DataFrame]:
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


def _read_price_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    timestamp_column: Optional[str] = None
    for candidate in ("timestamp", "datetime", "time"):
        if candidate in frame.columns:
            timestamp_column = candidate
            break
    if timestamp_column is None:
        return None
    try:
        frame["timestamp"] = pd.to_datetime(frame[timestamp_column], utc=True)
    except Exception:
        return None
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            return None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if frame.empty:
        return None
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "open", "high", "low", "close"]]


def _read_price_payload(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_price_json(path)
    if suffix == ".csv":
        return _read_price_csv(path)
    return None
    
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


def _parse_timestamp(value: object) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp


def _coerce_price(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _maybe_simulate_exit(
    trade_row: pd.Series,
    price_frame: pd.DataFrame,
    evaluation: Dict[str, Any],
    horizon_minutes: int,
) -> None:
    outcome = str(evaluation.get("validation_outcome", "")).lower()
    if outcome in RESOLVED_OUTCOMES_LOWER:
        return

    analysis_timestamp = _parse_timestamp(trade_row.get("analysis_timestamp"))
    if analysis_timestamp is None:
        return

    direction_raw = str(trade_row.get("signal", "")).lower()
    if direction_raw not in {"buy", "sell"}:
        return
    direction = 1 if direction_raw == "buy" else -1

    entry_price = _coerce_price(trade_row.get("entry_price"))
    if entry_price is None:
        entry_price = _coerce_price(trade_row.get("spot_price"))
    stop_loss = _coerce_price(trade_row.get("stop_loss"))
    take_profit = _coerce_price(trade_row.get("take_profit_1"))
    if take_profit is None:
        take_profit = _coerce_price(trade_row.get("take_profit_2"))

    if entry_price is None or stop_loss is None or take_profit is None:
        return

    risk = entry_price - stop_loss if direction == 1 else stop_loss - entry_price
    if not math.isfinite(risk) or risk <= 0:
        return

    evaluation_end = analysis_timestamp + pd.Timedelta(minutes=int(max(horizon_minutes, 1)))
    window_start = analysis_timestamp.floor("min")
    price_slice = price_frame[
        (price_frame["timestamp"] >= window_start)
        & (price_frame["timestamp"] <= evaluation_end)
    ]
    if price_slice.empty:
        return

    exit_row = price_slice.iloc[-1]
    exit_timestamp = exit_row["timestamp"]
    exit_close = _coerce_price(exit_row.get("close"))
    if exit_close is None:
        return

    fill_timestamp = _parse_timestamp(evaluation.get("fill_timestamp")) or analysis_timestamp
    time_to_outcome: Optional[float] = None
    if exit_timestamp is not None:
        time_to_outcome = (exit_timestamp - fill_timestamp).total_seconds() / 60.0

    if direction == 1:
        rr = (exit_close - entry_price) / risk
        max_favorable = (price_slice["high"].max() - entry_price) / risk
        max_adverse = (entry_price - price_slice["low"].min()) / risk
    else:
        rr = (entry_price - exit_close) / risk
        max_favorable = (entry_price - price_slice["low"].min()) / risk
        max_adverse = (price_slice["high"].max() - entry_price) / risk

    if not all(map(math.isfinite, [rr, max_favorable, max_adverse])):
        return

    epsilon = 1e-6
    if rr > epsilon:
        simulated_outcome = SIMULATED_TP
    elif rr < -epsilon:
        simulated_outcome = SIMULATED_SL
    else:
        simulated_outcome = SIMULATED_EXIT

    evaluation.update(
        {
            "validation_outcome": simulated_outcome,
            "validation_rr": float(rr),
            "max_favorable_excursion": float(max_favorable),
            "max_adverse_excursion": float(max_adverse),
            "time_to_outcome_minutes": float(time_to_outcome) if time_to_outcome is not None else None,
            "exit_timestamp": _serialise_timestamp(exit_timestamp),
            "validation_source": "simulated",
        }
    )
    if evaluation.get("fill_timestamp") is None:
        evaluation["fill_timestamp"] = _serialise_timestamp(fill_timestamp)
        
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


def _write_validation_reports(
    summary: Dict[str, Any],
    results_df: pd.DataFrame,
    target_dirs: Iterable[Path],
) -> Optional[Path]:
    summary_blob = json.dumps(summary, ensure_ascii=False, indent=2)
    if not summary_blob.endswith("\n"):
        summary_blob = f"{summary_blob}\n"
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    summary_path: Optional[Path] = None
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        summary_file = directory / "live_validation.json"
        summary_file.write_text(summary_blob, encoding="utf-8")
        details_file = directory / "live_validation.csv"
        details_file.write_text(csv_data, encoding="utf-8")
        if summary_path is None:
            summary_path = summary_file
    return summary_path


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
            _maybe_simulate_exit(row, price_history, evaluation, horizon_minutes)
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
    
    def _compute_performance(frame: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if frame.empty:
            return None
        resolved_mask = frame["validation_outcome"].astype(str).str.lower().isin(RESOLVED_OUTCOMES_LOWER)
        resolved_subset = frame[resolved_mask].copy()
        performance: Dict[str, Any] = {
            "trades": int(len(frame)),
            "resolved": int(len(resolved_subset)),
            "wins": 0,
            "losses": 0,
            "ambiguous": 0,
            "win_rate": None,
            "avg_validation_rr": None,
            "max_drawdown_rr": 0.0,
        }
        if resolved_subset.empty:
            return performance

        lower_outcomes = resolved_subset["validation_outcome"].astype(str).str.lower()
        win_mask = lower_outcomes.isin(WIN_OUTCOMES_LOWER)
        loss_mask = lower_outcomes.isin(LOSS_OUTCOMES_LOWER)
        ambiguous_mask = lower_outcomes.isin(AMBIGUOUS_OUTCOMES_LOWER)
        avg_rr_series = pd.to_numeric(resolved_subset["validation_rr"], errors="coerce")

        sort_column = None
        for candidate in ("analysis_timestamp", "fill_timestamp", "exit_timestamp"):
            if candidate in resolved_subset.columns:
                sort_column = candidate
                break
        resolved_sorted = resolved_subset.sort_values(sort_column) if sort_column else resolved_subset

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

        performance.update(
            {
                "wins": int(win_mask.sum()),
                "losses": int(loss_mask.sum()),
                "ambiguous": int(ambiguous_mask.sum()),
                "win_rate": float(win_mask.sum() / len(resolved_subset)) if len(resolved_subset) else None,
                "avg_validation_rr": float(avg_rr_series.mean())
                if not avg_rr_series.dropna().empty
                else None,
                "max_drawdown_rr": float(max_drawdown),
            }
        )
        return performance
        
    performance = _compute_performance(trade_df) or {}
    summary_avg_rr = performance.get("avg_validation_rr")
    summary = {
        "generated_utc": _now().isoformat(),
        "evaluated_trades": int(len(trade_df)),
        "resolved_trades": int(performance.get("resolved", 0)),
        "wins": int(performance.get("wins", 0)),
        "losses": int(performance.get("losses", 0)),
        "ambiguous": int(performance.get("ambiguous", 0)),
        "win_rate": performance.get("win_rate"),
        "avg_validation_rr": float(summary_avg_rr) if summary_avg_rr is not None else None,
        "max_drawdown_rr": float(performance.get("max_drawdown_rr", 0.0)),
    }

    rule_summary: Optional[Dict[str, Any]] = None
    if "probability_source" in trade_df.columns:
        rule_mask = trade_df["probability_source"].astype(str).str.lower().isin({"rule", "fallback"})
        rule_summary = _compute_performance(trade_df[rule_mask])
    summary["rule_based_performance"] = rule_summary
    
    asset_breakdown: Dict[str, Dict[str, Any]] = {}
    for asset, asset_df in trade_df.groupby("asset"):
        asset_performance = _compute_performance(asset_df) or {}
        asset_avg_rr = asset_performance.get("avg_validation_rr")
        asset_breakdown[asset] = {
            "trades": int(asset_performance.get("trades", len(asset_df))),
            "resolved": int(asset_performance.get("resolved", 0)),
            "win_rate": asset_performance.get("win_rate"),
            "avg_validation_rr": float(asset_avg_rr) if asset_avg_rr is not None else None,
        }
    summary["assets"] = asset_breakdown

    primary_reports_dir = Path(reports_dir)
    default_public_reports_dir = Path(public_dir) / "reports"
    target_dirs: List[Path] = []
    for candidate in (primary_reports_dir, default_public_reports_dir):
        candidate = candidate.resolve()
        if candidate not in target_dirs:
            target_dirs.append(candidate)

    return _write_validation_reports(summary, results_df, target_dirs)
    
    return summary_path


__all__ = ["update_live_validation"]
