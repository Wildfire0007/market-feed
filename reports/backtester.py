"""Lightweight live validation utilities driven by public JSON exports."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_candles(asset: str, public_dir: Path) -> Optional[pd.DataFrame]:
    path = public_dir / asset / "klines_5m.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        return None
    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        return None
    try:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    except Exception:
        return None
    numeric_cols = [col for col in ["open", "high", "low", "close"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])  # type: ignore[arg-type]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _risk_in_price(signal: str, entry: float, stop: float) -> Optional[float]:
    if not np.isfinite(entry) or not np.isfinite(stop):
        return None
    if signal.lower() == "buy":
        risk = entry - stop
    else:
        risk = stop - entry
    return float(risk) if risk > 0 else None


def _evaluate_trade_row(
    row: pd.Series,
    candles: pd.DataFrame,
    lookahead_hours: float = 48.0,
) -> Dict[str, Any]:
    signal = str(row.get("signal", "")).lower()
    if signal not in {"buy", "sell"}:
        return {
            "journal_id": row.get("journal_id"),
            "validation_outcome": None,
            "validation_rr": None,
            "max_favorable_excursion": None,
            "max_adverse_excursion": None,
            "time_to_outcome_minutes": None,
        }

    entry = row.get("entry_price")
    stop = row.get("stop_loss")
    target = row.get("take_profit_1")
    timestamp = _parse_iso(row.get("analysis_timestamp"))
    if not all(map(np.isfinite, [entry, stop, target])) or timestamp is None:
        return {
            "journal_id": row.get("journal_id"),
            "validation_outcome": "missing_levels",
            "validation_rr": None,
            "max_favorable_excursion": None,
            "max_adverse_excursion": None,
            "time_to_outcome_minutes": None,
        }

    entry = float(entry)
    stop = float(stop)
    target = float(target)
    horizon = timestamp + timedelta(hours=lookahead_hours)
    window = candles[(candles["datetime"] >= timestamp) & (candles["datetime"] <= horizon)]
    if window.empty:
        return {
            "journal_id": row.get("journal_id"),
            "validation_outcome": "insufficient_data",
            "validation_rr": None,
            "max_favorable_excursion": None,
            "max_adverse_excursion": None,
            "time_to_outcome_minutes": None,
        }

    risk = _risk_in_price(signal, entry, stop)
    if risk is None or risk <= 0:
        return {
            "journal_id": row.get("journal_id"),
            "validation_outcome": "invalid_risk",
            "validation_rr": None,
            "max_favorable_excursion": None,
            "max_adverse_excursion": None,
            "time_to_outcome_minutes": None,
        }

    mfe_ratio = 0.0
    mae_ratio = 0.0
    outcome = "open"
    minutes_to_outcome: Optional[float] = None

    for _, candle in window.iterrows():
        high = float(candle.get("high", np.nan))
        low = float(candle.get("low", np.nan))
        if not np.isfinite(high) or not np.isfinite(low):
            continue
        if signal == "buy":
            favorable = high - entry
            adverse = entry - low
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            favorable = entry - low
            adverse = high - entry
            stop_hit = high >= stop
            target_hit = low <= target
        mfe_ratio = max(mfe_ratio, favorable / risk if risk else 0.0)
        mae_ratio = max(mae_ratio, adverse / risk if risk else 0.0)
        if stop_hit and target_hit:
            outcome = "ambiguous"
            minutes_to_outcome = (candle["datetime"] - timestamp).total_seconds() / 60.0
            break
        if stop_hit:
            outcome = "stopped"
            minutes_to_outcome = (candle["datetime"] - timestamp).total_seconds() / 60.0
            break
        if target_hit:
            outcome = "tp1"
            minutes_to_outcome = (candle["datetime"] - timestamp).total_seconds() / 60.0
            break

    validation_rr = None
    if outcome == "tp1":
        validation_rr = (target - entry) / risk if signal == "buy" else (entry - target) / risk
    elif outcome == "stopped":
        validation_rr = -1.0

    return {
        "journal_id": row.get("journal_id"),
        "validation_outcome": outcome,
        "validation_rr": validation_rr,
        "max_favorable_excursion": float(mfe_ratio) if np.isfinite(mfe_ratio) else None,
        "max_adverse_excursion": float(mae_ratio) if np.isfinite(mae_ratio) else None,
        "time_to_outcome_minutes": minutes_to_outcome,
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

    results: List[Dict[str, Any]] = []
    for asset, asset_df in journal_df.groupby("asset"):
        candles = _load_candles(asset, public_dir)
        if candles is None:
            continue
        for _, row in asset_df.iterrows():
            result = _evaluate_trade_row(row, candles, lookahead_hours=lookahead_hours)
            result["asset"] = asset
            results.append(result)
    if not results:
        return None

    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=["journal_id"], how="any")

    journal_df = journal_df.merge(results_df, on="journal_id", how="left", suffixes=("", "_calc"))
    for column in [
        "validation_outcome",
        "validation_rr",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "time_to_outcome_minutes",
    ]:
        calc_column = f"{column}_calc"
        if calc_column in journal_df.columns:
            journal_df[column] = journal_df[calc_column]
            journal_df.drop(columns=[calc_column], inplace=True)
    journal_df.to_csv(journal_path, index=False)

    trade_df = journal_df[journal_df["signal"].str.lower().isin(["buy", "sell"])]
    resolved = trade_df[trade_df["validation_outcome"].isin(["tp1", "tp2", "stopped", "ambiguous"])]

    resolved_sorted = resolved.sort_values("analysis_timestamp")
    equity_curve: List[float] = []
    equity = 0.0
    for _, trade in resolved_sorted.iterrows():
        outcome = str(trade.get("validation_outcome", ""))
        rr_value = trade.get("validation_rr")
        if outcome == "tp1" and np.isfinite(rr_value):
            equity += float(rr_value)
        elif outcome == "stopped":
            equity -= 1.0
        elif outcome == "ambiguous":
            equity += 0.0
        equity_curve.append(equity)

    max_drawdown = 0.0
    peak = -math.inf
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)

    summary = {
        "generated_utc": _now().isoformat(),
        "evaluated_trades": int(len(trade_df)),
        "resolved_trades": int(len(resolved)),
        "wins": int(len(resolved[resolved["validation_outcome"] == "tp1"])),
        "losses": int(len(resolved[resolved["validation_outcome"] == "stopped"])),
        "ambiguous": int(len(resolved[resolved["validation_outcome"] == "ambiguous"])),
        "win_rate": float(
            len(resolved[resolved["validation_outcome"] == "tp1"]) / len(resolved)
        )
        if len(resolved)
        else None,
        "avg_validation_rr": float(resolved["validation_rr"].mean())
        if not resolved["validation_rr"].dropna().empty
        else None,
        "max_drawdown_rr": max_drawdown,
    }

    asset_breakdown: Dict[str, Dict[str, Any]] = {}
    for asset, asset_df in trade_df.groupby("asset"):
        asset_resolved = asset_df[asset_df["validation_outcome"].isin(["tp1", "tp2", "stopped", "ambiguous"])]
        asset_breakdown[asset] = {
            "trades": int(len(asset_df)),
            "resolved": int(len(asset_resolved)),
            "win_rate": float(
                len(asset_resolved[asset_resolved["validation_outcome"] == "tp1"]) / len(asset_resolved)
            )
            if len(asset_resolved)
            else None,
            "avg_validation_rr": float(asset_resolved["validation_rr"].mean())
            if not asset_resolved["validation_rr"].dropna().empty
            else None,
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
