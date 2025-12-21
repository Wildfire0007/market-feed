"""Trading journal helpers for manual execution and P&L attribution."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
JOURNAL_DIR = PUBLIC_DIR / "journal"
JOURNAL_FILE = JOURNAL_DIR / "trade_journal.csv"
SUMMARY_FILE = JOURNAL_DIR / "summary.json"

JOURNAL_COLUMNS: List[str] = [
    "journal_id",
    "asset",
    "analysis_timestamp",
    "signal",
    "mode",
    "probability",
    "model_probability",
    "realtime_confidence",
    "leverage",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "risk_reward",
    "initial_risk_abs",
    "spot_price",
    "spot_timestamp",
    "spot_latency_seconds",
    "precision_score",
    "anchor_bias",
    "probability_source",
    "notes",
    "validation_outcome",
    "validation_rr",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "time_to_outcome_minutes",
    "slippage",
    "execution_latency_ms",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_get(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    ref: Any = data
    for key in keys:
        if not isinstance(ref, dict) or key not in ref:
            return None
        ref = ref[key]
    return ref


def _normalise_row(asset: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    timeframes = diagnostics.get("timeframes") or {}
    spot_latency = None
    if isinstance(timeframes, dict):
        spot_meta = timeframes.get("spot")
        if isinstance(spot_meta, dict):
            spot_latency = spot_meta.get("latency_seconds")
    active_meta = payload.get("active_position_meta") or {}
    probability_model = payload.get("probability_model")
    probability_source = "ensemble" if probability_model is not None else "rule"
    notes = payload.get("reasons")
    if isinstance(notes, list):
        note_text = "; ".join(str(item) for item in notes)
    elif isinstance(notes, str):
        note_text = notes
    else:
        note_text = ""

    row = {
        "journal_id": str(uuid.uuid4()),
        "asset": asset.upper(),
        "analysis_timestamp": payload.get("retrieved_at_utc"),
        "signal": payload.get("signal"),
        "mode": _safe_get(payload, "gates", "mode"),
        "probability": payload.get("probability"),
        "model_probability": probability_model,
        "realtime_confidence": active_meta.get("realtime_confidence")
        if isinstance(active_meta, dict)
        else payload.get("realtime_confidence"),
        "leverage": payload.get("leverage"),
        "entry_price": payload.get("entry"),
        "stop_loss": payload.get("sl"),
        "take_profit_1": payload.get("tp1"),
        "take_profit_2": payload.get("tp2"),
        "risk_reward": payload.get("rr"),
        "initial_risk_abs": active_meta.get("initial_risk_abs") if isinstance(active_meta, dict) else None,
        "spot_price": _safe_get(payload, "spot", "price"),
        "spot_timestamp": _safe_get(payload, "spot", "utc"),
        "spot_latency_seconds": spot_latency,
        "precision_score": active_meta.get("precision_score") if isinstance(active_meta, dict) else None,
        "anchor_bias": active_meta.get("anchor_side") if isinstance(active_meta, dict) else None,
        "probability_source": probability_source,
        "notes": note_text,
        "validation_outcome": None,
        "validation_rr": None,
        "max_favorable_excursion": None,
        "max_adverse_excursion": None,
        "time_to_outcome_minutes": None,
        "slippage": None,
        "execution_latency_ms": None,
    }
    return row


def record_signal_event(asset: str, payload: Dict[str, Any]) -> Optional[Path]:
    if not isinstance(payload, dict):
        return None

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    row = _normalise_row(asset, payload)
    df = pd.DataFrame([row], columns=JOURNAL_COLUMNS)
    if JOURNAL_FILE.exists():
        df.to_csv(JOURNAL_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(JOURNAL_FILE, index=False)

    update_journal_summary()
    return JOURNAL_FILE


def update_journal_summary() -> Optional[Path]:
    if not JOURNAL_FILE.exists():
        return None
    try:
        df = pd.read_csv(JOURNAL_FILE)
    except Exception:
        return None
    if df.empty:
        summary_payload = {
            "generated_utc": _now().isoformat(),
            "total_signals": 0,
            "trade_candidates": 0,
            "resolved_trades": 0,
            "win_rate": None,
            "assets": {},
        }
    else:
        df["is_trade"] = df["signal"].str.lower().isin(["buy", "sell"])
        trade_df = df[df["is_trade"]]
        resolved = trade_df[trade_df["validation_outcome"].isin(["tp1", "tp2", "stopped"])]
        wins = resolved[resolved["validation_outcome"].isin(["tp1", "tp2"])]
        win_rate = float(len(wins) / len(resolved)) if len(resolved) else None
        asset_summary: Dict[str, Dict[str, Any]] = {}
        for asset, asset_df in trade_df.groupby("asset"):
            asset_resolved = asset_df[asset_df["validation_outcome"].isin(["tp1", "tp2", "stopped"])]
            asset_wins = asset_resolved[asset_resolved["validation_outcome"].isin(["tp1", "tp2"])]
            asset_summary[asset] = {
                "signals": int(len(asset_df)),
                "resolved": int(len(asset_resolved)),
                "wins": int(len(asset_wins)),
                "win_rate": float(len(asset_wins) / len(asset_resolved)) if len(asset_resolved) else None,
                "avg_validation_rr": float(asset_resolved["validation_rr"].mean())
                if not asset_resolved["validation_rr"].isna().all()
                else None,
            }
        summary_payload = {
            "generated_utc": _now().isoformat(),
            "total_signals": int(len(df)),
            "trade_candidates": int(len(trade_df)),
            "resolved_trades": int(len(resolved)),
            "win_rate": win_rate,
            "assets": asset_summary,
        }
    with SUMMARY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
    return SUMMARY_FILE


__all__ = ["record_signal_event", "update_journal_summary", "JOURNAL_FILE", "SUMMARY_FILE"]
