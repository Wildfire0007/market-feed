"""Volatility overlays for the analysis pipeline.

The helper exposes a single entry point ``load_volatility_overlay`` which
combines optional JSON metrics (generated offline) with realised volatility
estimates computed from the latest one minute candles.  The overlay is designed
so that callers can adjust risk parameters when implied volatility departs from
recent realised behaviour.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class VolatilityOverlay:
    implied_vol: Optional[float]
    realised_vol: Optional[float]
    realised_window_minutes: int
    regime: str
    premium: Optional[float]
    source_path: Optional[Path]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "implied_vol": self.implied_vol,
            "realised_vol": self.realised_vol,
            "realised_window_minutes": self.realised_window_minutes,
            "regime": self.regime,
            "premium": self.premium,
            "source_path": str(self.source_path) if self.source_path else None,
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _classify_regime(realised: Optional[float], implied: Optional[float]) -> str:
    if realised is None and implied is None:
        return "unknown"
    if realised is None:
        return "implied_only"
    if implied is None:
        if realised < 0.0004:
            return "suppressed"
        if realised > 0.002:
            return "elevated"
        return "historical"
    premium = implied - realised
    if implied <= 0 and realised <= 0:
        return "flat"
    ratio = implied / realised if realised else None
    if ratio is not None:
        if ratio >= 1.6:
            return "implied_extreme"
        if ratio >= 1.25:
            return "implied_elevated"
        if ratio <= 0.6:
            return "implied_discount"
    if premium > 0.0015:
        return "premium"
    if premium < -0.0015:
        return "discount"
    return "balanced"


def _compute_realised_vol(k1m: Optional[pd.DataFrame], window: int = 120) -> Optional[float]:
    if k1m is None or k1m.empty:
        return None
    closes = k1m["close"].astype(float).dropna()
    if closes.size < 10:
        return None
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    if window > 0:
        returns = returns.tail(window)
    std = returns.std()
    if std is None or np.isnan(std):
        return None
    return float(std)


def load_volatility_overlay(
    asset: str,
    outdir: Path,
    k1m: Optional[pd.DataFrame] = None,
    realised_window: int = 120,
) -> Dict[str, Any]:
    """Load implied/realised volatility metrics for ``asset``.

    Parameters
    ----------
    asset:
        Instrument identifier (case insensitive).
    outdir:
        Directory that contains the public artefacts for the asset.
    k1m:
        Latest 1 minute candle dataframe.  When provided the helper computes
        realised volatility using percentage returns.
    realised_window:
        Amount of recent one minute bars to use for the realised volatility
        estimate.
    """

    outdir = Path(outdir)
    json_path = outdir / "vol_metrics.json"
    implied = realised = premium = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            payload = {}
        implied = _safe_float(payload.get("implied_vol"))
        realised_external = _safe_float(payload.get("realised_vol"))
        if realised_external is not None:
            realised = realised_external
        premium = _safe_float(payload.get("vol_premium"))
    realised_internal = _compute_realised_vol(k1m, realised_window)
    if realised_internal is not None:
        realised = realised_internal
    regime = _classify_regime(realised, implied)
    overlay = VolatilityOverlay(
        implied_vol=implied,
        realised_vol=realised,
        realised_window_minutes=realised_window,
        regime=regime,
        premium=premium,
        source_path=json_path if json_path.exists() else None,
    )
    data = overlay.as_dict()
    if implied is not None and realised is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if realised != 0:
                data["implied_realised_ratio"] = float(implied / realised)
            else:
                data["implied_realised_ratio"] = None
    else:
        data["implied_realised_ratio"] = None
    return data


__all__ = ["load_volatility_overlay"]
