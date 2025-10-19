"""Volatility overlays for the analysis pipeline.

The helper exposes a single entry point ``load_volatility_overlay`` which
combines optional JSON metrics (generated offline) with realised volatility
estimates computed from the latest one minute candles.  The overlay is designed
so that callers can adjust risk parameters when implied volatility departs from
recent realised behaviour.

When executed as a script (``python -m volatility_metrics``) the module
generates fresh overlay snapshots for every configured asset so that
``analysis.py`` always finds up-to-date realised volatility estimates between
the trading and analysis stages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
DEFAULT_OVERLAY_FILENAME = "vol_overlay.json"


def _load_klines_frame(asset_dir: Path) -> Optional[pd.DataFrame]:
    path = asset_dir / "klines_1m.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None

    values: Optional[List[Dict[str, Any]]]
    if isinstance(payload, dict):
        raw_values = payload.get("values") or payload.get("data")
        values = raw_values if isinstance(raw_values, list) else None
    elif isinstance(payload, list):
        values = payload
    else:
        values = None

    if not values:
        return None

    try:
        frame = pd.DataFrame(values)
    except Exception:
        return None

    if frame.empty:
        return None

    frame = frame.rename(columns={str(col): str(col).lower() for col in frame.columns})
    if "close" not in frame.columns:
        return None

    for column in ("datetime", "time", "timestamp"):
        if column in frame.columns:
            try:
                frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
            except Exception:
                continue
            frame = frame.sort_values(column)
            break

    return frame


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    return path


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


def generate_overlay_snapshot(
    asset: str,
    public_dir: Path = PUBLIC_DIR,
    realised_window: int = 120,
    output_filename: str = DEFAULT_OVERLAY_FILENAME,
    write: bool = True,
) -> Optional[Dict[str, Any]]:
    """Create a volatility overlay snapshot for ``asset``.

    The helper loads the latest one minute candles, combines them with the
    implied metrics (if available) and optionally persists a JSON file that can
    be consumed by downstream tooling.
    """

    asset_dir = Path(public_dir) / asset.upper()
    asset_dir.mkdir(parents=True, exist_ok=True)
    k1m = _load_klines_frame(asset_dir)
    overlay = load_volatility_overlay(asset, asset_dir, k1m, realised_window)
    snapshot: Dict[str, Any] = dict(overlay)
    snapshot.update(
        {
            "asset": asset.upper(),
            "generated_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }
    )
    if k1m is not None:
        snapshot["samples"] = int(k1m.shape[0])
    if write:
        output_path = asset_dir / output_filename
        _write_json(output_path, snapshot)
        snapshot["output_path"] = str(output_path)
    return snapshot


def _infer_assets(explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [str(asset).upper() for asset in explicit if asset]
    try:
        from Trading import ASSETS as TRADING_ASSETS  # type: ignore

        return [str(asset).upper() for asset in TRADING_ASSETS.keys()]
    except Exception:
        try:
            directories = [
                path.name
                for path in PUBLIC_DIR.iterdir()
                if path.is_dir()
            ]
        except FileNotFoundError:
            return []
        return sorted({name.upper() for name in directories})


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate volatility overlays")
    parser.add_argument(
        "--assets",
        nargs="*",
        help="Assets to process (default: Trading.ASSETS or public directory)",
    )
    parser.add_argument(
        "--public-dir",
        default=str(PUBLIC_DIR),
        help="Directory containing the public artefacts",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Realised volatility window in minutes",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OVERLAY_FILENAME,
        help="Overlay JSON filename to write per asset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute overlays without writing files",
    )
    args = parser.parse_args(argv)

    public_dir = Path(args.public_dir)
    assets = _infer_assets(args.assets)
    if not assets:
        print("No assets detected for volatility overlay generation", file=sys.stderr)
        return 1

    exit_code = 0
    for asset in assets:
        try:
            snapshot = generate_overlay_snapshot(
                asset,
                public_dir=public_dir,
                realised_window=max(1, int(args.window)),
                output_filename=args.output,
                write=not args.dry_run,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Failed to build overlay for {asset}: {exc}", file=sys.stderr)
            exit_code = 2
            continue

        if snapshot is None:
            print(f"Skipping {asset}: missing candle data", file=sys.stderr)
            continue

        ratio = snapshot.get("implied_realised_ratio")
        regime = snapshot.get("regime")
        extra = f" ratio={ratio:.2f}" if isinstance(ratio, (int, float)) else ""
        target = snapshot.get("output_path") if snapshot.get("output_path") else "(dry-run)"
        print(f"Overlay for {asset} -> {target} [{regime}{extra}]")

    return exit_code


__all__ = [
    "load_volatility_overlay",
    "generate_overlay_snapshot",
    "cli",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(cli())
