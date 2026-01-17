# -*- coding: utf-8 -*-
"""
optimizer.py — Github Actions verzió.
Kizárólag számol és ment. A commitolást a workflow végzi.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# GitHub runneren a checkout helyéhez viszonyítunk
SCRIPT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = SCRIPT_DIR / "public"
OUTPUT_PATH = PUBLIC_DIR / "adaptive_params.json"

PREFERRED_KLINES = ("klines_4h.json", "klines_1h.json")
TIMEFRAME_REGEX = r"klines_(\d+)([mhd])\.json"

def _find_asset_dirs(base: Path) -> List[Path]:
    assets: List[Path] = []
    if not base.exists():
        return assets
    for entry in base.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if any(entry.glob("klines_*.json")):
            assets.append(entry)
    return assets

def _timeframe_to_minutes(value: int, unit: str) -> int:
    if unit == "m": return value
    if unit == "h": return value * 60
    if unit == "d": return value * 1440
    return value

def _select_kline_file(asset_dir: Path) -> Optional[Path]:
    for name in PREFERRED_KLINES:
        candidate = asset_dir / name
        if candidate.exists(): return candidate
    candidates: List[Tuple[int, Path]] = []
    for path in asset_dir.glob("klines_*.json"):
        match = pd.Series(path.name).str.extract(TIMEFRAME_REGEX).iloc[0]
        if match.isna().any(): continue
        amount = int(match[0])
        unit = str(match[1])
        minutes = _timeframe_to_minutes(amount, unit)
        candidates.append((minutes, path))
    if not candidates: return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]

def _load_klines(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception: return pd.DataFrame()
    values = payload.get("values") if isinstance(payload, dict) else payload
    if not isinstance(values, list): return pd.DataFrame()
    df = pd.DataFrame(values)
    if df.empty: return df
    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "high", "low", "close"])
    df = df.sort_values("timestamp")
    return df

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def _precision_for_price(price: float) -> int:
    if price < 10: return 5
    if price > 1000: return 1
    return 2

def _extract_ofi_zscores(data: Any) -> List[float]:
    scores: List[float] = []
    def _try_value(raw: Any) -> None:
        try:
            val = float(raw)
            if np.isfinite(val): scores.append(abs(val))
        except: pass
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                for key in ("ofi_z", "ofi_zscore", "zscore", "z_score", "imbalance_z"):
                    if key in row: _try_value(row[key]); break
            else: _try_value(row)
    elif isinstance(data, dict):
        if "values" in data and isinstance(data["values"], list): scores.extend(_extract_ofi_zscores(data["values"]))
        else:
            for key in ("ofi_z", "ofi_zscore", "zscore", "z_score", "imbalance_z"):
                if key in data: _try_value(data[key])
    return scores

def _load_ofi_trigger(asset_dir: Path) -> Optional[float]:
    candidates = [asset_dir/"order_flow.json", asset_dir/"ofi_zscores.json", asset_dir/"ofi.json"]
    for path in candidates:
        if not path.exists(): continue
        try:
            with path.open("r", encoding="utf-8") as handle: payload = json.load(handle)
        except: continue
        scores = _extract_ofi_zscores(payload)
        if scores: return float(np.nanpercentile(scores, 95))
    return None

def _compute_params_for_asset(asset_dir: Path) -> Optional[Dict[str, Any]]:
    asset = asset_dir.name
    kline_path = _select_kline_file(asset_dir)
    if not kline_path: return None
    df = _load_klines(kline_path)
    if df.empty: return None
    latest_ts = df["timestamp"].max()
    cutoff = latest_ts - pd.Timedelta(days=90)
    df_recent = df[df["timestamp"] >= cutoff]
    if df_recent.empty: return None
    atr_series = _compute_atr(df_recent, period=14).dropna()
    if atr_series.empty: return None
    atr_floor_raw = float(np.nanpercentile(atr_series.values, 20))
    last_close = float(df_recent["close"].iloc[-1])
    precision = _precision_for_price(last_close)
    atr_floor = round(atr_floor_raw, precision)
    ofi_trigger = _load_ofi_trigger(asset_dir)
    payload: Dict[str, Any] = {"atr_floor": atr_floor, "precision": precision, "last_updated": str(pd.Timestamp.now())}
    if ofi_trigger is not None and np.isfinite(ofi_trigger):
        payload["ofi_trigger"] = round(float(ofi_trigger), max(precision, 1))
    return payload

def build_adaptive_params() -> Dict[str, Dict[str, Any]]:
    params: Dict[str, Dict[str, Any]] = {}
    if not PUBLIC_DIR.exists(): return params
    for asset_dir in _find_asset_dirs(PUBLIC_DIR):
        try:
            result = _compute_params_for_asset(asset_dir)
            if result: params[asset_dir.name] = result
        except Exception: pass
    return params

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recalc", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    logging.basicConfig(level=logging.INFO)
    print(f"--- GITHUB OPTIMIZER ---\nDIR: {SCRIPT_DIR}")
    
    # GitHubon mindig kényszerítjük a számítást, ha nincs megadva flag, 
    # de a biztonság kedvéért itt hagyjuk a logikát.
    params = build_adaptive_params()
    
    if not params:
        print("FIGYELEM: Nem találtam adatokat a public mappában!")
        print("Ellenőrizd, hogy a repo tartalmazza-e a klines_*.json fájlokat.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, ensure_ascii=False, sort_keys=True)
    
    print(f"SIKER: Adaptive params generálva: {OUTPUT_PATH}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
