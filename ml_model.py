"""Utility helpers for gradient boosting probability estimates.

The analysis pipeline logs tabular feature snapshots and, when an asset has a
trained gradient boosting classifier saved under ``public/models/<asset>_gbm.pkl``,
this module scores the snapshot and returns a probability that the next signal
is profitable.  The model artefacts intentionally live next to the JSON
pipeline outputs so that re-training can be performed offline without shipping
large binaries into the repo.

If no model file is present the helpers degrade gracefully and simply return
``None`` while still writing the feature snapshot into a CSV log for future
labelling.
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
try:  # scikit-learn might be missing on lightweight runners
    from sklearn.ensemble import GradientBoostingClassifier
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    GradientBoostingClassifier = type("GradientBoostingClassifier", (), {})
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:  # joblib is optional on the analysis runner
    from joblib import load
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    load = None  # type: ignore[assignment]
    _JOBLIB_IMPORT_ERROR = exc
else:
    _JOBLIB_IMPORT_ERROR = None

_JOBLIB_WARNING_EMITTED = False
_SKLEARN_WARNING_EMITTED = False

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
MODEL_DIR = PUBLIC_DIR / "models"
FEATURE_LOG_DIR = PUBLIC_DIR / "ml_features"

# A consistent, minimal feature vector so that models can be re-trained offline
# without having to reverse engineer implicit column order.
MODEL_FEATURES: List[str] = [
    "p_score",
    "rel_atr",
    "ema21_slope",
    "bias_long",
    "bias_short",
    "momentum_vol_ratio",
    "order_flow_imbalance",
    "order_flow_pressure",
    "news_sentiment",
    "realtime_confidence",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _vectorise(features: Dict[str, Any]) -> np.ndarray:
    row = []
    for name in MODEL_FEATURES:
        value = features.get(name)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            row.append(0.0)
        else:
            row.append(float(value))
    return np.asarray(row, dtype=float)


def _model_path(asset: str) -> Path:
    return MODEL_DIR / f"{asset.upper()}_gbm.pkl"


def predict_signal_probability(asset: str, features: Dict[str, Any]) -> Optional[float]:
    """Returns the model probability (0..1) for the provided snapshot.

    Parameters
    ----------
    asset: str
        Instrument identifier.  Used to locate the persisted model artifact.
    features: Dict[str, Any]
        Feature dictionary; missing keys are imputed with zero.

    Returns
    -------
    Optional[float]
        ``None`` if no trained model is available, otherwise the probability
        that the analysed setup will finish in profit.
    """
    
    if load is None or _SKLEARN_IMPORT_ERROR is not None:
        global _JOBLIB_WARNING_EMITTED, _SKLEARN_WARNING_EMITTED
        if _JOBLIB_IMPORT_ERROR is not None and not _JOBLIB_WARNING_EMITTED:
            warnings.warn(
                "joblib is not available; skipping probability prediction. "
                "Install joblib to enable ML scoring.",
                RuntimeWarning,
            )
            _JOBLIB_WARNING_EMITTED = True
        if _SKLEARN_IMPORT_ERROR is not None and not _SKLEARN_WARNING_EMITTED:
            warnings.warn(
                "scikit-learn is not available; skipping probability prediction. "
                "Install scikit-learn to enable ML scoring.",
                RuntimeWarning,
            )
            _SKLEARN_WARNING_EMITTED = True
        return None
    
    model_file = _model_path(asset)
    if not model_file.exists():
        return None

    clf = load(model_file)
    if not isinstance(clf, GradientBoostingClassifier):
        # The artifact exists but is not a GBM; avoid crashing the analysis and
        # signal that the model should be rebuilt.
        return None

    vector = _vectorise(features).reshape(1, -1)
    try:
        proba = clf.predict_proba(vector)
    except Exception:
        return None
    if isinstance(proba, Iterable):
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])
        if arr.ndim == 1 and arr.size:
            return float(arr[-1])
    return None


def log_feature_snapshot(asset: str, features: Dict[str, Any]) -> Path:
    """Appends the feature vector to a CSV file for future labelling.

    The CSV structure matches ``MODEL_FEATURES`` and always contains a header.
    The return value is the path to the written file so that callers can emit a
    debug note if needed.
    """

    FEATURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_LOG_DIR / f"{asset.upper()}_features.csv"
    vector = _vectorise(features)
    df = pd.DataFrame([vector], columns=MODEL_FEATURES)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def export_feature_schema() -> Path:
    """Writes the canonical feature schema into ``public/ml_features/schema.json``."""

    FEATURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_LOG_DIR / "schema.json"
    payload = {
        "features": MODEL_FEATURES,
        "notes": "Each row corresponds to a single analysis snapshot."
        " Use the columns to assemble supervised labels offline.",
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


__all__ = [
    "MODEL_FEATURES",
    "predict_signal_probability",
    "log_feature_snapshot",
    "export_feature_schema",
]
