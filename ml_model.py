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

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    "order_flow_aggressor",
    "news_sentiment",
    "news_event_severity",
    "realtime_confidence",
    "volatility_ratio",
    "volatility_regime_flag",
    "precision_score",
    "precision_trigger_ready",
    "precision_trigger_arming",
    "precision_trigger_fire",
    "precision_trigger_confidence",
    "precision_order_flow_ready",
    "structure_flip_flag",
    "momentum_trail_activation_rr",
    "momentum_trail_lock_ratio",
    "momentum_trail_price",
]


@dataclass
class ProbabilityPrediction:
    """Container describing the calibrated probability output.

    Attributes
    ----------
    probability:
        Final calibrated probability after stacking and calibration.  ``None``
        when no model artefact is present.
    raw_probability:
        Direct model probability prior to stacking/calibration.  Useful for
        debugging and drift monitoring.
    threshold:
        Asset specific decision boundary, if available from the calibration
        manifest.
    metadata:
        Auxiliary diagnostics about the scoring pipeline (e.g. applied
        calibration method).  The structure is intentionally loose so callers
        can simply serialise it alongside analysis diagnostics.
    """

    probability: Optional[float] = None
    raw_probability: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

def _stack_config_path(asset: str) -> Path:
    return MODEL_DIR / f"{asset.upper()}_stack.json"


def _calibration_path(asset: str) -> Path:
    return MODEL_DIR / f"{asset.upper()}_calibration.json"


def missing_model_artifacts(assets: Iterable[str]) -> Dict[str, Path]:
    """Return a mapping of assets without an on-disk gradient boosting model."""

    missing: Dict[str, Path] = {}
    for asset in assets:
        symbol = str(asset).upper()
        path = _model_path(symbol)
        if not path.exists():
            missing[symbol] = path
    return missing
    

def _load_stack_config(asset: str) -> Optional[Dict[str, Any]]:
    path = _stack_config_path(asset)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_calibration(asset: str) -> Optional[Dict[str, Any]]:
    path = _calibration_path(asset)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _logit(value: float, epsilon: float = 1e-6) -> float:
    clipped = min(max(value, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return float(value > 0)


def _piecewise_linear(x: float, points: List[Tuple[float, float]]) -> float:
    if not points:
        return x
    points = sorted(points, key=lambda item: item[0])
    xs, ys = zip(*points)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return ys[-1]


def _apply_calibration(
    probability: Optional[float],
    raw_probability: Optional[float],
    features: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    if config is None:
        return probability, {}

    method = str(config.get("method", "platt")).lower()
    base = probability if probability is not None else raw_probability
    if base is None:
        return probability, {"method": method, "applied": False}

    calibrated = base
    details: Dict[str, Any] = {"method": method, "applied": True}

    if method in {"platt", "logistic"}:
        params = config.get("parameters") or {}
        slope = float(params.get("slope", 1.0) or 1.0)
        intercept = float(params.get("intercept", 0.0) or 0.0)
        domain = str(params.get("domain", "probability")).lower()
        details["parameters"] = {"slope": slope, "intercept": intercept, "domain": domain}

        if domain == "logit":
            transformed = slope * _logit(base) + intercept
        else:
            transformed = slope * base + intercept
        calibrated = _sigmoid(transformed)

        blend = float(config.get("blend", 1.0) or 1.0)
        if 0.0 <= blend < 1.0:
            calibrated = blend * calibrated + (1.0 - blend) * base
            details["blend"] = blend

    elif method in {"isotonic", "piecewise"}:
        raw_points = (
            config.get("points")
            or config.get("table")
            or config.get("pairs")
            or config.get("mapping")
        )
        points: List[Tuple[float, float]] = []
        if isinstance(raw_points, list):
            for item in raw_points:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    x_val = float(item[0])
                    y_val = float(item[1])
                except (TypeError, ValueError):
                    continue
                points.append((x_val, min(max(y_val, 0.0), 1.0)))
        calibrated = _piecewise_linear(base, points) if points else base
        details["points"] = points
    else:
        details["applied"] = False
        return probability, details

    calibrated = min(max(calibrated, 0.0), 1.0)
    return calibrated, details


def _determine_threshold(
    config: Optional[Dict[str, Any]], features: Dict[str, Any]
) -> Tuple[Optional[float], Dict[str, Any]]:
    if config is None:
        return None, {}
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        return None, {}

    meta: Dict[str, Any] = {"available": thresholds}
    try:
        threshold = float(thresholds.get("default"))
    except (TypeError, ValueError):
        threshold = None
    if threshold is None:
        return None, meta

    selection = "default"
    bias_long = float(features.get("bias_long", 0.0) or 0.0)
    bias_short = float(features.get("bias_short", 0.0) or 0.0)
    if bias_long > bias_short:
        value = thresholds.get("long")
        if value is not None:
            try:
                threshold = float(value)
                selection = "long"
            except (TypeError, ValueError):
                pass
    elif bias_short > bias_long:
        value = thresholds.get("short")
        if value is not None:
            try:
                threshold = float(value)
                selection = "short"
            except (TypeError, ValueError):
                pass

    overrides_meta: List[Dict[str, Any]] = []
    overrides = config.get("feature_thresholds")
    if isinstance(overrides, list):
        for override in overrides:
            if not isinstance(override, dict):
                continue
            feature = override.get("feature")
            if not feature:
                continue
            value = features.get(feature)
            if value is None:
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            min_val = override.get("min")
            max_val = override.get("max")
            if min_val is not None:
                try:
                    if value_float < float(min_val):
                        continue
                except (TypeError, ValueError):
                    continue
            if max_val is not None:
                try:
                    if value_float > float(max_val):
                        continue
                except (TypeError, ValueError):
                    continue
            try:
                threshold = float(override.get("threshold", threshold))
            except (TypeError, ValueError):
                continue
            overrides_meta.append(
                {
                    "feature": feature,
                    "value": value_float,
                    "threshold": threshold,
                }
            )
            selection = f"override:{feature}"
    meta["selection"] = selection
    if overrides_meta:
        meta["overrides"] = overrides_meta
    return threshold, meta


def _apply_stack(
    base_probability: Optional[float],
    features: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Optional[float]:
    if config is None:
        return base_probability
    method = (config.get("method") or "average").lower()
    components = config.get("components") or {}
    defaults = config.get("defaults") or {}

    values: Dict[str, float] = {}
    if base_probability is not None:
        values["gbm"] = float(base_probability)
    for name, meta in components.items():
        if isinstance(meta, dict) and meta.get("source") == "feature":
            feature_name = meta.get("name") or name
            try:
                values[name] = float(features.get(feature_name, defaults.get(name, 0.0)))
            except (TypeError, ValueError):
                values[name] = float(defaults.get(name, 0.0))
        elif isinstance(meta, dict) and meta.get("source") == "constant":
            try:
                values[name] = float(meta.get("value", defaults.get(name, 0.0)))
            except (TypeError, ValueError):
                values[name] = float(defaults.get(name, 0.0))

    if not values:
        return base_probability

    if method == "logistic":
        bias = float(config.get("bias", 0.0))
        weights = config.get("weights") or {}
        z = bias
        for key, weight in weights.items():
            try:
                z += float(weight) * float(values.get(key, defaults.get(key, 0.0)))
            except (TypeError, ValueError):
                continue
        try:
            return 1.0 / (1.0 + np.exp(-z))
        except OverflowError:
            return float(z > 0)

    if method == "weighted_average":
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in (config.get("weights") or {}).items():
            if key not in values:
                continue
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            total_weight += w
            weighted_sum += w * float(values[key])
        if total_weight > 0:
            return weighted_sum / total_weight

    # Fallback simple average
    return sum(values.values()) / len(values)



def predict_signal_probability(asset: str, features: Dict[str, Any]) -> ProbabilityPrediction:
    """Return calibrated probability estimates for an analysis snapshot."""

    metadata: Dict[str, Any] = {
        "asset": asset,
    }
   
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
        metadata["unavailable_reason"] = "sklearn_missing"
        return ProbabilityPrediction(metadata=metadata)
    
    model_file = _model_path(asset)
    if not model_file.exists():
        metadata["unavailable_reason"] = "model_missing"
        return ProbabilityPrediction(metadata=metadata)

    try:
        clf = load(model_file)
    except Exception as exc:  # pragma: no cover - corrupted artefacts
        metadata["unavailable_reason"] = f"model_load_error:{exc}"
        return ProbabilityPrediction(metadata=metadata)

    if not isinstance(clf, GradientBoostingClassifier):
        metadata["unavailable_reason"] = "model_type_mismatch"
        return ProbabilityPrediction(metadata=metadata)

    vector = _vectorise(features).reshape(1, -1)
    try:
        proba = clf.predict_proba(vector)
    except Exception as exc:  # pragma: no cover - scoring edge cases
        metadata["unavailable_reason"] = f"predict_error:{exc}"
        return ProbabilityPrediction(metadata=metadata)
        
    base_probability: Optional[float] = None
    if isinstance(proba, Iterable):
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            base_probability = float(arr[0, 1])
        elif arr.ndim == 1 and arr.size:
            base_probability = float(arr[-1])

    stack_config = _load_stack_config(asset)
    stacked_probability = _apply_stack(base_probability, features, stack_config)

    if base_probability is not None:
        metadata["raw_probability"] = base_probability
    if stacked_probability is not None and stacked_probability != base_probability:
        metadata["stacked_probability"] = stacked_probability

    if stack_config:
        metadata["stack"] = {
            "method": stack_config.get("method", "average"),
            "components": list((stack_config.get("components") or {}).keys()),
        }

    calibration_config = _load_calibration(asset)
    calibrated_probability, calibration_meta = _apply_calibration(
        stacked_probability,
        base_probability,
        features,
        calibration_config,
    )
    if calibration_meta:
        metadata["calibration"] = calibration_meta

    threshold, threshold_meta = _determine_threshold(calibration_config, features)
    if threshold_meta:
        metadata["threshold"] = threshold_meta

    return ProbabilityPrediction(
        probability=calibrated_probability,
        raw_probability=base_probability,
        threshold=threshold,
        metadata=metadata,
    )


def log_feature_snapshot(
    asset: str, features: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
) -> Path:
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

    if metadata:
        meta_path = path.with_suffix(path.suffix + ".meta.jsonl")
        try:
            record = {
                "asset": asset,
                "metadata": metadata,
            }
            with meta_path.open("a", encoding="utf-8") as meta_fh:
                json.dump(record, meta_fh, ensure_ascii=False)
                meta_fh.write("\n")
        except Exception:
            # Metadata logging should never block the analysis pipeline.
            pass
            
    try:
        from reports.feature_monitor import update_feature_drift_report

        try:
            update_feature_drift_report(asset, path, features)
        except Exception:
            # Feature monitor issues should not break the analysis pipeline.
            pass
    except Exception:
        # Optional dependency missing (e.g. module not packaged yet).
        pass
    return path


def export_feature_schema() -> Path:
    """Writes the canonical feature schema into ``public/ml_features/schema.json``."""

    FEATURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_LOG_DIR / "schema.json"
    payload = {
        "features": MODEL_FEATURES,
        "notes": (
            "Each row corresponds to a single analysis snapshot. "
            "Use the columns to assemble supervised labels offline."
        ),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


__all__ = [
    "MODEL_FEATURES",
    "ProbabilityPrediction",
    "predict_signal_probability",
    "log_feature_snapshot",
    "export_feature_schema",
    "missing_model_artifacts",
]
