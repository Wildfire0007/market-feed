#!/usr/bin/env python3
"""Train gradient boosting models and emit calibration manifests.

This utility stitches together the feature snapshots exported by ``analysis.py``
with manually curated trade outcomes.  The resulting gradient boosting model and
calibration metadata live under ``public/models`` so the live pipeline can load
probabilities without bundling large binaries in the repository.

A minimal workflow looks like:

1. Run the analysis pipeline to populate ``public/ml_features/<ASSET>_features.csv``.
2. Label each row offline (e.g. ``1`` for profitable, ``0`` for invalidated trades)
   and save the result as ``<ASSET>_labelled.csv`` in the same directory.
3. Execute ``scripts/train_models.py --asset EURUSD --dataset
   public/ml_features/EURUSD_labelled.csv`` to produce the model artefacts.

Calibration and decision thresholds are derived from a hold-out split of the
labelled data so the live system can apply Platt or isotonic scaling alongside
asset specific decision boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_model import FEATURE_LOG_DIR, MODEL_DIR, MODEL_FEATURES  # noqa: E402

DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_STRATEGY = "f1"
DEFAULT_CALIBRATION = "platt"


@dataclass
class Dataset:
    """Container for the assembled training dataset.

    Attributes
    ----------
    features:
        Normalised feature matrix restricted to :data:`MODEL_FEATURES`.
    labels:
        Binary targets derived from the label column.
    weights:
        Optional per-sample weights if a ``--weight-column`` was provided.
    missing_features:
        Sorted list of required feature columns that were absent from the
        original CSV input before zero filling.
    """

    features: pd.DataFrame
    labels: np.ndarray
    weights: Optional[np.ndarray]
    missing_features: List[str] = field(default_factory=list)


def _logit(value: float, epsilon: float = 1e-6) -> float:
    clipped = min(max(float(value), epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gradient boosting models")
    parser.add_argument(
        "--asset",
        required=True,
        help="Asset symbol (e.g. EURUSD, BTCUSD)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        dest="datasets",
        help=(
            "Path to a labelled CSV file. The file must contain the feature "
            "columns used by analysis.py and a binary label column. Repeat the "
            "flag to concatenate multiple files."
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column name containing binary trade outcomes (default: label)",
    )
    parser.add_argument(
        "--weight-column",
        help="Optional column name containing per-sample weights",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Directory to store trained models (default: public/models)",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["platt", "isotonic", "none"],
        default=DEFAULT_CALIBRATION,
        help="Calibration strategy for the validation split",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "balanced_accuracy", "youden", "none"],
        default=DEFAULT_STRATEGY,
        help="Optimisation criterion for decision thresholds",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction reserved for calibration/threshold estimation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed for the train/calibration split",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of boosting stages",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for gradient boosting",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of individual regression estimators",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample fraction for stochastic gradient boosting",
    )
    parser.add_argument(
        "--export-schema",
        action="store_true",
        help="Write the canonical feature schema alongside the model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not persist artefacts; useful for sanity checks",
    )
    return parser.parse_args(argv)


def _default_dataset(asset: str) -> Optional[Path]:
    candidate = FEATURE_LOG_DIR / f"{asset.upper()}_labelled.csv"
    return candidate if candidate.exists() else None


def _normalise_labels(series: pd.Series) -> np.ndarray:
    values = series.replace({True: 1, False: 0}).to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        values = np.where(finite_mask, values, 0.0)
    unique = np.unique(values)
    if set(unique).issubset({0.0, 1.0}):
        return values.astype(int)
    if set(unique).issubset({-1.0, 0.0, 1.0}):
        return (values > 0).astype(int)
    raise ValueError("Label column must be binary (0/1 or -1/1)")


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    working = df.copy()
    missing = [feature for feature in MODEL_FEATURES if feature not in working.columns]
    for feature in missing:
        working[feature] = 0.0
    numeric = working[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric.astype(float), missing


def load_dataset(paths: Iterable[Path], label_column: str, weight_column: str | None) -> Dataset:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No datasets supplied; use --dataset or create a labelled CSV")
    df = pd.concat(frames, ignore_index=True)

    if label_column not in df.columns:
        raise KeyError(f"Column '{label_column}' not present in dataset")

    labels = _normalise_labels(df[label_column])
    features, missing_features = _prepare_features(df)

    weights: Optional[np.ndarray] = None
    if weight_column:
        if weight_column not in df.columns:
            raise KeyError(f"Weight column '{weight_column}' missing from dataset")
        weights_series = pd.to_numeric(df[weight_column], errors="coerce").fillna(1.0)
        weights = weights_series.to_numpy(dtype=float)
        weights = np.clip(weights, a_min=1e-6, a_max=None)

    if missing_features:
        preview = ", ".join(missing_features[:6])
        if len(missing_features) > 6:
            preview += ", …"
        print(
            "Filling missing feature columns with zeros: "
            f"{preview} (total: {len(missing_features)})"
        )

    return Dataset(
        features=features,
        labels=labels,
        weights=weights,
        missing_features=missing_features,
    )


def _train_model(
    dataset: Dataset,
    args: argparse.Namespace,
) -> Tuple[GradientBoostingClassifier, pd.DataFrame, pd.DataFrame]:
    if args.validation_split < 0 or args.validation_split >= 1:
        raise ValueError("validation_split must be in [0, 1)")

    total_rows = len(dataset.features)
    if total_rows < 10:
        raise ValueError("Not enough samples; at least 10 rows required for training")

    if args.validation_split > 0:
        from sklearn.model_selection import train_test_split

        split_inputs: List[pd.DataFrame | np.ndarray] = [
            dataset.features,
            dataset.labels,
        ]
        if dataset.weights is not None:
            split_inputs.append(dataset.weights)

        split_outputs = train_test_split(
            *split_inputs,
            test_size=args.validation_split,
            random_state=args.random_state,
            stratify=dataset.labels if len(np.unique(dataset.labels)) > 1 else None,
        )

        X_train = split_outputs[0]
        X_val = split_outputs[1]
        y_train = split_outputs[2]
        y_val = split_outputs[3]
        if dataset.weights is not None:
            w_train = split_outputs[4]
            w_val = split_outputs[5]
        else:
            w_train = w_val = None
    else:
        X_train = dataset.features
        y_train = dataset.labels
        X_val = dataset.features
        y_val = dataset.labels
        w_train = w_val = dataset.weights

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    calibration_frame = X_val.copy()
    calibration_frame["label"] = y_val
    if w_val is not None:
        calibration_frame["weight"] = w_val

    training_frame = X_train.copy()
    training_frame["label"] = y_train
    if w_train is not None:
        training_frame["weight"] = w_train

    return clf, training_frame, calibration_frame


def _weighted_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray],
) -> Tuple[float, float, float, float]:
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    else:
        weights = weights.astype(float)
    tp = float(weights[(y_true == 1) & (y_pred == 1)].sum())
    tn = float(weights[(y_true == 0) & (y_pred == 0)].sum())
    fp = float(weights[(y_true == 0) & (y_pred == 1)].sum())
    fn = float(weights[(y_true == 1) & (y_pred == 0)].sum())
    return tp, fp, tn, fn


def _score_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray],
    strategy: str,
) -> float:
    tp, fp, tn, fn = _weighted_confusion(y_true, y_pred, weights)
    if strategy == "f1":
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else (2 * tp) / denom
    if strategy == "balanced_accuracy":
        tpr_denom = tp + fn
        tnr_denom = tn + fp
        tpr = 0.0 if tpr_denom == 0 else tp / tpr_denom
        tnr = 0.0 if tnr_denom == 0 else tn / tnr_denom
        return 0.5 * (tpr + tnr)
    if strategy == "youden":
        tpr_denom = tp + fn
        fpr_denom = fp + tn
        tpr = 0.0 if tpr_denom == 0 else tp / tpr_denom
        fpr = 0.0 if fpr_denom == 0 else fp / fpr_denom
        return tpr - fpr
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def _optimise_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray],
    strategy: str,
) -> Optional[float]:
    if strategy == "none":
        return None
    if len(probabilities) == 0 or len(np.unique(labels)) < 2:
        return None
    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    if candidates.size > 512:
        candidates = np.linspace(0.05, 0.95, 181)
    best_score = float("-inf")
    best_threshold = None
    for threshold in candidates:
        preds = (probabilities >= threshold).astype(int)
        score = _score_threshold(labels, preds, weights, strategy)
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _calibration_points(
    labels: np.ndarray,
    weights: Optional[np.ndarray],
    probs: np.ndarray,
    method: str,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    if method == "none":
        return None, {"method": "none", "applied": False}
    if len(np.unique(labels)) < 2:
        return None, {"method": method, "applied": False, "reason": "single_class"}

    meta: Dict[str, object] = {"method": method, "applied": True}

    if method == "platt":
        logits = np.array([_logit(p) for p in probs])
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(logits.reshape(-1, 1), labels, sample_weight=weights)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        config = {
            "method": "platt",
            "parameters": {"slope": slope, "intercept": intercept, "domain": "logit"},
        }
        return config, meta

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, labels, sample_weight=weights)
        points = [
            [float(x), float(y)]
            for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)
        ]
        config = {"method": "isotonic", "points": points}
        return config, meta

    raise ValueError(f"Unsupported calibration method: {method}")


def _evaluation_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    weights: Optional[np.ndarray],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "num_samples": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
    }
    if labels.size and len(np.unique(labels)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs, sample_weight=weights))
        except ValueError:
            pass
    if labels.size:
        try:
            metrics["brier_score"] = float(brier_score_loss(labels, probs, sample_weight=weights))
        except ValueError:
            pass
    return metrics


def _thresholds_from_frame(
    frame: pd.DataFrame,
    probs: np.ndarray,
    strategy: str,
) -> Dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    weights = frame["weight"].to_numpy(dtype=float) if "weight" in frame.columns else None
    thresholds: Dict[str, float] = {}

    default = _optimise_threshold(probs, labels, weights, strategy)
    if default is not None:
        thresholds["default"] = default

    bias_long = (
        frame["bias_long"] if "bias_long" in frame.columns else pd.Series(0.0, index=frame.index)
    )
    bias_short = (
        frame["bias_short"] if "bias_short" in frame.columns else pd.Series(0.0, index=frame.index)
    )
    long_mask = bias_long > bias_short
    short_mask = bias_short > bias_long

    if long_mask.any():
        long_probs = probs[long_mask.to_numpy()]
        long_labels = labels[long_mask.to_numpy()]
        long_weights = weights[long_mask.to_numpy()] if weights is not None else None
        long_threshold = _optimise_threshold(long_probs, long_labels, long_weights, strategy)
        if long_threshold is not None:
            thresholds["long"] = long_threshold

    if short_mask.any():
        short_probs = probs[short_mask.to_numpy()]
        short_labels = labels[short_mask.to_numpy()]
        short_weights = weights[short_mask.to_numpy()] if weights is not None else None
        short_threshold = _optimise_threshold(short_probs, short_labels, short_weights, strategy)
        if short_threshold is not None:
            thresholds["short"] = short_threshold

    return thresholds


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def persist_model(asset: str, clf: GradientBoostingClassifier, model_dir: Path, dry_run: bool) -> Path:
    ensure_directory(model_dir)
    target = model_dir / f"{asset.upper()}_gbm.pkl"
    if dry_run:
        print(f"DRY-RUN: would save model to {target}")
        return target

    from joblib import dump

    dump(clf, target)
    print(f"Saved model → {target}")
    return target


def persist_calibration(asset: str, payload: Dict[str, object], model_dir: Path, dry_run: bool) -> Path:
    ensure_directory(model_dir)
    target = model_dir / f"{asset.upper()}_calibration.json"
    if dry_run:
        print(f"DRY-RUN: would write calibration manifest to {target}")
        return target
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Saved calibration → {target}")
    return target


def maybe_export_schema(dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN: would refresh public/ml_features/schema.json")
        return
    from ml_model import export_feature_schema

    path = export_feature_schema()
    print(f"Exported feature schema → {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    asset = args.asset.upper()

    dataset_paths: List[Path] = []
    if args.datasets:
        dataset_paths = [Path(path) for path in args.datasets]
    else:
        default_path = _default_dataset(asset)
        if default_path is not None:
            dataset_paths = [default_path]
            print(f"Using default dataset: {default_path}")
    if not dataset_paths:
        raise SystemExit(
            "No datasets supplied. Provide --dataset or create public/ml_features/"
            f"{asset}_labelled.csv"
        )

    dataset = load_dataset(dataset_paths, args.label_column, args.weight_column)

    print(f"Loaded {len(dataset.features)} rows for asset {asset}")

    clf, train_frame, calibration_frame = _train_model(dataset, args)
    print(
        f"Training rows: {len(train_frame)}, calibration rows: {len(calibration_frame)}"
    )

    train_probs = clf.predict_proba(train_frame[MODEL_FEATURES])[:, 1]
    calib_probs = clf.predict_proba(calibration_frame[MODEL_FEATURES])[:, 1]

    train_labels = train_frame["label"].to_numpy()
    train_weights = (
        train_frame["weight"].to_numpy() if "weight" in train_frame else None
    )
    calib_labels = calibration_frame["label"].to_numpy()
    calib_weights = (
        calibration_frame["weight"].to_numpy() if "weight" in calibration_frame else None
    )

    train_metrics = _evaluation_metrics(train_labels, train_probs, train_weights)
    calib_metrics = _evaluation_metrics(calib_labels, calib_probs, calib_weights)

    print("Training metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value}")

    print("Calibration metrics:")
    for key, value in calib_metrics.items():
        print(f"  {key}: {value}")

    calibration_config: Dict[str, object] = {}
    if args.calibration_method != "none":
        config, calibration_meta = _calibration_points(
            calib_labels,
            calib_weights,
            calib_probs,
            args.calibration_method,
        )
        calibration_config.update(calibration_meta)
        if calibration_meta.get("applied"):
            print(f"Applied {args.calibration_method} calibration")
        else:
            reason = calibration_meta.get("reason", "insufficient data")
            print(f"Skipped calibration ({reason})")
        if config:
            calibration_config.update(config)
    else:
        calibration_config = {"method": "none", "applied": False}
        print("Calibration disabled (--calibration-method none)")

    thresholds = _thresholds_from_frame(calibration_frame, calib_probs, args.threshold_strategy)
    if thresholds:
        calibration_config["thresholds"] = thresholds
        print("Derived thresholds:")
        for name, value in thresholds.items():
            print(f"  {name}: {value}")

    calibration_config.setdefault("metrics", {})
    calibration_config["metrics"].update({
        "train": train_metrics,
        "calibration": calib_metrics,
    })

    model_dir = Path(args.model_dir).expanduser()
    persist_model(asset, clf, model_dir, args.dry_run)
    if calibration_config:
        persist_calibration(asset, calibration_config, model_dir, args.dry_run)

    if args.export_schema:
        maybe_export_schema(args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
