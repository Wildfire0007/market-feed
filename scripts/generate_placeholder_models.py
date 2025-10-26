#!/usr/bin/env python3
"""Bootstrap lightweight gradient boosting models from feature logs.

This helper converts the raw feature CSV exports stored under
``public/ml_features`` into synthetic gradient boosting classifiers.  The
resulting artefacts unblock the ``ml_model.predict_signal_probability`` path so
that the analysis pipeline can serve calibrated probabilities even when manually
labelled training sets are not yet available.

The command intentionally keeps the training logic deterministic and focuses on
simple heuristics driven by ``p_score`` so that we obtain a reasonable class
balance for every asset.  The generated models should be treated as placeholders
until proper, labelled datasets are curated offline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.analysis_settings import ASSETS
from ml_model import FEATURE_LOG_DIR, MODEL_DIR, MODEL_FEATURES


def _extract_feature_matrix(path: Path) -> pd.DataFrame:
    """Return a numeric feature matrix trimmed/padded to ``MODEL_FEATURES``."""

    if not path.exists():
        raise FileNotFoundError(f"Feature log not found: {path}")

    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            values: List[float] = []
            for idx in range(len(MODEL_FEATURES)):
                cell = row[idx] if idx < len(row) else ""
                try:
                    values.append(float(cell))
                except (TypeError, ValueError):
                    values.append(0.0)
            rows.append(values)

    if not rows:
        raise ValueError(f"Feature log {path} did not contain any rows")

    frame = pd.DataFrame(rows, columns=MODEL_FEATURES)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _derive_labels(features: pd.DataFrame) -> np.ndarray:
    """Produce a binary target array using simple ``p_score`` heuristics."""

    series = (features["p_score"].astype(float) >= 55.0).astype(int)
    if series.nunique() >= 2:
        return series.to_numpy()

    quantile = float(features["p_score"].quantile(0.65))
    series = (features["p_score"] >= quantile).astype(int)
    if series.nunique() >= 2:
        return series.to_numpy()

    median = float(features["p_score"].median())
    series = (features["p_score"] >= median).astype(int)
    if series.nunique() >= 2:
        return series.to_numpy()

    rng = np.random.default_rng(seed=13)
    return rng.integers(0, 2, size=len(features))


def _train_placeholder_model(features: pd.DataFrame, labels: np.ndarray) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=13,
    )
    model.fit(features.to_numpy(), labels)
    return model


def _iter_assets(selected: Iterable[str] | None) -> Iterable[str]:
    assets = ASSETS if selected is None else [asset.upper() for asset in selected]
    for asset in assets:
        yield asset.upper()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        help="Optional list of asset symbols to train (defaults to config assets)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for the generated models (default: public/models)",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_LOG_DIR,
        help="Directory containing feature CSV exports (default: public/ml_features)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.model_dir.mkdir(parents=True, exist_ok=True)

    for asset in _iter_assets(args.assets):
        feature_path = args.feature_dir / f"{asset}_features.csv"
        try:
            features = _extract_feature_matrix(feature_path)
            labels = _derive_labels(features)
            model = _train_placeholder_model(features, labels)
        except Exception as exc:  # pragma: no cover - CLI diagnostics
            print(f"[WARN] Skipping {asset}: {exc}")
            continue

        model_path = args.model_dir / f"{asset}_gbm.pkl"
        dump(model, model_path)
        print(f"[OK] Wrote placeholder model for {asset} -> {model_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
