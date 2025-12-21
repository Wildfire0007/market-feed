"""Generate deterministic GradientBoosting models for configured assets.

This helper provides loadable ML artefacts in environments where real labelled
trade data is unavailable. It builds a synthetic training set covering the
canonical MODEL_FEATURES and fits a small GradientBoostingClassifier for each
asset declared in the analysis settings.

The resulting pickles live under ``public/models/<ASSET>_gbm.pkl`` and are
suitable for pipeline sanity checks and local development. They are not meant
for production trading decisions.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.analysis_settings import ASSETS
from ml_model import MODEL_DIR, MODEL_FEATURES


def _seed_for_asset(asset: str) -> int:
    """Return a reproducible integer seed for the given asset symbol."""

    digest = hashlib.sha256(asset.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _synthetic_dataset(asset: str, *, rows: int = 400) -> pd.DataFrame:
    """Create a deterministic synthetic dataset covering MODEL_FEATURES."""

    rng = np.random.default_rng(_seed_for_asset(asset))
    data = rng.normal(loc=0.0, scale=1.0, size=(rows, len(MODEL_FEATURES)))
    frame = pd.DataFrame(data, columns=MODEL_FEATURES)

    # Generate a mildly informative label using a few stable features so the
    # model learns non-trivial structure without relying on real trades.
    signal = (
        0.4 * frame[MODEL_FEATURES[0]]
        - 0.25 * frame[MODEL_FEATURES[1]]
        + 0.15 * frame[MODEL_FEATURES[2]]
    )
    logits = signal + rng.normal(scale=0.5, size=len(frame))
    frame["label"] = (logits > 0).astype(int)
    return frame


def _train_model(asset: str, dataset: pd.DataFrame) -> GradientBoostingClassifier:
    """Fit a small GradientBoostingClassifier on the provided dataset."""

    clf = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        random_state=_seed_for_asset(asset),
    )
    clf.fit(dataset[MODEL_FEATURES], dataset["label"])
    return clf


def _persist_model(asset: str, clf: GradientBoostingClassifier) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{asset}_gbm.pkl"
    joblib.dump(clf, path)
    return path


def refresh_models(assets: Iterable[str]) -> None:
    for asset in assets:
        dataset = _synthetic_dataset(asset)
        clf = _train_model(asset, dataset)
        path = _persist_model(asset, clf)
        print(f"Saved synthetic model for {asset} → {path}")


if __name__ == "__main__":
    refresh_models(ASSETS)
