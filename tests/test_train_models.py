"""Unit tests for the training utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import train_models


def _write_csv(path, data):
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_dataset_fills_missing_features_and_weights(tmp_path):
    path = tmp_path / "weighted.csv"
    _write_csv(
        path,
        {
            "label": [1, 0, 1],
            "weight": [0.5, "", -3],
            "p_score": [60, 45, 55],
            "rel_atr": [0.2, 0.3, 0.1],
            "bias_long": [1.0, 0.0, 0.0],
        },
    )

    dataset = train_models.load_dataset([path], "label", "weight")

    assert dataset.features.shape == (3, len(train_models.MODEL_FEATURES))
    present = {"p_score", "rel_atr", "bias_long"}
    expected_missing = [
        feature for feature in train_models.MODEL_FEATURES if feature not in present
    ]
    assert dataset.missing_features == expected_missing

    for feature in expected_missing:
        assert np.allclose(dataset.features[feature].to_numpy(), 0.0)
    assert dataset.features["bias_long"].tolist() == [1.0, 0.0, 0.0]

    assert dataset.labels.tolist() == [1, 0, 1]
    assert dataset.weights is not None
    assert dataset.weights[0] == pytest.approx(0.5)
    assert dataset.weights[1] == pytest.approx(1.0)
    assert dataset.weights[2] == pytest.approx(1e-6)


def test_load_dataset_without_weight_column(tmp_path):
    path = tmp_path / "unweighted.csv"
    _write_csv(
        path,
        {
            "label": [1, 0],
            "p_score": [52, 47],
        },
    )

    dataset = train_models.load_dataset([path], "label", None)

    assert dataset.weights is None
    assert dataset.missing_features
    assert set(dataset.features.columns) == set(train_models.MODEL_FEATURES)


def test_load_dataset_raises_for_missing_weight_column(tmp_path):
    path = tmp_path / "missing_weight.csv"
    _write_csv(
        path,
        {
            "label": [1],
            "p_score": [60],
        },
    )

    with pytest.raises(KeyError):
        train_models.load_dataset([path], "label", "weight")
