from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(Path(__file__).resolve().parent.parent))

import ml_model


@pytest.fixture(autouse=True)
def restore_model_dir(monkeypatch):
    original_dir = ml_model.MODEL_DIR
    monkeypatch.setattr(ml_model, "MODEL_DIR", original_dir)
    yield


def test_inspect_model_artifact_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    result = ml_model.inspect_model_artifact("BTCUSD")
    assert result["status"] == "missing"
    assert Path(result["path"]).name == "BTCUSD_gbm.pkl"


def test_inspect_model_artifact_dependency_issue(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    path = tmp_path / "BTCUSD_gbm.pkl"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(ml_model, "load", None)
    monkeypatch.setattr(ml_model, "_SKLEARN_IMPORT_ERROR", ImportError("sklearn missing"))

    result = ml_model.inspect_model_artifact("BTCUSD")
    assert result["status"] == "dependency_unavailable"
    assert "sklearn missing" in result.get("detail", "")


def test_inspect_model_artifact_load_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    path = tmp_path / "BTCUSD_gbm.pkl"
    path.write_bytes(b"x")

    result = ml_model.inspect_model_artifact("BTCUSD")
    assert result["status"] == "load_error"
    assert "size_warning" in result


def test_inspect_model_artifact_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    from joblib import dump
    from sklearn.ensemble import GradientBoostingClassifier

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = GradientBoostingClassifier(random_state=1)
    clf.fit(X, y)

    path = tmp_path / "BTCUSD_gbm.pkl"
    dump(clf, path)

    result = ml_model.inspect_model_artifact("BTCUSD")
    assert result["status"] == "ok"
    assert result["model_class"] == "GradientBoostingClassifier"
    assert result["size_bytes"] == path.stat().st_size
