from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(Path(__file__).resolve().parent.parent))

import ml_model
from scripts import check_ml_readiness


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
    assert result["status"] == "placeholder"
    assert "placeholder" in result.get("detail", "").lower()
    

def test_inspect_model_artifact_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    path = tmp_path / "BTCUSD_gbm.pkl"
    path.write_text("placeholder", encoding="utf-8")

    result = ml_model.inspect_model_artifact("BTCUSD")
    assert result["status"] == "placeholder"
    assert "Placeholder artefact" in result.get("detail", "")


def test_inspect_model_artifact_load_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    path = tmp_path / "BTCUSD_gbm.pkl"
    path.write_bytes(os.urandom(ml_model.PLACEHOLDER_MAX_BYTES + 1))

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


def test_run_diagnostics_placeholder(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ml_model, "MODEL_DIR", tmp_path)
    path = tmp_path / "BTCUSD_gbm.pkl"
    path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(check_ml_readiness, "runtime_dependency_issues", lambda: {})

    exit_code = check_ml_readiness.run_diagnostics(["BTCUSD"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PLACEHOLDER" in captured.out
    assert "Töltsd fel a tényleges" in captured.out


def test_python_version_status_ok(monkeypatch):
    fake_sys = type("FakeSys", (), {"version_info": (3, 10, 4)})
    monkeypatch.setattr(check_ml_readiness, "sys", fake_sys)

    info = check_ml_readiness.python_version_status()
    assert info["status"] == "ok"
    assert info["current"].startswith("3.10")


def test_python_version_status_outdated(monkeypatch):
    fake_sys = type("FakeSys", (), {"version_info": (3, 8, 17)})
    monkeypatch.setattr(check_ml_readiness, "sys", fake_sys)

    info = check_ml_readiness.python_version_status()
    assert info["status"] == "outdated"
    assert info["recommended"].startswith(">= 3.9")
