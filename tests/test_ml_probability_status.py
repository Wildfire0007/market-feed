from pathlib import Path

import pytest

import ml_model


@pytest.fixture()
def temp_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_model_path(asset: str) -> Path:
        return tmp_path / f"{asset}_gbm.pkl"

    monkeypatch.setattr(ml_model, "_model_path", fake_model_path)
    monkeypatch.setattr(ml_model, "_load_stack_config", lambda *_: None)
    monkeypatch.setattr(ml_model, "_load_calibration", lambda *_: None)
    return tmp_path


def test_predict_signal_probability_marks_fallback_status(temp_model_dir: Path) -> None:
    prediction = ml_model.predict_signal_probability("EURUSD", {"p_score": 55.0})

    assert prediction.metadata.get("status") == "fallback"
    assert prediction.metadata.get("source") == "fallback"
    assert prediction.metadata.get("unavailable_reason") == "model_missing"
    assert prediction.metadata.get("fallback") is not None


def test_predict_signal_probability_marks_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_model, "_model_path", lambda *_: Path("/nonexistent/model.pkl"))
    monkeypatch.setattr(ml_model, "_load_stack_config", lambda *_: None)
    monkeypatch.setattr(ml_model, "_load_calibration", lambda *_: None)
    monkeypatch.setattr(ml_model, "_fallback_probability", lambda *_, **__: None)

    prediction = ml_model.predict_signal_probability("BTCUSD", {"p_score": 48.0})

    assert prediction.metadata.get("status") == "unavailable"
    assert prediction.metadata.get("source") == "unavailable"
    assert prediction.metadata.get("unavailable_reason") == "model_missing"


def test_predict_signal_probability_enabled_status(temp_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = temp_model_dir / "EURUSD_gbm.pkl"
    model_path.write_bytes(b"stub")

    class DummyModel:
        def predict_proba(self, _vector):
            return [[0.4, 0.6]]

    monkeypatch.setattr(ml_model, "GradientBoostingClassifier", DummyModel)
    monkeypatch.setattr(ml_model, "load", lambda *_: DummyModel())

    prediction = ml_model.predict_signal_probability("EURUSD", {"p_score": 60.0})

    assert prediction.metadata.get("status") == "enabled"
    assert prediction.metadata.get("source") == "sklearn"
    assert prediction.probability is not None
