import analysis
import pytest


def test_combine_probability_ignores_ml_when_disabled():
    base = 60.0
    ml_prob = 0.1  # would drag score down if applied

    combined = analysis._combine_probability(base, ml_prob, ml_enabled=False)

    assert combined == base / 100.0


def test_combine_probability_blends_when_enabled():
    base = 50.0
    ml_prob = 0.8

    combined = analysis._combine_probability(base, ml_prob, ml_enabled=True)

    assert combined == pytest.approx(0.6 * (base / 100.0) + 0.4 * ml_prob)
 


def test_load_adaptive_params_returns_empty_when_disabled(monkeypatch, tmp_path, caplog):
    analysis._load_adaptive_params.cache_clear()
    monkeypatch.setitem(analysis.SETTINGS, "adaptive_params_enabled", False)
    monkeypatch.setattr(analysis, "ADAPTIVE_PARAMS_PATH", tmp_path / "adaptive_params.json")
    analysis.ADAPTIVE_PARAMS_PATH.write_text('{"XAGUSD":{"atr_floor":1}}', encoding="utf-8")
    caplog.set_level("INFO", logger=analysis.LOGGER.name)

    assert analysis._load_adaptive_params() == {}
    assert "adaptive params disabled — measurement freeze" in caplog.text
    analysis._load_adaptive_params.cache_clear()
