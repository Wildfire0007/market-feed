import pandas as pd

import analysis


def _classify(monkeypatch, adx, slope_abs=0.0):
    monkeypatch.setattr(analysis, "latest_adx", lambda *args, **kwargs: adx)
    monkeypatch.setattr(
        analysis,
        "ema_slope_ok",
        lambda *args, **kwargs: (slope_abs >= 0.015, slope_abs, None),
    )
    frame = pd.DataFrame({"close": [1.0]})
    return analysis.RegimeClassifier().classify(frame, frame)["label"]


def test_regime_classifier_strong_adx_trending_without_ema_slope(monkeypatch):
    assert _classify(monkeypatch, 34.0, 0.0) == "TRENDING"
    assert _classify(monkeypatch, 41.0, 0.0) == "TRENDING"


def test_regime_classifier_missing_adx_unknown(monkeypatch):
    assert _classify(monkeypatch, None, 0.0) == "UNKNOWN"


def test_regime_classifier_range_and_choppy_bands(monkeypatch):
    assert _classify(monkeypatch, 15.0, 0.0) == "RANGING"
    assert _classify(monkeypatch, 21.0, 0.0) == "CHOPPY"


def test_unknown_regime_soft_penalty_skips_choppy_hard_block():
    source = open(analysis.__file__, encoding="utf-8").read()
    assert 'regime_label != "unknown"' in source
    assert "Regime: ADX nem elérhető — soft büntetés, hard block kihagyva" in source

