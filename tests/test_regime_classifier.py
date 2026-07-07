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


def test_regime_classifier_range_and_choppy_bands(monkeypatch):
    assert _classify(monkeypatch, 15.0, 0.0) == "RANGING"
    assert _classify(monkeypatch, 21.0, 0.0) == "CHOPPY"
