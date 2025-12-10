import importlib
import sys
import types

import pytest


def test_get_low_atr_override_merges_defaults(monkeypatch):
    import config.analysis_settings as settings

    monkeypatch.setattr(
        settings,
        "LOW_ATR_OVERRIDES",
        {"default": {"floor": 0.2, "tp_min_pct": 0.5}, "EURUSD": {"rr_required": 1.1}},
    )

    result = settings.get_low_atr_override("eurusd")

    assert result == {"floor": 0.2, "tp_min_pct": 0.5, "rr_required": 1.1}


@pytest.fixture()
def light_analysis(monkeypatch):
    dummy_dynamic = types.ModuleType("dynamic_logic")
    dummy_dynamic.DynamicScoreEngine = type("DynamicScoreEngine", (), {})
    dummy_dynamic.VolatilityManager = type("VolatilityManager", (), {})
    dummy_dynamic.apply_latency_relaxation = lambda *args, **kwargs: None
    dummy_dynamic.validate_dynamic_logic_config = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dynamic_logic", dummy_dynamic)

    dummy_anchor = types.ModuleType("active_anchor")
    dummy_anchor.load_anchor_state = lambda *args, **kwargs: {}
    dummy_anchor.record_anchor = lambda *args, **kwargs: None
    dummy_anchor.update_anchor_metrics = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "active_anchor", dummy_anchor)

    dummy_ml = types.ModuleType("ml_model")
    dummy_ml.ProbabilityPrediction = type("ProbabilityPrediction", (), {})
    dummy_ml.inspect_model_artifact = lambda *args, **kwargs: None
    dummy_ml.log_feature_snapshot = lambda *args, **kwargs: None
    dummy_ml.missing_model_artifacts = lambda *args, **kwargs: []
    dummy_ml.predict_signal_probability = lambda *args, **kwargs: None
    dummy_ml.runtime_dependency_issues = []
    monkeypatch.setitem(sys.modules, "ml_model", dummy_ml)

    dummy_news = types.ModuleType("news_feed")
    dummy_news.SentimentSignal = type("SentimentSignal", (), {})
    dummy_news.load_sentiment = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "news_feed", dummy_news)

    if "analysis" in sys.modules:
        del sys.modules["analysis"]
    analysis = importlib.import_module("analysis")
    return analysis


def test_tp_min_pct_for_applies_floor_override(light_analysis, monkeypatch):
    analysis = light_analysis

    monkeypatch.setattr(analysis, "get_tp_min_pct_value", lambda asset: 0.0025)
    monkeypatch.setattr(
        analysis,
        "get_low_atr_override",
        lambda asset: {"floor": 0.001, "tp_min_pct": 0.0015},
    )

    assert analysis.tp_min_pct_for("TEST", 0.0009, True) == pytest.approx(0.0015)
    assert analysis.tp_min_pct_for("TEST", 0.002, True) == pytest.approx(0.0025)
