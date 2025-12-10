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


def test_tp_min_pct_for_applies_floor_override(monkeypatch):
    import analysis

    monkeypatch.setattr(analysis, "get_tp_min_pct_value", lambda asset: 0.0025)
    monkeypatch.setattr(
        analysis,
        "get_low_atr_override",
        lambda asset: {"floor": 0.001, "tp_min_pct": 0.0015},
    )

    assert analysis.tp_min_pct_for("TEST", 0.0009, True) == pytest.approx(0.0015)
    assert analysis.tp_min_pct_for("TEST", 0.002, True) == pytest.approx(0.0025)
