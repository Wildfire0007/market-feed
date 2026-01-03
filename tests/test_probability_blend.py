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
 
