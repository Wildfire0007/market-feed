from scripts.label_trades import _wilson_interval


def test_wilson_interval_is_deterministic():
    lo, hi = _wilson_interval(3, 5)
    assert round(lo, 4) == 0.2307
    assert round(hi, 4) == 0.8824
