from analysis import _sniper_min_p_score_for_asset, SNIPER_MIN_P_SCORE


def test_sniper_min_p_score_has_metals_overrides():
    assert _sniper_min_p_score_for_asset("GOLD_CFD") == 30.0
    assert _sniper_min_p_score_for_asset("XAGUSD") == 28.0


def test_sniper_min_p_score_falls_back_to_default():
    assert _sniper_min_p_score_for_asset("BTCUSD") == SNIPER_MIN_P_SCORE
