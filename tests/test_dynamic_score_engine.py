import pytest

from dynamic_logic import DynamicScoreEngine


@pytest.mark.parametrize(
    "base_score, regime_points, expected_score",
    [
        (5.0, -10.0, 0.0),
        (25.0, -10.0, 15.0),
    ],
)
def test_dynamic_score_engine_applies_regime_penalty_and_clamps(
    base_score, regime_points, expected_score
):
    engine = DynamicScoreEngine(
        {
            "enabled": True,
            "p_score": {
                "regime_penalty": {
                    "enabled": True,
                    "points": regime_points,
                }
            },
        },
        validate_config=False,
    )

    score, notes, meta = engine.score(
        base_p_score=base_score,
        regime_data={"label": "CHOPPY"},
        volatility_data=None,
    )

    assert score == expected_score
    assert meta["final_score"] == expected_score
+    assert meta["regime_penalty"] == {"label": "CHOPPY", "points": regime_points}
+    assert any("regime CHOPPY" in note for note in notes)
