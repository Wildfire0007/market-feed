import importlib
import sys
from pathlib import Path

import pytest


def _reload_settings(monkeypatch, profile=None):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config.analysis_settings as settings

    if profile is None:
        monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    else:
        monkeypatch.setenv("ENTRY_THRESHOLD_PROFILE", profile)
    settings.reload_config()
    return importlib.reload(settings)


def test_default_profile_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch)

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "relaxed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "relaxed"
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(58.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(48.0)
    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(0.95)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.9)

    baseline = settings.describe_entry_threshold_profile("baseline")
    assert baseline["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(60.0)
    assert baseline["atr_threshold_multiplier"]["default"] == pytest.approx(1.0)

    _reload_settings(monkeypatch)


def test_baseline_profile_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="baseline")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "baseline"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "baseline"
    assert profile["p_score_min"]["default"] == pytest.approx(60.0)
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(60.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(52.0)
    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(1.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(1.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(1.0)


def test_relaxed_profile_override(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="relaxed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "relaxed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "relaxed"
    assert profile["p_score_min"]["by_asset"]["GOLD_CFD"] == pytest.approx(60.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(48.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.9)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(1.0)
    # Non-overridden assets fall back to the profile defaults.
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(58.0)

    # You can also inspect another profile without changing the active one.
    baseline = settings.describe_entry_threshold_profile("baseline")
    assert baseline["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(60.0)

    # The helper exposes the list of available profiles for documentation
    # and UI surfaces.
    assert set(settings.list_entry_threshold_profiles()) >= {
        "baseline",
        "relaxed",
        "suppressed",
    }

    # Restore the default profile for subsequent tests.
    _reload_settings(monkeypatch)


def test_intraday_bias_and_atr_overrides(monkeypatch):
    settings = _reload_settings(monkeypatch)

    assert settings.INTRADAY_ATR_RELAX["EURUSD"] == pytest.approx(0.85)
    assert settings.INTRADAY_ATR_RELAX["GOLD_CFD"] == pytest.approx(0.9)
    assert settings.INTRADAY_ATR_RELAX["BTCUSD"] == pytest.approx(0.85)

    eurusd_bias = settings.INTRADAY_BIAS_RELAX["EURUSD"]
    assert eurusd_bias["allow_neutral"] is True
    eurusd_scenarios = {
        (scenario["direction"], tuple(scenario["requires"]))
        for scenario in eurusd_bias["scenarios"]
    }
    assert ("long", ("micro_bos_long", "atr_ok")) in eurusd_scenarios
    assert ("short", ("micro_bos_short", "atr_ok")) in eurusd_scenarios

    nvda_bias = settings.INTRADAY_BIAS_RELAX["NVDA"]
    assert any(
        scenario["direction"] == "long" and "atr_strong" in scenario["requires"]
        for scenario in nvda_bias["scenarios"]
    )
    assert all("label" in scenario for scenario in nvda_bias["scenarios"])

    btc_bias = settings.INTRADAY_BIAS_RELAX["BTCUSD"]
    assert any(
        scenario["direction"] == "long" and "momentum_volume" in scenario["requires"]
        for scenario in btc_bias["scenarios"]
    )

    _reload_settings(monkeypatch)


def test_btc_profile_overrides(monkeypatch):
    baseline_settings = _reload_settings(monkeypatch, profile="baseline")

    assert "BTCUSD" in baseline_settings.ENABLE_MOMENTUM_ASSETS

    overrides = baseline_settings.BTC_PROFILE_OVERRIDES
    assert overrides["baseline"]["atr_floor_usd"] == pytest.approx(150.0)
    assert overrides["baseline"]["tp_min_pct"] == pytest.approx(0.008)
    assert overrides["baseline"]["sl_buffer"]["atr_mult"] == pytest.approx(0.3)
    assert overrides["baseline"]["sl_buffer"]["abs_min"] == pytest.approx(150.0)
    assert overrides["baseline"]["rr"]["trend_core"] == pytest.approx(1.9)
    assert overrides["baseline"]["rr"]["range_core"] == pytest.approx(1.5)
    assert overrides["baseline"]["rr"]["range_momentum"] == pytest.approx(1.5)
    assert overrides["baseline"]["rr"]["range_size_scale"] == pytest.approx(0.6)
    assert overrides["baseline"]["rr"]["range_time_stop"] == 20
    assert overrides["baseline"]["rr"]["range_breakeven"] == pytest.approx(0.35)
    assert overrides["baseline"]["bias_relax"]["ofi_sub_threshold"] == pytest.approx(1.2)
    assert overrides["baseline"]["structure"]["ofi_gate"] == pytest.approx(0.8)
    assert overrides["baseline"]["momentum_override"]["rr_min"] == pytest.approx(1.6)
    assert overrides["baseline"]["momentum_override"]["max_slippage_r"] == pytest.approx(0.25)
    assert overrides["baseline"]["momentum_override"]["no_chase_r"] == pytest.approx(0.25)

    relaxed = overrides["relaxed"]
    assert relaxed["rr"]["range_time_stop"] == 15
    assert relaxed["momentum_override"]["max_slippage_r"] == pytest.approx(0.22)
    assert relaxed["momentum_override"]["no_chase_r"] == pytest.approx(0.35)
    assert relaxed["atr_floor_usd"] == pytest.approx(135.0)
    assert relaxed["atr_percentiles"]["open"] == pytest.approx(0.4)
    assert relaxed["atr_percentiles"]["mid"] == pytest.approx(0.4)
    assert relaxed["tp_min_pct"] == pytest.approx(0.007)
    assert relaxed["sl_buffer"]["atr_mult"] == pytest.approx(0.28)
    assert relaxed["sl_buffer"]["abs_min"] == pytest.approx(135.0)
    assert relaxed["bias_relax"]["vwap_ofi_threshold"] == pytest.approx(1.0)

    suppressed = overrides["suppressed"]
    assert suppressed["rr"]["range_breakeven"] == pytest.approx(0.25)
    assert suppressed["momentum_override"]["no_chase_r"] == pytest.approx(0.32)
    assert suppressed["atr_floor_usd"] == pytest.approx(120.0)
    assert suppressed["atr_percentiles"]["open"] == pytest.approx(0.3)
    assert suppressed["atr_percentiles"]["mid"] == pytest.approx(0.28)
    assert suppressed["tp_min_pct"] == pytest.approx(0.006)
    assert suppressed["sl_buffer"]["atr_mult"] == pytest.approx(0.26)
    assert suppressed["sl_buffer"]["abs_min"] == pytest.approx(120.0)
    assert suppressed["momentum_override"]["rr_min"] == pytest.approx(1.45)
    assert suppressed["range_guard_requires_override"] is True

    _reload_settings(monkeypatch)
