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

    # Alapértelmezett aktív profil most már 'baseline'
    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "baseline"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "baseline"
    assert profile["p_score_min"]["default"] == pytest.approx(50.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(48.0)
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(50.0)
    assert profile["p_score_min"]["by_asset"]["XAGUSD"] == pytest.approx(53.0)
    assert profile["p_score_min"]["by_asset"]["NVDA"] == pytest.approx(48.0)
    assert profile["p_score_min"]["by_asset"]["USOIL"] == pytest.approx(48.0)
    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(0.95)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.9)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.92)

    suppressed = settings.describe_entry_threshold_profile("suppressed")
    assert suppressed["p_score_min"]["default"] == pytest.approx(46.0)
    assert suppressed["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(46.0)
    assert suppressed["atr_threshold_multiplier"]["default"] == pytest.approx(0.88)
    assert suppressed["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.82)

    _reload_settings(monkeypatch)


def test_suppressed_profile_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="suppressed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "suppressed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "suppressed"
    assert profile["p_score_min"]["default"] == pytest.approx(46.0)
    assert profile["p_score_min"]["by_asset"]["USOIL"] == pytest.approx(43.0)
    assert profile["p_score_min"]["by_asset"]["XAGUSD"] == pytest.approx(46.0)
    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(0.88)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.85)


def test_relaxed_profile_override(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="relaxed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "relaxed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "relaxed"

    # RELAXED értékek
    assert profile["p_score_min"]["by_asset"]["GOLD_CFD"] == pytest.approx(48.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(46.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.88)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.88)

    # Nem felülírt eszköz fallback a profil defaultjára (EURUSD -> 50.0)
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(48.0)

    # Baseline ellenőrzés (aktív)
    baseline = settings.describe_entry_threshold_profile("baseline")
    assert baseline["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(50.0)

    # Profilok listája
    assert set(settings.list_entry_threshold_profiles()) >= {
        "baseline",
        "relaxed",
        "suppressed",
    }

    _reload_settings(monkeypatch)


def test_intraday_bias_and_atr_overrides(monkeypatch):
    settings = _reload_settings(monkeypatch)

    # INTRADAY_ATR_RELAX új értékek
    assert settings.INTRADAY_ATR_RELAX["EURUSD"] == pytest.approx(0.75)
    assert settings.INTRADAY_ATR_RELAX["GOLD_CFD"] == pytest.approx(0.80)
    assert settings.INTRADAY_ATR_RELAX["BTCUSD"] == pytest.approx(0.75)

    # Bias relax struktúra és címkék változatlan logikával
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
    assert any(
        scenario["direction"] == "long" and "bos5m_long" in scenario["requires"]
        for scenario in btc_bias["scenarios"]
    )

    _reload_settings(monkeypatch)


def test_btc_profile_overrides(monkeypatch):
    baseline_settings = _reload_settings(monkeypatch, profile="baseline")

    # Momentum eszközök listája tartalmazza a BTCUSD-t
    assert "BTCUSD" in baseline_settings.ENABLE_MOMENTUM_ASSETS

    overrides = baseline_settings.BTC_PROFILE_OVERRIDES

    # BASELINE BTC override-ok (új értékek)
    assert overrides["baseline"]["atr_floor_usd"] == pytest.approx(85.0)
    assert overrides["baseline"]["tp_min_pct"] == pytest.approx(0.007)
    assert overrides["baseline"]["sl_buffer"]["atr_mult"] == pytest.approx(0.28)
    assert overrides["baseline"]["sl_buffer"]["abs_min"] == pytest.approx(90.0)
    assert overrides["baseline"]["rr"]["trend_core"] == pytest.approx(1.8)
    assert overrides["baseline"]["rr"]["range_core"] == pytest.approx(1.5)
    assert overrides["baseline"]["rr"]["range_momentum"] == pytest.approx(1.4)
    assert overrides["baseline"]["rr"]["range_size_scale"] == pytest.approx(0.55)
    assert overrides["baseline"]["rr"]["range_time_stop"] == 24
    assert overrides["baseline"]["rr"]["range_breakeven"] == pytest.approx(0.32)
    assert overrides["baseline"]["bias_relax"]["ofi_sub_threshold"] == pytest.approx(1.1)
    assert overrides["baseline"]["structure"]["ofi_gate"] == pytest.approx(0.8)
    assert overrides["baseline"]["momentum_override"]["rr_min"] == pytest.approx(1.5)
    assert overrides["baseline"]["momentum_override"]["max_slippage_r"] == pytest.approx(0.22)
    assert overrides["baseline"]["momentum_override"]["no_chase_r"] == pytest.approx(0.25)

    # RELAXED BTC override-ok – új értékek
    relaxed = overrides["relaxed"]
    assert relaxed["rr"]["range_time_stop"] == 18
    assert relaxed["momentum_override"]["max_slippage_r"] == pytest.approx(0.24)
    assert relaxed["momentum_override"]["no_chase_r"] == pytest.approx(0.22)
    assert relaxed["atr_floor_usd"] == pytest.approx(78.0)
    assert relaxed["atr_percentiles"]["open"] == pytest.approx(0.38)
    assert relaxed["atr_percentiles"]["mid"] == pytest.approx(0.36)
    assert relaxed["tp_min_pct"] == pytest.approx(0.0065)
    assert relaxed["sl_buffer"]["atr_mult"] == pytest.approx(0.26)
    assert relaxed["sl_buffer"]["abs_min"] == pytest.approx(85.0)
    assert relaxed["bias_relax"]["vwap_ofi_threshold"] == pytest.approx(0.9)

    # SUPPRESSED BTC override-ok (maradtak)
    suppressed = overrides["suppressed"]
    assert suppressed["rr"]["range_breakeven"] == pytest.approx(0.24)
    assert suppressed["momentum_override"]["no_chase_r"] == pytest.approx(0.18)
    assert suppressed["atr_floor_usd"] == pytest.approx(70.0)
    assert suppressed["atr_percentiles"]["open"] == pytest.approx(0.32)
    assert suppressed["atr_percentiles"]["mid"] == pytest.approx(0.30)
    assert suppressed["tp_min_pct"] == pytest.approx(0.006)
    assert suppressed["sl_buffer"]["atr_mult"] == pytest.approx(0.24)
    assert suppressed["sl_buffer"]["abs_min"] == pytest.approx(80.0)
    assert suppressed["momentum_override"]["rr_min"] == pytest.approx(1.35)
    assert suppressed["range_guard_requires_override"] is True

    _reload_settings(monkeypatch)
