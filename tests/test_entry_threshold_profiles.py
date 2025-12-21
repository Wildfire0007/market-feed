import importlib
import json
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

    default_name = settings.ENTRY_THRESHOLD_PROFILE_NAME
    profile = settings.describe_entry_threshold_profile()
    assert default_name == "relaxed"
    assert profile["name"] == default_name
    assert profile == settings.describe_entry_threshold_profile(default_name)
    assert default_name in settings.list_entry_threshold_profiles()

    assert profile["p_score_min"]["default"] == pytest.approx(34.0)
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(28.0)
    assert profile["p_score_min"]["by_asset"]["GOLD_CFD"] == pytest.approx(30.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(30.0)
    assert profile["p_score_min"]["by_asset"]["NVDA"] == pytest.approx(30.0)
    assert profile["p_score_min"]["by_asset"]["USOIL"] == pytest.approx(30.0)
    assert profile["p_score_min"]["by_asset"]["XAGUSD"] == pytest.approx(30.0)

    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(0.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["GOLD_CFD"] == pytest.approx(0.5)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.0)

    # Fib toleranciák és ATR floor (relaxed / aktív profil)
    assert settings.get_fib_tolerance("BTCUSD") == pytest.approx(0.022)
    assert settings.get_fib_tolerance("EURUSD") == pytest.approx(0.010)
    assert settings.get_fib_tolerance("GOLD_CFD") == pytest.approx(0.014)
    assert settings.get_fib_tolerance("NVDA") == pytest.approx(0.016)
    assert settings.get_fib_tolerance("USOIL") == pytest.approx(0.016)
    assert settings.get_fib_tolerance("XAGUSD") == pytest.approx(0.016)
    assert settings.get_atr_abs_min("BTCUSD") == pytest.approx(50.0)
    
    baseline = settings.describe_entry_threshold_profile("baseline")
    assert baseline["p_score_min"]["default"] == pytest.approx(34.0)
    assert baseline["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(34.0)
    assert baseline["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(32.0)
    assert baseline["p_score_min"]["by_asset"]["XAGUSD"] == pytest.approx(36.0)
    assert baseline["p_score_min"]["by_asset"]["NVDA"] == pytest.approx(34.0)
    assert baseline["p_score_min"]["by_asset"]["USOIL"] == pytest.approx(34.0)
    assert baseline["atr_threshold_multiplier"]["default"] == pytest.approx(0.70)
    assert baseline["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.70)
    assert baseline["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.70)

    suppressed = settings.describe_entry_threshold_profile("suppressed")
    assert suppressed["p_score_min"]["default"] == pytest.approx(32.0)
    assert suppressed["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(32.0)
    assert suppressed["atr_threshold_multiplier"]["default"] == pytest.approx(0.90)
    assert suppressed["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.95)

    _reload_settings(monkeypatch)


def test_suppressed_profile_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="suppressed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "suppressed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "suppressed"
    assert profile["p_score_min"]["default"] == pytest.approx(32.0)
    assert profile["p_score_min"]["by_asset"]["USOIL"] == pytest.approx(32.0)
    assert profile["p_score_min"]["by_asset"]["XAGUSD"] == pytest.approx(34.0)
    assert profile["atr_threshold_multiplier"]["default"] == pytest.approx(0.90)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.95)

    # Suppressed fib toleranciák
    assert settings.get_fib_tolerance("BTCUSD") == pytest.approx(0.012)
    assert settings.get_fib_tolerance("EURUSD") == pytest.approx(0.006)
    assert settings.get_fib_tolerance("GOLD_CFD") == pytest.approx(0.008)
    assert settings.get_fib_tolerance("NVDA") == pytest.approx(0.01)
    assert settings.get_fib_tolerance("USOIL") == pytest.approx(0.01)
    assert settings.get_fib_tolerance("XAGUSD") == pytest.approx(0.01)


def test_relaxed_profile_override(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="relaxed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "relaxed"
    profile = settings.describe_entry_threshold_profile()
    assert profile["name"] == "relaxed"

    # RELAXED értékek
    assert profile["p_score_min"]["by_asset"]["GOLD_CFD"] == pytest.approx(30.0)
    assert profile["p_score_min"]["by_asset"]["BTCUSD"] == pytest.approx(30.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["USOIL"] == pytest.approx(0.0)
    assert profile["atr_threshold_multiplier"]["by_asset"]["BTCUSD"] == pytest.approx(0.0)
    
    # EURUSD override a relaxed profilban
    assert profile["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(28.0)

    # Baseline ellenőrzés (aktív)
    baseline = settings.describe_entry_threshold_profile("baseline")
    assert baseline["p_score_min"]["by_asset"]["EURUSD"] == pytest.approx(32.0)

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
    assert settings.INTRADAY_ATR_RELAX["EURUSD"] == pytest.approx(0.90)
    assert settings.INTRADAY_ATR_RELAX["GOLD_CFD"] == pytest.approx(1.0)
    assert settings.INTRADAY_ATR_RELAX["BTCUSD"] == pytest.approx(0.80)
    assert settings.INTRADAY_ATR_RELAX["NVDA"] == pytest.approx(1.0)

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
    assert any(
        scenario["direction"] == "short" and scenario["requires"] == ["bos5m_short", "atr_ok"]
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
    assert any(
        scenario["direction"] == "short" and scenario["requires"] == ["micro_bos_short", "atr_ok"]
        for scenario in btc_bias["scenarios"]
    )

    _reload_settings(monkeypatch)


def test_btc_profile_overrides(monkeypatch):
    baseline_settings = _reload_settings(monkeypatch, profile="baseline")

    # Momentum eszközök listája tartalmazza a BTCUSD-t
    assert "BTCUSD" in baseline_settings.ENABLE_MOMENTUM_ASSETS

    overrides = baseline_settings.BTC_PROFILE_OVERRIDES

    # BASELINE BTC override-ok (új értékek)
    assert overrides["baseline"]["atr_floor_usd"] == pytest.approx(80.0)
    assert overrides["baseline"]["tp_min_pct"] == pytest.approx(0.007)
    assert overrides["baseline"]["sl_buffer"]["atr_mult"] == pytest.approx(0.28)
    assert overrides["baseline"]["sl_buffer"]["abs_min"] == pytest.approx(90.0)
    assert overrides["baseline"]["rr"]["trend_core"] == pytest.approx(1.6)
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
    assert overrides["baseline"]["precision"]["score_min"] == pytest.approx(52.0)

    # RELAXED BTC override-ok – új értékek
    relaxed = overrides["relaxed"]
    assert relaxed["rr"]["range_time_stop"] == 18
    assert relaxed["momentum_override"]["max_slippage_r"] == pytest.approx(0.24)
    assert relaxed["momentum_override"]["no_chase_r"] == pytest.approx(0.22)
    assert relaxed["atr_floor_usd"] == pytest.approx(75.0)
    assert relaxed["atr_percentiles"]["open"] == pytest.approx(0.38)
    assert relaxed["atr_percentiles"]["mid"] == pytest.approx(0.36)
    assert relaxed["tp_min_pct"] == pytest.approx(0.0065)
    assert relaxed["sl_buffer"]["atr_mult"] == pytest.approx(0.26)
    assert relaxed["sl_buffer"]["abs_min"] == pytest.approx(85.0)
    assert relaxed["bias_relax"]["vwap_ofi_threshold"] == pytest.approx(0.9)
    assert relaxed["precision"]["score_min"] == pytest.approx(50.0)

    # SUPPRESSED BTC override-ok (maradtak)
    suppressed = overrides["suppressed"]
    assert suppressed["rr"]["range_breakeven"] == pytest.approx(0.24)
    assert suppressed["momentum_override"]["no_chase_r"] == pytest.approx(0.18)
    assert suppressed["atr_floor_usd"] == pytest.approx(75.0)
    assert suppressed["atr_percentiles"]["open"] == pytest.approx(0.32)
    assert suppressed["atr_percentiles"]["mid"] == pytest.approx(0.30)
    assert suppressed["tp_min_pct"] == pytest.approx(0.006)
    assert suppressed["sl_buffer"]["atr_mult"] == pytest.approx(0.24)
    assert suppressed["sl_buffer"]["abs_min"] == pytest.approx(80.0)
    assert suppressed["momentum_override"]["rr_min"] == pytest.approx(1.35)
    assert suppressed["momentum_override"]["max_slippage_r"] == pytest.approx(0.24)
    assert suppressed["range_guard_requires_override"] is True


def test_rr_minimum_settings(monkeypatch):
    settings = _reload_settings(monkeypatch)

    assert settings.CORE_RR_MIN["default"] == pytest.approx(1.3)
    assert settings.CORE_RR_MIN["EURUSD"] == pytest.approx(1.3)
    assert settings.CORE_RR_MIN["GOLD_CFD"] == pytest.approx(1.3)
    assert settings.CORE_RR_MIN["USOIL"] == pytest.approx(1.3)
    assert settings.CORE_RR_MIN["NVDA"] == pytest.approx(1.4)
    assert settings.CORE_RR_MIN["BTCUSD"] == pytest.approx(1.25)
    assert settings.CORE_RR_MIN["XAGUSD"] == pytest.approx(1.4)

    assert settings.MOMENTUM_RR_MIN["default"] == pytest.approx(1.1)
    assert settings.MOMENTUM_RR_MIN["EURUSD"] == pytest.approx(1.15)
    assert settings.MOMENTUM_RR_MIN["GOLD_CFD"] == pytest.approx(1.1)
    assert settings.MOMENTUM_RR_MIN["USOIL"] == pytest.approx(1.1)
    assert settings.MOMENTUM_RR_MIN["NVDA"] == pytest.approx(1.2)
    assert settings.MOMENTUM_RR_MIN["BTCUSD"] == pytest.approx(1.1)
    assert settings.MOMENTUM_RR_MIN["XAGUSD"] == pytest.approx(1.1)
    

def test_profile_specific_helpers(monkeypatch):
    baseline = _reload_settings(monkeypatch, profile="baseline")
    assert baseline.get_atr_period("EURUSD") == 13
    assert baseline.get_spread_max_atr_pct("EURUSD") == pytest.approx(0.60)
    assert baseline.get_spread_max_atr_pct("USOIL") == pytest.approx(0.55)
    assert baseline.get_spread_max_atr_pct("NVDA") == pytest.approx(0.50)
    assert baseline.get_spread_max_atr_pct("BTCUSD") == pytest.approx(0.70)
    assert baseline.get_spread_max_atr_pct("XAGUSD") == pytest.approx(0.50)
    assert baseline.get_fib_tolerance("GOLD_CFD") == pytest.approx(0.01)
    assert baseline.get_fib_tolerance("NVDA") == pytest.approx(0.012)
    assert baseline.get_fib_tolerance("EURUSD") == pytest.approx(0.008)
    assert baseline.get_fib_tolerance("USOIL") == pytest.approx(0.012)
    assert baseline.get_fib_tolerance("XAGUSD") == pytest.approx(0.012)
    assert baseline.get_atr_abs_min("EURUSD") == pytest.approx(0.00025)
    assert baseline.get_atr_abs_min("GOLD_CFD") == pytest.approx(0.60)
    assert baseline.get_atr_abs_min("USOIL") == pytest.approx(0.14)
    assert baseline.get_atr_abs_min("NVDA") == pytest.approx(0.45)
    assert baseline.get_atr_abs_min("BTCUSD") == pytest.approx(55.0)
    assert baseline.get_atr_abs_min("XAGUSD") == pytest.approx(0.06)
    assert baseline.get_max_risk_pct("NVDA") == pytest.approx(1.3)
    assert baseline.get_bos_lookback("EURUSD") == 28

    suppressed = _reload_settings(monkeypatch, profile="suppressed")
    assert suppressed.get_atr_period("EURUSD") == 12
    assert suppressed.get_atr_abs_min("EURUSD") == pytest.approx(0.00030)
    assert suppressed.get_spread_max_atr_pct("BTCUSD") == pytest.approx(0.75)
    assert suppressed.get_spread_max_atr_pct("NVDA") == pytest.approx(0.55)
    assert suppressed.get_spread_max_atr_pct("USOIL") == pytest.approx(0.60)
    assert suppressed.get_fib_tolerance("BTCUSD") == pytest.approx(0.012)
    assert suppressed.get_fib_tolerance("GOLD_CFD") == pytest.approx(0.008)
    assert suppressed.get_fib_tolerance("NVDA") == pytest.approx(0.01)
    assert suppressed.get_max_risk_pct("BTCUSD") == pytest.approx(1.2)
    assert suppressed.get_bos_lookback("BTCUSD") == 24

    relaxed = _reload_settings(monkeypatch, profile="relaxed")
    assert relaxed.get_spread_max_atr_pct("GOLD_CFD") == pytest.approx(0.55)
    assert relaxed.get_bos_lookback(None) == 28
    assert relaxed.get_fib_tolerance("GOLD_CFD") == pytest.approx(0.014)
    assert relaxed.get_fib_tolerance("NVDA") == pytest.approx(0.016)
    assert relaxed.get_max_risk_pct("GOLD_CFD") == pytest.approx(1.8)

    _reload_settings(monkeypatch)


def test_entry_profile_routing_helpers(monkeypatch, tmp_path):
    settings = _reload_settings(monkeypatch)
    default_name = settings.ENTRY_THRESHOLD_PROFILE_NAME
    monkeypatch.setattr(settings, "_ENTRY_PROFILE_SCHEDULE", {})
    monkeypatch.setattr(settings, "_current_tod_bucket", lambda *_args, **_kwargs: None)

    custom_map_path = tmp_path / "asset_profile_map.json"
    custom_map_path.write_text(
        json.dumps(
            {
                "BTCUSD": "suppressed",
                "EXTRA": "relaxed",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "_ASSET_PROFILE_MAP_PATH", custom_map_path)
    settings._load_asset_entry_profile_map.cache_clear()

    mapped_name = settings.get_entry_threshold_profile_name_for_asset("BTCUSD")
    assert mapped_name == "suppressed"
    mapped_profile = settings.get_entry_threshold_profile_for_asset("BTCUSD")
    assert mapped_profile == settings.ENTRY_THRESHOLD_PROFILES["suppressed"]

    unmapped_name = settings.get_entry_threshold_profile_name_for_asset("EURUSD")
    assert unmapped_name == default_name
    unmapped_profile = settings.get_entry_threshold_profile_for_asset("EURUSD")
    assert unmapped_profile == settings.ENTRY_THRESHOLD_PROFILES[default_name]

    custom_map_path.write_text(json.dumps({"BTCUSD": "unknown"}), encoding="utf-8")
    settings._load_asset_entry_profile_map.cache_clear()
    fallback_name = settings.get_entry_threshold_profile_name_for_asset("BTCUSD")
    assert fallback_name == default_name
    fallback_profile = settings.get_entry_threshold_profile_for_asset("BTCUSD")
    assert fallback_profile == settings.ENTRY_THRESHOLD_PROFILES[default_name]

    _reload_settings(monkeypatch)


def test_rr_relax_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="suppressed")

    assert settings.RR_RELAX_ENABLED is False
    assert settings.RR_RELAX_RANGE_MOMENTUM == pytest.approx(1.35)
    assert settings.RR_RELAX_RANGE_CORE == pytest.approx(1.5)
    assert settings.RR_RELAX_ATR_RATIO_TRIGGER == pytest.approx(0.92)
    
def test_risk_template_helpers(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="baseline")
    assert settings.get_tp_min_pct_value("EURUSD") == pytest.approx(0.0013)
    assert settings.get_tp_min_abs_value("NVDA") == pytest.approx(0.8)
    assert settings.get_tp_net_min("GOLD_CFD") == pytest.approx(0.0035)
    sl_buffer = settings.get_sl_buffer_config("USOIL")
    assert sl_buffer["abs_min"] == pytest.approx(0.35)
    assert settings.get_max_slippage_r("XAGUSD") == pytest.approx(0.22)
    assert settings.get_spread_max_atr_pct("BTCUSD") == pytest.approx(0.70)
    assert settings.get_spread_max_atr_pct("USOIL") == pytest.approx(0.55)
    assert settings.get_spread_max_atr_pct("NVDA") == pytest.approx(0.50)
    assert settings.get_spread_max_atr_pct("XAGUSD") == pytest.approx(0.50)

    suppressed = _reload_settings(monkeypatch, profile="suppressed")
    assert suppressed.get_tp_min_pct_value("EURUSD") == pytest.approx(0.0013)
    assert suppressed.get_tp_net_min("BTCUSD") == pytest.approx(0.006)
    sl_buffer_supp = suppressed.get_sl_buffer_config("XAGUSD")
    assert sl_buffer_supp["atr_mult"] == pytest.approx(0.22)
    assert suppressed.get_max_slippage_r("USOIL") == pytest.approx(0.18)
    assert suppressed.get_spread_max_atr_pct("BTCUSD") == pytest.approx(0.75)
    assert suppressed.get_spread_max_atr_pct("NVDA") == pytest.approx(0.55)

    relaxed = _reload_settings(monkeypatch, profile="relaxed")
    gold_risk = relaxed.get_risk_template("GOLD_CFD")
    assert gold_risk["core_rr_min"] == pytest.approx(1.4)
    assert gold_risk["momentum_rr_min"] == pytest.approx(1.4)
    assert gold_risk["tp_min_pct"] == pytest.approx(0.0025)
    assert "tp_net_min" not in gold_risk

    suppressed_after = _reload_settings(monkeypatch, profile="suppressed")
    assert suppressed_after.get_spread_max_atr_pct("USOIL") == pytest.approx(0.60)
    
    _reload_settings(monkeypatch)
