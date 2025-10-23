"""Load analysis asset configuration from JSON.

This module centralises the asset specific knobs that used to live in
``analysis.py`` so we can manage them without editing the strategy code.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging

LOGGER = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("analysis_settings.json")
_ENV_OVERRIDE = "ANALYSIS_CONFIG_FILE"


class AnalysisConfigError(RuntimeError):
    """Raised when the analysis configuration is missing or invalid."""


def _resolve_config_path(path: Optional[str] = None) -> Path:
    """Resolve the configuration path with environment fallbacks."""
    candidate = path or os.getenv(_ENV_OVERRIDE)
    if candidate:
        cfg_path = Path(candidate)
        if not cfg_path.is_absolute():
            cfg_path = _DEFAULT_CONFIG_PATH.parent / cfg_path
        return cfg_path
    return _DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)
def _load_raw_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(path)
    if not cfg_path.exists():
        raise AnalysisConfigError(f"Analysis config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise AnalysisConfigError(f"Invalid JSON in config {cfg_path}: {exc}") from exc

    LOGGER.debug("Loaded analysis configuration from %s", cfg_path)
    return data


def reload_config(path: Optional[str] = None) -> None:
    """Force the next ``load_config`` call to reload the JSON file."""
    _load_raw_config.cache_clear()
    _load_raw_config(path)


def _convert_sequence(sequence: Optional[Iterable[Sequence[Any]]]) -> Optional[List[Tuple[Any, ...]]]:
    if sequence is None:
        return None
    return [tuple(item) for item in sequence]


def _build_session_windows(raw: Dict[str, Any]) -> Dict[str, Dict[str, Optional[List[Tuple[int, int, int, int]]]]]:
    processed: Dict[str, Dict[str, Optional[List[Tuple[int, int, int, int]]]]] = {}
    for asset, windows in raw.items():
        processed[asset] = {}
        for window_type, entries in windows.items():
            processed[asset][window_type] = _convert_sequence(entries)
    return processed


def _build_session_time_rules(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    processed: Dict[str, Dict[str, Any]] = {}
    for asset, meta in raw.items():
        processed[asset] = {
            "sunday_open_minute": int(meta.get("sunday_open_minute", 0)),
            "friday_close_minute": int(meta.get("friday_close_minute", 0)),
            "daily_breaks": _convert_sequence(meta.get("daily_breaks")),
        }
    return processed


@lru_cache(maxsize=1)
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return a validated analysis configuration dictionary."""
    raw = _load_raw_config(path)

    assets = raw.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AnalysisConfigError("Config must define a non-empty 'assets' list")

    leverage = raw.get("leverage", {})
    missing_leverage = [asset for asset in assets if asset not in leverage]
    if missing_leverage:
        raise AnalysisConfigError(f"Missing leverage configuration for: {missing_leverage}")

    config: Dict[str, Any] = dict(raw)
    config["assets"] = assets
    config["leverage"] = leverage
    config["session_windows_utc"] = _build_session_windows(raw.get("session_windows_utc", {}))
    config["session_time_rules"] = _build_session_time_rules(raw.get("session_time_rules", {}))
    config["ema_slope_sign_enforced"] = set(raw.get("ema_slope_sign_enforced", []))
    config["enable_momentum_assets"] = set(raw.get("enable_momentum_assets", []))

    return config


def _get_config_value(key: str) -> Any:
    cfg = load_config()
    return cfg.get(key)


def _normalize_threshold_map(raw_map: Any, default_value: float) -> Dict[str, float]:
    mapping: Dict[str, float] = {"default": float(default_value)}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            try:
                mapping[str(key)] = float(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid threshold value %r for %s in entry profile",
                    value,
                    key,
                )
    return mapping


def _normalize_entry_profile(name: str, raw_profile: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw_profile, dict):
        LOGGER.warning(
            "Entry threshold profile '%s' must be a mapping; falling back to defaults",
            name,
        )
        raw_profile = {}
    return {
        "p_score_min": _normalize_threshold_map(raw_profile.get("p_score_min"), 60.0),
        "atr_threshold_multiplier": _normalize_threshold_map(
            raw_profile.get("atr_threshold_multiplier"), 1.0
        ),
    }


# Module level shortcuts used by ``analysis.py`` and helper scripts.
ASSETS: List[str] = load_config()["assets"]
LEVERAGE: Dict[str, float] = load_config()["leverage"]
_ENTRY_THRESHOLD_PROFILES_RAW: Dict[str, Any] = dict(
    _get_config_value("entry_threshold_profiles") or {}
)
_DEFAULT_ENTRY_PROFILE_NAME = "baseline"
if _DEFAULT_ENTRY_PROFILE_NAME not in _ENTRY_THRESHOLD_PROFILES_RAW:
    _ENTRY_THRESHOLD_PROFILES_RAW[_DEFAULT_ENTRY_PROFILE_NAME] = {}

ENTRY_THRESHOLD_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    name: _normalize_entry_profile(name, profile)
    for name, profile in _ENTRY_THRESHOLD_PROFILES_RAW.items()
}

if _DEFAULT_ENTRY_PROFILE_NAME not in ENTRY_THRESHOLD_PROFILES:
    ENTRY_THRESHOLD_PROFILES[_DEFAULT_ENTRY_PROFILE_NAME] = _normalize_entry_profile(
        _DEFAULT_ENTRY_PROFILE_NAME, {}
    )

_ENV_PROFILE = os.getenv("ENTRY_THRESHOLD_PROFILE")
_CFG_ACTIVE_PROFILE = _get_config_value("active_entry_threshold_profile")
_ACTIVE_PROFILE_NAME = (
    _ENV_PROFILE
    or (_CFG_ACTIVE_PROFILE if isinstance(_CFG_ACTIVE_PROFILE, str) else None)
    or _DEFAULT_ENTRY_PROFILE_NAME
)
if _ACTIVE_PROFILE_NAME not in ENTRY_THRESHOLD_PROFILES:
    LOGGER.warning(
        "Unknown entry threshold profile '%s'; falling back to '%s'",
        _ACTIVE_PROFILE_NAME,
        _DEFAULT_ENTRY_PROFILE_NAME,
    )
    _ACTIVE_PROFILE_NAME = _DEFAULT_ENTRY_PROFILE_NAME

ENTRY_THRESHOLD_PROFILE_NAME: str = _ACTIVE_PROFILE_NAME
_ACTIVE_PROFILE = ENTRY_THRESHOLD_PROFILES[ENTRY_THRESHOLD_PROFILE_NAME]
_P_SCORE_PROFILE = _ACTIVE_PROFILE.get("p_score_min", {})
_ATR_MULT_PROFILE = _ACTIVE_PROFILE.get("atr_threshold_multiplier", {})

P_SCORE_MIN_DEFAULT: float = float(_P_SCORE_PROFILE.get("default", 60.0))
P_SCORE_MIN_ASSET: Dict[str, float] = {
    key: float(value)
    for key, value in _P_SCORE_PROFILE.items()
    if key != "default"
}

ATR_THRESHOLD_MULT_DEFAULT: float = float(_ATR_MULT_PROFILE.get("default", 1.0))
ATR_THRESHOLD_MULT_ASSET: Dict[str, float] = {
    key: float(value)
    for key, value in _ATR_MULT_PROFILE.items()
    if key != "default"
}

ATR_LOW_TH_DEFAULT: float = float(_get_config_value("atr_low_threshold_default"))
ATR_LOW_TH_ASSET: Dict[str, float] = dict(_get_config_value("atr_low_threshold") or {})
GOLD_HIGH_VOL_WINDOWS = _get_config_value("gold_high_vol_windows") or []
GOLD_LOW_VOL_TH: float = float(_get_config_value("gold_low_vol_threshold") or 0.0)
SMT_PENALTY_VALUE: int = int(_get_config_value("smt_penalty_value") or 0)
SMT_REQUIRED_BARS: int = int(_get_config_value("smt_required_bars") or 0)
SMT_AUTO_CONFIG: Dict[str, Any] = dict(_get_config_value("smt_auto_config") or {})
TP_NET_MIN_DEFAULT: float = float(_get_config_value("tp_net_min_default") or 0.0)
TP_NET_MIN_ASSET: Dict[str, float] = dict(_get_config_value("tp_net_min") or {})
TP_MIN_PCT: Dict[str, float] = dict(_get_config_value("tp_min_pct") or {})
TP_MIN_ABS: Dict[str, float] = dict(_get_config_value("tp_min_abs") or {})
SL_BUFFER_RULES: Dict[str, Any] = dict(_get_config_value("sl_buffer_rules") or {})
MIN_RISK_ABS: Dict[str, float] = dict(_get_config_value("min_risk_abs") or {})
ACTIVE_INVALID_BUFFER_ABS: Dict[str, float] = dict(_get_config_value("active_invalid_buffer_abs") or {})
ASSET_COST_MODEL: Dict[str, Any] = dict(_get_config_value("asset_cost_model") or {})
DEFAULT_COST_MODEL: Dict[str, Any] = dict(_get_config_value("default_cost_model") or {})
COST_MULT_DEFAULT: float = float(_get_config_value("cost_mult_default") or 0.0)
COST_MULT_HIGH_VOL: float = float(_get_config_value("cost_mult_high_vol") or 0.0)
ATR5_MIN_MULT: float = float(_get_config_value("atr5_min_mult") or 0.0)
ATR_VOL_HIGH_REL: float = float(_get_config_value("atr_vol_high_rel") or 0.0)
EMA_SLOPE_TH_DEFAULT: float = float(_get_config_value("ema_slope_th_default") or 0.0)
EMA_SLOPE_TH_ASSET: Dict[str, float] = dict(_get_config_value("ema_slope_th_asset") or {})
EMA_SLOPE_SIGN_ENFORCED: set = load_config()["ema_slope_sign_enforced"]
ATR_ABS_MIN: Dict[str, float] = dict(_get_config_value("atr_abs_min") or {})
CORE_RR_MIN: Dict[str, float] = dict(_get_config_value("core_rr_min") or {})
MOMENTUM_RR_MIN: Dict[str, float] = dict(_get_config_value("momentum_rr_min") or {})
FX_TP_TARGETS: Dict[str, float] = dict(_get_config_value("fx_tp_targets") or {})
NVDA_EXTENDED_ATR_REL: float = float(_get_config_value("nvda_extended_atr_rel") or 0.0)
NVDA_MOMENTUM_ATR_REL: float = float(_get_config_value("nvda_momentum_atr_rel") or 0.0)
SRTY_MOMENTUM_ATR_REL: float = float(_get_config_value("srty_momentum_atr_rel") or 0.0)
ENABLE_MOMENTUM_ASSETS: set = load_config()["enable_momentum_assets"]
SESSION_WINDOWS_UTC: Dict[str, Dict[str, Optional[List[Tuple[int, int, int, int]]]]] = load_config()[
    "session_windows_utc"
]
SPOT_MAX_AGE_SECONDS: Dict[str, int] = {
    k: int(v) for k, v in dict(_get_config_value("spot_max_age_seconds") or {}).items()
}
INTERVENTION_WATCH_DEFAULT: Dict[str, Any] = dict(_get_config_value("intervention_watch_default") or {})
SESSION_WEEKDAYS: Dict[str, Any] = dict(_get_config_value("session_weekdays") or {})
SESSION_TIME_RULES: Dict[str, Dict[str, Any]] = load_config()["session_time_rules"]


def get_p_score_min(asset: str) -> float:
    """Return the configured minimum P-score for the active profile."""

    return P_SCORE_MIN_ASSET.get(asset, P_SCORE_MIN_DEFAULT)


def get_atr_threshold_multiplier(asset: str) -> float:
    """Return the ATR threshold multiplier for the active profile."""

    return ATR_THRESHOLD_MULT_ASSET.get(asset, ATR_THRESHOLD_MULT_DEFAULT)


def list_entry_threshold_profiles() -> List[str]:
    """Return the sorted list of entry threshold profile names."""

    return sorted(ENTRY_THRESHOLD_PROFILES.keys())


def describe_entry_threshold_profile(profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the resolved threshold values for the requested entry profile.

    Parameters
    ----------
    profile_name:
        Optional explicit profile name. When omitted, the currently active
        profile (``ENTRY_THRESHOLD_PROFILE_NAME``) is described.

    Returns
    -------
    dict
        A mapping that exposes the profile ``name`` and the effective
        ``p_score_min`` and ``atr_threshold_multiplier`` values. Each section
        contains the configured ``default`` value, the per-asset overrides that
        are present in the JSON, and a ``by_asset`` dictionary that lists the
        numbers applied to every asset declared in the analysis configuration.
    """

    target_name = profile_name or ENTRY_THRESHOLD_PROFILE_NAME
    profile = ENTRY_THRESHOLD_PROFILES.get(target_name)
    if profile is None:
        raise AnalysisConfigError(
            f"Unknown entry threshold profile '{target_name}'"
        )

    def _resolve(metric_key: str, implicit_default: float) -> Dict[str, Any]:
        raw_map: Dict[str, float] = profile.get(metric_key, {})
        default_value = float(raw_map.get("default", implicit_default))
        overrides = {
            str(asset): float(value)
            for asset, value in raw_map.items()
            if asset != "default"
        }
        by_asset = {
            asset: float(raw_map.get(asset, default_value))
            for asset in ASSETS
        }
        return {
            "default": default_value,
            "overrides": overrides,
            "by_asset": by_asset,
        }

    return {
        "name": target_name,
        "p_score_min": _resolve("p_score_min", 60.0),
        "atr_threshold_multiplier": _resolve("atr_threshold_multiplier", 1.0),
    }
