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


# Module level shortcuts used by ``analysis.py`` and helper scripts.
ASSETS: List[str] = load_config()["assets"]
LEVERAGE: Dict[str, float] = load_config()["leverage"]
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
