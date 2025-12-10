"""Load analysis asset configuration from JSON.

This module centralises the asset specific knobs that used to live in
``analysis.py`` so we can manage them without editing the strategy code.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

import logging

LOGGER = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("analysis_settings.json")
_ASSET_PROFILE_MAP_PATH = Path(__file__).with_name("asset_profile_map.json")
_ENV_OVERRIDE = "ANALYSIS_CONFIG_FILE"

SequenceTuple = Tuple[Any, ...]
T = TypeVar("T")


def _now_utc() -> datetime:
    """Return the current UTC time (split out for test overrides)."""

    return datetime.now(timezone.utc)


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


def _convert_sequence(
    sequence: Optional[Iterable[Sequence[Any]]]
) -> Optional[List[SequenceTuple]]:
    """Convert nested sequences to tuples while preserving ``None``."""
    
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



    
def _optional_int(value: Any, *, field: str, asset: str) -> Optional[int]:
    """Return ``value`` converted to ``int`` when possible, otherwise ``None``.

    The configuration uses ``null`` to express the absence of a time bound.  When
    that happens (or when an invalid value sneaks in) we want the analysis layer
    to behave as if there is no restriction instead of defaulting to midnight.
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Ignoring invalid %s value %r for asset %s in session_time_rules",
            field,
            value,
            asset,
        )
        return None


def _build_session_time_rules(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    processed: Dict[str, Dict[str, Any]] = {}
    for asset, meta in raw.items():
        processed[asset] = {
            "sunday_open_minute": _optional_int(
                meta.get("sunday_open_minute"), field="sunday_open_minute", asset=asset
            ),
            "friday_close_minute": _optional_int(
                meta.get("friday_close_minute"), field="friday_close_minute", asset=asset
            ),
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


_SESSION_BUCKET_PROXIES_RAW: Dict[str, str] = {
    str(key).upper(): str(value).upper()
    for key, value in dict(_get_config_value("session_bucket_proxies") or {}).items()
    if value is not None
}


def is_intraday_relax_enabled(asset: str) -> bool:
    """Return whether intraday prerequisite relaxation is enabled for ``asset``."""

    raw = _get_config_value("enable_intraday_relax") or {}
    default_value = False
    if isinstance(raw, dict):
        default_value = bool(raw.get("default", False))
        return bool(raw.get(asset, default_value))
    return bool(raw)


def get_intraday_relax_size_scale(asset: str) -> float:
    """Return the risk scaling factor applied when intraday relax is active."""

    raw = _get_config_value("intraday_relax_size_scale") or {}
    default_value = 0.6
    if isinstance(raw, dict):
        try:
            default_value = float(raw.get("default", default_value))
        except (TypeError, ValueError):
            default_value = 0.6
        value = raw.get(asset, default_value)
    else:
        value = raw or default_value
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = default_value
    clamped_default = max(0.1, min(float(default_value), 1.0))
    return max(0.1, min(scale, clamped_default if scale == default_value else 1.0))


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


def _normalize_session_status_profiles(raw_map: Any) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_map, dict):
        return profiles

    bool_fields = (
        "force_session_closed",
        "open",
        "entry_open",
        "within_window",
        "within_entry_window",
        "within_monitor_window",
        "weekday_ok",
        "market_closed_assumed",
        "auto_activate_weekend",
    )
    str_fields = (
        "status",
        "status_note",
        "market_closed_reason",
        "next_open_utc",
    )

    for name, meta in raw_map.items():
        if not isinstance(meta, dict):
            profiles[str(name)] = {}
            continue
        profile_cfg: Dict[str, Any] = {}
        for field in bool_fields:
            if field in meta:
                profile_cfg[field] = bool(meta.get(field))
        for field in str_fields:
            if field in meta and meta.get(field) is not None:
                profile_cfg[field] = str(meta.get(field))

        notes_raw = meta.get("notes")
        if isinstance(notes_raw, list):
            notes: List[str] = []
            for item in notes_raw:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        notes.append(text)
            if notes:
                profile_cfg["notes"] = notes

        tags_raw = meta.get("tags")
        if isinstance(tags_raw, list):
            tags: List[str] = []
            for item in tags_raw:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        tags.append(text)
            if tags:
                profile_cfg["tags"] = tags

        context_raw = meta.get("context")
        if isinstance(context_raw, dict):
            profile_cfg["context"] = {str(key): value for key, value in context_raw.items()}

        assets_raw = meta.get("assets")
        if isinstance(assets_raw, list):
            assets: List[str] = []
            for item in assets_raw:
                if isinstance(item, str):
                    text = item.strip().upper()
                    if text and text not in assets:
                        assets.append(text)
            if assets:
                profile_cfg["assets"] = assets

        profiles[str(name)] = profile_cfg

    return profiles


def _normalize_dynamic_logic(raw: Any) -> Dict[str, Any]:
    default_cfg: Dict[str, Any] = {
        "enabled": False,
        "p_score": {},
        "soft_gates": {
            "atr": {"enabled": False, "tolerance_pct": 0.15, "penalty_max": 6.0}
        },
        "latency_relaxation": {
            "profiles": {"strict": {"limit": None, "penalty": 0}},
            "asset_map": {"default": "strict"},
        },
    }

    if not isinstance(raw, dict):
        return default_cfg

    cfg: Dict[str, Any] = {"enabled": bool(raw.get("enabled"))}
    for key in ("p_score", "soft_gates", "latency_relaxation"):
        if isinstance(raw.get(key), dict):
            cfg[key] = dict(raw[key])
    # Guarantee baseline strict profiles even when the config omits them to
    # preserve backwards compatibility with older JSON files.
    latency_cfg = cfg.setdefault("latency_relaxation", {})
    if isinstance(latency_cfg, dict):
        profiles = latency_cfg.setdefault("profiles", {})
        if isinstance(profiles, dict):
            profiles.setdefault("strict", {"limit": None, "penalty": 0})
        latency_cfg.setdefault("asset_map", {}).setdefault("default", "strict")
    soft_cfg = cfg.setdefault("soft_gates", {})
    if isinstance(soft_cfg, dict):
        atr_cfg = soft_cfg.setdefault("atr", {}) if isinstance(soft_cfg.get("atr"), dict) else soft_cfg.setdefault("atr", {})
        atr_cfg.setdefault("enabled", False)
        atr_cfg.setdefault("tolerance_pct", 0.15)
        atr_cfg.setdefault("penalty_max", 6.0)
    cfg.setdefault("p_score", {})
    return cfg


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


def _normalize_profiled_numeric_map(
    raw_map: Any, cast: Callable[[Any], T], default_value: T
) -> Dict[str, Dict[str, T]]:
    profiles: Dict[str, Dict[str, T]] = {}
    if not isinstance(raw_map, dict):
        return profiles

    for profile_name, mapping in raw_map.items():
        if not isinstance(mapping, dict):
            continue
        profile_default = default_value
        if "default" in mapping:
            try:
                value = mapping.get("default")
                if value is not None:
                    profile_default = cast(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid default %r for profile %s", mapping.get("default"), profile_name
                )
        overrides: Dict[str, T] = {}
        for asset, value in mapping.items():
            if asset == "default":
                continue
            try:
                if value is None:
                    continue
                overrides[str(asset)] = cast(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid override %r for %s in profile %s", value, asset, profile_name
                )
        profiles[str(profile_name)] = {"default": profile_default, "by_asset": overrides}
    return profiles


def _normalize_fib_tolerance_profiles(raw_map: Any) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_map, dict):
        return profiles

    for profile_name, mapping in raw_map.items():
        if not isinstance(mapping, dict):
            continue
        default_value = 0.02
        by_asset: Dict[str, float] = {}
        by_class: Dict[str, float] = {}

        if "default" in mapping:
            try:
                default_raw = mapping.get("default")
                if default_raw is not None:
                    default_value = float(default_raw)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid fib_tolerance default %r for profile %s",
                    mapping.get("default"),
                    profile_name,
                )

        classes_map = mapping.get("classes")
        if isinstance(classes_map, dict):
            for class_name, class_value in classes_map.items():
                try:
                    if class_value is None:
                        continue
                    by_class[str(class_name)] = float(class_value)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid fib_tolerance class override %r for %s in profile %s",
                        class_value,
                        class_name,
                        profile_name,
                    )

        for asset_key, value in mapping.items():
            if asset_key in {"default", "classes"}:
                continue
            try:
                if value is None:
                    continue
                by_asset[str(asset_key)] = float(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid fib_tolerance override %r for %s in profile %s",
                    value,
                    asset_key,
                    profile_name,
                )

        profiles[str(profile_name)] = {
            "default": default_value,
            "by_asset": by_asset,
            "by_class": by_class,
        }

    return profiles


def _normalize_risk_templates(raw_map: Any) -> Dict[str, Dict[str, Any]]:
    """Normalise risk template profiles and coerce numeric fields."""

    if not isinstance(raw_map, dict):
        return {}

    numeric_fields = {
        "risk_pct",
        "core_rr_min",
        "momentum_rr_min",
        "tp_net_min",
        "tp_min_pct",
        "tp_min_abs",
        "max_slippage_r",
        "max_spread_atr_pct",
    }
    int_fields = {"cooldown_minutes", "max_concurrent"}

    def normalise_leaf(meta: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not isinstance(meta, dict):
            return result

        for key, value in meta.items():
            if key in numeric_fields:
                try:
                    if value is not None:
                        result[key] = float(value)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid numeric risk template value %r for %s", value, key
                    )
            elif key in int_fields:
                try:
                    if value is not None:
                        result[key] = int(float(value))
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid integer risk template value %r for %s", value, key
                    )
            elif key == "sl_buffer" and isinstance(value, dict):
                buf: Dict[str, float] = {}
                atr_val = value.get("atr_mult")
                abs_val = value.get("abs_min")
                try:
                    if atr_val is not None:
                        buf["atr_mult"] = float(atr_val)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid atr_mult %r in sl_buffer template", atr_val
                    )
                try:
                    if abs_val is not None:
                        buf["abs_min"] = float(abs_val)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring invalid abs_min %r in sl_buffer template", abs_val
                    )
                if buf:
                    result["sl_buffer"] = buf
            elif key == "sl_tp_rule" and isinstance(value, str):
                text = value.strip()
                if text:
                    result["sl_tp_rule"] = text
            else:
                result[key] = value

        return result

    templates: Dict[str, Dict[str, Any]] = {}
    for profile_name, mapping in raw_map.items():
        if not isinstance(mapping, dict):
            continue
        profile_cfg: Dict[str, Any] = {"default": {}, "assets": {}}
        default_meta = mapping.get("default")
        if isinstance(default_meta, dict):
            profile_cfg["default"] = normalise_leaf(default_meta)
        assets_meta = mapping.get("assets")
        if isinstance(assets_meta, dict):
            asset_cfg: Dict[str, Dict[str, Any]] = {}
            for asset_key, meta in assets_meta.items():
                if not isinstance(meta, dict):
                    continue
                asset_cfg[str(asset_key).upper()] = normalise_leaf(meta)
            profile_cfg["assets"] = asset_cfg
        templates[str(profile_name)] = profile_cfg

    baseline_cfg = templates.setdefault("baseline", {"default": {}, "assets": {}})
    baseline_cfg.setdefault("default", {})
    baseline_cfg.setdefault("assets", {})

    return templates


def _normalize_low_atr_overrides(raw_map: Any) -> Dict[str, Dict[str, float]]:
    """Validate ATR floor overrides used for adaptive TP and RR thresholds."""

    result: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw_map, dict):
        return result

    for asset, meta in raw_map.items():
        if not isinstance(meta, dict):
            continue
        asset_key = "default" if str(asset).lower() == "default" else str(asset).upper()
        cfg: Dict[str, float] = {}
        for key in ("floor", "rel_atr_floor"):
            value = _safe_float(meta.get(key))
            if value is not None:
                cfg["floor"] = value
                break
        tp_override = _safe_float(meta.get("tp_min_pct"))
        if tp_override is not None:
            cfg["tp_min_pct"] = tp_override
        rr_override = _safe_float(meta.get("rr_required"))
        if rr_override is not None:
            cfg["rr_required"] = rr_override
        if cfg:
            result[asset_key] = cfg
    return result


def _normalize_bias_relax(raw_map: Any) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_map, dict):
        return result

    for asset, meta in raw_map.items():
        if not isinstance(meta, dict):
            continue
        allow_neutral = bool(meta.get("allow_neutral"))
        scenarios: List[Dict[str, Any]] = []
        raw_scenarios = meta.get("scenarios")
        if isinstance(raw_scenarios, list):
            for item in raw_scenarios:
                if not isinstance(item, dict):
                    continue
                direction = str(item.get("direction", "")).lower()
                if direction not in {"long", "short"}:
                    continue
                raw_requires = item.get("requires")
                requires: List[str] = []
                if isinstance(raw_requires, list):
                    for requirement in raw_requires:
                        if isinstance(requirement, str):
                            requires.append(requirement)
                if not requires:
                    continue
                label = item.get("label")
                label_value = str(label) if isinstance(label, str) else None
                scenarios.append(
                    {
                        "direction": direction,
                        "requires": requires,
                        "label": label_value,
                    }
                )
        result[str(asset)] = {
            "allow_neutral": allow_neutral,
            "scenarios": scenarios,
        }
    return result


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_btc_profiles(raw_map: Any) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_map, dict):
        return result

    for profile_name, meta in raw_map.items():
        if not isinstance(meta, dict):
            continue
        profile_cfg: Dict[str, Any] = {}

        floor_val = _safe_float(meta.get("atr_floor_usd"))
        if floor_val is not None:
            profile_cfg["atr_floor_usd"] = floor_val

        percentiles_raw = meta.get("atr_percentiles")
        if isinstance(percentiles_raw, dict):
            percentiles: Dict[str, float] = {}
            for bucket, value in percentiles_raw.items():
                val = _safe_float(value)
                if val is not None:
                    percentiles[str(bucket)] = val
            if percentiles:
                profile_cfg["atr_percentiles"] = percentiles

        rr_raw = meta.get("rr")
        if isinstance(rr_raw, dict):
            rr_cfg: Dict[str, Any] = {}
            for key in (
                "trend_core",
                "trend_momentum",
                "range_core",
                "range_momentum",
                "range_size_scale",
                "range_breakeven",
            ):
                val = _safe_float(rr_raw.get(key))
                if val is not None:
                    rr_cfg[key] = val
            time_stop = rr_raw.get("range_time_stop")
            try:
                if time_stop is not None:
                    rr_cfg["range_time_stop"] = int(time_stop)
            except (TypeError, ValueError):
                pass
            if rr_cfg:
                profile_cfg["rr"] = rr_cfg

        tp_min_val = _safe_float(meta.get("tp_min_pct"))
        if tp_min_val is not None:
            profile_cfg["tp_min_pct"] = tp_min_val

        sl_raw = meta.get("sl_buffer")
        if isinstance(sl_raw, dict):
            sl_cfg: Dict[str, float] = {}
            for key in ("atr_mult", "abs_min"):
                val = _safe_float(sl_raw.get(key))
                if val is not None:
                    sl_cfg[key] = val
            if sl_cfg:
                profile_cfg["sl_buffer"] = sl_cfg

        bias_raw = meta.get("bias_relax")
        if isinstance(bias_raw, dict):
            bias_cfg: Dict[str, float] = {}
            for key, value in bias_raw.items():
                val = _safe_float(value)
                if val is not None:
                    bias_cfg[str(key)] = val
            if bias_cfg:
                profile_cfg["bias_relax"] = bias_cfg

        structure_raw = meta.get("structure")
        if isinstance(structure_raw, dict):
            structure_cfg: Dict[str, float] = {}
            for key, value in structure_raw.items():
                val = _safe_float(value)
                if val is not None:
                    structure_cfg[str(key)] = val
            if structure_cfg:
                profile_cfg["structure"] = structure_cfg

        momentum_raw = meta.get("momentum_override")
        if isinstance(momentum_raw, dict):
            momentum_cfg: Dict[str, float] = {}
            for key, value in momentum_raw.items():
                val = _safe_float(value)
                if val is not None:
                    momentum_cfg[str(key)] = val
            if momentum_cfg:
                profile_cfg["momentum_override"] = momentum_cfg

        rr_min_raw = meta.get("rr_min")
        if isinstance(rr_min_raw, dict):
            rr_min_cfg: Dict[str, float] = {}
            for key, value in rr_min_raw.items():
                val = _safe_float(value)
                if val is not None:
                    rr_min_cfg[str(key)] = val
            if rr_min_cfg:
                profile_cfg["rr_min"] = rr_min_cfg

        precision_raw = meta.get("precision")
        if isinstance(precision_raw, dict):
            precision_cfg: Dict[str, Any] = {}
            score_min = _safe_float(precision_raw.get("score_min"))
            if score_min is not None:
                precision_cfg["score_min"] = score_min
            for key in ("ready_timeout_minutes", "arming_timeout_minutes"):
                try:
                    timeout_val = precision_raw.get(key)
                    if timeout_val is not None:
                        precision_cfg[key] = int(timeout_val)
                except (TypeError, ValueError):
                    continue
            if precision_cfg:
                profile_cfg["precision"] = precision_cfg

        no_chase_val = _safe_float(meta.get("no_chase_r"))
        if no_chase_val is not None:
            profile_cfg["no_chase_r"] = no_chase_val
            
        if bool(meta.get("range_guard_requires_override")):
            profile_cfg["range_guard_requires_override"] = True

        if profile_cfg:
            result[str(profile_name)] = profile_cfg

    return result


# Module level shortcuts used by ``analysis.py`` and helper scripts.
ASSETS: List[str] = load_config()["assets"]
_ASSET_FILTER_ENV = os.getenv("ANALYSIS_ASSET_FILTER", "").strip()
if _ASSET_FILTER_ENV:
    _asset_filter = {
        item.strip().upper()
        for item in _ASSET_FILTER_ENV.split(",")
        if item and item.strip()
    }
    if _asset_filter:
        ASSETS = [
            asset
            for asset in ASSETS
            if asset.upper() in _asset_filter
        ]
LEVERAGE: Dict[str, float] = load_config()["leverage"]
DYNAMIC_LOGIC: Dict[str, Any] = _normalize_dynamic_logic(
    _get_config_value("dynamic_logic")
)
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

@lru_cache(maxsize=1)
def _load_asset_entry_profile_map() -> Dict[str, str]:
    """Return a normalised mapping of assets to entry profile names."""

    if not _ASSET_PROFILE_MAP_PATH.exists():
        return {}
    try:
        with _ASSET_PROFILE_MAP_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "Failed to load asset profile map from %s: %s", _ASSET_PROFILE_MAP_PATH, exc
        )
        return {}

    mapping: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return mapping

    for asset, profile in raw.items():
        if not isinstance(asset, str) or not asset.strip():
            continue
        asset_key = asset.strip().upper()
        if not isinstance(profile, str) or not profile.strip():
            continue
        profile_name = profile.strip()
        mapping[asset_key] = profile_name

    return mapping
    
_ENV_PROFILE = os.getenv("ENTRY_THRESHOLD_PROFILE")
_ENV_ACTIVE_PROFILE = os.getenv("ACTIVE_ENTRY_PROFILE")
_CFG_ACTIVE_PROFILE = _get_config_value("active_entry_threshold_profile")
_CFG_WEEKDAY_PROFILE = _get_config_value("weekday_entry_threshold_profile")
_ENTRY_PROFILE_SCHEDULE = _get_config_value("entry_profile_schedule_by_tod") or {}

def _weekday_profile_candidate() -> Optional[str]:
    try:
        candidate = str(_CFG_WEEKDAY_PROFILE)
    except Exception:
        return None
    if not candidate:
        return None
    now = _now_utc()
    if now.weekday() < 5:
        return candidate
    return None


_ACTIVE_PROFILE_NAME = (
    _ENV_ACTIVE_PROFILE
    or _ENV_PROFILE
    or _weekday_profile_candidate()
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


def _env_flag_enabled(flag_name: Optional[str], *, default: bool = False) -> bool:
    """Return whether an environment flag is enabled."""

    if not flag_name:
        return default
    raw = os.getenv(str(flag_name))
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _tod_buckets_from_config() -> List[Tuple[str, range]]:
    profiles = _get_config_value("atr_percentile_min_by_tod_profiles") or {}
    profile_cfg = profiles.get(ENTRY_THRESHOLD_PROFILE_NAME) or profiles.get(
        _DEFAULT_ENTRY_PROFILE_NAME
    )
    buckets: List[Tuple[str, range]] = []
    if not isinstance(profile_cfg, dict):
        return buckets
    for bucket in profile_cfg.get("buckets", []) or []:
        if not isinstance(bucket, dict):
            continue
        name = str(bucket.get("name") or "").strip()
        if not name:
            continue
        try:
            start_minute = int(bucket.get("start_minute", 0))
            end_minute = int(bucket.get("end_minute", 0))
        except (TypeError, ValueError):
            continue
        start_minute = max(0, start_minute)
        end_minute = max(start_minute, min(1440, end_minute))
        buckets.append((name, range(start_minute, end_minute)))
    return buckets


def _session_bucket(asset: Optional[str], now: Optional[datetime] = None) -> Optional[str]:
    ts = now or _now_utc()
    minute_of_day = ts.hour * 60 + ts.minute
    asset_key = str(asset or "").upper()
    proxy_asset = _SESSION_BUCKET_PROXIES_RAW.get(asset_key, asset_key)
    rules = SESSION_TIME_RULES.get(proxy_asset)
    if not isinstance(rules, dict):
        return None
    open_minute = rules.get("sunday_open_minute")
    close_minute = rules.get("friday_close_minute")
    if open_minute is None or close_minute is None:
        return None
    try:
        open_minute = int(open_minute)
        close_minute = int(close_minute)
    except (TypeError, ValueError):
        return None
    if minute_of_day < open_minute or minute_of_day > close_minute:
        return None
    session_span = max(1, close_minute - open_minute)
    segment = max(1, session_span // 3)
    if minute_of_day < open_minute + segment:
        return "open"
    if minute_of_day < close_minute - segment:
        return "mid"
    return "close"


def _current_tod_bucket(asset: Optional[str] = None, now: Optional[datetime] = None) -> Optional[str]:
    session_bucket = _session_bucket(asset, now)
    if session_bucket:
        return session_bucket
    ts = now or _now_utc()
    minute_of_day = ts.hour * 60 + ts.minute
    for name, bucket_range in _tod_buckets_from_config():
        if minute_of_day in bucket_range:
            return name
    return None


def time_of_day_bucket(asset: str, now: Optional[datetime] = None) -> Optional[str]:
    """Return the intraday bucket label for ``asset`` at ``now``."""

    return _current_tod_bucket(asset, now)


def _resolve_scheduled_profile(asset: str, now: Optional[datetime] = None) -> Optional[str]:
    bucket = _current_tod_bucket(asset, now)
    if not bucket:
        return None
    schedule_default = (_ENTRY_PROFILE_SCHEDULE.get("default") or {}).get(bucket)
    asset_cfg = (_ENTRY_PROFILE_SCHEDULE.get("assets") or {}).get(asset.upper(), {})
    schedule_asset = None
    if isinstance(asset_cfg, dict):
        schedule_asset = asset_cfg.get(bucket)
    profile_name = schedule_asset or schedule_default
    if profile_name and profile_name in ENTRY_THRESHOLD_PROFILES:
        return profile_name
    return None    
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


def get_entry_threshold_profile_name_for_asset(asset: str) -> str:
    """Return the entry threshold profile name configured for ``asset``."""

    asset_key = str(asset or "").upper()
    mapped_profile: Optional[str] = None
    
    if asset_key:
        mapping = _load_asset_entry_profile_map()
        profile_name = mapping.get(asset_key)
        if profile_name:
            if profile_name in ENTRY_THRESHOLD_PROFILES:
                if profile_name != ENTRY_THRESHOLD_PROFILE_NAME:
                    return profile_name
                mapped_profile = profile_name
            else:
                LOGGER.warning(
                    "Asset %s mapped to unknown entry profile '%s'; using %s",
                    asset_key,
                    profile_name,
                    ENTRY_THRESHOLD_PROFILE_NAME,
                )

    scheduled_profile = _resolve_scheduled_profile(asset_key)
    if scheduled_profile:
        return scheduled_profile
    if mapped_profile:
        return mapped_profile
    return ENTRY_THRESHOLD_PROFILE_NAME


def get_entry_threshold_profile_for_asset(asset: str) -> Dict[str, Dict[str, float]]:
    """Return the entry threshold profile dictionary for ``asset``."""

    profile_name = get_entry_threshold_profile_name_for_asset(asset)
    return ENTRY_THRESHOLD_PROFILES[profile_name]
    
BTC_PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = _normalize_btc_profiles(
    _get_config_value("btc_profile_overrides")
)

_SESSION_STATUS_PROFILES_RAW: Dict[str, Any] = dict(
    _get_config_value("session_status_profiles") or {}
)
SESSION_STATUS_PROFILES: Dict[str, Dict[str, Any]] = _normalize_session_status_profiles(
    _SESSION_STATUS_PROFILES_RAW
)
if "default" not in SESSION_STATUS_PROFILES:
    SESSION_STATUS_PROFILES["default"] = {}

_SESSION_STATUS_PROFILE_ASSETS: Dict[str, Set[str]] = {
    name: set(meta.get("assets", [])) for name, meta in SESSION_STATUS_PROFILES.items()
}


def _auto_session_status_profile(
    profiles: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    now = _now_utc()
    if now.weekday() >= 5:
        for name, meta in profiles.items():
            if meta.get("auto_activate_weekend"):
                return name
    return None


def _resolve_active_session_status_profile() -> str:
    env_value_raw = os.getenv("SESSION_STATUS_PROFILE")
    env_value = env_value_raw.strip() if isinstance(env_value_raw, str) else None
    cfg_value_raw = _get_config_value("active_session_status_profile")
    cfg_value = cfg_value_raw.strip() if isinstance(cfg_value_raw, str) else None
    auto_value = _auto_session_status_profile(SESSION_STATUS_PROFILES)

    candidates = [
        (env_value, "env"),
        (auto_value, "auto"),
        (cfg_value, "config"),
        ("default", "default"),
    ]
    for candidate, source in candidates:
        if not candidate:
            continue
        if candidate in SESSION_STATUS_PROFILES:
            return candidate
        if source == "env":
            LOGGER.warning(
                "Unknown session status profile '%s'; falling back to auto/config/default",
                candidate,
            )
    return "default"


SESSION_STATUS_PROFILE_NAME: str = "default"
SESSION_STATUS_PROFILE: Dict[str, Any] = {}


def refresh_session_status_profile() -> None:
    """Re-evaluate the active session status profile."""

    global SESSION_STATUS_PROFILE_NAME, SESSION_STATUS_PROFILE

    active = _resolve_active_session_status_profile()
    SESSION_STATUS_PROFILE_NAME = active
    SESSION_STATUS_PROFILE = dict(SESSION_STATUS_PROFILES.get(active, {}))


refresh_session_status_profile()


def session_status_profile_targets_asset(
    profile_name: Optional[str], asset: Optional[str]
) -> bool:
    asset_key = str(asset or "").strip().upper()
    if not asset_key:
        return True
    if not profile_name:
        return True
    assets = _SESSION_STATUS_PROFILE_ASSETS.get(profile_name)
    if not assets:
        return True
    return asset_key in assets


def _weekday_allows_auto_override(asset_key: str, weekday: int) -> bool:
    """Return ``True`` when the configured weekdays include ``weekday``."""

    allowed = SESSION_WEEKDAYS.get(asset_key)
    if not allowed:
        return False
    if isinstance(allowed, (set, frozenset)):
        iterable = allowed
    elif isinstance(allowed, (list, tuple)):
        iterable = allowed
    else:
        iterable = [allowed]
    for item in iterable:
        try:
            if int(item) == weekday:
                return True
        except (TypeError, ValueError):
            continue
    return False


def resolve_session_status_for_asset(
    asset: Optional[str], *, when: Optional[datetime] = None, weekday_ok: Optional[bool] = None
) -> Tuple[str, Dict[str, Any]]:
    """Return the active session status profile metadata for ``asset``.

    Parameters
    ----------
    asset:
        Asset symbol whose session profile should be resolved.
    when:
        Optional timestamp that represents the evaluation context.  This allows
        callers (for example historical test probes) to override the
        auto-activated weekend profile when the examined time belongs to a
        trading day.
    weekday_ok:
        Optional pre-computed weekday flag.  When ``True`` we treat the
        evaluation instant as an allowed trading day regardless of ``when``.
    """

    asset_key = str(asset or "").strip().upper()
    profile_name = SESSION_STATUS_PROFILE_NAME
    if not session_status_profile_targets_asset(profile_name, asset_key):
        default_profile = dict(SESSION_STATUS_PROFILES.get("default", {}))
        return "default", default_profile
        
    profile = dict(SESSION_STATUS_PROFILE)
    if profile_name != "default" and profile.get("auto_activate_weekend"):
        should_fallback = False
        if weekday_ok is True:
            should_fallback = True
        elif weekday_ok is None and when is not None:
            try:
                weekday = when.weekday()
            except Exception:
                weekday = None
            if weekday is not None:
                if _weekday_allows_auto_override(asset_key, weekday):
                    should_fallback = True
                elif not SESSION_WEEKDAYS.get(asset_key) and weekday < 5:
                    should_fallback = True
        if should_fallback:
            default_profile = dict(SESSION_STATUS_PROFILES.get("default", {}))
            return "default", default_profile

    return profile_name, profile
    
ATR_LOW_TH_DEFAULT: float = float(_get_config_value("atr_low_threshold_default"))
ATR_LOW_TH_ASSET: Dict[str, float] = dict(_get_config_value("atr_low_threshold") or {})
GOLD_HIGH_VOL_WINDOWS = _get_config_value("gold_high_vol_windows") or []
GOLD_HIGH_VOL_TH: float = float(_get_config_value("gold_high_vol_threshold") or 0.0)
GOLD_LOW_VOL_TH: float = float(_get_config_value("gold_low_vol_threshold") or 0.0)
SMT_PENALTY_VALUE: int = int(_get_config_value("smt_penalty_value") or 0)
SMT_REQUIRED_BARS: int = int(_get_config_value("smt_required_bars") or 0)
SMT_AUTO_CONFIG: Dict[str, Any] = dict(_get_config_value("smt_auto_config") or {})
TP_NET_MIN_DEFAULT: float = float(_get_config_value("tp_net_min_default") or 0.0)
TP_NET_MIN_ASSET: Dict[str, float] = dict(_get_config_value("tp_net_min") or {})
TP_MIN_PCT: Dict[str, float] = dict(_get_config_value("tp_min_pct") or {})
TP_MIN_ABS: Dict[str, float] = dict(_get_config_value("tp_min_abs") or {})
SL_BUFFER_RULES: Dict[str, Any] = dict(_get_config_value("sl_buffer_rules") or {})
RISK_TEMPLATES: Dict[str, Dict[str, Any]] = _normalize_risk_templates(
    _get_config_value("risk_templates")
)
MIN_RISK_ABS: Dict[str, float] = dict(_get_config_value("min_risk_abs") or {})
ACTIVE_INVALID_BUFFER_ABS: Dict[str, float] = dict(_get_config_value("active_invalid_buffer_abs") or {})
ASSET_COST_MODEL: Dict[str, Any] = dict(_get_config_value("asset_cost_model") or {})
DEFAULT_COST_MODEL: Dict[str, Any] = dict(_get_config_value("default_cost_model") or {})
COST_MULT_DEFAULT: float = float(_get_config_value("cost_mult_default") or 0.0)
COST_MULT_HIGH_VOL: float = float(_get_config_value("cost_mult_high_vol") or 0.0)
ATR5_MIN_MULT: float = float(_get_config_value("atr5_min_mult") or 0.0)
_ATR5_MIN_MULT_ASSET_RAW = dict(_get_config_value("atr5_min_mult_asset") or {})
ATR5_MIN_MULT_ASSET: Dict[str, float] = {}
for key, value in _ATR5_MIN_MULT_ASSET_RAW.items():
    try:
        ATR5_MIN_MULT_ASSET[str(key)] = float(value)
    except (TypeError, ValueError):
        continue
ATR_VOL_HIGH_REL: float = float(_get_config_value("atr_vol_high_rel") or 0.0)
EMA_SLOPE_TH_DEFAULT: float = float(_get_config_value("ema_slope_th_default") or 0.0)
EMA_SLOPE_TH_ASSET: Dict[str, float] = dict(_get_config_value("ema_slope_th_asset") or {})
EMA_SLOPE_SIGN_ENFORCED: set = load_config()["ema_slope_sign_enforced"]
ATR_ABS_MIN: Dict[str, float] = dict(_get_config_value("atr_abs_min") or {})
XAGUSD_ATR_5M_FLOOR: float = float(_get_config_value("xagusd_atr_5m_floor") or 0.0)
XAGUSD_ATR_5M_FLOOR_ENABLED: bool = bool(
    _get_config_value("xagusd_atr_5m_floor_enabled")
)
CORE_RR_MIN: Dict[str, float] = dict(_get_config_value("core_rr_min") or {})
MOMENTUM_RR_MIN: Dict[str, float] = dict(_get_config_value("momentum_rr_min") or {})
FX_TP_TARGETS: Dict[str, float] = dict(_get_config_value("fx_tp_targets") or {})
NVDA_EXTENDED_ATR_REL: float = float(_get_config_value("nvda_extended_atr_rel") or 0.0)
NVDA_MOMENTUM_ATR_REL: float = float(_get_config_value("nvda_momentum_atr_rel") or 0.0)
NVDA_DAILY_ATR_MULTIPLIER: float = float(_get_config_value("nvda_daily_atr_multiplier") or 0.0)
NVDA_DAILY_ATR_MIN: float = float(_get_config_value("nvda_daily_atr_min") or 0.0)
NVDA_DAILY_ATR_STRONG: float = float(_get_config_value("nvda_daily_atr_strong") or 0.0)
NVDA_LOW_ATR_P_SCORE_ADD: float = float(_get_config_value("nvda_low_atr_p_score_add") or 0.0)
NVDA_RR_BANDS: Dict[str, Any] = dict(_get_config_value("nvda_rr_bands") or {})
_NVDA_STOP_ATR = dict(_get_config_value("nvda_stop_atr_mult") or {})
NVDA_STOP_ATR_MIN: float = float(_NVDA_STOP_ATR.get("min") or 0.0)
NVDA_STOP_ATR_MAX: float = float(_NVDA_STOP_ATR.get("max") or 0.0)
NVDA_POSITION_SCALE: Dict[str, float] = {
    key: float(value)
    for key, value in dict(_get_config_value("nvda_position_scale") or {}).items()
}

LOW_ATR_OVERRIDES: Dict[str, Dict[str, float]] = _normalize_low_atr_overrides(
    _get_config_value("low_atr_overrides")
)


_REALTIME_PRICE_GUARD_FALLBACK: Dict[str, float] = {"limit_pct": 0.01, "min_abs": 0.1}


def _normalize_price_guard_map(raw: Any) -> Dict[str, Dict[str, float]]:
    """Validate and normalise realtime price guard thresholds."""

    guard_map: Dict[str, Dict[str, float]] = {
        "default": dict(_REALTIME_PRICE_GUARD_FALLBACK)
    }
    if not isinstance(raw, dict):
        return guard_map

    for key, meta in raw.items():
        guard_key = "default" if str(key).lower() == "default" else str(key).upper()
        if not isinstance(meta, dict):
            LOGGER.warning(
                "Ignoring realtime_price_guard entry for %s — expected mapping, got %r",
                guard_key,
                type(meta).__name__,
            )
            continue
        limit_val = meta.get("limit_pct")
        min_abs_val = meta.get("min_abs")
        try:
            limit_pct = float(limit_val)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid limit_pct %r for realtime_price_guard[%s] — using fallback",
                limit_val,
                guard_key,
            )
            continue
        if limit_pct <= 0:
            LOGGER.warning(
                "Non-positive limit_pct %s for realtime_price_guard[%s] — using fallback",
                limit_pct,
                guard_key,
            )
            continue
        try:
            min_abs = float(min_abs_val) if min_abs_val is not None else _REALTIME_PRICE_GUARD_FALLBACK["min_abs"]
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid min_abs %r for realtime_price_guard[%s] — using fallback",
                min_abs_val,
                guard_key,
            )
            min_abs = _REALTIME_PRICE_GUARD_FALLBACK["min_abs"]
        guard_map[guard_key] = {
            "limit_pct": float(limit_pct),
            "min_abs": max(0.0, float(min_abs)),
        }

    if "default" not in guard_map:
        guard_map["default"] = dict(_REALTIME_PRICE_GUARD_FALLBACK)
    else:
        guard_map["default"].setdefault("min_abs", _REALTIME_PRICE_GUARD_FALLBACK["min_abs"])

    return guard_map


REALTIME_PRICE_GUARD: Dict[str, Dict[str, float]] = _normalize_price_guard_map(
    _get_config_value("realtime_price_guard")
)


def get_realtime_price_guard(asset: Optional[str]) -> Dict[str, float]:
    """Return the configured realtime price guard thresholds for ``asset``."""

    asset_key = str(asset or "").upper()
    if asset_key and asset_key in REALTIME_PRICE_GUARD:
        return dict(REALTIME_PRICE_GUARD[asset_key])
    return dict(REALTIME_PRICE_GUARD.get("default", _REALTIME_PRICE_GUARD_FALLBACK))


def get_low_atr_override(asset: str) -> Dict[str, float]:
    """Return ATR floor overrides merged with the default mapping."""

    base = dict(LOW_ATR_OVERRIDES.get("default", {}))
    asset_cfg = LOW_ATR_OVERRIDES.get(str(asset).upper(), {})
    base.update(asset_cfg)
    return base


def _resolve_momentum_assets() -> Set[str]:
    """Return the configured momentum-capable assets with fallbacks."""

    try:
        raw_assets = load_config().get("enable_momentum_assets", [])
    except AnalysisConfigError:
        raw_assets = []
    except Exception:
        raw_assets = []

    enabled: Set[str] = {
        str(asset)
        for asset in raw_assets
        if isinstance(asset, str)
    }
    enabled.add("BTCUSD")
    return enabled


ENABLE_MOMENTUM_ASSETS: Set[str] = _resolve_momentum_assets()
INTRADAY_ATR_RELAX: Dict[str, float] = {
    k: float(v) for k, v in dict(_get_config_value("intraday_atr_relax") or {}).items()
}
INTRADAY_BIAS_RELAX: Dict[str, Dict[str, Any]] = _normalize_bias_relax(
    _get_config_value("intraday_bias_relax")
)
SESSION_WINDOWS_UTC: Dict[str, Dict[str, Optional[List[Tuple[int, int, int, int]]]]] = load_config()[
    "session_windows_utc"
]
SPOT_MAX_AGE_SECONDS: Dict[str, int] = {
    k: int(v) for k, v in dict(_get_config_value("spot_max_age_seconds") or {}).items()
}
_RAW_LATENCY_GUARD = dict(_get_config_value("data_latency_guard") or {})
DATA_LATENCY_GUARD: Dict[str, Dict[str, Any]] = {}
for key, value in _RAW_LATENCY_GUARD.items():
    if isinstance(value, dict):
        DATA_LATENCY_GUARD[str(key).upper()] = dict(value)
INTERVENTION_WATCH_DEFAULT: Dict[str, Any] = dict(_get_config_value("intervention_watch_default") or {})
SESSION_WEEKDAYS: Dict[str, Any] = dict(_get_config_value("session_weekdays") or {})
SESSION_TIME_RULES: Dict[str, Dict[str, Any]] = load_config()["session_time_rules"]
SESSION_BUCKET_PROXIES: Dict[str, str] = dict(_SESSION_BUCKET_PROXIES_RAW)
_ATR_PERCENTILE_PROFILES: Dict[str, Any] = dict(
    _get_config_value("atr_percentile_min_by_tod_profiles") or {}
)


def _resolve_atr_percentiles() -> Dict[str, Any]:
    if _ATR_PERCENTILE_PROFILES:
        profile_cfg = dict(
            _ATR_PERCENTILE_PROFILES.get(ENTRY_THRESHOLD_PROFILE_NAME) or {}
        )
        if not profile_cfg:
            profile_cfg = dict(_ATR_PERCENTILE_PROFILES.get("baseline") or {})
        if profile_cfg:
            return profile_cfg
    return dict(_get_config_value("atr_percentile_min_by_tod") or {})


ATR_PERCENTILE_TOD: Dict[str, Any] = _resolve_atr_percentiles()
P_SCORE_TIME_BONUS: Dict[str, Any] = dict(_get_config_value("p_score_time_bonus") or {})
ADX_RR_BANDS: Dict[str, Any] = dict(_get_config_value("adx_rr_bands") or {})
ASSET_CLASS_MAP: Dict[str, str] = {
    str(asset).upper(): str(class_name)
    for asset, class_name in dict(_get_config_value("asset_classes") or {}).items()
    if isinstance(class_name, str)
}

_RAW_SPREAD_MAX = dict(_get_config_value("spread_max_atr_pct") or {})
SPREAD_MAX_ATR_PCT: Dict[str, float] = {}
for key, value in _RAW_SPREAD_MAX.items():
    try:
        SPREAD_MAX_ATR_PCT[str(key)] = float(value)
    except (TypeError, ValueError):
        continue

_ATR_PERIOD_PROFILES = _normalize_profiled_numeric_map(
    _get_config_value("atr_period_profiles"), lambda value: int(float(value)), 14
)
_ATR_ABS_MIN_PROFILES = _normalize_profiled_numeric_map(
    _get_config_value("atr_abs_min_profiles"), float, 0.0
)
_SPREAD_PROFILE_DEFAULT = float(SPREAD_MAX_ATR_PCT.get("default", 0.0) or 0.0)
_SPREAD_MAX_ATR_PCT_PROFILES = _normalize_profiled_numeric_map(
    _get_config_value("spread_max_atr_pct_profiles"), float, _SPREAD_PROFILE_DEFAULT
)
_FIB_TOLERANCE_PROFILES = _normalize_fib_tolerance_profiles(
    _get_config_value("fib_tolerance_profiles")
)
_FIB_RELAX_FLAG: Optional[str] = _get_config_value("fib_relax_env_flag")
_FIB_RELAX_DEFAULT: bool = bool(_get_config_value("fib_relax_default_enabled"))
FIB_RELAX_ENABLED: bool = _env_flag_enabled(
    _FIB_RELAX_FLAG or "RELAX_FIB", default=_FIB_RELAX_DEFAULT
)
_RR_RELAX_CONFIG: Dict[str, Any] = dict(_get_config_value("rr_relax_settings") or {})
RR_RELAX_ENABLED: bool = _env_flag_enabled(
    str(_RR_RELAX_CONFIG.get("env_flag") or "RR_RELAX").strip() or None,
    default=bool(_RR_RELAX_CONFIG.get("enabled", False)),
)
RR_RELAX_RANGE_MOMENTUM: float = float(_RR_RELAX_CONFIG.get("range_momentum", 1.35))
RR_RELAX_RANGE_CORE: float = float(_RR_RELAX_CONFIG.get("range_core", 1.5))
RR_RELAX_ATR_RATIO_TRIGGER: float = float(_RR_RELAX_CONFIG.get("atr_ratio_trigger", 0.92))
_MAX_RISK_PCT_PROFILES = _normalize_profiled_numeric_map(
    _get_config_value("max_risk_pct_profiles"), float, 1.5
)
_BOS_LOOKBACK_PROFILES = _normalize_profiled_numeric_map(
    _get_config_value("bos_lookback_profiles"), lambda value: int(float(value)), 30
)

_RAW_VWAP_BAND_MULT = dict(_get_config_value("vwap_band_mult") or {})
VWAP_BAND_MULT: Dict[str, float] = {}
for key, value in _RAW_VWAP_BAND_MULT.items():
    try:
        VWAP_BAND_MULT[str(key)] = float(value)
    except (TypeError, ValueError):
        continue

_RAW_OFI_Z = dict(_get_config_value("ofi_z_th") or {})
OFI_Z_SETTINGS: Dict[str, float] = {}
for key, value in _RAW_OFI_Z.items():
    try:
        OFI_Z_SETTINGS[str(key)] = float(value)
    except (TypeError, ValueError):
        continue

NEWS_MODE_SETTINGS: Dict[str, Any] = dict(_get_config_value("news_mode_settings") or {})
FUNDING_RATE_RULES: Dict[str, Any] = dict(_get_config_value("funding_rate_rules") or {})


def get_p_score_min(asset: str) -> float:
    """Return the configured minimum P-score for the active profile."""

    return P_SCORE_MIN_ASSET.get(asset, P_SCORE_MIN_DEFAULT)


def is_momentum_asset(asset: str) -> bool:
    """Return ``True`` when ``asset`` participates in the momentum strategy."""

    return asset in _resolve_momentum_assets()


def get_atr_threshold_multiplier(asset: str) -> float:
    """Return the ATR threshold multiplier for the active profile."""

    return ATR_THRESHOLD_MULT_ASSET.get(asset, ATR_THRESHOLD_MULT_DEFAULT)


def get_atr_period(asset: str, profile: Optional[str] = None) -> int:
    """Return the ATR calculation window (bars) for the requested profile."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    profile_cfg = _ATR_PERIOD_PROFILES.get(profile_name) or {}
    overrides = profile_cfg.get("by_asset", {})
    if asset in overrides:
        return int(overrides[asset])
    default_val = profile_cfg.get("default")
    if default_val not in (None, 0):
        return int(default_val)
    baseline_default = _ATR_PERIOD_PROFILES.get("baseline", {}).get("default")
    if baseline_default not in (None, 0):
        return int(baseline_default)
    return 14


def get_atr_abs_min(asset: str, profile: Optional[str] = None) -> Optional[float]:
    """Return the minimum absolute ATR threshold for the profile if configured."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    profile_cfg = _ATR_ABS_MIN_PROFILES.get(profile_name) or {}
    overrides = profile_cfg.get("by_asset", {})
    value = overrides.get(asset)
    if value not in (None, 0):
        return float(value)
    default_val = profile_cfg.get("default")
    if default_val not in (None, 0):
        return float(default_val)
    base = ATR_ABS_MIN.get(asset)
    if base is not None:
        return float(base)
    return None


def get_risk_template(asset: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Return the merged risk template for ``asset`` and profile."""

    asset_key = str(asset or "").upper()
    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    merged: Dict[str, Any] = {}

    def merge(source: Optional[Dict[str, Any]]) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            if isinstance(value, dict):
                existing = merged.get(key)
                if isinstance(existing, dict):
                    existing.update(value)
                else:
                    merged[key] = dict(value)
            else:
                merged[key] = value

    baseline_cfg = RISK_TEMPLATES.get("baseline") or {}
    merge(baseline_cfg.get("default"))
    merge((baseline_cfg.get("assets") or {}).get(asset_key))

    profile_cfg = RISK_TEMPLATES.get(profile_name)
    if profile_cfg and profile_name != "baseline":
        merge(profile_cfg.get("default"))
        merge((profile_cfg.get("assets") or {}).get(asset_key))
    elif profile_cfg:
        merge((profile_cfg.get("assets") or {}).get(asset_key))

    return merged


def get_tp_net_min(asset: str, profile: Optional[str] = None) -> float:
    template_value = get_risk_template(asset, profile).get("tp_net_min")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid tp_net_min %r in risk template for %s", template_value, asset
            )
    return float(TP_NET_MIN_ASSET.get(asset, TP_NET_MIN_DEFAULT))


def get_tp_min_pct_value(asset: str, profile: Optional[str] = None) -> float:
    template_value = get_risk_template(asset, profile).get("tp_min_pct")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid tp_min_pct %r in risk template for %s", template_value, asset
            )
    return float(TP_MIN_PCT.get(asset, TP_MIN_PCT.get("default", 0.0)))


def get_tp_min_abs_value(asset: str, profile: Optional[str] = None) -> float:
    template_value = get_risk_template(asset, profile).get("tp_min_abs")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid tp_min_abs %r in risk template for %s", template_value, asset
            )
    base = TP_MIN_ABS.get(asset)
    if base is None:
        base = TP_MIN_ABS.get("default", 0.0)
    return float(base)


def get_sl_buffer_config(asset: str, profile: Optional[str] = None) -> Dict[str, float]:
    template_buffer = get_risk_template(asset, profile).get("sl_buffer")
    result: Dict[str, float] = {}
    if isinstance(template_buffer, dict):
        for key in ("atr_mult", "abs_min"):
            value = template_buffer.get(key)
            if value is None:
                continue
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Invalid %s value %r in sl_buffer template for %s",
                    key,
                    value,
                    asset,
                )
    base_rules: Optional[Dict[str, Any]] = None
    if asset in SL_BUFFER_RULES:
        raw_rule = SL_BUFFER_RULES.get(asset)
        if isinstance(raw_rule, dict):
            base_rules = raw_rule
    if base_rules is None:
        raw_default = SL_BUFFER_RULES.get("default")
        if isinstance(raw_default, dict):
            base_rules = raw_default
    if base_rules:
        for key in ("atr_mult", "abs_min"):
            if key in result:
                continue
            value = base_rules.get(key)
            if value is None:
                continue
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
    return result


def get_max_slippage_r(asset: str, profile: Optional[str] = None) -> Optional[float]:
    template_value = get_risk_template(asset, profile).get("max_slippage_r")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid max_slippage_r %r in risk template for %s",
                template_value,
                asset,
            )
    return None


def get_spread_max_atr_pct(asset: str, profile: Optional[str] = None) -> float:
    """Return the maximum spread/ATR ratio for the given profile."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    template_value = get_risk_template(asset, profile_name).get("max_spread_atr_pct")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid max_spread_atr_pct %r in risk template for %s",
                template_value,
                asset,
            )
    profile_cfg = _SPREAD_MAX_ATR_PCT_PROFILES.get(profile_name) or {}
    overrides = profile_cfg.get("by_asset", {})
    value = overrides.get(asset)
    if value is not None:
        return float(value)
    default_val = profile_cfg.get("default")
    if default_val is not None and default_val > 0:
        return float(default_val)
    base = SPREAD_MAX_ATR_PCT.get(asset, SPREAD_MAX_ATR_PCT.get("default", 0.0))
    return float(base or 0.0)


def get_fib_tolerance(asset: str, profile: Optional[str] = None) -> float:
    """Return the Fib tolerance fraction for the profile/asset combination."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    fib_profile = profile_name
    if profile_name == "relaxed" and not FIB_RELAX_ENABLED:
        fib_profile = "baseline"
    profile_cfg = _FIB_TOLERANCE_PROFILES.get(fib_profile) or {}
    overrides = profile_cfg.get("by_asset", {})
    asset_key = str(asset or "").upper()
    value = overrides.get(asset_key)
    if value is not None and value > 0:
        return float(value)
    class_overrides = profile_cfg.get("by_class", {})
    asset_class = ASSET_CLASS_MAP.get(asset_key)
    if asset_class:
        class_value = class_overrides.get(asset_class)
        if class_value is not None and class_value > 0:
            return float(class_value)
    default_val = profile_cfg.get("default")
    if default_val is not None and default_val > 0:
        return float(default_val)
    baseline_default = _FIB_TOLERANCE_PROFILES.get("baseline", {}).get("default")
    if baseline_default is not None and baseline_default > 0:
        return float(baseline_default)
    return 0.02


def get_max_risk_pct(asset: str, profile: Optional[str] = None) -> float:
    """Return the maximum per-trade risk percentage for the active profile."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    template_value = get_risk_template(asset, profile_name).get("risk_pct")
    if template_value is not None:
        try:
            return float(template_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid risk_pct %r in risk template for %s", template_value, asset
            )
    profile_cfg = _MAX_RISK_PCT_PROFILES.get(profile_name) or {}
    overrides = profile_cfg.get("by_asset", {})
    value = overrides.get(asset)
    if value is not None and value > 0:
        return float(value)
    default_val = profile_cfg.get("default")
    if default_val is not None and default_val > 0:
        return float(default_val)
    baseline_default = _MAX_RISK_PCT_PROFILES.get("baseline", {}).get("default")
    if baseline_default is not None and baseline_default > 0:
        return float(baseline_default)
    return 1.5


def get_bos_lookback(asset: Optional[str] = None, profile: Optional[str] = None) -> int:
    """Return the BOS/CHoCH lookback window in bars for the selected profile."""

    profile_name = profile or ENTRY_THRESHOLD_PROFILE_NAME
    profile_cfg = _BOS_LOOKBACK_PROFILES.get(profile_name) or {}
    overrides = profile_cfg.get("by_asset", {})
    if asset and asset in overrides:
        return int(overrides[asset])
    default_val = profile_cfg.get("default")
    if default_val not in (None, 0):
        return int(default_val)
    baseline_default = _BOS_LOOKBACK_PROFILES.get("baseline", {}).get("default")
    if baseline_default not in (None, 0):
        return int(baseline_default)
    return 30


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
    
