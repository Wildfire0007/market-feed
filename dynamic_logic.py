"""Dynamic, opt-in analysis logic helpers.

This module isolates optional scoring and guardrail adjustments from the
monolithic ``analysis.py`` workflow so they can evolve independently of the
core data plumbing.  Each helper is defensive by default and returns metadata
that callers can surface in diagnostics.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _coerce_float(
    value: Any,
    *,
    path: str,
    warnings: List[str],
    allow_negative: bool = True,
    allow_none: bool = True,
) -> Optional[float]:
    """Convert ``value`` to float with validation side effects."""

    if value is None and allow_none:
        return None
    coerced = _safe_float(value)
    if coerced is None:
        warnings.append(f"Invalid numeric value at {path}: {value!r}")
        return None
    if not allow_negative and coerced < 0:
        warnings.append(f"Negative value not allowed at {path}: {coerced}")
        return None
    return coerced


def _is_enabled(config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get("enabled"))


def validate_dynamic_logic_config(
    config: Any, *, logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a sanitized dynamic logic config and validation warnings."""

    log = logger or LOGGER
    warnings: List[str] = []
    default_cfg: Dict[str, Any] = {
        "enabled": False,
        "p_score": {
            "volatility_bonus": {"enabled": False, "z_threshold": None, "points": None},
            "momentum_boost": {"enabled": False, "adx_min": None, "points": None},
            "regime_penalty": {"enabled": False, "adx_max": None, "points": None},
        },
        "soft_gates": {
            "atr": {"enabled": False, "tolerance_pct": 0.15, "penalty_max": 6.0},
            "p_score": {"enabled": False, "gap_max": 2.0, "size_scale": 0.7, "rr_add": 0.15},
        },
        "latency_relaxation": {
            "profiles": {"strict": {"limit": None, "penalty": 0.0}},
            "asset_map": {"default": "strict"},
        },
    }

    if not isinstance(config, dict):
        return default_cfg, warnings

    sanitized: Dict[str, Any] = {"enabled": bool(config.get("enabled", False))}

    p_score_cfg: Dict[str, Any] = {}
    raw_p_score = config.get("p_score")
    if isinstance(raw_p_score, dict):
        vol_raw = raw_p_score.get("volatility_bonus")
        if isinstance(vol_raw, dict):
            vol_cfg: Dict[str, Any] = {"enabled": bool(vol_raw.get("enabled"))}
            vol_cfg["z_threshold"] = _coerce_float(
                vol_raw.get("z_threshold"), path="p_score.volatility_bonus.z_threshold", warnings=warnings
            )
            vol_cfg["points"] = _coerce_float(
                vol_raw.get("points"), path="p_score.volatility_bonus.points", warnings=warnings
            )
            p_score_cfg["volatility_bonus"] = vol_cfg

        momentum_raw = raw_p_score.get("momentum_boost")
        if isinstance(momentum_raw, dict):
            momentum_cfg: Dict[str, Any] = {"enabled": bool(momentum_raw.get("enabled"))}
            momentum_cfg["adx_min"] = _coerce_float(
                momentum_raw.get("adx_min"),
                path="p_score.momentum_boost.adx_min",
                warnings=warnings,
            )
            momentum_cfg["points"] = _coerce_float(
                momentum_raw.get("points"), path="p_score.momentum_boost.points", warnings=warnings
            )
            p_score_cfg["momentum_boost"] = momentum_cfg

        regime_raw = raw_p_score.get("regime_penalty")
        if isinstance(regime_raw, dict):
            regime_cfg: Dict[str, Any] = {"enabled": bool(regime_raw.get("enabled"))}
            regime_cfg["adx_max"] = _coerce_float(
                regime_raw.get("adx_max"),
                path="p_score.regime_penalty.adx_max",
                warnings=warnings,
            )
            regime_cfg["points"] = _coerce_float(
                regime_raw.get("points"), path="p_score.regime_penalty.points", warnings=warnings
            )
            p_score_cfg["regime_penalty"] = regime_cfg

    sanitized["p_score"] = p_score_cfg

    soft_gates_cfg: Dict[str, Any] = {}
    raw_soft = config.get("soft_gates")
    if isinstance(raw_soft, dict):
        atr_raw = raw_soft.get("atr")
        if isinstance(atr_raw, dict):
            atr_cfg: Dict[str, Any] = {"enabled": bool(atr_raw.get("enabled"))}
            atr_cfg["tolerance_pct"] = _coerce_float(
                atr_raw.get("tolerance_pct"),
                path="soft_gates.atr.tolerance_pct",
                warnings=warnings,
                allow_negative=False,
            )
            atr_cfg["penalty_max"] = _coerce_float(
                atr_raw.get("penalty_max"),
                path="soft_gates.atr.penalty_max",
                warnings=warnings,
                allow_negative=False,
            )
            soft_gates_cfg["atr"] = atr_cfg
        p_score_raw = raw_soft.get("p_score")
        if isinstance(p_score_raw, dict):
            p_score_soft_cfg: Dict[str, Any] = {"enabled": bool(p_score_raw.get("enabled"))}
            p_score_soft_cfg["gap_max"] = _coerce_float(
                p_score_raw.get("gap_max"),
                path="soft_gates.p_score.gap_max",
                warnings=warnings,
                allow_negative=False,
            )
            p_score_soft_cfg["size_scale"] = _coerce_float(
                p_score_raw.get("size_scale"),
                path="soft_gates.p_score.size_scale",
                warnings=warnings,
                allow_negative=False,
            )
            p_score_soft_cfg["rr_add"] = _coerce_float(
                p_score_raw.get("rr_add"),
                path="soft_gates.p_score.rr_add",
                warnings=warnings,
                allow_negative=False,
            )
            soft_gates_cfg["p_score"] = p_score_soft_cfg
    sanitized["soft_gates"] = soft_gates_cfg

    latency_cfg: Dict[str, Any] = {"profiles": {}, "asset_map": {}}
    raw_latency = config.get("latency_relaxation")
    if isinstance(raw_latency, dict):
        profiles_raw = raw_latency.get("profiles") if isinstance(raw_latency.get("profiles"), dict) else {}
        profiles: Dict[str, Dict[str, Any]] = {}
        for name, meta in profiles_raw.items():
            if not isinstance(meta, dict):
                warnings.append(f"Ignoring latency profile {name!r}: expected mapping")
                continue
            profile_cfg: Dict[str, Any] = {}
            profile_cfg["limit"] = _coerce_float(
                meta.get("limit"), path=f"latency_relaxation.profiles.{name}.limit", warnings=warnings
            )
            profile_cfg["penalty"] = _coerce_float(
                meta.get("penalty"),
                path=f"latency_relaxation.profiles.{name}.penalty",
                warnings=warnings,
                allow_none=False,
            )
            profiles[str(name)] = profile_cfg
        if "strict" not in profiles:
            profiles["strict"] = {"limit": None, "penalty": 0.0}
        latency_cfg["profiles"] = profiles

        asset_map_raw = raw_latency.get("asset_map")
        if isinstance(asset_map_raw, dict):
            latency_cfg["asset_map"] = {
                str(asset).upper(): str(profile)
                for asset, profile in asset_map_raw.items()
                if isinstance(asset, str) and isinstance(profile, str)
            }
    else:
        latency_cfg["profiles"] = {"strict": {"limit": None, "penalty": 0.0}}
        latency_cfg["asset_map"] = {"default": "strict"}

    if "default" not in latency_cfg.get("asset_map", {}):
        latency_cfg.setdefault("asset_map", {})["default"] = "strict"

    sanitized["latency_relaxation"] = latency_cfg

    for warning in warnings:
        log.warning(warning)

    return sanitized, warnings


def apply_dynamic_p_score(
    current_score: float,
    config: Optional[Dict[str, Any]],
    *,
    volatility_overlay: Optional[Dict[str, Any]] = None,
    adx_value: Optional[float] = None,
    adx_regime: Optional[str] = None,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """Return a score adjusted with the dynamic logic configuration.

    The helper is intentionally conservative: if any inputs are missing or the
    configuration is disabled no changes are applied.
    """

    if not _is_enabled(config):
        return current_score, [], {}

    updated_score = current_score
    notes: List[str] = []
    meta: Dict[str, Any] = {}

    p_score_cfg = config.get("p_score", {}) if isinstance(config, dict) else {}

    # Volatility-driven bonus when implied volatility materially exceeds
    # realised observations.
    vol_cfg = p_score_cfg.get("volatility_bonus") if isinstance(p_score_cfg, dict) else {}
    z_threshold = vol_cfg.get("z_threshold") if isinstance(vol_cfg, dict) else None
    vol_points = vol_cfg.get("points") if isinstance(vol_cfg, dict) else None
    overlay_ratio = None
    if isinstance(volatility_overlay, dict):
        overlay_ratio = _safe_float(volatility_overlay.get("implied_realised_ratio"))
    if (
        _is_enabled(vol_cfg)
        and z_threshold is not None
        and vol_points is not None
        and overlay_ratio is not None
        and overlay_ratio >= z_threshold
    ):
        updated_score += vol_points
        notes.append(
            f"Dinamikus P-score: emelt volatilitás prémium (+{vol_points:.1f})"
        )
        meta["volatility_bonus"] = {
            "ratio": overlay_ratio,
            "threshold": z_threshold,
            "points": vol_points,
        }

    # ADX-driven momentum bump.
    momentum_cfg = p_score_cfg.get("momentum_boost") if isinstance(p_score_cfg, dict) else {}
    adx_min = momentum_cfg.get("adx_min") if isinstance(momentum_cfg, dict) else None
    momentum_points = momentum_cfg.get("points") if isinstance(momentum_cfg, dict) else None
    if _is_enabled(momentum_cfg) and adx_min is not None and momentum_points is not None:
        adx_value_normalized = _safe_float(adx_value)
        if adx_value_normalized is not None and adx_value_normalized >= adx_min:
            updated_score += momentum_points
            notes.append(
                f"Dinamikus P-score: ADX momentum boost (+{momentum_points:.1f})"
            )
            meta["momentum_boost"] = {
                "adx": adx_value_normalized,
                "threshold": adx_min,
                "points": momentum_points,
            }

    # Conditional regime penalty for low-momentum environments.
    regime_cfg = p_score_cfg.get("regime_penalty") if isinstance(p_score_cfg, dict) else {}
    adx_cap = regime_cfg.get("adx_max") if isinstance(regime_cfg, dict) else None
    regime_points = regime_cfg.get("points") if isinstance(regime_cfg, dict) else None
    adx_value_normalized = _safe_float(adx_value)
    if _is_enabled(regime_cfg) and adx_cap is not None and regime_points is not None:
        if adx_value_normalized is not None and adx_value_normalized <= adx_cap:
            updated_score += regime_points
            notes.append(
                f"Regime penalty: alacsony ADX ({adx_value_normalized:.1f}) ({regime_points:+.1f})"
            )
            meta["regime_penalty"] = {
                "adx": adx_value_normalized,
                "threshold": adx_cap,
                "points": regime_points,
                "regime": adx_regime,
            }

    return updated_score, notes, meta


class DynamicScoreEngine:
    """Compute dynamic P-score adjustments in a dedicated workflow."""

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, *, validate_config: bool = True
    ) -> None:
        if validate_config:
            sanitized, _ = validate_dynamic_logic_config(config or {})
            self.config = sanitized
        else:
            self.config = config or {}

    def score(
        self,
        base_p_score: float,
        regime_data: Optional[Dict[str, Any]],
        volatility_data: Optional[Dict[str, Any]],
        *,
        atr_soft_gate_penalty: float = 0.0,
        latency_penalty: float = 0.0,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        """Return final P-score with regime/volatility/penalty adjustments."""

        score = _safe_float(base_p_score) or 0.0
        notes: List[str] = []
        meta: Dict[str, Any] = {"base_score": score}

        p_score_cfg = self.config.get("p_score") if isinstance(self.config, dict) else {}

        regime_cfg = p_score_cfg.get("regime_penalty") if isinstance(p_score_cfg, dict) else {}
        regime_points = regime_cfg.get("points") if isinstance(regime_cfg, dict) else None
        regime_label = str((regime_data or {}).get("label") or "").upper()
        if _is_enabled(regime_cfg) and regime_points is not None and regime_label == "CHOPPY":
            score += regime_points
            meta["regime_penalty"] = {
                "label": regime_label,
                "points": regime_points,
            }
            adjustment = regime_points
            if adjustment < 0:
                notes.append(
                    f"P-Score reduced by {abs(adjustment):.1f} due to regime {regime_label}"
                )
            else:
                notes.append(
                    f"P-Score increased by {adjustment:.1f} due to regime {regime_label}"
                )

        vol_cfg = p_score_cfg.get("volatility_bonus") if isinstance(p_score_cfg, dict) else {}
        vol_threshold = vol_cfg.get("z_threshold") if isinstance(vol_cfg, dict) else None
        vol_bonus = vol_cfg.get("points") if isinstance(vol_cfg, dict) else None
        volatility_z = _safe_float((volatility_data or {}).get("volatility_z"))
        if volatility_z is None:
            volatility_z = _safe_float((volatility_data or {}).get("implied_realised_ratio"))
        if (
            _is_enabled(vol_cfg)
            and vol_threshold is not None
            and vol_bonus is not None
            and volatility_z is not None
            and volatility_z > vol_threshold
        ):
            score += vol_bonus
            meta["volatility_bonus"] = {
                "volatility_z": volatility_z,
                "threshold": vol_threshold,
                "points": vol_bonus,
            }
            notes.append(
                f"P-Score increased by {vol_bonus:.1f} from volatility z-score {volatility_z:.2f}"
            )

        atr_penalty = _safe_float(atr_soft_gate_penalty) or 0.0
        if atr_penalty:
            score -= atr_penalty
            meta["atr_soft_gate_penalty"] = atr_penalty
            notes.append(
                f"P-Score reduced by {atr_penalty:.1f} due to ATR Soft Gate tolerance"
            )

        latency_pen = _safe_float(latency_penalty) or 0.0
        if latency_pen:
            score -= latency_pen
            meta["latency_penalty"] = latency_pen
            notes.append(
                f"P-Score reduced by {latency_pen:.1f} due to relaxed latency guard"
            )

        score = max(0.0, min(100.0, score))
        meta["final_score"] = score
        return score, notes, meta


def apply_atr_soft_gate(
    atr_ok: bool,
    rel_atr: Optional[float],
    atr_threshold: Optional[float],
    config: Optional[Dict[str, Any]],
) -> Tuple[bool, float, Dict[str, Any]]:
    """Relax the ATR gate when the deficit is within tolerance.

    Returns a tuple of (effective_atr_ok, penalty, meta).
    """

    if not _is_enabled(config):
        return atr_ok, 0.0, {}

    if atr_ok:
        return True, 0.0, {"soft_gate": False}

    tolerance_pct = config.get("tolerance_pct") if isinstance(config, dict) else None
    penalty_max = config.get("penalty_max") if isinstance(config, dict) else None
    rel_atr_val = _safe_float(rel_atr)
    atr_threshold_val = _safe_float(atr_threshold)

    if (
        tolerance_pct is None
        or penalty_max is None
        or rel_atr_val is None
        or atr_threshold_val is None
        or tolerance_pct <= 0
        or atr_threshold_val <= 0
    ):
        return atr_ok, 0.0, {}

    soft_lower_bound = atr_threshold_val * (1 - tolerance_pct)
    if rel_atr_val < soft_lower_bound:
        return atr_ok, 0.0, {}

    deficit = max(0.0, atr_threshold_val - rel_atr_val)
    deficit_ratio = deficit / atr_threshold_val if atr_threshold_val else 0.0
    penalty = min(penalty_max, penalty_max * (deficit_ratio / tolerance_pct))

    meta = {
        "soft_gate": True,
        "rel_atr": rel_atr_val,
        "threshold": atr_threshold_val,
        "tolerance_pct": tolerance_pct,
        "penalty": penalty,
    }
    return True, penalty, meta


class VolatilityManager:
    """Evaluate volatility gates with soft tolerance handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def evaluate(self, rel_atr: Optional[float], threshold: Optional[float]) -> Dict[str, Any]:
        """Return gate decision, penalties and metadata for ATR soft-gating.

        The evaluation follows three tiers:
        - PASS when the ATR meets or exceeds the threshold (zero penalty).
        - SOFT PASS when the deficit is within tolerance (penalty applied).
        - FAIL when the deficit exceeds tolerance.
        """

        rel_atr_val = _safe_float(rel_atr)
        threshold_val = _safe_float(threshold)
        tolerance_pct = self.config.get("tolerance_pct") if isinstance(self.config, dict) else None
        penalty_max = self.config.get("penalty_max") if isinstance(self.config, dict) else None
        enabled = _is_enabled(self.config)
        # Fall back to sane defaults when the config block is incomplete to keep
        # soft-gating behaviour backwards compatible.
        if enabled:
            tolerance_pct = tolerance_pct if tolerance_pct is not None else 0.15
            penalty_max = penalty_max if penalty_max is not None else 6.0

        meta: Dict[str, Any] = {
            "enabled": bool(enabled),
            "rel_atr": rel_atr_val,
            "threshold": threshold_val,
            "tolerance_pct": tolerance_pct,
            "penalty_max": penalty_max,
        }

        if threshold_val is None or threshold_val <= 0 or rel_atr_val is None:
            meta["mode"] = "invalid"
            return {"ok": False, "penalty": 0.0, "warning": None, "meta": meta}

        diff_ratio = (threshold_val - rel_atr_val) / threshold_val
        meta["diff_ratio"] = diff_ratio

        # Disabled/legacy: fall back to binary gating.
        if not enabled or tolerance_pct is None or penalty_max is None or tolerance_pct <= 0:
            meta["mode"] = "binary_pass" if diff_ratio <= 0 else "binary_fail"
            return {
                "ok": diff_ratio <= 0,
                "penalty": 0.0,
                "warning": None,
                "meta": meta,
            }

        if diff_ratio <= 0:
            meta["mode"] = "pass"
            return {"ok": True, "penalty": 0.0, "warning": None, "meta": meta}

        if diff_ratio <= tolerance_pct:
            penalty = min(penalty_max, (diff_ratio / tolerance_pct) * penalty_max)
            meta.update({"mode": "soft_pass", "penalty": penalty})
            return {
                "ok": True,
                "penalty": penalty,
                "warning": "ATR Soft Gate",
                "meta": meta,
            }

        meta["mode"] = "fail"
        return {"ok": False, "penalty": 0.0, "warning": None, "meta": meta}


def apply_latency_relaxation(
    asset: str,
    guard_status: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
    *,
    profile: Optional[str] = None,
    latency_seconds: Optional[float] = None,
    strict_limit_seconds: Optional[float] = None,
) -> Tuple[float, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Apply latency relaxation rules.

    Returns a tuple of (penalty, relaxation_meta, effective_guard_status). When
    the guard is relaxed the returned ``effective_guard_status`` is ``None`` so
    callers can skip blocking logic.
    """

    if not _is_enabled(config):
        return 0.0, None, guard_status

    profiles = config.get("profiles") if isinstance(config, dict) else None
    asset_map = config.get("asset_map") if isinstance(config, dict) else None
    if not isinstance(profiles, dict) or not isinstance(asset_map, dict):
        return 0.0, None, guard_status

    asset_key = str(asset or "").upper()
    profile_name = str(asset_map.get(asset_key) or asset_map.get("default") or "").strip()
    if not profile_name:
        return 0.0, None, guard_status

    profile_cfg = profiles.get(profile_name)
    if not isinstance(profile_cfg, dict):
        return 0.0, None, guard_status

    relaxed_limit = profile_cfg.get("limit") if isinstance(profile_cfg, dict) else None
    penalty = profile_cfg.get("penalty") if isinstance(profile_cfg, dict) else 0.0
    if relaxed_limit is None:
        return penalty, None, guard_status

    age_seconds = _safe_float(latency_seconds)
    if age_seconds is None:
        age_seconds = _safe_float(guard_status.get("age_seconds")) if guard_status else None
    base_limit = _safe_float(strict_limit_seconds)
    if base_limit is None and guard_status:
        base_limit = _safe_float(guard_status.get("limit_seconds"))
    if age_seconds is None:
        return 0.0, None, guard_status

    if base_limit is None:
        base_limit = relaxed_limit

    meta = {
        "profile": profile_name,
        "age_seconds": age_seconds,
        "base_limit_seconds": base_limit,
        "relaxed_limit_seconds": relaxed_limit,
        "penalty": penalty,
        "mode": "pass",
    }

    if relaxed_limit > base_limit and age_seconds > base_limit and age_seconds <= relaxed_limit:
        meta["mode"] = "penalized"
        return penalty, meta, None

    if age_seconds > relaxed_limit:
        updated_guard = dict(guard_status) if isinstance(guard_status, dict) else {}
        updated_guard.setdefault("asset", asset_key)
        updated_guard.setdefault("feed", "k1m")
        updated_guard["age_seconds"] = age_seconds
        updated_guard["limit_seconds"] = relaxed_limit
        meta["mode"] = "block_trade"
        return penalty, meta, updated_guard

    return 0.0, None, guard_status
