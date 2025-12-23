# -*- coding: utf-8 -*- 
"""
analysis.py — TD-only intraday jelzésképző (lokális JSON-okból).
Forrás: Trading.py által generált fájlok a public/<ASSET>/ alatt.
Kimenet:
  public/<ASSET>/signal.json      — "buy" / "sell" / "no entry" + okok
  public/analysis_summary.json    — összesített státusz
  public/analysis.html            — egyszerű HTML kivonat
"""

import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import date, datetime, timezone, timedelta
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Budapest")

from logging_utils import ensure_json_file_handler

LOGGER = logging.getLogger(__name__)

# Track optional dependencies that fell back to no-op stubs so the summary
# can surface degraded guardrails.
MISSING_OPTIONAL_DEPENDENCIES: Set[str] = set()
OPTIONAL_DEPENDENCY_ISSUES: List[str] = []


POSITION_SIZE_SCALE_FLOOR_BY_ASSET = {
    "EURUSD": 0.05,
    "XAGUSD": 0.03,
    "GOLD_CFD": 0.05,
    "USOIL": 0.04,
    "BTCUSD": 0.04,
    # NVDA itt nem kritikus, de legyen konzisztens:
    "NVDA": 0.05,
}

ATR_ABS_MIN_OVERRIDE = {
    "USOIL": 0.16,
}


def _log_optional_dependency_warning(name: str, exc: Exception) -> None:
    MISSING_OPTIONAL_DEPENDENCIES.add(name)
    OPTIONAL_DEPENDENCY_ISSUES.append(f"{name}: {exc}")
    LOGGER.warning("Optional dependency missing: %s (%s) — using fallback", name, exc)

from config import analysis_settings as settings
from dynamic_logic import (
    DynamicScoreEngine,
    VolatilityManager,
    apply_latency_relaxation,
    validate_dynamic_logic_config,
)

from active_anchor import load_anchor_state, record_anchor, update_anchor_metrics
from ml_model import (
    ProbabilityPrediction,
    inspect_model_artifact,
    log_feature_snapshot,
    missing_model_artifacts,
    predict_signal_probability,
    runtime_dependency_issues,
)
from news_feed import SentimentSignal, load_sentiment
import position_tracker

try:  # Optional monitoring utilities; keep analysis resilient if absent.
    from reports.trade_journal import record_signal_event
except Exception as exc:  # pragma: no cover - optional dependency guard
    _log_optional_dependency_warning("reports.trade_journal.record_signal_event", exc)

    def record_signal_event(*_args, **_kwargs):
        return None

try:
    from reports.monitoring import (
        update_signal_health_report,
        update_data_latency_report,
        record_latency_alert,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    _log_optional_dependency_warning("reports.monitoring", exc)

    def update_signal_health_report(*_args, **_kwargs):
        return None

    def update_data_latency_report(*_args, **_kwargs):
        return None

    def record_latency_alert(*_args, **_kwargs):
        return None

try:
    from reports.precision_monitor import update_precision_gate_report
except Exception as exc:  # pragma: no cover - optional dependency guard
    _log_optional_dependency_warning("reports.precision_monitor.update_precision_gate_report", exc)

    def update_precision_gate_report(*_args, **_kwargs):
        return None

try:
    from reports.backtester import update_live_validation
except Exception as exc:  # pragma: no cover - optional dependency guard
    _log_optional_dependency_warning("reports.backtester.update_live_validation", exc)

    def update_live_validation(*_args, **_kwargs):
        return None

try:
    from volatility_metrics import load_volatility_overlay
except Exception as exc:  # pragma: no cover - optional helper
    _log_optional_dependency_warning("volatility_metrics.load_volatility_overlay", exc)

    def load_volatility_overlay(asset: str, outdir: Path, k1m: Optional[Any] = None) -> Dict[str, Any]:
        return {}

try:
    from reports.pipeline_monitor import (
        record_analysis_run,
        finalize_analysis_run,
        get_pipeline_log_path,
        DEFAULT_MAX_LAG_SECONDS as PIPELINE_MAX_LAG_SECONDS,
        record_ml_model_status,
        get_run_logging_context,
    )
except Exception as exc:  # pragma: no cover - optional helper
    _log_optional_dependency_warning("reports.pipeline_monitor", exc)
    record_analysis_run = None
    finalize_analysis_run = None
    get_pipeline_log_path = None
    PIPELINE_MAX_LAG_SECONDS = None
    record_ml_model_status = None
    get_run_logging_context = lambda *a, **k: {}

import pandas as pd
import numpy as np

# === BTCUSD 5m intraday finomhangolás – profilkonstansok ===

BTC_OFI_Z = {"trigger": 0.8, "strong": 1.0, "weakening": -0.4, "lookback_bars": 60}
BTC_ADX_TREND_MIN = 20.0

# ATR floor + napszak-percentilis (TOD = time-of-day buckets) – baseline/relaxed/suppressed
BTC_ATR_FLOOR_USD = {
    "baseline": 80.0,
    "relaxed": 80.0,
    "intraday": 80.0,
    "suppressed": 75.0,
}
BTC_ATR_PCT_TOD = {  # percentilis minimum a nap adott szakaszára
    "baseline": {"open": 0.34, "mid": 0.34, "close": 0.34},
    "relaxed": {"open": 0.34, "mid": 0.34, "close": 0.34},
    "intraday": {"open": 0.34, "mid": 0.34, "close": 0.34},
    "suppressed": {"open": 0.32, "mid": 0.32, "close": 0.32},
}

# P-score / RR / TP / SL / no-chase per profile
BTC_P_SCORE_MIN = {"baseline": 34, "relaxed": 34, "intraday": 34, "suppressed": 32}
BTC_RR_MIN_TREND = {"baseline": 1.50, "relaxed": 1.50, "intraday": 1.50, "suppressed": 1.45}
BTC_RR_MIN_RANGE = {"baseline": 1.50, "relaxed": 1.50, "intraday": 1.50, "suppressed": 1.45}

BTC_RANGE_SIZE_SCALE = {"baseline": 0.50, "relaxed": 0.50, "intraday": 0.50, "suppressed": 0.48}
BTC_RANGE_TIME_STOP_MIN = {"baseline": 20, "relaxed": 20, "intraday": 20, "suppressed": 18}
BTC_RANGE_BE_TRIGGER_R = {"baseline": 0.26, "relaxed": 0.26, "intraday": 0.26, "suppressed": 0.24}

BTC_TP_MIN_PCT = {"baseline": 0.0062, "relaxed": 0.0062, "intraday": 0.0062, "suppressed": 0.0060}
BTC_SL_ATR_MULT = {"baseline": 0.24, "relaxed": 0.24, "intraday": 0.24, "suppressed": 0.24}
BTC_SL_ABS_MIN = 80.0  # USD

# Momentum override és no-chase (slippage tiltó R-ben)
BTC_MOMENTUM_RR_MIN = {"baseline": 1.50, "relaxed": 1.45, "intraday": 1.40, "suppressed": 1.35}
BTC_NO_CHASE_R = {"baseline": 0.25, "relaxed": 0.22, "intraday": 0.20, "suppressed": 0.18}

# Runtime state for BTC momentum overrides and guardrails. Populated lazily
# during analysis to support helper utilities and optional real-time adapters.
_BTC_MOMENTUM_RUNTIME: Dict[str, Any] = {}
_BTC_NO_CHASE_LIMITS: Dict[str, float] = {}

# Precision állapot figyelése a timeout diagnosztikához.
_PRECISION_RUNTIME: Dict[str, Dict[str, Any]] = {}

# Precision pipeline alsó határ (ne blokkoljon túl korán)
BTC_NO_CHASE_R = {"baseline": 0.25, "relaxed": 0.22, "intraday": 0.20, "suppressed": 0.18}
BTC_PRECISION_SOFT_DELTA = 2.0


def btc_precision_state(profile: str, asset: str, score: float, trigger_ready: bool) -> str:
    if asset != "BTCUSD":
        return "none"
    profile_key = profile if profile in BTC_PRECISION_MIN else "baseline"
    threshold = BTC_PRECISION_MIN.get(profile_key, BTC_PRECISION_MIN["baseline"])
    soft_block_threshold = threshold - BTC_PRECISION_SOFT_DELTA
    if score >= threshold:
        return "precision_arming" if trigger_ready else "precision_ready"
    if score >= soft_block_threshold and trigger_ready:
        return "precision_soft_block"
    return "none"  # ne blokkoljon önmagában


def calculate_position_size_multiplier(
    regime_label: str, atr_gate_ok: bool, p_score: float
) -> float:
    """Dynamic position sizing based on soft gate quality.

    Base size is 1.0 and sequential multipliers are applied for
    choppy regimes, low volatility, and sub-70 P-scores. The resulting
    multiplier is clamped to a minimum of 0.0; callers should block
    entries if the applied size would fall below 0.25.
    """

    multiplier = 1.0
    if regime_label.lower() == "choppy":
        multiplier *= 0.5
    if not atr_gate_ok:
        multiplier *= 0.5
    if p_score < 70:
        multiplier *= 0.5
    return max(multiplier, 0.0)


def classify_gate_failure(gate: str) -> str:
    """Categorise gates into critical or soft buckets.

    Critical gates hard-block the entry because they materially impact
    execution quality or risk management. Soft gates express alignment
    preferences; they should apply score/size penalties instead of
    preventing a trade outright.
    """

    critical = {"session", "session_open", "data_integrity", "spread", "spread_guard", "risk_reward"}
    return "critical" if gate in critical else "soft"

# --- Elemzendő eszközök ---
from config.analysis_settings import (
    ACTIVE_INVALID_BUFFER_ABS,
    ASSET_COST_MODEL,
    ATR5_MIN_MULT,
    ATR5_MIN_MULT_ASSET,
    ATR_LOW_TH_ASSET,
    ATR_LOW_TH_DEFAULT,
    DATA_LATENCY_GUARD,
    ATR_VOL_HIGH_REL,
    ATR_PERCENTILE_TOD,
    ASSETS,
    COST_MULT_DEFAULT,
    COST_MULT_HIGH_VOL,
    ADX_RR_BANDS,
    FUNDING_RATE_RULES,
    CORE_RR_MIN,
    DEFAULT_COST_MODEL,
    ENABLE_MOMENTUM_ASSETS,
    ENTRY_THRESHOLD_PROFILE_NAME,
    EMA_SLOPE_SIGN_ENFORCED,
    EMA_SLOPE_TH_ASSET,
    EMA_SLOPE_TH_DEFAULT,
    get_intraday_relax_size_scale,
    is_intraday_relax_enabled,
    BTC_PROFILE_OVERRIDES,
    NEWS_MODE_SETTINGS,
    FX_TP_TARGETS,
    GOLD_HIGH_VOL_TH,
    GOLD_HIGH_VOL_WINDOWS,
    GOLD_LOW_VOL_TH,
    INTERVENTION_WATCH_DEFAULT,
    INTRADAY_ATR_RELAX,
    INTRADAY_BIAS_RELAX,
    LEVERAGE,
    MIN_RISK_ABS,
    MOMENTUM_RR_MIN,
    XAGUSD_ATR_5M_FLOOR,
    XAGUSD_ATR_5M_FLOOR_ENABLED,
    OFI_Z_SETTINGS,
    NVDA_EXTENDED_ATR_REL,
    NVDA_MOMENTUM_ATR_REL,
    NVDA_DAILY_ATR_MULTIPLIER,
    NVDA_DAILY_ATR_MIN,
    NVDA_DAILY_ATR_STRONG,
    NVDA_LOW_ATR_P_SCORE_ADD,
    NVDA_RR_BANDS,
    NVDA_STOP_ATR_MIN,
    NVDA_STOP_ATR_MAX,
    NVDA_POSITION_SCALE,
    P_SCORE_TIME_BONUS,
    SESSION_TIME_RULES,
    SESSION_WEEKDAYS,
    SESSION_WINDOWS_UTC,
    resolve_session_status_for_asset,
    is_momentum_asset,
    SMT_AUTO_CONFIG,
    SMT_PENALTY_VALUE,
    SMT_REQUIRED_BARS,
    SPOT_MAX_AGE_SECONDS,
    VWAP_BAND_MULT,
    get_atr_abs_min,
    get_atr_period,
    get_atr_threshold_multiplier,
    get_bos_lookback,
    get_fib_tolerance,
    get_max_slippage_r,
    get_max_risk_pct,
    get_p_score_min,
    get_realtime_price_guard,
    get_sl_buffer_config,
    get_spread_max_atr_pct,
    get_low_atr_override,
    get_tp_min_abs_value,
    get_tp_min_pct_value,
    get_tp_net_min,
    get_risk_template,
    load_config,
)

def _entry_threshold_profile_name(asset: str) -> str:
    try:
        profile_name = settings.get_entry_threshold_profile_name_for_asset(asset)
    except Exception:
        profile_name = None
    analysis_default = globals().get("ENTRY_THRESHOLD_PROFILE_NAME", ENTRY_THRESHOLD_PROFILE_NAME)
    try:
        settings_default = settings.ENTRY_THRESHOLD_PROFILE_NAME  # type: ignore[attr-defined]
    except Exception:
        settings_default = ENTRY_THRESHOLD_PROFILE_NAME
    # If the analysis module explicitly overrides the active profile (e.g. via
    # monkeypatching in tests), prefer that value over any asset-specific mapping
    # from the settings module. This keeps runtime overrides authoritative while
    # still defaulting to the configured mapping when no override is present.
    if analysis_default and analysis_default != settings_default:
        return str(analysis_default)
    if not profile_name or profile_name == settings_default:
        profile_name = analysis_default
    return str(profile_name)


BTC_PROFILE_CONFIG: Dict[str, Any] = dict(
    BTC_PROFILE_OVERRIDES.get(_entry_threshold_profile_name("BTCUSD")) or {}
)

SETTINGS: Dict[str, Any] = load_config()


def btc_rr_min_with_adx(profile: str, asset: str, adx_5m_val: float) -> Tuple[Optional[float], Dict[str, Any]]:
    """Return BTC RR-min override + metadata based on 5m ADX regime."""

    if asset != "BTCUSD":
        return (None, {})

    profile_key = profile if profile in BTC_RR_MIN_TREND else "baseline"
    if adx_5m_val is None or not np.isfinite(adx_5m_val):
        return (None, {})

    if adx_5m_val >= BTC_ADX_TREND_MIN:
        rr_val = BTC_RR_MIN_TREND[profile_key]
        return (rr_val, {"mode": "trend"})

    rr_val = BTC_RR_MIN_RANGE[profile_key]
    return (
        rr_val,
        {
            "mode": "range",
            "size_scale": BTC_RANGE_SIZE_SCALE[profile_key],
            "time_stop": BTC_RANGE_TIME_STOP_MIN[profile_key],
            "be_trigger_r": BTC_RANGE_BE_TRIGGER_R[profile_key],
        },
    )


try:  # pragma: no cover - optional micro structure utility
    structure_ok_5m  # type: ignore[name-defined]
except NameError:  # pragma: no cover - stub fallback
    def structure_ok_5m(asset: str, side: str) -> bool:
        """TODO: Replace with micro structure confirmation implementation."""

        return False


try:  # pragma: no cover - optional VWAP retest utility
    vwap_retest_ok  # type: ignore[name-defined]
except NameError:  # pragma: no cover - stub fallback
    def vwap_retest_ok(asset: str, side: str) -> bool:
        """TODO: Replace with VWAP reclaim/reject confirmation implementation."""

        return False


try:  # pragma: no cover - optional OFI utility
    ofi_zscore  # type: ignore[name-defined]
except NameError:  # pragma: no cover - stub fallback
    def ofi_zscore(asset: str, lookback: int) -> float:
        """TODO: Replace with order flow imbalance z-score implementation."""

        return -999.0


def btc_core_triggers_ok(asset: str, side: str) -> Tuple[bool, Dict[str, Any]]:
    """BTC core: mikró-BOS (1m+retest), VWAP reclaim/reject (5m), OFI z-score (runtime) — EITHER-OF(2)."""

    if asset != "BTCUSD" or side not in {"long", "short"}:
        return False, {"bos_ok": False, "vwap_ok": False, "ofi_ok": False, "ofi_z": None}

    runtime_state = globals().get("_BTC_MOMENTUM_RUNTIME")
    if not isinstance(runtime_state, dict):
        runtime_state = {}
    z_value = runtime_state.get("ofi_z")
    try:
        z_float = float(z_value) if z_value is not None else float("nan")
    except (TypeError, ValueError):
        z_float = float("nan")

    try:
        ofi_threshold = float(BTC_OFI_Z.get("trigger", 1.0))
    except (TypeError, ValueError):
        ofi_threshold = 1.0
    if ofi_threshold < 0:
        ofi_threshold = abs(ofi_threshold)

    if np.isfinite(z_float):
        if side == "long":
            ofi_ok: Optional[bool] = z_float >= ofi_threshold
        else:
            ofi_ok = z_float <= -ofi_threshold
    else:
        ofi_ok = None

    micro_state = globals().get("_BTC_STRUCT_MICRO")
    if not isinstance(micro_state, dict):
        micro_state = {}
    vwap_state = globals().get("_BTC_STRUCT_VWAP")
    if not isinstance(vwap_state, dict):
        vwap_state = {}

    def _normalise(flag: Any) -> Optional[bool]:
        if flag is None:
            return None
        try:
            return bool(flag)
        except Exception:
            return None

    bos_ok = _normalise(micro_state.get(side))
    vwap_ok = _normalise(vwap_state.get(side))

    components = {"bos": bos_ok, "vwap": vwap_ok, "ofi": ofi_ok}
    available = [name for name, value in components.items() if value is not None]
    positives = [name for name, value in components.items() if value is True]
    required_hits = 2 if len(available) >= 2 else 1
    gate_ok = bool(positives) and len(positives) >= required_hits

    return (
        gate_ok,
        {
            "bos_ok": bool(bos_ok) if bos_ok is not None else False,
            "vwap_ok": bool(vwap_ok) if vwap_ok is not None else False,
            "ofi_ok": bool(ofi_ok) if ofi_ok is not None else False,
            "ofi_z": None if not np.isfinite(z_float) else z_float,
            "available": available,
            "missing": [name for name, value in components.items() if value is None],
            "required_hits": required_hits,
            "hits": len(positives),
        },
    )


def btc_momentum_override(profile: str, asset: str, side: str, atr_ok: bool) -> Tuple[bool, Optional[float], str]:
    """Return the BTC momentum override status, RR override and descriptor."""

    if asset != "BTCUSD" or not is_momentum_asset(asset) or side not in {"long", "short"}:
        return (False, None, "")

    ema_ok = False
    ema_cross_fn = globals().get("ema_cross_5m")
    if callable(ema_cross_fn):
        try:
            ema_ok = bool(ema_cross_fn(asset, fast=9, slow=21, side=side))
        except Exception:
            ema_ok = False
    if not ema_ok:
        state = globals().get("_BTC_MOMENTUM_RUNTIME", {})
        if isinstance(state, dict):
            ema_state = state.get("ema_cross")
            if isinstance(ema_state, dict):
                ema_ok = bool(ema_state.get(side))

    z_value: Optional[float] = None
    ofi_fn = globals().get("ofi_zscore")
    if callable(ofi_fn):
        try:
            lookback = int(BTC_OFI_Z.get("lookback_bars", 60))
        except (TypeError, ValueError):
            lookback = 60
        try:
            z_val = ofi_fn(asset, lookback=lookback)
        except Exception:
            z_val = None
        else:
            try:
                z_value = float(z_val)
            except (TypeError, ValueError):
                z_value = None
    if z_value is None:
        state = globals().get("_BTC_MOMENTUM_RUNTIME", {})
        if isinstance(state, dict):
            cached = state.get("ofi_z")
            try:
                z_value = float(cached) if cached is not None else None
            except (TypeError, ValueError):
                z_value = None

    if z_value is None or not np.isfinite(z_value):
        z_value = float("nan")

    cfg_state = globals().get("_BTC_MOMENTUM_RUNTIME", {})
    ofi_strong_th = BTC_OFI_Z.get("strong", 0.0)
    rr_override: Optional[float] = None
    if isinstance(cfg_state, dict):
        cfg = cfg_state.get("cfg")
        if isinstance(cfg, dict):
            strong_raw = cfg.get("ofi_strong")
            if strong_raw is not None:
                try:
                    ofi_strong_th = float(strong_raw)
                except (TypeError, ValueError):
                    pass
            rr_raw = cfg.get("rr_min")
            if rr_raw is not None:
                try:
                    rr_override = float(rr_raw)
                except (TypeError, ValueError):
                    rr_override = None

    if rr_override is None:
        profile_cfg = BTC_PROFILE_OVERRIDES.get(profile)
        if not profile_cfg and isinstance(BTC_PROFILE_CONFIG, dict):
            if profile == _btc_active_profile():
                profile_cfg = BTC_PROFILE_CONFIG
        rr_min_cfg = None
        if isinstance(profile_cfg, dict):
            rr_section = profile_cfg.get("rr_min")
            if isinstance(rr_section, dict):
                rr_min_cfg = rr_section.get("momentum")
        if rr_min_cfg is None and isinstance(BTC_PROFILE_CONFIG, dict):
            rr_section = BTC_PROFILE_CONFIG.get("rr_min")
            if isinstance(rr_section, dict):
                rr_min_cfg = rr_section.get("momentum")
        if rr_min_cfg is not None:
            try:
                rr_override = float(rr_min_cfg)
            except (TypeError, ValueError):
                rr_override = None
    if rr_override is None:
        rr_override = float(
            BTC_MOMENTUM_RR_MIN.get(profile, BTC_MOMENTUM_RR_MIN.get("baseline", 1.4))
        )

    if not np.isfinite(ofi_strong_th):
        ofi_strong_th = BTC_OFI_Z.get("strong", 0.0)

    if np.isnan(z_value):
        ofi_strong = False
    elif side == "long":
        ofi_strong = z_value >= ofi_strong_th
    else:
        ofi_strong = z_value <= -ofi_strong_th

    if ema_ok and ofi_strong and atr_ok:
        desc = f"ema9x21+ofi_z={z_value:.2f}" if np.isfinite(z_value) else "ema9x21+ofi"
        return (True, rr_override, desc)
    return (False, None, "")


def btc_no_chase_violated(profile: str, entry_price: float, trigger_price: float, sl_price: float) -> bool:
    """Return ``True`` when the BTC momentum entry chases beyond the R limit."""

    profile_key = profile if profile in BTC_NO_CHASE_R else "baseline"
    limit_value: Optional[float] = None
    profile_cfg = BTC_PROFILE_OVERRIDES.get(profile)
    if not profile_cfg and isinstance(BTC_PROFILE_CONFIG, dict):
        if profile == _btc_active_profile():
            profile_cfg = BTC_PROFILE_CONFIG
    if isinstance(profile_cfg, dict):
        limit_raw = profile_cfg.get("no_chase_r")
        if limit_raw is not None:
            try:
                limit_value = float(limit_raw)
            except (TypeError, ValueError):
                limit_value = None
    if limit_value is None and isinstance(BTC_PROFILE_CONFIG, dict):
        if profile == _btc_active_profile():
            limit_raw = BTC_PROFILE_CONFIG.get("no_chase_r")
            if limit_raw is not None:
                try:
                    limit_value = float(limit_raw)
                except (TypeError, ValueError):
                    limit_value = None
    if limit_value is None:
        limit_value = BTC_NO_CHASE_R.get(profile_key, BTC_NO_CHASE_R["baseline"])
    limit_map = globals().get("_BTC_NO_CHASE_LIMITS", {})
    if isinstance(limit_map, dict) and profile in limit_map:
        try:
            limit_value = float(limit_map[profile])
        except (TypeError, ValueError):
            limit_value = BTC_NO_CHASE_R.get(profile_key, BTC_NO_CHASE_R["baseline"])
    r = abs(entry_price - trigger_price) / max(1e-9, abs(trigger_price - sl_price))
    return r > limit_value


def btc_gate_margins(asset: str, ctx) -> dict:
    """
    Számolja, mennyivel maradtunk el a belépési küszöböktől:
    - ATR_gate: rel_atr vs atr_gate_th (érték + arány)
    - P-score:  P vs p_min
    - OFI:      |z| vs trigger
    - ADX:      adx vs trend_min
    - RR:       rr vs rr_min
    - TP1:      tp_pct vs tp_min_pct
    ctx: tartalmazza az épp számolt értékeket (rel_atr, atr_gate_th, P, p_min, z, adx, rr, rr_min, tp_pct, tp_min_pct)
    """

    m: Dict[str, float] = {}
    try:
        rel_atr_val = float(ctx["rel_atr"])
        atr_gate_th = float(ctx["atr_gate_th"])
        m["atr_gap"] = rel_atr_val - atr_gate_th
        if atr_gate_th != 0:
            m["atr_ratio"] = rel_atr_val / max(atr_gate_th, 1e-9)
    except Exception:
        pass
    try:
        m["p_gap"] = float(ctx["P"]) - float(ctx["p_min"])
    except Exception:
        pass
    try:
        z = float(ctx.get("ofi_z", float("nan")))
        trig = float(ctx.get("ofi_trig", 1.0))
        m["ofi_gap"] = abs(z) - trig
    except Exception:
        pass
    try:
        m["adx_gap"] = float(ctx["adx"]) - float(ctx["adx_trend_min"])
    except Exception:
        pass
    try:
        m["rr_gap"] = float(ctx["rr"]) - float(ctx["rr_min"])
    except Exception:
        pass
    try:
        m["tp_gap"] = float(ctx["tp_pct"]) - float(ctx["tp_min_pct"])
    except Exception:
        pass
    return m


def btc_sl_tp_checks(
    profile: str,
    asset: str,
    atr_5m: float,
    entry: float,
    sl: float,
    tp1: float,
    rr_min: float,
    info: Dict[str, Any],
    blockers: List[str],
) -> None:
    if asset != "BTCUSD":
        return
    profile_key = profile if profile in BTC_SL_ATR_MULT else "baseline"

    relaxed_blockers: Set[str] = set()
    if profile_key == "intraday":
        relaxed_blockers = {"sl_too_tight", "tp_min_pct"}

    sp_limit = get_spread_max_atr_pct(asset, profile)
    spread_func = globals().get("spread_usd")
    sp = spread_func(asset) if callable(spread_func) else 0.0
    if sp > atr_5m * sp_limit:
        blockers.append("spread_gate")

    abs_min_default = BTC_SL_ABS_MIN
    sl_cfg = None
    try:
        overrides = globals().get("BTC_PROFILE_OVERRIDES")
    except NameError:
        overrides = None
    if isinstance(overrides, dict):
        profile_section = overrides.get(profile_key) or overrides.get(profile)
        if isinstance(profile_section, dict):
            sl_cfg = profile_section.get("sl_buffer")
    if not sl_cfg:
        try:
            sl_cfg = (BTC_PROFILE_CONFIG or {}).get("sl_buffer")
        except Exception:
            sl_cfg = None

    if isinstance(sl_cfg, dict):
        try:
            abs_min_default = max(abs_min_default, float(sl_cfg.get("abs_min", abs_min_default)))
        except Exception:
            pass

    sl_buf = max(atr_5m * BTC_SL_ATR_MULT[profile_key], abs_min_default)
    sl_dist = abs(entry - sl)
    info["sl_buffer"] = float(sl_dist)
    info["sl_buffer_min"] = float(sl_buf)
    info["sl_buffer_abs_min"] = float(abs_min_default)
    info["btc_profile"] = profile_key
    info["spread_usd"] = float(sp)
    info["spread_limit_atr_mult"] = float(sp_limit)
    if sl_dist < sl_buf:
        if "sl_too_tight" not in relaxed_blockers:
            blockers.append("sl_too_tight")
        else:
            info.setdefault("relaxed_checks", []).append("sl_buffer")

    tp_min = BTC_TP_MIN_PCT.get(profile_key, BTC_TP_MIN_PCT["baseline"])
    tp_pct = abs(tp1 - entry) / entry if entry else 0.0
    info["tp_pct"] = tp_pct
    info["tp_min_pct"] = tp_min
    if tp_pct < tp_min:
        if "tp_min_pct" not in relaxed_blockers:
            blockers.append("tp_min_pct")
        else:
            info.setdefault("relaxed_checks", []).append("tp_min_pct")

    rr = abs(tp1 - entry) / max(1e-9, abs(entry - sl))
    info["rr"] = rr
    info["rr_min"] = rr_min
    if rr < rr_min:
        blockers.append("rr_min")
    info["spread_usd"] = sp


def _btc_active_profile() -> str:
    profile = _entry_threshold_profile_name("BTCUSD")
    if profile not in BTC_P_SCORE_MIN:
        return "baseline"
    return profile


def _btc_profile_section(name: str) -> Dict[str, Any]:
    section = BTC_PROFILE_CONFIG.get(name)
    defaults: Dict[str, Any] = {}
    profile = _btc_active_profile()
    if name == "momentum_override":
        defaults = {
            "ofi_z": BTC_OFI_Z["trigger"],
            "ofi_strong": BTC_OFI_Z["strong"],
            "ofi_weakening": BTC_OFI_Z["weakening"],
            "lookback_bars": BTC_OFI_Z["lookback_bars"],
            "rr_min": BTC_MOMENTUM_RR_MIN.get(profile),
            "no_chase_r": BTC_NO_CHASE_R.get(profile),
        }
    elif name == "rr":
        defaults = {
            "trend_core": BTC_RR_MIN_TREND.get(profile),
            "trend_momentum": BTC_MOMENTUM_RR_MIN.get(profile),
            "range_core": BTC_RR_MIN_RANGE.get(profile),
            "range_momentum": BTC_RR_MIN_RANGE.get(profile),
            "range_size_scale": BTC_RANGE_SIZE_SCALE.get(profile),
            "range_time_stop": BTC_RANGE_TIME_STOP_MIN.get(profile),
            "range_breakeven": BTC_RANGE_BE_TRIGGER_R.get(profile),
        }
    elif name == "structure":
        defaults = {
            "ofi_gate": BTC_OFI_Z["trigger"],
        }
    result = dict(defaults)
    if isinstance(section, dict):
        result.update(section)
    return result


def _precision_profile_config(asset: str) -> Dict[str, Any]:
    if asset != "BTCUSD":
        return {}

    profile = _btc_active_profile()
    cfg: Dict[str, Any] = {}
    if isinstance(BTC_PROFILE_CONFIG, dict):
        section = BTC_PROFILE_CONFIG.get("precision")
        if isinstance(section, dict):
            cfg.update(section)

    precision_min = (cfg or {}).get("score_min")
    if precision_min is None:
        precision_min = BTC_PRECISION_MIN.get(profile, BTC_PRECISION_MIN.get("baseline"))
    try:
        if precision_min is not None:
            precision_min = float(precision_min)
    except (TypeError, ValueError):
        precision_min = BTC_PRECISION_MIN.get(profile, BTC_PRECISION_MIN.get("baseline"))
    if precision_min is not None:
        cfg["score_min"] = precision_min
    return cfg


def get_precision_score_threshold(asset: str) -> float:
    cfg = _precision_profile_config(asset)
    value = cfg.get("score_min")
    if value is not None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Ignoring invalid precision score_min %r for asset %s", value, asset
            )
        else:
            if np.isfinite(val) and val > 0:
                return val
    asset_key = str(asset or "").upper()
    profile_name = _entry_threshold_profile_name(asset_key)
    profile_overrides = PRECISION_SCORE_PROFILE_OVERRIDES.get(asset_key)
    if isinstance(profile_overrides, dict):
        try:
            profile_value = profile_overrides.get(profile_name)
        except Exception:
            profile_value = None
        if profile_value is not None:
            try:
                val = float(profile_value)
            except (TypeError, ValueError):
                val = None
            if val is not None and np.isfinite(val) and val > 0:
                return val
    return float(PRECISION_SCORE_THRESHOLD_DEFAULT)


def get_precision_timeouts(asset: str) -> Dict[str, int]:
    cfg = _precision_profile_config(asset)

    def _safe_timeout(value: Any, default: int) -> int:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return default
        return max(1, parsed)

    ready_default = int(PRECISION_READY_TIMEOUT_DEFAULT)
    arming_default = int(PRECISION_ARMING_TIMEOUT_DEFAULT)
    ready = _safe_timeout(cfg.get("ready_timeout_minutes"), ready_default)
    arming_raw = cfg.get("arming_timeout_minutes")
    arming = _safe_timeout(arming_raw, arming_default)
    if arming_raw is None:
        arming = max(arming, ready)
    return {"ready": ready, "arming": max(arming, ready)}

# Az asset-specifikus küszöböket a config/analysis_settings.json állomány
# szolgáltatja, így új eszköz felvételekor elegendő azt módosítani.

MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")
BUDAPEST_TIMEZONE = ZoneInfo("Europe/Budapest")
RULE_TIMEZONES: Dict[str, ZoneInfo] = {"EURUSD": ZoneInfo("America/New_York")}

PUBLIC_DIR = "public"
_ANALYSIS_BASE_DIR = Path(__file__).resolve().parent
ENTRY_GATE_LOG_DIR: Path = (
    _ANALYSIS_BASE_DIR / PUBLIC_DIR / "debug" / "entry_gates"
).resolve()
ENTRY_GATE_STATS_PATH: Path = (
    _ANALYSIS_BASE_DIR / PUBLIC_DIR / "debug" / "entry_gate_stats.json"
).resolve()
ENTRY_GATE_GAP_LOG_PATH: Path = (
    _ANALYSIS_BASE_DIR / PUBLIC_DIR / "debug" / "entry_gate_gap_log.jsonl"
).resolve()
PROB_STACK_SNAPSHOT_FILENAME = "probability_stack_snapshot.json"
PROB_STACK_EXPORT_FILENAME = "probability_stack.json"
PROB_STACK_GAP_ENV_DISABLE = "DISABLE_PROB_STACK_GAP_FALLBACK"
PROB_STACK_GAP_STALE_MINUTES = 10
LATENCY_GUARD_PROFILE_LIMIT_SECONDS: Dict[str, int] = {"suppressed": 420}
ML_FEATURE_SNAPSHOT_DIRNAME = "ml_features"
PRECISION_SCORE_PROFILE_OVERRIDES: Dict[str, Dict[str, float]] = {
    "NVDA": {"suppressed": 52.0},
    "GOLD_CFD": {"relaxed": 50.0},
}
ENTRY_GATE_EXTRA_LOGS_DISABLE = "DISABLE_ENTRY_GATE_EXTRA_LOGS"
MACRO_DATA_DIR = Path("data") / "macro"
MACRO_LOCKOUT_FILE = MACRO_DATA_DIR / "lockout_by_asset.json"


def _parse_utc_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = value.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
    except Exception:
        try:
            dt = pd.to_datetime(value).to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_MACRO_LOCKOUT_CACHE: Optional[Tuple[float, Dict[str, List[Dict[str, Any]]]]] = None


def _load_macro_lockout_windows() -> Dict[str, List[Dict[str, Any]]]:
    global _MACRO_LOCKOUT_CACHE
    path = MACRO_LOCKOUT_FILE
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    mtime = stat.st_mtime
    if _MACRO_LOCKOUT_CACHE and _MACRO_LOCKOUT_CACHE[0] == mtime:
        return _MACRO_LOCKOUT_CACHE[1]
    raw = load_json(str(path))
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for asset, events in raw.items():
            if not isinstance(events, list):
                continue
            asset_key = str(asset).upper()
            normalized_events: List[Dict[str, Any]] = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                ts_value = event.get("ts_release_utc") or event.get("ts")
                ts_release = _parse_utc_timestamp(str(ts_value)) if ts_value else None
                if ts_release is None:
                    continue
                try:
                    pre = max(0, int(event.get("pre") or event.get("pre_seconds") or 0))
                except (TypeError, ValueError):
                    pre = 0
                try:
                    post = max(0, int(event.get("post") or event.get("post_seconds") or 0))
                except (TypeError, ValueError):
                    post = 0
                start = ts_release - timedelta(seconds=pre)
                end = ts_release + timedelta(seconds=post)
                label = (
                    event.get("label")
                    or event.get("event")
                    or event.get("id")
                    or "Macro event"
                )
                normalized_events.append(
                    {
                        "id": event.get("id"),
                        "provider": event.get("provider"),
                        "label": str(label),
                        "start": start,
                        "end": end,
                        "release": ts_release,
                        "pre": pre,
                        "post": post,
                    }
                )
            if normalized_events:
                normalized_events.sort(key=lambda item: item.get("release") or datetime.min.replace(tzinfo=timezone.utc))
                normalized[asset_key] = normalized_events
    _MACRO_LOCKOUT_CACHE = (mtime, normalized)
    return normalized

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value_norm = value.strip().lower()
    return value_norm in {"1", "true", "yes", "on"}


ENABLE_SENTIMENT_PROBABILITY = _env_flag("ENABLE_SENTIMENT_PROBABILITY", default=False)
ENABLE_ML_PROBABILITY = _env_flag("ENABLE_ML_PROBABILITY", default=False)
SUPPRESS_ML_MODEL_WARNINGS = _env_flag("SUPPRESS_ML_MODEL_WARNINGS", default=False)
# ``PIPELINE_ENV`` allows operators to expose environment-specific toggles.
PIPELINE_ENV = (os.getenv("PIPELINE_ENV", "") or "").strip().lower()
_ML_OVERRIDE_DISABLED_ENVS = {"ci", "qa", "test"}
_ML_OVERRIDE_DEFAULT = PIPELINE_ENV not in _ML_OVERRIDE_DISABLED_ENVS
# Allow hard-disabling ML scoring regardless of legacy flags.  The CI runner
# doesn't currently set this but it gives operators a one-line escape hatch.
if _env_flag("DISABLE_ML_PROBABILITY", default=False) or _env_flag(
    "FORCE_DISABLE_ML", default=False
):
    ENABLE_ML_PROBABILITY = False
# ``USE_ML`` was the legacy toggle for enabling machine learning scoring.  CI
# still exports it, so treat it as a backwards compatible alias to avoid
# silently disabling probabilities when only ``USE_ML`` is set.
if not ENABLE_ML_PROBABILITY and _env_flag("USE_ML", default=False):
    ENABLE_ML_PROBABILITY = True

# --- Temporary override ---------------------------------------------------
# A modell tréningje még folyamatban van, ezért az ML valószínűség számítást
# alapértelmezetten csak az éles környezetben tiltjuk le.  A ``PIPELINE_ENV``
# változó CI/QA környezetben automatikusan visszakapcsolja a scoringot, így
# amint elérhető az artefakt, ott nem szükséges manuálisan beavatkozni.
ML_PROBABILITY_MANUAL_OVERRIDE = _env_flag(
    "ML_PROBABILITY_MANUAL_OVERRIDE",
    default=_ML_OVERRIDE_DEFAULT,
)
ML_PROBABILITY_MANUAL_REASON = os.getenv(
    "ML_PROBABILITY_MANUAL_REASON",
    "ML valószínűség manuálisan letiltva: modell tréning alatt",
)
if ML_PROBABILITY_MANUAL_OVERRIDE:
    ENABLE_ML_PROBABILITY = False
ML_PROBABILITY_ACTIVE = ENABLE_ML_PROBABILITY
ML_PROBABILITY_PLACEHOLDER_BLOCKLIST: Set[str] = set()
ML_PROBABILITY_PLACEHOLDER_INFO: Dict[str, Dict[str, Any]] = {}
FIB_TOL = 0.02

# --- Kereskedési/egz. küszöbök (RR/TP) ---
MIN_R_CORE      = 2.0
MIN_R_MOMENTUM  = 1.6

# --- EURUSD-specifikus volatilitási paraméterek ----------------------------
EURUSD_PIP = 0.0001
EURUSD_ATR_PIPS_TARGET = 16.0
EURUSD_ATR_PIPS_LOW = 14.0
EURUSD_ATR_PIPS_HIGH = 20.0
EURUSD_ATR_REF = EURUSD_ATR_PIPS_TARGET * EURUSD_PIP
EURUSD_P_SCORE_LOW_VOL_ADD = 6.0
EURUSD_P_SCORE_MOMENTUM_ADD = 2.0
EURUSD_ATR_POSITION_MULT = 1.25
EURUSD_POSITION_SCALE_MIN = 0.4
EURUSD_POSITION_SCALE_MAX = 1.3
USOIL_ATR1H_BREAKOUT_MIN = 1.0
USOIL_ATR1H_TARGET = 1.5
USOIL_ATR1H_PARABOLIC = 2.2
USOIL_P_SCORE_LOW_ATR_ADD = 6.0
USOIL_P_SCORE_SIDEWAYS_ADD = 4.0
USOIL_MOMENTUM_STOP_MULT = 2.5
USOIL_VWAP_BIAS_LOOKBACK = 240
USOIL_VWAP_BIAS_RATIO = 0.65
USOIL_GAP_VWAP_MIN_PCT = 0.6
TP1_R   = 2.0
TP2_R   = 3.0
TP1_R_MOMENTUM = 1.5
TP2_R_MOMENTUM = 2.4
MIN_STOPLOSS_PCT = 0.01
# Momentum structure check window (5m candles × lookback bars)
DEFAULT_BOS_LOOKBACK = get_bos_lookback()
# Number of recent bars to inspect for EMA cross momentum confirmation
MOMENTUM_BARS = 12

ATR_PERCENTILE_LOOKBACK_DAYS = int(ATR_PERCENTILE_TOD.get("lookback_days") or 0)
ATR_PERCENTILE_BUCKETS = list(ATR_PERCENTILE_TOD.get("buckets") or [])
ATR_PERCENTILE_OVERLAP = dict(ATR_PERCENTILE_TOD.get("overlap") or {})
ATR_PERCENTILE_RANGE_FLOOR = float(ATR_PERCENTILE_TOD.get("range_floor_percentile") or 0.0)
ATR_PERCENTILE_RANGE_ADX = float(ATR_PERCENTILE_TOD.get("range_adx_threshold") or 0.0)

P_SCORE_TIME_WINDOWS = dict(P_SCORE_TIME_BONUS.get("default") or {})
P_SCORE_OFI_BONUS = float(P_SCORE_TIME_BONUS.get("ofi_bonus") or 0.0)

ADX_TREND_BAND = dict(ADX_RR_BANDS.get("trend") or {})
ADX_RANGE_BAND = dict(ADX_RR_BANDS.get("range") or {})
ADX_TREND_MIN = float(ADX_TREND_BAND.get("adx_min") or 0.0)
ADX_RANGE_MAX = float(ADX_RANGE_BAND.get("adx_max") or ADX_TREND_MIN)
ADX_TREND_CORE_RR = float(ADX_TREND_BAND.get("core_rr_min") or MIN_R_CORE)
ADX_TREND_MOM_RR = float(ADX_TREND_BAND.get("momentum_rr_min") or MIN_R_MOMENTUM)
ADX_RANGE_CORE_RR = float(ADX_RANGE_BAND.get("core_rr_min") or MIN_R_CORE)
ADX_RANGE_MOM_RR = float(ADX_RANGE_BAND.get("momentum_rr_min") or MIN_R_MOMENTUM)
ADX_RANGE_SIZE_SCALE = float(ADX_RANGE_BAND.get("size_scale") or 1.0)
ADX_RANGE_TIME_STOP = int(ADX_RANGE_BAND.get("time_stop_minutes") or 0)
ADX_RANGE_BE_TRIGGER = float(ADX_RANGE_BAND.get("breakeven_trigger_r") or 0.0)
ADX_RANGE_GIVEBACK = float(ADX_RANGE_BAND.get("giveback_ratio") or 0.0)

OFI_Z_TRIGGER = float(OFI_Z_SETTINGS.get("trigger") or 0.0)
OFI_Z_WEAKENING = float(OFI_Z_SETTINGS.get("weakening") or 0.0)
OFI_Z_LOOKBACK = int(OFI_Z_SETTINGS.get("lookback_bars") or 0)

VWAP_TREND_BAND = float(VWAP_BAND_MULT.get("trend") or 0.8)
VWAP_MEAN_REVERT_BAND = float(VWAP_BAND_MULT.get("mean_revert") or 2.0)

NEWS_LOCKOUT_MINUTES_DEFAULT = int(NEWS_MODE_SETTINGS.get("lockout_minutes") or 0)
NEWS_STABILISATION_MINUTES_DEFAULT = int(NEWS_MODE_SETTINGS.get("stabilisation_minutes") or 0)
NEWS_SEVERITY_THRESHOLD_DEFAULT = float(NEWS_MODE_SETTINGS.get("severity_threshold") or 1.0)
NEWS_CALENDAR_FILES: List[str] = list(NEWS_MODE_SETTINGS.get("calendar_files") or [])
NEWS_ASSET_SETTINGS: Dict[str, Dict[str, Any]] = {
    str(asset): dict(settings)
    for asset, settings in NEWS_MODE_SETTINGS.items()
    if isinstance(settings, dict)
}
# Period used for EMA slope calculations
EMA_SLOPE_PERIOD = 21
# Historical window (number of bars) considered for EMA slope thresholds
EMA_SLOPE_LOOKBACK = 12
# Baseline EMA slope threshold used when no asset override is configured
EMA_SLOPE_TH = EMA_SLOPE_TH_DEFAULT
REFRESH_TIPS = (
    "Az analysis.py mindig a legutóbbi ZÁRT gyertyával számol (5m: max. ~5 perc késés).",
    "CI/CD-ben kösd össze a Trading és Analysis futást: az analysis job csak a trading után induljon (needs: trading).",
    "A kliens kéréséhez adj cache-busting query paramot (pl. ?v=<timestamp>) és no-store cache-control fejlécet.",
    "Cloudflare Worker stale policy: 5m feedre állítsd 120s-re, hogy hamar átjöjjön az új jel.",
    "A dashboard stabilizáló (2 azonos jel + 10 perc cooldown) lassíthatja a kártya frissítését — lazítsd, ha realtime kell."
)
LATENCY_PROFILE_FILENAME = "latency_profile.json"
LATENCY_GUARD_STATE_FILENAME = "latency_guard_state.json"
ORDER_FLOW_LOOKBACK_MIN = 120
ORDER_FLOW_IMBALANCE_TH = 0.6
ORDER_FLOW_PRESSURE_TH = 0.7
PRECISION_FLOW_IMBALANCE_MARGIN = 1.1
PRECISION_FLOW_PRESSURE_MARGIN = 1.1
PRECISION_FLOW_STATS_FILE = Path(
    os.getenv(
        "PRECISION_FLOW_STATS_FILE",
        os.path.join(PUBLIC_DIR, "monitoring", "precision_gates_by_asset.csv"),
    )
)
PRECISION_FLOW_TARGET_RATIO = float(os.getenv("PRECISION_FLOW_TARGET_RATIO", "0.14"))
PRECISION_FLOW_SCALE_MIN = float(os.getenv("PRECISION_FLOW_SCALE_MIN", "0.6"))
PRECISION_FLOW_MARGIN_MIN = float(os.getenv("PRECISION_FLOW_MARGIN_MIN", "0.9"))
PRECISION_FLOW_STRENGTH_BASE = float(os.getenv("PRECISION_FLOW_STRENGTH_BASE", "0.74"))
PRECISION_FLOW_STRENGTH_MIN = float(os.getenv("PRECISION_FLOW_STRENGTH_MIN", "0.65"))
PRECISION_FLOW_STALLED_EPS = float(os.getenv("PRECISION_FLOW_STALLED_EPS", "0.002"))
PRECISION_FLOW_STALLED_DELTA_EPS = float(os.getenv("PRECISION_FLOW_STALLED_DELTA_EPS", "0.05"))
SPOT_REALTIME_TTL_SECONDS = int(os.getenv("SPOT_REALTIME_TTL_SECONDS", "300"))
PRECISION_SCORE_THRESHOLD_DEFAULT = 55.0
PRECISION_SCORE_THRESHOLD = PRECISION_SCORE_THRESHOLD_DEFAULT
PRECISION_READY_TIMEOUT_DEFAULT = 15
PRECISION_ARMING_TIMEOUT_DEFAULT = 20
PRECISION_TRIGGER_NEAR_MULT = 0.2
REALTIME_JUMP_MULT = 2.0
MICRO_BOS_P_BONUS = 8.0
MOMENTUM_ATR_REL = 0.0005
MOMENTUM_VOLUME_RECENT = 6
MOMENTUM_VOLUME_BASE = 30
MOMENTUM_VOLUME_RATIO_TH = 1.05
MOMENTUM_TRAIL_TRIGGER_R = 1.2
MOMENTUM_TRAIL_LOCK = 0.5
ANCHOR_P_SCORE_DELTA_WARN = 10.0
ANCHOR_ATR_DROP_RATIO = 0.75
INTRADAY_EXHAUSTION_PCT = 0.82


def _btc_precision_thresholds_from_config() -> Dict[str, float]:
    """Build per-profile precision score minimums from the settings map."""

    thresholds: Dict[str, float] = {}
    for profile_name, meta in BTC_PROFILE_OVERRIDES.items():
        if not isinstance(meta, dict):
            continue
        precision_cfg = meta.get("precision") or {}
        try:
            score_min = float(precision_cfg.get("score_min"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(score_min) and score_min > 0:
            thresholds[str(profile_name)] = score_min

    # Ensure we always have a usable baseline fallback
    if "baseline" not in thresholds:
        thresholds["baseline"] = float(PRECISION_SCORE_THRESHOLD_DEFAULT)
    return thresholds


BTC_PRECISION_MIN: Dict[str, float] = _btc_precision_thresholds_from_config()
INTRADAY_ATR_EXHAUSTION = 0.75
INTRADAY_COMPRESSION_TH = 0.45
INTRADAY_EXPANSION_TH = 1.25
INTRADAY_BALANCE_LOW = 0.35
INTRADAY_BALANCE_HIGH = 0.65
OPENING_RANGE_MINUTES = 45
ANCHOR_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
TF_STALE_TOLERANCE = {"k1m": 240, "k5m": 900, "k1h": 5400, "k4h": 21600}
CRITICAL_STALE_FRAMES = {
    "k1m": "k1m: jelzés korlátozva",
    "k5m": "k5m: jelzés korlátozva",
    "k1h": "k1h: jelzés korlátozva",
    "k4h": "k4h: jelzés korlátozva",
}
RELAXED_STALE_FRAMES: Dict[str, Set[str]] = {
    "USOIL": {"k1m", "k5m", "spot"},
    "NVDA": {"k1m", "k5m", "k1h", "k4h"},
}
INTERVENTION_CONFIG_FILENAME = "intervention_watch.json"
INTERVENTION_STATE_FILENAME = "intervention_state.json"
INTERVENTION_NEWS_FILENAME = "intervention_news.json"
INTERVENTION_SUMMARY_FILENAME = "intervention_summary.json"
INTERVENTION_P_SCORE_ADD = 5.0


def current_anchor_state() -> Dict[str, Dict[str, Any]]:
    """Return the persisted anchor state in a defensive manner."""
    try:
        return load_anchor_state()
    except Exception:
        return {}

# A TP/SL, ATR és session-specifikus határok a fenti importból érkeznek:
# pl. TP_NET_MIN_ASSET, TP_MIN_PCT, SL_BUFFER_RULES, SESSION_WINDOWS_UTC stb.
# -------------------------- segédek -----------------------------------

def nowiso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_utc_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    return None


def normalize_generated_utc(value: Any, *, field: str) -> str:
    parsed = parse_utc_timestamp(value)
    if parsed is None:
        LOGGER.error("Érvénytelen időbélyeg", extra={"mezo": field, "ertek": value})
        parsed = datetime.now(timezone.utc)
    return to_utc_iso(parsed)


def _should_use_realtime_spot(
    rt_price: Optional[Any],
    rt_ts: Optional[datetime],
    spot_ts: Optional[datetime],
    now_utc: datetime,
    max_age_seconds: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether the realtime spot snapshot should override the last spot close."""

    meta_base = {"max_age_seconds": float(max_age_seconds)}
    if rt_price is None:
        return False, dict(meta_base, reason="missing_price")
    if rt_ts is None:
        return False, dict(meta_base, reason="missing_timestamp")
    if rt_ts.tzinfo is None:
        rt_ts = rt_ts.replace(tzinfo=timezone.utc)
    age_seconds = (now_utc - rt_ts).total_seconds()
    if age_seconds < 0:
        age_seconds = 0.0
    if age_seconds > float(max_age_seconds):
        return False, dict(meta_base, reason="stale", age_seconds=age_seconds)
    if spot_ts is not None and rt_ts <= spot_ts:
        return False, dict(meta_base, reason="not_newer", age_seconds=age_seconds)
    return True, dict(meta_base, reason="ok", age_seconds=age_seconds)


def _normalize_realtime_meta(
    meta: Dict[str, Any], *, max_age_seconds: int, now_utc: Optional[datetime] = None
) -> Dict[str, Any]:
    """Return a sanitised realtime meta payload that avoids stale blockers."""

    normalised = dict(meta)
    try:
        age_seconds = float(normalised.get("age_seconds", 0.0))
    except (TypeError, ValueError):
        age_seconds = 0.0
    used = bool(normalised.get("used"))
    if not used and age_seconds and max_age_seconds > 0:
        stale_ceiling = float(max_age_seconds) * 4.0
        if age_seconds >= stale_ceiling:
            normalised["age_seconds_original"] = age_seconds
            normalised["age_seconds"] = None
            normalised["cleared_stale_meta"] = True
            if now_utc:
                normalised["cleared_at_utc"] = now_utc.replace(microsecond=0).isoformat().replace(
                    "+00:00", "Z"
                )
            normalised.setdefault("reason", "stale")
    return normalised


def _extract_bid_ask_spread(snapshot: Any) -> Optional[float]:
    """Return the bid/ask spread from a realtime snapshot when available."""

    if not isinstance(snapshot, dict):
        return None
    bid_val = safe_float(snapshot.get("bid") or snapshot.get("best_bid"))
    ask_val = safe_float(snapshot.get("ask") or snapshot.get("best_ask"))
    if bid_val is None or ask_val is None:
        return None
    try:
        spread = float(ask_val) - float(bid_val)
    except (TypeError, ValueError):
        return None
    if spread < 0:
        return None
    return spread


def _resolve_spread_for_entry(
    spot_realtime: Any,
    spot_price_reference: Optional[float],
    rt_price: Optional[float],
    use_realtime: bool,
) -> Optional[float]:
    """Determine the spread guard input used during entry evaluation.

    The spread gate should only consider realtime samples when they are
    actively used for pricing.  Otherwise stale forced snapshots can inject
    arbitrarily large spreads and block entries despite fresh OHLC data.
    """

    if not use_realtime:
        return None
    spread = _extract_bid_ask_spread(spot_realtime)
    if spread is not None:
        return spread
    if rt_price is None or spot_price_reference is None:
        return None
    try:
        return abs(float(rt_price) - float(spot_price_reference))
    except (TypeError, ValueError):
        return None
  
def now_utctime_hm() -> Tuple[int,int]:
    t = datetime.now(timezone.utc)
    return t.hour, t.minute


def _entry_gate_log_payload(
    symbol: str,
    bar_time: Optional[datetime],
    reasons: Sequence[str],
    gate: str = "entry_gate",
) -> Dict[str, Any]:
    ts = bar_time
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    bud_ts = ts.astimezone(LOCAL_TZ)

    normalized: List[str] = []
    for reason in reasons:
        if reason is None:
            continue
        reason_text = str(reason).strip()
        if reason_text:
            normalized.append(reason_text)
    unique_reasons = list(dict.fromkeys(normalized))
    result = "rejected" if unique_reasons else "accepted"

    return {
        "symbol": symbol,
        "asset": str(symbol).upper(),
        "gate": gate,
        "timestamp": ts.isoformat(),
        "utc_ts": ts.isoformat(),
        "bud_ts": bud_ts.isoformat(),
        "reasons": unique_reasons,
        "reason": unique_reasons[0] if unique_reasons else None,
        "result": result,
    }


def log_entry_gate_decision(
    symbol: str,
    bar_time: Optional[datetime],
    reasons: Sequence[str],
    gate: str = "entry_gate",
) -> None:
    """Append a JSONL entry describing the gate outcome for a candidate."""

    try:
        ENTRY_GATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = _entry_gate_log_payload(symbol, bar_time, reasons, gate=gate)
        ts = _parse_utc_timestamp(payload.get("timestamp")) or datetime.now(timezone.utc)
        log_path = ENTRY_GATE_LOG_DIR / f"entry_gates_{ts.date().isoformat()}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    except Exception:
        LOGGER.debug("entry_gate_log_failed", exc_info=True)


def _append_entry_gate_stats(summary: Dict[str, Any]) -> None:
    if os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        return
    try:
        ENTRY_GATE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = load_json(str(ENTRY_GATE_STATS_PATH)) if ENTRY_GATE_STATS_PATH.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        asset_key = str(summary.get("asset") or summary.get("symbol") or "UNKNOWN").upper()
        asset_entries = existing.setdefault(asset_key, [])
        asset_entries.append(summary)
        save_json(str(ENTRY_GATE_STATS_PATH), existing)
        _render_entry_gate_chart(existing)
    except Exception:
        LOGGER.debug("entry_gate_stats_append_failed", exc_info=True)


def _append_gate_gap_log(entries: Sequence[Dict[str, Any]]) -> None:
    if os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        return
    if not entries:
        return
    try:
        ENTRY_GATE_GAP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ENTRY_GATE_GAP_LOG_PATH.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    except Exception:
        LOGGER.debug("entry_gate_gap_log_failed", exc_info=True)


def _render_entry_gate_chart(stats: Dict[str, Any]) -> None:
    """Render a lightweight HTML bar chart for entry gate rejections."""

    try:
        target_dir = ENTRY_GATE_LOG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        html_path = target_dir / "index.html"
        rows: List[str] = []
        for asset, entries in sorted(stats.items()):
            if not isinstance(entries, list):
                continue
            reason_counts: Dict[str, int] = {}
            for item in entries:
                if not isinstance(item, dict):
                    continue
                for reason in item.get("missing") or item.get("precision_hiany") or []:
                    reason_text = str(reason)
                    reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
            total = sum(reason_counts.values()) or 1
            for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                pct = (count / total) * 100.0
                rows.append(
                    f"<div class='row'><span class='asset'>{asset}</span>"
                    f"<span class='reason'>{reason}</span>"
                    f"<span class='bar'><span style='width:{pct:.1f}%'></span></span>"
                    f"<span class='count'>{count}</span></div>"
                )
        content = "\n".join(rows) if rows else "<p>Nincs gate elutasítási adat.</p>"
        html = f"""
<!doctype html>
<html lang='hu'>
<head>
  <meta charset='utf-8'>
  <title>Entry gate statisztika</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    .row {{ display: grid; grid-template-columns: 90px 1fr 260px 60px; gap: 10px; align-items: center; }}
    .asset {{ font-weight: bold; }}
    .reason {{ color: #333; }}
    .bar {{ background:#f2f2f2; border-radius:4px; height:10px; position: relative; }}
    .bar span {{ background:#c0392b; display:block; height:10px; border-radius:4px; }}
    .count {{ text-align:right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <h2>Entry gate elutasítások (összesített)</h2>
  {content}
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
    except Exception:
        LOGGER.debug("entry_gate_chart_render_failed", exc_info=True)


def _gate_timestamp_fields(timestamp: Optional[datetime]) -> Dict[str, str]:
    ts_ref = timestamp or datetime.now(timezone.utc)
    if ts_ref.tzinfo is None:
        ts_ref = ts_ref.replace(tzinfo=timezone.utc)
    ts_utc = ts_ref.astimezone(timezone.utc)
    ts_fmt = "%Y-%m-%d %H:%M:%S"
    return {
        "timestamp_utc": ts_utc.strftime(ts_fmt),
        "timestamp_bud": ts_utc.astimezone(LOCAL_TZ).strftime(ts_fmt),
    }


def _precision_ready_elapsed_seconds(
    ready_since: Optional[datetime], now_runtime: Optional[datetime] = None
) -> Optional[float]:
    """Return elapsed seconds a precision ready állapot kezdete óta."""

    if ready_since is None:
        return None
    if not isinstance(ready_since, datetime):
        return None
    now = (now_runtime or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        return (now - ready_since).total_seconds()
    except Exception:
        return None


def _log_gate_summary(asset: str, decision: Dict[str, Any]) -> None:
    """Emit a structured log line describing the gate evaluation outcome."""

    try:
        gates = decision.get("gates") or {}
        entry_meta = decision.get("entry_thresholds") or {}
        active_meta = decision.get("active_position_meta") or {}
        context_hu = decision.get("entry_gate_context_hu") if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE) else {}
        ts_fields = _gate_timestamp_fields(parse_utc_timestamp(decision.get("retrieved_at_utc")))
        summary = {
            "asset": asset,
            "profile": entry_meta.get("profile"),
            "mode": gates.get("mode"),
            "missing": list(gates.get("missing") or []),
            "p_score": decision.get("probability_raw"),
            "p_score_min": entry_meta.get("p_score_min_effective")
            or entry_meta.get("p_score_min")
            or entry_meta.get("p_score_min_base"),
            "atr_rel": active_meta.get("atr5_rel"),
            "atr_threshold": entry_meta.get("atr_threshold_effective"),
            "atr_ok": entry_meta.get("atr_ratio_ok"),
            "spread_ok": entry_meta.get("spread_gate_ok"),
            "liquidity_ok": decision.get("momentum_liquidity_ok"),
            "rr_min_core": entry_meta.get("rr_min_core"),
            "rr_min_momentum": entry_meta.get("rr_min_momentum"),
            **ts_fields,
        }
        summary["atr_kuszob_hatasos"] = entry_meta.get("atr_threshold_effective")
        risk_meta = entry_meta.get("risk_guard") or {}
        summary["risk_max_pct"] = risk_meta.get("max_risk_pct")
        summary["risk_spread_ok"] = risk_meta.get("spread_gate_ok")
        summary["cost_model"] = risk_meta.get("cost_model")
        guard_meta = entry_meta.get("latency_guard") or {}
        summary["kesleltetesi_vedo_kor_mp"] = guard_meta.get("age_seconds")
        summary["kesleltetesi_vedo_limit_mp"] = guard_meta.get("limit_seconds")
        precision_missing = [
            item for item in summary["missing"] if isinstance(item, str) and "precision" in item.lower()
        ]
        if precision_missing:
            summary["precision_hiany"] = precision_missing
        if isinstance(context_hu, dict):
            summary.update(context_hu)
        LOGGER.info("gate_summary", extra={"asset": asset, "gate_summary": summary})
        _append_entry_gate_stats(summary)
        gap_records: List[Dict[str, Any]] = []
        profile = summary.get("profile")
        timestamp_utc = ts_fields.get("timestamp_utc")

        def _add_gap_record(
            kapu: str,
            value: Optional[Any],
            threshold: Optional[Any],
            ok: Optional[bool] = None,
        ) -> None:
            try:
                numeric_value = float(value) if value is not None else None
            except (TypeError, ValueError):
                numeric_value = None
            try:
                numeric_threshold = float(threshold) if threshold is not None else None
            except (TypeError, ValueError):
                numeric_threshold = None
            gap = None
            if (
                numeric_value is not None
                and numeric_threshold is not None
                and not (np.isnan(numeric_value) or np.isnan(numeric_threshold))
            ):
                gap = numeric_value - numeric_threshold
            gap_records.append(
                {
                    "asset": asset,
                    "kapu": kapu,
                    "profil": profile,
                    "érték": numeric_value,
                    "küszöb": numeric_threshold,
                    "rés": gap,
                    "ok": bool(ok) if ok is not None else None,
                    "timestamp_utc": timestamp_utc,
                }
            )

        _add_gap_record("p_score", summary.get("p_score"), summary.get("p_score_min"), None)
        if isinstance(context_hu, dict):
            atr_ctx = context_hu.get("atr_kapu") or {}
            _add_gap_record(
                "atr",
                atr_ctx.get("rel_atr"),
                atr_ctx.get("threshold"),
                atr_ctx.get("ok"),
            )
            spread_ctx = context_hu.get("spread_kapu") or {}
            _add_gap_record(
                "spread",
                spread_ctx.get("spread_ratio_atr"),
                spread_ctx.get("spread_limit_atr"),
                spread_ctx.get("ok"),
            )
            momentum_ctx = context_hu.get("momentum_kapu") or {}
            _add_gap_record(
                "momentum",
                momentum_ctx.get("momentum_score"),
                momentum_ctx.get("threshold"),
                momentum_ctx.get("ok"),
            )
        _append_gate_gap_log([entry for entry in gap_records if entry])
    except Exception:
        LOGGER.debug("gate_summary_logging_failed", exc_info=True)


def _build_entry_count_summary(
    asset_results: Dict[str, Any], window_days: int = 14
) -> Dict[str, Any]:
    """Aggregate buy/sell jeleket nap/eszkoz bontásban az utolsó ablakra."""

    counts_by_day: Dict[str, Dict[str, int]] = {}
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=window_days)
    journal_path = Path(PUBLIC_DIR) / "journal" / "trade_journal.csv"

    def _bump(day_key: str, asset_key: str) -> None:
        asset_bucket = counts_by_day.setdefault(day_key, {})
        asset_bucket[asset_key] = asset_bucket.get(asset_key, 0) + 1

    if journal_path.exists():
        try:
            with journal_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    signal = (row.get("signal") or "").strip().lower()
                    if signal not in {"buy", "sell"}:
                        continue
                    ts = parse_utc_timestamp(row.get("analysis_timestamp"))
                    if ts is None:
                        continue
                    if ts < cutoff:
                        continue
                    day_key = ts.astimezone(timezone.utc).date().isoformat()
                    asset_key = (row.get("asset") or "").strip().upper()
                    if not asset_key:
                        continue
                    _bump(day_key, asset_key)
        except Exception:
            LOGGER.debug("entry_count_aggregation_failed", exc_info=True)

    for asset_key, payload in asset_results.items():
        if not isinstance(payload, dict):
            continue
        signal = (payload.get("signal") or payload.get("decision") or "").strip().lower()
        if signal not in {"buy", "sell"}:
            continue
        ts = parse_utc_timestamp(payload.get("retrieved_at_utc")) or now_utc
        day_key = ts.astimezone(timezone.utc).date().isoformat()
        _bump(day_key, asset_key.upper())

    totals: Dict[str, int] = {}
    for _, assets in counts_by_day.items():
        for asset_key, count in assets.items():
            totals[asset_key] = totals.get(asset_key, 0) + count

    return {
        "generated_utc": nowiso(),
        "window_days": window_days,
        "by_day_asset": counts_by_day,
        "by_asset_total": totals,
    }


def _btc_time_of_day_bucket(minute: int) -> str:
    if minute < 8 * 60:
        return "open"
    if minute < 16 * 60:
        return "mid"
    return "close"


def time_of_day_bucket(now_utc: Optional[datetime]) -> str:
    """Return the BTC intraday bucket name ("open"/"mid"/"close")."""

    try:
        bucket = settings.time_of_day_bucket("BTCUSD", now_utc)
    except Exception:
        bucket = None
    if bucket:
        return str(bucket)

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    minute = now_utc.hour * 60 + now_utc.minute
    return _btc_time_of_day_bucket(minute)


def atr_percentile_value(
    asset: str,
    percentile: float,
    lookback_days: int = 20,
    tod: Optional[str] = None,
) -> Optional[float]:
    """Return the ATR value at the requested percentile.

    TODO: Replace with a real historical ATR percentile lookup (asset/TOD aware).
    """

    return None


def btc_atr_gate_threshold(profile: str, asset: str, now_utc: Optional[datetime]) -> float:
    """Return the absolute ATR threshold for BTC given the current profile/bucket."""

    if asset != "BTCUSD":
        return 0.0

    floor: Optional[float] = None
    profile_cfg = BTC_PROFILE_OVERRIDES.get(profile)
    if not profile_cfg and isinstance(BTC_PROFILE_CONFIG, dict):
        if profile == _btc_active_profile():
            profile_cfg = BTC_PROFILE_CONFIG
    if isinstance(profile_cfg, dict):
        floor_raw = profile_cfg.get("atr_floor_usd")
        if floor_raw is not None:
            try:
                floor = float(floor_raw)
            except (TypeError, ValueError):
                floor = None
    if floor is None:
        floor = BTC_ATR_FLOOR_USD.get(profile)
    if floor is None:
        floor = BTC_ATR_FLOOR_USD.get("baseline", 0.0)
    tod = time_of_day_bucket(now_utc)
    pct = None
    if isinstance(profile_cfg, dict):
        pct_section = profile_cfg.get("atr_percentiles")
        if isinstance(pct_section, dict):
            pct = pct_section.get(tod)
    if pct is None:
        pct_cfg = BTC_ATR_PCT_TOD.get(profile) or BTC_ATR_PCT_TOD.get("baseline", {})
        pct = pct_cfg.get(tod, 0.0)

    percentile_fn = globals().get("atr_percentile_value")
    percentile_value: Optional[float] = None
    if callable(percentile_fn):
        try:
            percentile_value = percentile_fn(asset, pct, lookback_days=20, tod=tod)
        except Exception:
            percentile_value = None

    threshold_candidates = []
    floor_usd = None
    try:
        floor_usd = float(floor)
        threshold_candidates.append(floor_usd)
    except (TypeError, ValueError):
        floor_usd = None
    if percentile_value is not None:
        try:
            threshold_candidates.append(float(percentile_value))
        except (TypeError, ValueError):
            pass
    threshold = max(threshold_candidates) if threshold_candidates else 0.0
    return threshold


def btc_atr_gate_ok(
    profile: str, asset: str, atr_5m_value: float, now_utc: Optional[datetime]
) -> bool:
    if asset != "BTCUSD":
        return True

    threshold = btc_atr_gate_threshold(profile, asset, now_utc)
    try:
        atr_value = float(atr_5m_value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(atr_value):
        return False
    return atr_value >= threshold


def df_last_timestamp(df: pd.DataFrame) -> Optional[datetime]:
    if df.empty:
        return None
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        ts = idx[-1]
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.to_pydatetime()
    return None


def ensure_closed_candles(df: pd.DataFrame,
                          now: Optional[datetime] = None,
                          tolerance_seconds: int = 5) -> pd.DataFrame:
    """Return a dataframe that contains only candles that have already closed.

    A Twelve Data time_series lekérés teljesen zárt gyertyákat ad vissza, de a
    feldolgozás során védjük magunkat a jövőbe eső (pl. cache/proxy hiba miatti)
    időbélyegek ellen. Ha a legutolsó gyertya timestampje a jelenhez képest
    jövőbeni, eltávolítjuk, különben megtartjuk az összes sort.

    Parameters
    ----------
    df: pd.DataFrame
        Idősor, datetime index-szel.
    now: Optional[datetime]
        Aktuális idő (UTC). Ha nincs megadva, a rendszer idejét használjuk.
    tolerance_seconds: int
        Ennyi másodpercet engedünk a jövő irányába a szerverórák közti pici
        csúszás miatt.
    """

    if df.empty:
        return df.copy()

    if now is None:
        now = datetime.now(timezone.utc)

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df.copy()

    cutoff = now + timedelta(seconds=max(tolerance_seconds, 0))
    mask = idx <= cutoff

    if mask.all():
        return df.copy()

    # Távolítsuk el a jövőbeli sorokat; ha mindet kidobtuk volna, adjunk üres df-et.
    filtered = df.loc[mask]
    return filtered.copy() if not filtered.empty else df.iloc[0:0].copy()

def file_mtime(path: str) -> Optional[str]:
    try:
        return to_utc_iso(datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc))
    except Exception:
        return None

def session_windows_utc(asset: str) -> Tuple[
    Optional[List[Tuple[int, int, int, int]]],
    Optional[List[Tuple[int, int, int, int]]],
]:
    cfg = SESSION_WINDOWS_UTC.get(asset)
    if not cfg:
        return None, None
    if isinstance(cfg, dict):
        entry_windows = cfg.get("entry")
        monitor_windows = cfg.get("monitor")
    else:  # visszafelé-kompatibilis: ha listát kapnánk, kezeljük entry-ként
        entry_windows = cfg  # type: ignore[assignment]
        monitor_windows = cfg  # type: ignore[assignment]
    if entry_windows is None and monitor_windows is not None:
        entry_windows = monitor_windows
    if monitor_windows is None and entry_windows is not None:
        monitor_windows = entry_windows
    return entry_windows, monitor_windows

def _min_of_day(hour: int, minute: int) -> int:
    """Return minutes from midnight, clamped to the valid daily range."""

    total = hour * 60 + minute
    return max(0, min(total, 23 * 60 + 59))


def in_any_window_utc(windows: Optional[List[Tuple[int, int, int, int]]], h: int, m: int) -> bool:
    if not windows:
        return True
    minutes = _min_of_day(h, m)
    for sh, sm, eh, em in windows:
        s = _min_of_day(sh, sm)
        e = _min_of_day(eh, em)
        if s <= minutes <= e:
            return True
    return False
def minute_in_interval(minute: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


def _convert_rule_minute_to_utc(
    minute: int, rule_date: date, source_tz: Optional[ZoneInfo]
) -> int:
    """Return the UTC minute-of-day for a local trading rule timestamp.

    Some session rules (e.g., FX Sunday open) are defined in a market's local
    timezone such as America/New_York to track DST transitions. Convert those
    minutes to UTC for the given rule_date. If conversion fails or no source
    timezone is provided, fall back to the original minute value.
    """

    if source_tz is None:
        return minute

    try:
        local_dt = datetime.combine(
            rule_date, dtime(minute // 60, minute % 60, tzinfo=source_tz)
        )
        utc_dt = local_dt.astimezone(timezone.utc)
        return utc_dt.hour * 60 + utc_dt.minute
    except Exception:
        return minute


def format_utc_minute(minute: int) -> str:
    minute = max(0, min(23 * 60 + 59, minute))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def format_local_range(start_dt: datetime, end_dt: datetime) -> List[str]:
    return [start_dt.strftime("%H:%M"), end_dt.strftime("%H:%M")]


def convert_windows_to_local(
    windows: Optional[List[Tuple[int, int, int, int]]],
    tz: ZoneInfo = MARKET_TIMEZONE,
) -> Optional[List[List[str]]]:
    if not windows:
        return None
    today_utc = datetime.now(timezone.utc).date()
    result: List[List[str]] = []
    for sh, sm, eh, em in windows:
        start_dt = datetime.combine(today_utc, dtime(sh, sm, tzinfo=timezone.utc))
        end_dt = datetime.combine(today_utc, dtime(eh, em, tzinfo=timezone.utc))
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        start_local = start_dt.astimezone(tz)
        end_local = end_dt.astimezone(tz)
        result.append(format_local_range(start_local, end_local))
    return result


def convert_minutes_to_local_range(
    start: int, end: int, tz: ZoneInfo = MARKET_TIMEZONE
) -> List[str]:
    today_utc = datetime.now(timezone.utc).date()
    start_dt = datetime.combine(today_utc, dtime(start // 60, start % 60, tzinfo=timezone.utc))
    end_dt = datetime.combine(today_utc, dtime(end // 60, end % 60, tzinfo=timezone.utc))
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    start_local = start_dt.astimezone(tz)
    end_local = end_dt.astimezone(tz)
    return format_local_range(start_local, end_local)

def session_weekday_ok(asset: str, now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    allowed = SESSION_WEEKDAYS.get(asset)
    if not allowed:
        return True
    return now.weekday() in allowed

def next_session_open(asset: str, now: Optional[datetime] = None) -> Optional[datetime]:
    if now is None:
        now = datetime.now(timezone.utc)

    entry_windows, _ = session_windows_utc(asset)
    weekdays = SESSION_WEEKDAYS.get(asset)

    if not entry_windows:
        entry_windows = [(0, 0, 23, 59)]

    for day_offset in range(0, 8):
        day = (now + timedelta(days=day_offset)).date()
        if weekdays and day.weekday() not in weekdays:
            continue

        for sh, sm, eh, em in entry_windows:
            start_dt = datetime.combine(day, dtime(sh, sm, tzinfo=timezone.utc))
            end_dt = datetime.combine(day, dtime(eh, em, tzinfo=timezone.utc))

            if day_offset == 0:
                if end_dt <= now:
                    continue
                if now < start_dt:
                    return start_dt
            else:
                return start_dt

    return None

def session_state(asset: str, now: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    h, m = now_utc.hour, now_utc.minute
    now_budapest = now_utc.astimezone(BUDAPEST_TIMEZONE)
    minute_of_day = h * 60 + m
    entry_windows, monitor_windows = session_windows_utc(asset)
    monitor_ok = in_any_window_utc(monitor_windows, h, m)
    entry_window_ok = in_any_window_utc(entry_windows, h, m)
    weekday_ok = session_weekday_ok(asset, now_utc)
    status_profile_name, status_profile = resolve_session_status_for_asset(
        asset, when=now_utc, weekday_ok=weekday_ok
    )
    if not isinstance(status_profile, dict):
        status_profile = {}
    next_open_override: Optional[str] = None
    profile_notes: List[str] = []
    profile_tags: List[str] = []
    profile_context: Dict[str, Any] = {}

    special_status: Optional[str] = None
    special_note: Optional[str] = None
    special_reason: Optional[str] = None
    break_active = False
    break_window: Optional[Tuple[int, int]] = None
    news_lockout = False

    rules = SESSION_TIME_RULES.get(asset, {})
    rule_tz = RULE_TIMEZONES.get(asset)
    rule_date = now_utc.date()

    sunday_open_raw = rules.get("sunday_open_minute")
    try:
        sunday_open = int(sunday_open_raw) if sunday_open_raw is not None else None
    except (TypeError, ValueError):
        sunday_open = None
    if sunday_open is not None:
        sunday_open = _convert_rule_minute_to_utc(sunday_open, rule_date, rule_tz)
    sunday_open_local: Optional[List[str]] = None
    if sunday_open is not None:
        sunday_open_local = convert_minutes_to_local_range(
            sunday_open, (sunday_open + 1) % (24 * 60), tz=BUDAPEST_TIMEZONE
        )
    if sunday_open is not None and now_utc.weekday() == 6 and minute_of_day < sunday_open:
        monitor_ok = False
        entry_window_ok = False
        special_status = "closed_out_of_hours"
        special_reason = "sunday_open_pending"
        special_note = (
            "Piac zárva (vasárnapi nyitás "
            f"{format_utc_minute(sunday_open)} UTC / {sunday_open_local[0]} Budapest után)"
        )
      
    daily_breaks_raw = rules.get("daily_breaks") or []
    daily_breaks: List[Tuple[int, int]] = []
    for start, end in daily_breaks_raw:
        try:
            start_m = int(start)
            end_m = int(end)
        except (TypeError, ValueError):
            continue
        start_m = _convert_rule_minute_to_utc(start_m, rule_date, rule_tz)
        end_m = _convert_rule_minute_to_utc(end_m, rule_date, rule_tz)
        daily_breaks.append((start_m, end_m))

    if special_status is None:
        for start, end in daily_breaks:
            if minute_in_interval(minute_of_day, start, end):
                monitor_ok = False
                entry_window_ok = False
                break_active = True
                break_window = (start, end)
                special_status = "maintenance"
                special_reason = "daily_break"
                start_s = format_utc_minute(start)
                end_s = format_utc_minute(end)
                special_note = f"Piac zárva (napi karbantartás {start_s}–{end_s} UTC)"
                break

    if (
        special_status is None
        and entry_window_ok
        and monitor_windows
        and not monitor_ok
    ):
        news_lockout = True
        special_status = "news_lockout"
        special_reason = "monitor_window"
        special_note = "Piac zárva (hír/monitor ablakon kívül)"

    friday_close_raw = rules.get("friday_close_minute")
    try:
        friday_close = int(friday_close_raw) if friday_close_raw is not None else None
    except (TypeError, ValueError):
        friday_close = None
    if friday_close is not None:
        friday_close = _convert_rule_minute_to_utc(friday_close, rule_date, None)
    if friday_close is not None and now_utc.weekday() == 4 and minute_of_day >= friday_close:
        monitor_ok = False
        entry_window_ok = False
        special_status = "closed_out_of_hours"
        special_reason = "friday_close"
        special_note = f"Piac zárva (pénteki zárás {format_utc_minute(friday_close)} UTC)"

    open_now = monitor_ok and weekday_ok
    if asset == "BTCUSD":
        entry_open = entry_window_ok and weekday_ok
    else:
        entry_open = entry_window_ok and weekday_ok and (
            monitor_ok or not monitor_windows
        )

    if status_profile:
        if status_profile.get("force_session_closed"):
            monitor_ok = False
            entry_window_ok = False
            open_now = False
            entry_open = False
        if "open" in status_profile:
            open_now = bool(status_profile.get("open"))
        if "entry_open" in status_profile:
            entry_open = bool(status_profile.get("entry_open"))
        if "within_monitor_window" in status_profile:
            monitor_ok = bool(status_profile.get("within_monitor_window"))
        if "within_entry_window" in status_profile:
            entry_window_ok = bool(status_profile.get("within_entry_window"))
        elif "within_window" in status_profile:
            entry_window_ok = bool(status_profile.get("within_window"))
        if "weekday_ok" in status_profile:
            weekday_ok = bool(status_profile.get("weekday_ok"))
        next_open_raw = status_profile.get("next_open_utc")
        if isinstance(next_open_raw, str) and next_open_raw.strip():
            next_open_override = next_open_raw.strip()
        notes_raw = status_profile.get("notes")
        if isinstance(notes_raw, list):
            profile_notes = [
                note.strip()
                for note in notes_raw
                if isinstance(note, str) and note.strip()
            ]
        tags_raw = status_profile.get("tags")
        if isinstance(tags_raw, list):
            profile_tags = [
                tag.strip()
                for tag in tags_raw
                if isinstance(tag, str) and tag.strip()
            ]
        context_raw = status_profile.get("context")
        if isinstance(context_raw, dict):
            profile_context = {str(key): value for key, value in context_raw.items()}

    info: Dict[str, Any] = {
        "open": open_now,
        "entry_open": entry_open,
        "within_window": entry_window_ok,
        "within_entry_window": entry_window_ok,
        "within_monitor_window": monitor_ok,
        "weekday_ok": weekday_ok,
        "now_utc": now_utc.isoformat(),
        "now_budapest": now_budapest.isoformat(),
        "windows_utc": entry_windows,
    }
    if monitor_windows and monitor_windows != entry_windows:
        info["monitor_windows_utc"] = monitor_windows
    info["time_zone"] = "Europe/Berlin"
    info["time_zone_budapest"] = "Europe/Budapest"
    entry_local = convert_windows_to_local(entry_windows)
    if entry_local:
        info["windows_local_cet"] = entry_local
    entry_local_budapest = convert_windows_to_local(entry_windows, tz=BUDAPEST_TIMEZONE)
    if entry_local_budapest:
        info["windows_local_budapest"] = entry_local_budapest
    monitor_local = convert_windows_to_local(monitor_windows)
    if monitor_local and monitor_windows != entry_windows:
        info["monitor_windows_local_cet"] = monitor_local
    monitor_local_budapest = convert_windows_to_local(
        monitor_windows, tz=BUDAPEST_TIMEZONE
    )
    if monitor_local_budapest and monitor_windows != entry_windows:
        info["monitor_windows_local_budapest"] = monitor_local_budapest
    allowed = SESSION_WEEKDAYS.get(asset)
    if allowed:
        info["allowed_weekdays"] = list(allowed)
    info["open"] = open_now
    info["entry_open"] = entry_open
    info["within_window"] = entry_window_ok
    info["within_entry_window"] = entry_window_ok
    info["within_monitor_window"] = monitor_ok
    info["weekday_ok"] = weekday_ok
    info["status_profile"] = status_profile_name
    market_closed_reason: Optional[str] = None

    if status_profile.get("force_session_closed"):
        info["status_profile_forced"] = True
    if profile_tags:
        info["status_profile_tags"] = profile_tags
    if profile_context:
        info["status_profile_context"] = profile_context
    if not weekday_ok:
        status = "closed_weekend"
        status_note = "Piac zárva (hétvége)"
        market_closed_reason = "weekend"
    elif not open_now:
        status = "closed_out_of_hours"
        status_note = "Piac zárva (nyitáson kívül)"
        market_closed_reason = "outside_hours"
    elif not entry_open:
        status = "open_entry_limited"
        status_note = "Piac nyitva (csak pozíciómenedzsment, entry ablak zárva)"
    else:
        status = "open"
        status_note = "Piac nyitva"
    if special_status:
        status = special_status
    if special_note:
        status_note = special_note
    if special_reason:
        market_closed_reason = special_reason
    if status_profile:
        override_status = status_profile.get("status")
        override_note = status_profile.get("status_note")
        if override_status:
            status = str(override_status)
        if override_note:
            status_note = str(override_note)
        if "market_closed_reason" in status_profile:
            market_closed_reason = str(status_profile.get("market_closed_reason"))
        if "market_closed_assumed" in status_profile:
            info["market_closed_assumed"] = bool(
                status_profile.get("market_closed_assumed")
            )
    if break_active:
        info["daily_break_active"] = True
    if break_window:
        info["daily_break_window_utc"] = [format_utc_minute(break_window[0]), format_utc_minute(break_window[1])]
        info["daily_break_window_cet"] = convert_minutes_to_local_range(*break_window)
        info["daily_break_window_budapest"] = convert_minutes_to_local_range(
            *break_window, tz=BUDAPEST_TIMEZONE
        )
    if special_reason:
        info["special_closure_reason"] = special_reason
    if sunday_open is not None:
        info["sunday_open_utc"] = format_utc_minute(sunday_open)
        info["sunday_open_budapest"] = sunday_open_local[0] if sunday_open_local else None
    if news_lockout:
        info["news_lockout"] = True
    next_open_calculated = next_session_open(asset, now_utc)
    if next_open_calculated:
        info["next_session_open_utc"] = next_open_calculated.isoformat()
        info["next_session_open_budapest"] = next_open_calculated.astimezone(
            BUDAPEST_TIMEZONE
        ).isoformat()
    info["status"] = status
    info["status_note"] = status_note
    if market_closed_reason:
        info["market_closed_reason"] = market_closed_reason
    if profile_notes:
        notes_bucket = info.setdefault("notes", [])
        for note in profile_notes:
            if note not in notes_bucket:
                notes_bucket.append(note)
    if next_open_override:
        info["next_open_utc"] = next_open_override
    elif next_open_calculated:
        info["next_open_utc"] = next_open_calculated.isoformat()
    return entry_open, info

def session_ok(asset: str) -> bool:
    ok, _ = session_state(asset)
    return ok


def format_atr_hint(asset: str, atr_value: Optional[float]) -> Optional[str]:
    if atr_value is None or not np.isfinite(atr_value):
        return None
    if atr_value == 0:
        return None
    if asset == "EURUSD":
        return f"1h ATR ≈ {atr_value:.4f}"
    if asset == "BTCUSD":
        if atr_value >= 500:
            return f"1h ATR ≈ {atr_value:.0f}"
        if atr_value >= 100:
            return f"1h ATR ≈ {atr_value:.1f}"
        return f"1h ATR ≈ {atr_value:.2f}"
    if atr_value >= 100:
        return f"1h ATR ≈ {atr_value:.0f}"
    if atr_value >= 10:
        return f"1h ATR ≈ {atr_value:.1f}"
    return f"1h ATR ≈ {atr_value:.2f}"


def format_price_compact(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    abs_val = abs(numeric)
    if abs_val >= 1000:
        decimals = 1
    elif abs_val >= 100:
        decimals = 2
    elif abs_val >= 10:
        decimals = 3
    else:
        decimals = 4
    return f"{numeric:.{decimals}f}"


def translate_gate_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    base_map = {
        "session": "Piac zárva – várj a kereskedési ablakra.",
        "regime": "Trend filter nem támogatja az irányt.",
        "bias": "Nincs egyértelmű bias – várj megerősítésre.",
        "atr": "ATR feltétel nem teljesült – volatilitás nem megfelelő.",
        "liquidity": "Forgalmi feltételek nem teljesültek – várj nagyobb volumenre.",
        "order_flow": "Order flow nincs összhangban a belépővel.",
        "order_flow_pressure": "Order flow nyomás gyenge – várj további erősítésre.",
        "momentum_trigger": "Momentum trigger még nem aktív.",
        "bos5m": "5m struktúra nem erősíti meg a setupot.",
        "tp_min_profit": "TP1 távolság túl kicsi – keress jobb kockázat/hozam arányt.",
        "min_stoploss": "Stop távolság túl kicsi – szélesítsd a védelmi sávot.",
        "intervention_watch": "Crypto Watch tiltja az új belépőt.",
        "ml_confidence": "ML valószínűség alacsony – belépő blokkolva.",
        "precision_flow_alignment": "Precision belépő: order flow megerősítés hiányzik.",
        "precision_trigger_sync": "Precision belépő: trigger szinkronra vár.",
        "intraday_range_guard": "Intraday range telített – várj visszahúzódásra.",
        "spread_gate": "Spread gate: aktuális spread túl széles a setuphoz.",
        "sl_too_tight": "Stop-loss puffer túl szűk – növeld a kockázati sávot.",
        "tp_min_pct": "TP1 cél nem éri el a profil szerinti minimum százalékot.",
        "rr_min": "RR arány nem teljesíti a profil követelményét.",
        "atr_gate": "BTC ATR kapu blokkolja a setupot.",
        "triggers": "BTC trigger feltételek hiányosak.",
        "no_chase": "No-chase szabály sérül – ne üldözd az árat.",
        "no_chase_core": "Core no-chase szabály sérül – ne üldözd az árat.",
    }
    if normalized in base_map:
        return base_map[normalized]
    if normalized.startswith("rr_math>="):
        threshold = label.split(">=", 1)[1]
        return f"RR arány nem éri el a {threshold} célt."
    if normalized.startswith("tp1_net>="):
        threshold = label.split(">=", 1)[1]
        return f"TP1 nettó profit érje el a {threshold} küszöböt."
    if normalized.startswith("precision_score>="):
        threshold = label.split(">=", 1)[1]
        return f"Precision score érje el a {threshold} szintet."
    return label


def translate_urgency(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    mapping = {
        "immediate": "azonnal",
        "fast": "sürgősen",
        "normal": "normál ütemben",
        "monitor": "folyamatos megfigyeléssel",
    }
    return mapping.get(str(value).lower(), str(value))


def translate_severity(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    mapping = {
        "critical": "kritikus",
        "high": "magas",
        "elevated": "emelt",
        "moderate": "mérsékelt",
        "caution": "figyelmeztetés",
        "info": "információs",
    }
    return mapping.get(str(value).lower(), str(value))


def short_text(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_action_plan(
    asset: str,
    decision: str,
    session_meta: Optional[Dict[str, Any]],
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    rr: Optional[float],
    leverage: Optional[float],
    probability: Optional[int],
    precision_plan: Optional[Dict[str, Any]],
    execution_playbook: Optional[List[Dict[str, Any]]],
    position_note: Optional[str],
    exit_signal: Optional[Dict[str, Any]],
    missing: Optional[List[str]],
    reasons: Optional[List[str]],
    last_computed_risk: Optional[float],
    momentum_trailing_plan: Optional[Dict[str, Any]],
    intraday_profile: Optional[Dict[str, Any]],
    btc_profile: Optional[str] = None,
    entry_thresholds: Optional[Dict[str, Any]] = None,
    entry_thresholds_meta: Optional[Dict[str, Any]] = None,
    entry_gate_context_hu: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session_meta = session_meta or {}
    execution_playbook = execution_playbook or []
    intraday_profile = intraday_profile or {}
    entry_thresholds_meta = entry_thresholds_meta or {}
    entry_gate_context_hu = entry_gate_context_hu or {}
    analysis_now = datetime.now(timezone.utc)
    gate_extra_context: Dict[str, Any] = {}
    btc_profile_name = btc_profile if isinstance(btc_profile, str) else None
    blockers_raw = [str(item) for item in (missing or []) if item]
    blockers: List[str] = []
    for raw in blockers_raw:
        translated = translate_gate_label(raw)
        if translated not in blockers:
            blockers.append(translated)

    priority_rank = {"low": 0, "medium": 1, "high": 2}
    status_rank = {
        "standby": 0,
        "await_session": 1,
        "monitor_trigger": 2,
        "execute_entry": 3,
        "manage_position": 4,
    }
    current_priority = "low"
    current_status = "standby"

    plan: Dict[str, Any] = {
        "status": current_status,
        "priority": current_priority,
        "summary": "",
        "steps": [],
        "blockers": blockers,
        "blockers_raw": blockers_raw,
        "notes": [],
        "context": {
            "asset": asset,
            "decision": decision,
            "probability": probability,
            "session_open": bool(session_meta.get("open")),
            "session_entry_open": bool(session_meta.get("entry_open")),
            "session_status": session_meta.get("status"),
        },
    }

    def append_blocker(label: str) -> None:
        raw_label = (label or "").strip().lower()
        if not raw_label:
            return
        if raw_label not in blockers_raw:
            blockers_raw.append(raw_label)
        translated = translate_gate_label(raw_label)
        if translated not in blockers:
            blockers.append(translated)

    if intraday_profile:
        plan["context"]["intraday_profile"] = {
            "range_state": intraday_profile.get("range_state"),
            "range_position": intraday_profile.get("range_position"),
            "range_vs_atr": intraday_profile.get("range_vs_atr"),
            "range_guard": intraday_profile.get("range_guard"),
            "opening_break": intraday_profile.get("opening_break"),
        }

    if session_meta.get("next_open_utc"):
        plan["context"]["next_session_open_utc"] = session_meta.get("next_open_utc")
      
    notes: List[str] = []
    summary_parts: List[str] = []
    order_counter = 1

    def add_note(text: Optional[str]) -> None:
        if not text:
            return
        note_clean = text.strip()
        if note_clean and note_clean not in notes:
            notes.append(note_clean)

    btc_entry_summary: Optional[Dict[str, Any]] = None
    if asset == "BTCUSD":
        summary_blockers: List[str] = []
        try:
            profile_name = btc_profile or (
                entry_thresholds.get("btc_profile")
                if isinstance(entry_thresholds, dict)
                else None
            )
            if not profile_name:
                profile_name = _btc_active_profile()

            atr_value_raw = (
                entry_thresholds.get("btc_atr_value_usd")
                if isinstance(entry_thresholds, dict)
                else None
            )
            try:
                atr_value = float(atr_value_raw) if atr_value_raw is not None else float("nan")
            except (TypeError, ValueError):
                atr_value = float("nan")
            atr_threshold = (
                entry_thresholds.get("btc_atr_gate_threshold_usd")
                if isinstance(entry_thresholds, dict)
                else None
            )
            atr_ok_meta = (
                entry_thresholds.get("btc_atr_gate_ok")
                if isinstance(entry_thresholds, dict)
                else None
            )
            atr_ok = bool(atr_ok_meta) if atr_ok_meta is not None else btc_atr_gate_ok(
                profile_name,
                asset,
                atr_value,
                datetime.now(timezone.utc),
            )

            adx_raw = (
                entry_thresholds.get("adx_value")
                if isinstance(entry_thresholds, dict)
                else None
            )
            try:
                adx_val = float(adx_raw) if adx_raw is not None else 25.0
            except (TypeError, ValueError):
                adx_val = 25.0

            default_rr = rr if isinstance(rr, (int, float)) else None
            if default_rr is None:
                default_rr = 1.5
            rr_min, rr_info = btc_rr_min_with_adx(profile_name, asset, float(adx_val))
            rr_effective = rr_min if rr_min is not None else default_rr

            side = None
            if decision == "buy":
                side = "long"
            elif decision == "sell":
                side = "short"

            core_ok = False
            trig_info: Dict[str, Any] = {}
            if side:
                try:
                    core_ok, trig_info = btc_core_triggers_ok(asset, side)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("BTC core trigger summary failed: %s", exc)
                    core_ok = False
                    trig_info = {}

            mom_ok = False
            mom_rr: Optional[float] = None
            mom_note = ""
            if side:
                try:
                    mom_ok, mom_rr, mom_note = btc_momentum_override(
                        profile_name,
                        asset,
                        side,
                        atr_ok,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("BTC momentum override summary failed: %s", exc)
                    mom_ok = False
                    mom_rr = None
                    mom_note = ""

            enter_mode: Optional[str] = None
            if mom_ok:
                enter_mode = "momentum"
                if mom_rr is not None:
                    rr_effective = float(mom_rr)
            elif core_ok:
                enter_mode = "core"

            if not atr_ok:
                summary_blockers.append("atr_gate")
                append_blocker("atr_gate")
            if not (core_ok or mom_ok):
                summary_blockers.append("triggers")
                append_blocker("triggers")

            momentum_no_chase = (
                entry_thresholds.get("momentum_no_chase")
                if isinstance(entry_thresholds, dict)
                else None
            )
            if isinstance(momentum_no_chase, dict) and momentum_no_chase.get("violated"):
                summary_blockers.append("no_chase")
                append_blocker("no_chase")

            extra_checks: Dict[str, Any] = {}
            if decision in {"buy", "sell"} and enter_mode and entry is not None and sl is not None and tp1 is not None:
                blockers_local: List[str] = []
                info_payload: Dict[str, Any] = {
                    "profile": profile_name,
                    "mode": enter_mode,
                    "side": side,
                    "entry": float(entry),
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "rr_required": float(rr_effective) if rr_effective is not None else None,
                }
                atr_for_checks = atr_value if np.isfinite(atr_value) else 0.0
                btc_sl_tp_checks(
                    profile_name,
                    asset,
                    float(atr_for_checks),
                    float(entry),
                    float(sl),
                    float(tp1),
                    float(rr_effective) if rr_effective is not None else default_rr,
                    info_payload,
                    blockers_local,
                )
                if asset == "BTCUSD" and enter_mode == "core":
                    profile_for_no_chase = profile_name or _btc_active_profile()
                    slip_limit = BTC_NO_CHASE_R.get(
                        profile_for_no_chase, BTC_NO_CHASE_R["baseline"]
                    )
                    limit_map = globals().get("_BTC_NO_CHASE_LIMITS", {})
                    if (
                        isinstance(limit_map, dict)
                        and profile_for_no_chase in limit_map
                    ):
                        try:
                            slip_limit = float(limit_map[profile_for_no_chase])
                        except (TypeError, ValueError):
                            slip_limit = BTC_NO_CHASE_R.get(
                                profile_for_no_chase, BTC_NO_CHASE_R["baseline"]
                            )
                    trigger_reference = last5_close if last5_close is not None else entry
                    if (
                        trigger_reference is not None
                        and entry is not None
                        and sl is not None
                    ):
                        try:
                            entry_price = float(entry)
                            trigger_price = float(trigger_reference)
                            sl_price = float(sl)
                        except (TypeError, ValueError):
                            entry_price = trigger_price = sl_price = None
                        if (
                            entry_price is not None
                            and trigger_price is not None
                            and sl_price is not None
                        ):
                            slip_r = abs(entry_price - trigger_price) / max(
                                1e-9, abs(trigger_price - sl_price)
                            )
                            violated_no_chase = slip_r > slip_limit
                            info_payload["no_chase_core_r"] = float(slip_r)
                            info_payload["no_chase_core_limit_r"] = float(slip_limit)
                            if asset == "BTCUSD" and violated_no_chase:
                                info_payload["no_chase_r"] = {
                                    "slip_r": float(slip_r),
                                    "limit_r": float(slip_limit),
                                }
                            if violated_no_chase and "no_chase_core" not in blockers_local:
                                blockers_local.append("no_chase_core")
                atr_gate_ratio = float("nan")
                if atr_threshold and price_for_calc:
                    try:
                        atr_gate_ratio = float(atr_threshold) / float(price_for_calc)
                    except Exception:
                        atr_gate_ratio = float("nan")
                rel_atr_val = float("nan")
                if rel_atr is not None:
                    try:
                        rel_atr_val = float(rel_atr)
                    except Exception:
                        rel_atr_val = float("nan")
                rr_actual = info_payload.get("rr")
                rr_min_val = info_payload.get("rr_min")
                tp_pct_val = info_payload.get("tp_pct")
                tp_min_val = info_payload.get("tp_min_pct")
                ctx = {
                    "rel_atr": rel_atr_val,
                    "atr_gate_th": atr_gate_ratio,
                    "P": float(P) if P is not None else float("nan"),
                    "p_min": float(p_score_min_local)
                    if p_score_min_local is not None
                    else float("nan"),
                    "ofi_z": ofi_zscore if ofi_zscore is not None else float("nan"),
                    "ofi_trig": BTC_OFI_Z.get("trigger", 1.0),
                    "adx": float(adx_val) if adx_val is not None else float("nan"),
                    "adx_trend_min": float(BTC_ADX_TREND_MIN),
                    "rr": float(rr_actual) if rr_actual is not None else float("nan"),
                    "rr_min": float(rr_min_val) if rr_min_val is not None else float("nan"),
                    "tp_pct": float(tp_pct_val) if tp_pct_val is not None else float("nan"),
                    "tp_min_pct": float(tp_min_val) if tp_min_val is not None else float("nan"),
                }
                info_payload["gate_margins"] = btc_gate_margins(asset, ctx)
                extra_checks = info_payload
                for blocker in blockers_local:
                    summary_blockers.append(blocker)
                    append_blocker(blocker)

            if mom_rr is not None:
                rr_effective = float(mom_rr)

            summary_note_parts = [
                f"mód: {enter_mode or 'nincs'}",
                f"ATR kapu: {'OK' if atr_ok else 'blokkol'}",
                f"core trigger: {'OK' if core_ok else 'hiányzik'}",
                f"momentum override: {'OK' if mom_ok else 'inaktív'}",
            ]
            if rr_effective is not None:
                summary_note_parts.append(f"RR min: {float(rr_effective):.2f}")
            if mom_note:
                summary_note_parts.append(mom_note)
            add_note("Belépési döntés – összefogás: " + "; ".join(summary_note_parts))

            btc_entry_summary = {
                "profile": profile_name,
                "enter_mode": enter_mode,
                "atr_gate": {
                    "ok": bool(atr_ok),
                    "value": float(atr_value) if np.isfinite(atr_value) else None,
                    "threshold": float(atr_threshold) if atr_threshold is not None else None,
                },
                "core_triggers": {"ok": bool(core_ok), **trig_info},
                "momentum_override": {
                    "ok": bool(mom_ok),
                    "rr_min": float(mom_rr) if mom_rr is not None else None,
                    "note": mom_note or None,
                },
                "rr_min_effective": float(rr_effective) if rr_effective is not None else None,
                "rr_from_adx": float(rr_min) if rr_min is not None else None,
                "rr_info": rr_info,
            }
            if isinstance(momentum_no_chase, dict):
                btc_entry_summary["momentum_no_chase"] = momentum_no_chase
            if extra_checks:
                btc_entry_summary["sl_tp_checks"] = extra_checks
            if isinstance(rr_info, dict) and rr_info.get("mode") == "range":
                btc_entry_summary["range_constraints"] = {
                    "size_scale": rr_info.get("size_scale"),
                    "time_stop": rr_info.get("time_stop"),
                    "be_trigger_r": rr_info.get("be_trigger_r"),
                }
            if summary_blockers:
                btc_entry_summary["blockers"] = summary_blockers
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to compile BTC entry summary: %s", exc)
            btc_entry_summary = None
          

    for note in intraday_profile.get("notes", []):
        add_note(note)

    def add_step(
        category: str,
        instruction: Optional[str],
        *,
        source: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal order_counter
        if not instruction:
            return
        step: Dict[str, Any] = {
            "order": order_counter,
            "category": category,
            "instruction": instruction,
        }
        if source:
            step["source"] = source
        if extra:
            for key, value in extra.items():
                if value is not None:
                    step[key] = value
        plan["steps"].append(step)
        order_counter += 1

    def update_state(new_status: str, new_priority: str) -> None:
        nonlocal current_status, current_priority
        if status_rank.get(new_status, 0) > status_rank.get(current_status, 0):
            current_status = new_status
            current_priority = new_priority
        elif status_rank.get(new_status, 0) == status_rank.get(current_status, 0):
            if priority_rank.get(new_priority, 0) > priority_rank.get(current_priority, 0):
                current_priority = new_priority

    session_open = bool(session_meta.get("open"))
    entry_open = bool(session_meta.get("entry_open"))
    next_open = session_meta.get("next_open_utc")

    if probability is not None:
        add_note(f"Összesített valószínűség: {probability}%.")
    if leverage is not None:
        add_note(f"Ajánlott tőkeáttétel: ×{float(leverage):.1f}.")

    if rr is not None and np.isfinite(rr):
        add_note(f"Célzott RR: {float(rr):.2f}R.")

    risk_pct: Optional[float] = None
    if last_computed_risk is not None and entry is not None:
        try:
            risk_pct = abs(float(last_computed_risk)) / abs(float(entry)) * 100.0
        except (ZeroDivisionError, TypeError, ValueError):
            risk_pct = None
    if risk_pct is not None and np.isfinite(risk_pct):
        add_note(
            f"Belépő–stop távolság ≈ {risk_pct:.2f}% ({abs(float(last_computed_risk)):.5f} ár)."
        )
        risk_cap = get_max_risk_pct(asset)
        add_note(f"Maximális számla kockázat: {risk_cap:.1f}%.")
        try:
            entry_thresholds_meta.setdefault("risk_guard", risk_guard_meta)[
                "risk_pct_estimate"
            ] = float(risk_pct)
        except Exception:
            entry_thresholds_meta.setdefault("risk_guard", {})[
                "risk_pct_estimate"
            ] = float(risk_pct)

    if momentum_trailing_plan:
        activation = momentum_trailing_plan.get("activation_rr")
        lock_ratio = momentum_trailing_plan.get("lock_ratio")
        if activation is not None:
            add_note(
                f"Momentum trail aktiválás: {float(activation):.2f}R, lock {float(lock_ratio or 0) * 100:.0f}%"
            )

    if precision_plan:
        score = precision_plan.get("score")
        threshold_raw = precision_plan.get("score_threshold")
        try:
            threshold_val = float(threshold_raw)
        except (TypeError, ValueError):
            threshold_val = float(PRECISION_SCORE_THRESHOLD_DEFAULT)
        threshold_display = (
            f"{int(round(threshold_val))}"
            if float(threshold_val).is_integer()
            else f"{threshold_val:.1f}"
        )
        confidence = precision_plan.get("confidence")
        trigger_state = precision_plan.get("trigger_state")
        if score is not None:
            add_note(
                f"Precision score: {float(score):.2f} / küszöb {threshold_display}."
            )
        if confidence:
            add_note(f"Precision confidence: {confidence}.")
        if trigger_state:
            add_note(f"Precision trigger állapot: {trigger_state}.")

    if position_note:
        add_note(position_note)

    for item in (reasons or [])[:6]:
        add_note(item)

    if exit_signal:
        severity_label = translate_severity(exit_signal.get("severity")) or ""
        state_label = str(exit_signal.get("state") or exit_signal.get("action") or "")
        state_map = {
            "hard_exit": "hard exit",
            "scale_out": "részleges zárás",
            "warn": "figyelmeztetés",
            "trail_guard": "trail védelem",
        }
        state_display = state_map.get(state_label, state_label.replace("_", " "))
        direction = exit_signal.get("direction")
        summary_parts.append(
            short_text(
                f"{severity_label.capitalize() if severity_label else 'Exit'} jelzés: {state_display}"
                + (f" ({direction})" if direction else "")
            )
        )
        exit_actions = exit_signal.get("actions") or []

        def describe_exit_action(action: Dict[str, Any]) -> str:
            action_type = str(action.get("type") or "").lower()
            size = action.get("size")
            urgency = translate_urgency(action.get("urgency"))
            extra_parts: List[str] = []
            if urgency:
                extra_parts.append(f"sürgősség: {urgency}")
            if action_type == "close":
                if size is None or float(size) >= 0.999:
                    base = "Zárd a teljes pozíciót"
                else:
                    base = f"Zárd a pozíció {float(size) * 100:.0f}%-át"
            elif action_type == "scale_out":
                pct = float(size) * 100 if size is not None else 50.0
                base = f"Realizálj {pct:.0f}% pozíciót"
            elif action_type == "tighten_stop":
                if direction == "long":
                    base = "Húzd a stop-loss-t a legutóbbi swing low alá"
                elif direction == "short":
                    base = "Húzd a stop-loss-t a legutóbbi swing high fölé"
                else:
                    base = "Szorítsd a stop-loss sávot a legfrissebb struktúra mögé"
            elif action_type == "monitor":
                base = "Fokozott monitorozás 5m/1m idősíkon"
            else:
                base = f"Hajtsd végre a(z) {action_type} műveletet"
            if extra_parts:
                base += " (" + ", ".join(extra_parts) + ")"
            return base

        for action in exit_actions:
            add_step(
                "exit",
                describe_exit_action(action),
                source="position_exit_signal",
                extra={k: v for k, v in action.items() if v is not None},
            )
        if exit_signal.get("reasons"):
            for reason in exit_signal.get("reasons", [])[:4]:
                add_note(reason)
        update_state(
            "manage_position",
            "high"
            if (exit_signal.get("severity") in {"critical", "high"} or exit_signal.get("action") == "close")
            else "medium",
        )

    entry_present = any((step or {}).get("step") == "entry" for step in execution_playbook)

    if execution_playbook:
        for pb_step in execution_playbook:
            description = pb_step.get("description")
            if not description:
                continue
            category = str(pb_step.get("step") or "playbook")
            extra = {
                key: pb_step.get(key)
                for key in (
                    "risk_abs",
                    "confidence",
                    "rr",
                    "trigger_rr",
                    "lock_ratio",
                    "entry_window",
                    "trigger_levels",
                )
                if pb_step.get(key) is not None
            }
            add_step(category, description, source="execution_playbook", extra=extra)

    if decision in {"buy", "sell"} and entry is not None and sl is not None:
        direction_label = "long" if decision == "buy" else "short"
        summary_text = f"Új {direction_label} setup kész ({probability}% valószínűség)"
        if not entry_open:
            summary_text += " – belépési ablak zárva"
        summary_parts.append(short_text(summary_text))
        stop_instruction = (
            "Stop-loss szint: "
            + format_price_compact(sl)
            + (" (long)" if decision == "buy" else " (short)")
        )
        add_step(
            "risk",
            stop_instruction,
            source="levels",
            extra={"stop_loss": float(sl)},
        )
        update_state("execute_entry" if entry_open else "await_session", "high" if entry_open else "medium")
        if not entry_present:
            add_step(
                "entry",
                f"Nyiss {direction_label} pozíciót {format_price_compact(entry)} környékén.",
                source="levels",
                extra={"entry_price": float(entry)},
            )
        if tp1 is not None:
            add_step(
                "take_profit",
                f"TP1: {format_price_compact(tp1)} (részleges zárás).",
                source="levels",
                extra={"tp1": float(tp1)},
            )
        if tp2 is not None:
            add_step(
                "take_profit",
                f"TP2: {format_price_compact(tp2)} (végcél).",
                source="levels",
                extra={"tp2": float(tp2)},
            )

    precision_direction = (precision_plan or {}).get("direction")
    precision_trigger_state = str((precision_plan or {}).get("trigger_state") or "")
    precision_profile_for_state: Optional[str] = None
    precision_state = "none"
    if precision_plan:
        score_value = safe_float(precision_plan.get("score")) or 0.0
        trigger_ready_flag = bool(precision_plan.get("trigger_ready")) or precision_trigger_state in {
            "arming",
            "fire",
        }
        profile_for_precision: Optional[str] = None
        if asset == "BTCUSD":
            if btc_profile_name and btc_profile_name in BTC_PRECISION_MIN:
                profile_for_precision = btc_profile_name
            else:
                profile_for_precision = _btc_active_profile()
        elif btc_profile_name:
            profile_for_precision = btc_profile_name
        precision_profile_for_state = profile_for_precision or "baseline"
        precision_state = btc_precision_state(
            precision_profile_for_state,
            asset,
            score_value,
            trigger_ready_flag,
        )
    if precision_state == "none" and decision in {"precision_ready", "precision_arming"}:
        precision_state = decision
    now_runtime = datetime.now(timezone.utc)
    runtime_bucket = _PRECISION_RUNTIME.setdefault(asset, {})
    ready_since = runtime_bucket.get("ready_since")
    arming_since = runtime_bucket.get("arming_since")
    if not isinstance(ready_since, datetime):
        ready_since = None
    if not isinstance(arming_since, datetime):
        arming_since = None
    precision_timeout_triggered = False
    timeout_minutes = 10
    runtime_bucket["events"] = int(runtime_bucket.get("events", 0)) + 1
    runtime_bucket.setdefault("hits_ready", 0)
    runtime_bucket.setdefault("hits_arming", 0)
    runtime_bucket.setdefault("hits_soft_block", 0)
    if runtime_bucket.get("last_state") == "precision_ready":
        elapsed = _precision_ready_elapsed_seconds(ready_since, now_runtime)
        if elapsed is not None and elapsed > timeout_minutes * 60:
            precision_timeout_triggered = True
            if precision_state == "precision_ready":
                precision_state = "none"
    if precision_state == "precision_ready":
        if runtime_bucket.get("last_state") != "precision_ready":
            runtime_bucket["ready_since"] = now_runtime
        runtime_bucket["last_state"] = "precision_ready"
        runtime_bucket["hits_ready"] = int(runtime_bucket.get("hits_ready", 0)) + 1
        runtime_bucket.pop("arming_since", None)
    elif precision_state == "precision_arming":
        if runtime_bucket.get("last_state") != "precision_arming":
            runtime_bucket["arming_since"] = now_runtime
        runtime_bucket["last_state"] = "precision_arming"
        runtime_bucket["hits_arming"] = int(runtime_bucket.get("hits_arming", 0)) + 1
        runtime_bucket.pop("ready_since", None)
    else:
        runtime_bucket["last_state"] = precision_state
        runtime_bucket.pop("ready_since", None)
        runtime_bucket.pop("arming_since", None)
        if precision_state == "precision_soft_block":
            runtime_bucket["hits_soft_block"] = int(runtime_bucket.get("hits_soft_block", 0)) + 1
    if precision_timeout_triggered and isinstance(entry_thresholds, dict):
        entry_thresholds["precision_timeout"] = {"minutes": timeout_minutes}
    precision_gate_snapshot = entry_thresholds_meta.get("precision_gate_state")
    if isinstance(precision_gate_snapshot, dict):
        precision_gate_snapshot["precision_state"] = precision_state
        precision_gate_snapshot["precision_soft_block"] = precision_state == "precision_soft_block"
        precision_gate_snapshot["precision_profile"] = precision_profile_for_state
        precision_gate_snapshot["btc_precision_threshold"] = BTC_PRECISION_MIN.get(
            precision_profile_for_state or "baseline",
            BTC_PRECISION_MIN.get("baseline"),
        )
        counters = runtime_bucket if isinstance(runtime_bucket, dict) else {}
        ready_ts_fields = _gate_timestamp_fields(ready_since) if ready_since else {}
        arming_ts_fields = _gate_timestamp_fields(arming_since) if arming_since else {}
        precision_gate_snapshot.update(
            {
                "precision_ready_since": ready_ts_fields or None,
                "precision_arming_since": arming_ts_fields or None,
                "precision_events": counters.get("events"),
                "precision_hits_ready": counters.get("hits_ready"),
                "precision_hits_arming": counters.get("hits_arming"),
                "precision_hits_soft_block": counters.get("hits_soft_block"),
            }
        )
        _log_precision_gate_summary(
            asset,
            precision_gate_snapshot,
            precision_gate_snapshot.get("flow_blockers"),
        )
        if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
            entry_gate_context_hu["precision_kapu"] = {
                **precision_gate_snapshot,
                **_gate_timestamp_fields(analysis_now),
            }
            gate_extra_context.update(entry_gate_context_hu)

    if precision_state in {"precision_ready", "precision_arming"} and precision_plan:
        label = "Precision trigger előkészítés"
        if precision_state == "precision_arming":
            label = "Precision trigger aktív"
        summary_parts.append(short_text(label))
        window = precision_plan.get("entry_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            add_step(
                "precision_window",
                (
                    f"Figyeld a {'long' if precision_direction == 'buy' else 'short' if precision_direction == 'sell' else 'trade'} belépő zónát"
                    f" {format_price_compact(window[0])}–{format_price_compact(window[1])}."
                ),
                source="precision",
                extra={"entry_window": list(window)},
            )
        if precision_plan.get("stop_loss") is not None:
            add_step(
                "precision_stop",
                f"Precision SL előkészítése: {format_price_compact(precision_plan['stop_loss'])}.",
                source="precision",
                extra={"stop_loss": float(precision_plan["stop_loss"])},
            )
        trigger_levels = precision_plan.get("trigger_levels") or {}
        if trigger_levels:
            add_step(
                "precision_trigger",
                "Aktiváld az árriasztást a precision trigger szintekre.",
                source="precision",
                extra={"trigger_levels": trigger_levels},
            )
        update_state("monitor_trigger", "high" if precision_state == "precision_arming" else "medium")

    if precision_state and precision_state != "none":
        plan["context"]["precision_state"] = precision_state
    if precision_trigger_state:
        plan["context"]["precision_trigger_state"] = precision_trigger_state
    if precision_direction:
        plan["context"]["precision_direction"] = precision_direction

    if btc_entry_summary:
        plan["context"]["btc_entry_summary"] = btc_entry_summary

    if not summary_parts:
        if decision == "no entry":
            if blockers:
                summary_parts.append("Nincs belépő – dolgozd le a hiányzó feltételeket.")
            else:
                summary_parts.append("Nincs belépő – figyeld a következő megerősítést.")
        elif exit_signal:
            summary_parts.append("Pozíció menedzsment folyamatban.")
        else:
            summary_parts.append(f"Állapot: {decision}")

    if (
        (decision in {"buy", "sell"})
        or (precision_state in {"precision_ready", "precision_arming"})
    ) and not entry_open:
        session_msg = "Belépési ablak zárva – várj a megnyitásig."
        if next_open:
            session_msg += f" Következő nyitás: {next_open}."
        add_step("session", session_msg, source="session", extra={"next_open_utc": next_open})

    plan["summary"] = " | ".join(summary_parts)
    plan["notes"] = notes
    update_state(current_status, current_priority)
    plan["status"] = current_status
    plan["priority"] = current_priority
    return plan


def derive_position_management_note(
    asset: str,
    session_meta: Dict[str, Any],
    regime_ok: bool,
    effective_bias: str,
    structure_flag: str,
    atr1h: Optional[float],
    anchor_bias: Optional[str],
    anchor_timestamp: Optional[str],
    anchor_record: Optional[Dict[str, Any]],
    current_p_score: Optional[float],
    current_rel_atr: Optional[float],
    current_atr5: Optional[float],
    current_price: Optional[float],
    invalid_level_buy: Optional[float],
    invalid_level_sell: Optional[float],
    invalid_buffer_abs: Optional[float],
    current_signal: Optional[str],
    sentiment_signal: Optional[SentimentSignal] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not session_meta.get("open"):
        return None, None
    if session_meta.get("status") in {"maintenance", "closed_out_of_hours"}:
        return None, None

    monitor_only = not session_meta.get("entry_open")
    anchor_active = anchor_bias in {"long", "short"}
    if not monitor_only and not anchor_active:
        return None, None

    bias = effective_bias or "neutral"

    hint_parts: List[str] = []
    if anchor_active:
        anchor_note = f"aktív {anchor_bias} pozíció"
        anchor_dt = parse_utc_timestamp(anchor_timestamp)
        if anchor_dt:
            local_dt = anchor_dt.astimezone(MARKET_TIMEZONE)
            anchor_note += f", nyitva: {local_dt.strftime('%Y-%m-%d %H:%M')} helyi idő"
        hint_parts.append(anchor_note)

    atr_hint = format_atr_hint(asset, atr1h)
    if atr_hint:
        hint_parts.append(atr_hint)

    hint_suffix = f" ({'; '.join(hint_parts)})" if hint_parts else ""
    prefix = "Pozíciómenedzsment: "

    prev_p = safe_float((anchor_record or {}).get("p_score")) if anchor_record else None
    prev_atr5 = safe_float((anchor_record or {}).get("atr5")) if anchor_record else None
    prev_rel_atr = safe_float((anchor_record or {}).get("rel_atr")) if anchor_record else None

    def deterioration_messages() -> List[str]:
        notes: List[str] = []
        if prev_p is not None and current_p_score is not None:
            if current_p_score <= prev_p - ANCHOR_P_SCORE_DELTA_WARN:
                notes.append(
                    f"P-score {prev_p:.0f}→{current_p_score:.0f} (−{prev_p - current_p_score:.0f}) → pozíciócsökkentés 50% / SL-szűkítés"
                )
        if prev_atr5 is not None and current_atr5 is not None and prev_atr5 > 0:
            if current_atr5 < prev_atr5 * ANCHOR_ATR_DROP_RATIO:
                notes.append("ATR5 jelentősen csökkent → profitvédelem (trail vagy részleges zárás) ajánlott")
        if prev_rel_atr is not None and current_rel_atr is not None:
            if not np.isnan(prev_rel_atr) and not np.isnan(current_rel_atr):
                if current_rel_atr < prev_rel_atr * ANCHOR_ATR_DROP_RATIO:
                    notes.append("Relatív ATR zsugorodik → pozícióvédelem szükséges")
        return notes

    if anchor_active and bias in {"long", "short"} and anchor_bias != bias:
        base = (
            prefix
            + f"aktív {anchor_bias} pozíció a jelenlegi bias ({bias}) ellen → defenzív menedzsment, részleges zárás vagy szoros SL"
        )
        det = deterioration_messages()
        if det:
            base += " — " + "; ".join(det)
        return base + hint_suffix, None

    direction = anchor_bias or bias
    base_message: Optional[str]
    exit_signal: Optional[Dict[str, Any]] = None

    current_price_val = safe_float(current_price)
    anchor_sl = safe_float((anchor_record or {}).get("stop_loss")) if anchor_record else None
    anchor_entry = safe_float((anchor_record or {}).get("entry_price") or (anchor_record or {}).get("price"))
    invalid_buy_val = safe_float(invalid_level_buy)
    invalid_sell_val = safe_float(invalid_level_sell)
    invalid_buffer_val = safe_float(invalid_buffer_abs)
    initial_risk_abs = safe_float((anchor_record or {}).get("initial_risk_abs"))
    max_fav_abs = safe_float((anchor_record or {}).get("max_favorable_excursion"))
    current_pnl_abs = safe_float((anchor_record or {}).get("current_pnl_abs"))
    current_pnl_r = safe_float((anchor_record or {}).get("current_pnl_r"))
    drift_state = (anchor_record or {}).get("drift_state") or (anchor_record or {}).get("anchor_drift_state")

    if (
        current_pnl_abs is None
        and anchor_entry is not None
        and current_price_val is not None
        and anchor_bias in {"long", "short"}
    ):
        if anchor_bias == "long":
            current_pnl_abs = current_price_val - anchor_entry
        else:
            current_pnl_abs = anchor_entry - current_price_val

    max_fav_r: Optional[float] = None
    if initial_risk_abs and initial_risk_abs > 0:
        if max_fav_abs is not None:
            max_fav_r = max_fav_abs / initial_risk_abs
        if current_pnl_r is None and current_pnl_abs is not None:
            current_pnl_r = current_pnl_abs / initial_risk_abs

    profit_drawdown_ratio: Optional[float] = None
    if max_fav_abs is not None and max_fav_abs > 0 and current_pnl_abs is not None:
        profit_drawdown_ratio = (max_fav_abs - current_pnl_abs) / max_fav_abs

    tolerance = 0.0
    if current_atr5 is not None and np.isfinite(current_atr5) and current_atr5 > 0:
        tolerance = max(tolerance, float(current_atr5) * 0.25)
    if invalid_buffer_val is not None and np.isfinite(invalid_buffer_val) and invalid_buffer_val > 0:
        tolerance = max(tolerance, float(invalid_buffer_val) * 0.15)
    if anchor_entry is not None and np.isfinite(anchor_entry):
        tolerance = max(tolerance, abs(anchor_entry) * 0.00015)

    def _triggered(price: Optional[float], side: str) -> bool:
        if price is None or not np.isfinite(price) or current_price_val is None:
            return False
        if side == "long":
            return current_price_val <= price + tolerance
        if side == "short":
            return current_price_val >= price - tolerance
        return False

    exit_reasons: List[str] = []

    if anchor_active and current_price_val is not None:
        if anchor_bias == "long":
            if _triggered(anchor_sl, "long"):
                exit_reasons.append("stop-loss zóna sérült")
            if _triggered(invalid_buy_val, "long"):
                exit_reasons.append("strukturális invalid szint elesett")
        elif anchor_bias == "short":
            if _triggered(anchor_sl, "short"):
                exit_reasons.append("stop-loss zóna sérült")
            if _triggered(invalid_sell_val, "short"):
                exit_reasons.append("strukturális invalid szint elesett")

    if exit_reasons:
        det = deterioration_messages()
        message = prefix + (
            f"aktív {anchor_bias} pozíció invalidálódott → hard exit szükséges"
        )
        message += " (" + "; ".join(exit_reasons) + ")"
        if det:
            message += " — " + "; ".join(det)
        message += hint_suffix
        exit_signal = {
            "state": "hard_exit",
            "severity": "critical",
            "action": "close",
            "reasons": exit_reasons,
            "trigger_price": current_price_val,
            "triggered_at": nowiso(),
        }
        exit_signal["actions"] = [
            {"type": "close", "size": 1.0, "urgency": "immediate"}
        ]
        exit_signal["category"] = "structural_invalid"
        if anchor_entry is not None and np.isfinite(anchor_entry):
            exit_signal["entry_price"] = anchor_entry
        if anchor_sl is not None and np.isfinite(anchor_sl):
            exit_signal["stop_loss"] = anchor_sl
        exit_signal["direction"] = anchor_bias
        if det:
            exit_signal["deterioration"] = det
        return message, exit_signal

    signal_norm = (current_signal or "").strip().lower()
    red_signal = signal_norm in {"no entry", "no"}

    structure_flip = False
    bias_flip = False
    reversal_reasons: List[str] = []
    reversal_strength = 0.0

    if anchor_bias == "long" and structure_flag == "bos_down":
        structure_flip = True
        reversal_strength += 1.0
        reversal_reasons.append("5m struktúra lefelé fordult")
    elif anchor_bias == "short" and structure_flag == "bos_up":
        structure_flip = True
        reversal_strength += 1.0
        reversal_reasons.append("5m struktúra felfelé fordult")

    if anchor_bias == "long" and bias == "short":
        bias_flip = True
        reversal_strength += 1.0
        reversal_reasons.append("Bias short-ra fordult")
    elif anchor_bias == "short" and bias == "long":
        bias_flip = True
        reversal_strength += 1.0
        reversal_reasons.append("Bias long-ra fordult")

    regime_break = not regime_ok
    if regime_break:
        reversal_strength += 1.0
        reversal_reasons.append("Trend filter kikapcsolt")

    drift_deteriorating = (drift_state == "deteriorating")
    if drift_deteriorating:
        reversal_strength += 1.0
        reversal_reasons.append("Anchor drift romlik")

    pscore_collapse = current_p_score is not None and current_p_score <= 35
    if pscore_collapse:
        reversal_reasons.append(f"P-score {current_p_score:.0f}")

    profit_risk = False
    profit_reasons: List[str] = []
    if max_fav_r is not None and max_fav_r >= 0.6:
        if current_pnl_r is not None and current_pnl_r <= max_fav_r * 0.35:
            profit_risk = True
        if current_pnl_r is not None and current_pnl_r <= 0.0:
            profit_risk = True
        if profit_drawdown_ratio is not None and profit_drawdown_ratio >= 0.65:
            profit_risk = True
    if current_pnl_r is not None and current_pnl_r <= -0.7:
        profit_risk = True

    if profit_risk:
        if max_fav_r is not None and current_pnl_r is not None:
            if current_pnl_r >= 0:
                profit_reasons.append(f"MFE {max_fav_r:.2f}R → most {current_pnl_r:.2f}R")
                if (
                    profit_drawdown_ratio is not None
                    and profit_drawdown_ratio >= 0.0
                    and max_fav_r >= 0.6
                ):
                    pct = max(0.0, min(100.0, profit_drawdown_ratio * 100.0))
                    if pct >= 50.0:
                        profit_reasons.append(f"profit {pct:.0f}% visszaadva")
            else:
                profit_reasons.append(
                    f"MFE {max_fav_r:.2f}R után pozíció {current_pnl_r:.2f}R"
                )
                if profit_drawdown_ratio is not None and profit_drawdown_ratio > 1.0:
                    profit_reasons.append("Profit teljesen elolvadt")
        elif max_fav_abs is not None and current_pnl_abs is not None:
            if current_pnl_abs >= 0:
                profit_reasons.append(
                    f"MFE {max_fav_abs:.1f} → most {current_pnl_abs:.1f}"
                )
                if (
                    profit_drawdown_ratio is not None
                    and profit_drawdown_ratio >= 0.0
                    and max_fav_abs > 0
                ):
                    pct = max(0.0, min(100.0, profit_drawdown_ratio * 100.0))
                    if pct >= 50.0:
                        profit_reasons.append(f"profit {pct:.0f}% visszaadva")
            else:
                profit_reasons.append(
                    f"MFE {max_fav_abs:.1f} után pozíció {current_pnl_abs:.1f}"
                )
                if profit_drawdown_ratio is not None and profit_drawdown_ratio > 1.0:
                    profit_reasons.append("Profit teljesen elolvadt")
        elif current_pnl_r is not None:
            profit_reasons.append(f"aktuális PnL {current_pnl_r:.2f}R")

    strong_reversal = (structure_flip and bias_flip) or reversal_strength >= 2.0
    profit_guard_trigger = structure_flip and profit_risk

    if (
        anchor_active
        and red_signal
        and (strong_reversal or profit_guard_trigger)
        and (profit_risk or pscore_collapse)
    ):
        context_parts = list(dict.fromkeys(reversal_reasons + profit_reasons))
        if not context_parts and pscore_collapse and current_p_score is not None:
            context_parts.append(f"P-score {current_p_score:.0f}")
        exit_actions: List[Dict[str, Any]] = []
        exit_state = "scale_out"
        exit_phrase = "profitvédelemként részleges zárás javasolt"
        if profit_drawdown_ratio is not None and profit_drawdown_ratio >= 0.85:
            exit_state = "hard_exit"
            exit_phrase = "profitvédelemként gyors teljes zárás javasolt"
            exit_actions.append({"type": "close", "size": 1.0, "urgency": "fast"})
        else:
            exit_actions.append({"type": "scale_out", "size": 0.5})
        exit_actions.append({"type": "tighten_stop"})
        message = prefix + (
            f"aktív {anchor_bias} pozíció piros jelzésben → {exit_phrase}"
        )
        if context_parts:
            message += " (" + "; ".join(context_parts) + ")"
        det = deterioration_messages()
        if det:
            message += " — " + "; ".join(det)
        message += hint_suffix
        exit_signal = {
            "state": exit_state,
            "severity": "high" if exit_state == "hard_exit" else "elevated",
            "action": "close" if exit_state == "hard_exit" else "scale_out",
            "category": "reversal_guard",
            "reasons": context_parts or profit_reasons or reversal_reasons,
            "trigger_price": current_price_val,
            "triggered_at": nowiso(),
            "direction": anchor_bias,
            "signal": current_signal,
        }
        exit_signal["actions"] = exit_actions
        if anchor_entry is not None and np.isfinite(anchor_entry):
            exit_signal["entry_price"] = anchor_entry
        if anchor_sl is not None and np.isfinite(anchor_sl):
            exit_signal["stop_loss"] = anchor_sl
        if max_fav_abs is not None and np.isfinite(max_fav_abs):
            exit_signal["max_favorable_abs"] = max_fav_abs
        if max_fav_r is not None and np.isfinite(max_fav_r):
            exit_signal["max_favorable_r"] = max_fav_r
        if current_pnl_abs is not None and np.isfinite(current_pnl_abs):
            exit_signal["current_pnl_abs"] = current_pnl_abs
        if current_pnl_r is not None and np.isfinite(current_pnl_r):
            exit_signal["current_pnl_r"] = current_pnl_r
        if profit_drawdown_ratio is not None and np.isfinite(profit_drawdown_ratio):
            exit_signal["profit_drawdown_ratio"] = profit_drawdown_ratio
        if current_p_score is not None and np.isfinite(current_p_score):
            exit_signal["p_score"] = current_p_score
        if drift_state:
            exit_signal["drift_state"] = drift_state
        return message, exit_signal

    trail_state: Optional[str] = None
    trail_reasons: List[str] = []
    trail_actions: List[Dict[str, Any]] = []
    if exit_signal is None:
        trail_log = (anchor_record or {}).get("trail_log") if anchor_record else None
        pnl_values: List[float] = []
        trail_times: List[Optional[datetime]] = []
        if isinstance(trail_log, list) and trail_log:
            for entry in trail_log[-12:]:
                pnl_val = safe_float(entry.get("pnl_r"))
                if pnl_val is None:
                    pnl_val = safe_float(entry.get("pnl_abs"))
                    if pnl_val is not None and initial_risk_abs and initial_risk_abs > 0:
                        pnl_val = pnl_val / initial_risk_abs
                ts = parse_utc_timestamp(entry.get("timestamp"))
                if pnl_val is not None and np.isfinite(pnl_val):
                    pnl_values.append(float(pnl_val))
                    trail_times.append(ts)
        if pnl_values:
            peak = max(pnl_values)
            latest = pnl_values[-1]
            drop_from_peak = peak - latest
            recent_drop = 0.0
            if len(pnl_values) >= 3:
                recent_drop = pnl_values[-3] - latest
            if peak >= 0.8 and drop_from_peak >= 0.45:
                trail_state = "scale_out"
                trail_reasons.append(
                    f"MFE {peak:.2f}R → jelenleg {latest:.2f}R (−{drop_from_peak:.2f}R)"
                )
            elif peak >= 0.4 and drop_from_peak >= 0.25:
                trail_state = "warn"
                trail_reasons.append(
                    f"MFE {peak:.2f}R → jelenleg {latest:.2f}R (−{drop_from_peak:.2f}R)"
                )
            if recent_drop >= 0.4:
                note = f"Az utolsó ~3 mérésben {recent_drop:.2f}R visszaadás"
                if note not in trail_reasons:
                    trail_reasons.append(note)
                trail_state = trail_state or "scale_out"
            if trail_times and len(trail_times) >= 2:
                t_now = trail_times[-1]
                t_prev = trail_times[0]
                if t_now and t_prev:
                    span_min = (t_now - t_prev).total_seconds() / 60.0
                    if span_min <= 30 and drop_from_peak >= 0.35:
                        note = "Gyors profit visszaadás 30 percen belül"
                        if note not in trail_reasons:
                            trail_reasons.append(note)
                        trail_state = trail_state or "scale_out"
        if trail_state:
            if trail_state == "scale_out":
                trail_actions = [
                    {"type": "scale_out", "size": 0.3},
                    {"type": "tighten_stop"},
                ]
            else:
                trail_actions = [
                    {"type": "tighten_stop"},
                    {"type": "monitor"},
                ]

    if direction == "long":
        if regime_ok and structure_flag == "bos_up":
            base_message = "long trend aktív → pozíció tartható, SL igazítás az 1h swing alatt"
        elif regime_ok:
            base_message = "long bias, de friss BOS nincs → részleges realizálás vagy szorosabb SL"
        else:
            base_message = "long kitettség gyenge trendben → méretcsökkentés vagy zárás mérlegelendő"
    elif direction == "short":
        if regime_ok and structure_flag == "bos_down":
            base_message = "short trend aktív → pozíció tartható, SL az 1h csúcsa felett"
        elif regime_ok:
            base_message = "short bias, de szerkezeti megerősítés hiányzik → részleges realizálás / SL szűkítés"
        else:
            base_message = "short kitettség gyenge trendben → méretcsökkentés vagy zárás mérlegelendő"
    else:
        base_message = "nincs egyértelmű bias → defenzív menedzsment, részleges zárás vagy szoros SL"

    det_notes = deterioration_messages()
    if det_notes:
        base_message += " — " + "; ".join(det_notes)

    sentiment_state: Optional[str] = None
    sentiment_reasons: List[str] = []
    sentiment_actions: List[Dict[str, Any]] = []
    sentiment_severity_label: Optional[str] = None
    if (
        (exit_signal is None or exit_signal.get("state") != "hard_exit")
        and anchor_active
        and sentiment_signal
        and anchor_bias in {"long", "short"}
    ):
        severity = sentiment_signal.effective_severity
        try:
            sentiment_score = float(sentiment_signal.score)
        except (TypeError, ValueError):
            sentiment_score = 0.0
        if severity >= 0.55 and sentiment_score:
            direction_factor = 1.0 if anchor_bias == "long" else -1.0
            effective_score = sentiment_score * direction_factor
            if effective_score <= -0.25:
                base_reason = (
                    f"Sentiment {sentiment_score:+.2f} ellentétes a {anchor_bias} pozícióval"
                )
                sentiment_reasons.append(base_reason)
                if sentiment_signal.headline:
                    headline_note = short_text(f"Hír: {sentiment_signal.headline}")
                    if headline_note not in sentiment_reasons:
                        sentiment_reasons.append(headline_note)
                if effective_score <= -0.55 or severity >= 0.85:
                    sentiment_state = "scale_out"
                    sentiment_actions = [
                        {"type": "scale_out", "size": 0.5},
                        {"type": "tighten_stop"},
                    ]
                    sentiment_severity_label = "high" if severity >= 0.85 else "moderate"
                else:
                    sentiment_state = "warn"
                    sentiment_actions = [
                        {"type": "tighten_stop"},
                        {"type": "monitor"},
                    ]
                    sentiment_severity_label = "moderate" if severity >= 0.7 else "caution"

                if base_message:
                    base_message += " — Sentiment kockázat: " + "; ".join(sentiment_reasons)
                else:
                    base_message = "Sentiment kockázat aktív"

                if exit_signal is None:
                    exit_signal = {
                        "state": sentiment_state or "warn",
                        "severity": sentiment_severity_label or ("high" if severity >= 0.85 else "moderate"),
                        "action": "scale_out"
                        if sentiment_state == "scale_out"
                        else "tighten_stop",
                        "reasons": list(dict.fromkeys(sentiment_reasons)),
                        "trigger_price": current_price_val,
                        "triggered_at": nowiso(),
                        "direction": anchor_bias,
                        "signal": current_signal,
                    }
                    exit_signal["category"] = "sentiment_risk"
                    exit_signal["actions"] = sentiment_actions or [
                        {"type": "tighten_stop"},
                        {"type": "monitor"},
                    ]
                    if anchor_entry is not None and np.isfinite(anchor_entry):
                        exit_signal["entry_price"] = anchor_entry
                    if current_pnl_r is not None and np.isfinite(current_pnl_r):
                        exit_signal["current_pnl_r"] = current_pnl_r
                    if max_fav_r is not None and np.isfinite(max_fav_r):
                        exit_signal["max_favorable_r"] = max_fav_r
                    if initial_risk_abs is not None and np.isfinite(initial_risk_abs):
                        exit_signal["initial_risk_abs"] = initial_risk_abs
                else:
                    existing_reasons = exit_signal.get("reasons") or []
                    for reason in sentiment_reasons:
                        if reason not in existing_reasons:
                            existing_reasons.append(reason)
                    exit_signal["reasons"] = existing_reasons
                    if exit_signal.get("state") != "hard_exit" and sentiment_state:
                        exit_signal["state"] = sentiment_state
                        exit_signal["action"] = (
                            "scale_out" if sentiment_state == "scale_out" else "tighten_stop"
                        )
                        exit_signal["severity"] = sentiment_severity_label or exit_signal.get("severity")
                    if sentiment_actions:
                        exit_actions_existing = exit_signal.setdefault("actions", [])
                        for action in sentiment_actions:
                            if action not in exit_actions_existing:
                                exit_actions_existing.append(action)
                    if exit_signal.get("category") not in {"structural_invalid"}:
                        exit_signal["category"] = exit_signal.get("category") or "sentiment_risk"

                if exit_signal is not None:
                    exit_signal["sentiment_score"] = sentiment_score
                    exit_signal["sentiment_severity"] = severity
                    if sentiment_signal.bias:
                        exit_signal["sentiment_bias"] = sentiment_signal.bias
                    if sentiment_signal.headline:
                        headline_short = short_text(sentiment_signal.headline)
                        headlines = exit_signal.setdefault("headlines", [])
                        if headline_short not in headlines:
                            headlines.append(headline_short)
                    if sentiment_signal.category:
                        exit_signal.setdefault("sentiment_category", sentiment_signal.category)

    if exit_signal is None and trail_state:
        exit_signal = {
            "state": trail_state,
            "severity": "moderate" if trail_state == "scale_out" else "caution",
            "action": trail_state,
            "reasons": trail_reasons,
            "trigger_price": current_price_val,
            "triggered_at": nowiso(),
            "direction": anchor_bias,
            "signal": current_signal,
        }
        exit_signal["category"] = "trail_guard"
        exit_signal["actions"] = trail_actions
        if anchor_entry is not None and np.isfinite(anchor_entry):
            exit_signal["entry_price"] = anchor_entry
        if current_pnl_r is not None and np.isfinite(current_pnl_r):
            exit_signal["current_pnl_r"] = current_pnl_r
        if max_fav_r is not None and np.isfinite(max_fav_r):
            exit_signal["max_favorable_r"] = max_fav_r
        if initial_risk_abs is not None and np.isfinite(initial_risk_abs):
            exit_signal["initial_risk_abs"] = initial_risk_abs
        if trail_reasons:
            base_message += " — Exit jelzés: " + "; ".join(trail_reasons)

    return (prefix + base_message + hint_suffix, exit_signal) if base_message else (None, exit_signal)

def atr_low_threshold(asset: str) -> float:
    h, m = now_utctime_hm()
    if asset == "GOLD_CFD":
        high_vol_base = GOLD_HIGH_VOL_TH or ATR_LOW_TH_ASSET.get(asset) or ATR_LOW_TH_DEFAULT
        low_vol_base = GOLD_LOW_VOL_TH or ATR_LOW_TH_ASSET.get(asset) or ATR_LOW_TH_DEFAULT
        if in_any_window_utc(GOLD_HIGH_VOL_WINDOWS, h, m):
            base = high_vol_base
        else:
            base = low_vol_base
    else:
        base = ATR_LOW_TH_ASSET.get(asset, ATR_LOW_TH_DEFAULT)
    multiplier = get_atr_threshold_multiplier(asset)
    return base * multiplier

def tp_min_pct_for(asset: str, rel_atr: float, session_flag: bool) -> float:
    base = get_tp_min_pct_value(asset)
    if asset == "BTCUSD":
        profile = _btc_active_profile()
        override_tp = BTC_TP_MIN_PCT.get(profile)
        if override_tp is not None:
            return float(override_tp)
    if np.isnan(rel_atr):
        return base
    low_atr_cfg = get_low_atr_override(asset)
    floor = safe_float(low_atr_cfg.get("floor"))
    tp_override = safe_float(low_atr_cfg.get("tp_min_pct"))
    if floor is not None and tp_override is not None and rel_atr < floor:
        return min(base, tp_override)
    return base

def tp_net_min_for(asset: str) -> float:
    return get_tp_net_min(asset)


def pip_size(asset: str) -> float:
    if asset.endswith("JPY"):
        return 0.01
    return 0.0001


def compute_cost_components(asset: str, entry: float, overnight_days: int) -> Tuple[float, float]:
    if entry <= 0:
        return 0.0, 0.0
    model = ASSET_COST_MODEL.get(asset, DEFAULT_COST_MODEL)
    mtype = model.get("type", "pct")
    if mtype == "pip":
        pip = pip_size(asset)
        rt = float(model.get("round_trip_pips", 0.0) or 0.0) * pip / entry
        overnight = float(model.get("overnight_pips", 0.0) or 0.0) * pip * overnight_days / entry
    else:
        rt = float(model.get("round_trip_pct", 0.0) or 0.0)
        overnight = float(model.get("overnight_pct", 0.0) or 0.0) * overnight_days
    return rt, overnight


def estimate_overnight_days(asset: str, now: Optional[datetime] = None) -> int:
    if now is None:
        now = datetime.now(timezone.utc)
    wd = now.weekday()
    if asset == "EURUSD":
        return 3 if wd == 2 else 1   # FX: szerdai tripla díj
    if asset in {"GOLD_CFD", "USOIL", "NVDA", "XAGUSD"}:
        return 3 if wd == 4 else 1   # hétvégi elszámolás pénteken
    return 1


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _nvda_precision_override_ready(
    asset: str,
    *,
    precision_plan: Optional[Dict[str, Any]],
    precision_threshold: float,
    spread_gate_ok: bool,
    session_entry_open: bool,
    risk_guard_allowed: bool,
    base_core_ok: bool,
) -> bool:
    if asset != "NVDA" or not base_core_ok:
        return False
    if not (precision_plan and spread_gate_ok and session_entry_open and risk_guard_allowed):
        return False
    if str(precision_plan.get("trigger_state") or "").lower() != "fire":
        return False
    score_val = safe_float(precision_plan.get("score")) or 0.0
    try:
        threshold_val = float(precision_threshold)
    except Exception:
        threshold_val = precision_threshold
    return score_val >= threshold_val


def _guard_realtime_price_stats(
    asset: str,
    stats: Dict[str, Any],
    reference_candidates: Sequence[Optional[float]],
) -> Dict[str, Any]:
    """Clamp realtime statistics that drift too far from trusted spot prices."""

    if not isinstance(stats, dict):
        return stats

    guard_cfg = get_realtime_price_guard(asset)
    limit_pct_raw = guard_cfg.get("limit_pct")
    min_abs_raw = guard_cfg.get("min_abs")

    try:
        limit_pct = float(limit_pct_raw)
    except (TypeError, ValueError):
        return stats
    if limit_pct <= 0:
        return stats

    try:
        min_abs = float(min_abs_raw) if min_abs_raw is not None else 0.0
    except (TypeError, ValueError):
        min_abs = 0.0
    min_abs = max(0.0, min_abs)

    reference: Optional[float] = None
    for candidate in reference_candidates:
        ref_val = safe_float(candidate)
        if ref_val is not None and ref_val > 0:
            reference = ref_val
            break

    if reference is None:
        return stats

    allowed_abs = max(min_abs, abs(reference) * limit_pct)
    if allowed_abs <= 0:
        return stats

    sanitized = dict(stats)
    adjustments: Dict[str, Dict[str, float]] = {}
    for key in ("min_price", "max_price", "mean_price"):
        price_val = safe_float(stats.get(key))
        if price_val is None:
            continue
        if abs(price_val - reference) > allowed_abs:
            sanitized[key] = float(reference)
            deviation_pct = None
            if reference != 0:
                deviation_pct = abs(price_val - reference) / abs(reference)
            adj_meta: Dict[str, float] = {"original": float(price_val)}
            if deviation_pct is not None and np.isfinite(deviation_pct):
                adj_meta["deviation_pct"] = float(deviation_pct)
            adjustments[key] = adj_meta

    if not adjustments:
        return stats

    sanitized["price_guard"] = {
        "reference": float(reference),
        "limit_pct": float(limit_pct),
        "min_abs": float(min_abs),
        "effective_limit": float(allowed_abs),
        "adjusted": sorted(adjustments.keys()),
        "original_values": adjustments,
    }
    return sanitized


_PRECISION_FLOW_CACHE: Dict[str, Any] = {"mtime": None, "data": {}}


def _clamp01(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _load_precision_flow_stats(
    path: Path = PRECISION_FLOW_STATS_FILE,
) -> Dict[str, Dict[str, float]]:
    """Return cached precision flow blocking ratios by asset."""

    global _PRECISION_FLOW_CACHE
    try:
        mtime = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        _PRECISION_FLOW_CACHE = {"mtime": None, "data": {}}
        return {}

    if _PRECISION_FLOW_CACHE.get("mtime") == mtime:
        cached = _PRECISION_FLOW_CACHE.get("data")
        return cached if isinstance(cached, dict) else {}

    try:
        frame = pd.read_csv(path)
    except Exception:
        _PRECISION_FLOW_CACHE = {"mtime": mtime, "data": {}}
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    total_signals_sum = 0.0
    flow_sum = 0.0
    for row in frame.to_dict(orient="records"):
        asset = str(row.get("asset", "")).strip().upper()
        if not asset:
            continue
        total = safe_float(row.get("total_signals")) or 0.0
        if total <= 0:
            total = safe_float(row.get("no_entry_signals")) or 0.0
        flow_hits = safe_float(row.get("precision_flow_alignment")) or 0.0
        ratio = _clamp01(flow_hits / total) if total > 0 else 0.0
        stats[asset] = {
            "ratio": ratio,
            "total": float(total),
            "flow_hits": float(flow_hits),
        }
        total_signals_sum += max(total, 0.0)
        flow_sum += max(flow_hits, 0.0)

    if total_signals_sum > 0:
        stats["__GLOBAL__"] = {
            "ratio": _clamp01(flow_sum / total_signals_sum),
            "total": float(total_signals_sum),
            "flow_hits": float(flow_sum),
        }

    _PRECISION_FLOW_CACHE = {"mtime": mtime, "data": stats}
    return stats


def get_precision_flow_rules(asset: str) -> Dict[str, float]:
    """Return adaptive thresholds for precision flow alignment."""

    stats = _load_precision_flow_stats()
    asset_key = str(asset or "").strip().upper()
    meta = stats.get(asset_key) or stats.get("__GLOBAL__") or {}
    ratio = float(meta.get("ratio") or 0.0)
    target = max(PRECISION_FLOW_TARGET_RATIO, 0.01)
    overshoot = max(0.0, ratio - target)
    overshoot_ratio = min(overshoot / target, 2.0)
    scale = 1.0 - 0.45 * overshoot_ratio
    scale = max(PRECISION_FLOW_SCALE_MIN, min(1.0, scale))

    imbalance_th = ORDER_FLOW_IMBALANCE_TH * scale
    pressure_th = ORDER_FLOW_PRESSURE_TH * scale

    imbalance_margin = 1.0 + (PRECISION_FLOW_IMBALANCE_MARGIN - 1.0) * scale
    imbalance_margin = max(PRECISION_FLOW_MARGIN_MIN, imbalance_margin)
    pressure_margin = 1.0 + (PRECISION_FLOW_PRESSURE_MARGIN - 1.0) * scale
    pressure_margin = max(PRECISION_FLOW_MARGIN_MIN, pressure_margin)

    strength_floor = PRECISION_FLOW_STRENGTH_BASE * (1.0 - 0.25 * overshoot_ratio)
    strength_floor = max(PRECISION_FLOW_STRENGTH_MIN, min(1.0, strength_floor))

    min_signals = 1
    if overshoot_ratio >= 0.8:
        min_signals = 0

    return {
        "imbalance_threshold": float(imbalance_th),
        "pressure_threshold": float(pressure_th),
        "imbalance_margin": float(imbalance_margin),
        "pressure_margin": float(pressure_margin),
        "min_signals": int(min_signals),
        "strength_floor": float(strength_floor),
        "block_ratio": float(ratio),
    }


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def parse_hhmm(value: str) -> Optional[int]:
    if not value or not isinstance(value, str):
        return None
    try:
        hour, minute = value.split(":", 1)
        return int(hour) * 60 + int(minute)
    except Exception:
        return None


def in_utc_range(now_utc: datetime, start: str, end: str) -> bool:
    start_min = parse_hhmm(start)
    end_min = parse_hhmm(end)
    if start_min is None or end_min is None:
        return False
    minute_now = now_utc.hour * 60 + now_utc.minute
    if start_min <= end_min:
        return start_min <= minute_now < end_min
    # Átívelés éjfélen
    return minute_now >= start_min or minute_now < end_min


def intervention_band(score: int, bands: Dict[str, List[int]]) -> str:
    try:
        items = [
            (name, int(bounds[0]), int(bounds[1]))
            for name, bounds in bands.items()
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2
        ]
    except Exception:
        items = []
    if items:
        items.sort(key=lambda x: x[1])
        for name, low, high in items:
            if low <= score <= high:
                return name
    if score >= 80:
        return "EXTREME" if "EXTREME" in bands else "IMMINENT"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "ELEVATED"
    return "LOW"


def _sentiment_normalizer_rollback_enabled() -> bool:
    raw = str(os.getenv("SENTIMENT_NORMALIZER_ROLLBACK", "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _normalize_btcusd_sentiment(signal: SentimentSignal) -> float:
    if _sentiment_normalizer_rollback_enabled():
        bias = (signal.bias or "").lower()
        direction = 1.0
        if any(flag in bias for flag in ("bear", "risk_off", "usd_bullish")):
            direction = -1.0
        elif any(flag in bias for flag in ("btc_bullish", "risk_on", "bull")):
            direction = 1.0
        return signal.score * direction

    bias: Optional[str] = None
    score: Optional[float] = None

    if isinstance(signal, SentimentSignal):
        bias = signal.bias
        score = signal.score
    elif hasattr(signal, "bias") and hasattr(signal, "score"):
        bias = getattr(signal, "bias", None)
        score = getattr(signal, "score", None)
    elif isinstance(signal, (list, tuple)):
        if len(signal) >= 2:
            first, second = signal[0], signal[1]
            if isinstance(first, (int, float)) and isinstance(second, str):
                score = first
                bias = second
            elif isinstance(first, str) and isinstance(second, (int, float)):
                bias = first
                score = second
            else:
                try:
                    score = float(first)
                except Exception:
                    score = None
                bias = str(second) if second is not None else None

    if score is None or bias is None:
        LOGGER.warning(
            "Érvénytelen BTCUSD sentiment típus: %s", type(signal), extra={"sentiment_type_error": True}
        )
        bias = "neutral"
        score = 0.0

    bias_value = (bias or "").lower()
    direction = 0.0 if bias_value == "neutral" else 1.0
    if any(flag in bias_value for flag in ("bear", "risk_off", "usd_bullish")):
        direction = -1.0
    elif any(flag in bias_value for flag in ("btc_bullish", "risk_on", "bull")):
        direction = 1.0

    try:
        score_val = float(score)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Sentiment score nem konvertálható: %r", score, extra={"sentiment_type_error": True}
        )
        score_val = 0.0

    # When the bias is bullish, treat the score as an absolute magnitude to avoid
    # double-negating tuples where the score is already negative. Bearish biases
    # keep the original sign behaviour so that negative scores flip to positive
    # and positive scores flip to negative, matching legacy expectations.
    if direction > 0 and score_val < 0:
        score_val = abs(score_val)

    return score_val * direction


def _sentiment_points_btcusd(
    signal: Optional[SentimentSignal], cfg: Dict[str, Any]
) -> Tuple[Optional[float], float]:
    if signal is None:
        return None, 0.0
    normalized = _normalize_btcusd_sentiment(signal)
    threshold = float(cfg.get("sentiment_min_abs", 0.2) or 0.0)
    if abs(normalized) < threshold:
        return normalized, 0.0
    weight_pos = float(cfg.get("sentiment_weight_positive", 7.0) or 0.0)
    weight_neg = float(cfg.get("sentiment_weight_negative", 5.0) or 0.0)
    weight = weight_pos if normalized >= 0 else weight_neg
    severity = signal.effective_severity
    return normalized, normalized * weight * severity


def _percent_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / previous * 100.0


def compute_btcusd_intraday_watch(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    now_utc: datetime,
    config: Dict[str, Any],
    news_flag: int = 0,
    sentiment_signal: Optional[SentimentSignal] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    cfg = dict(config.get("crypto_watch", config))
    metrics: Dict[str, Any] = {
        "price": None,
        "roc_30m_pct": None,
        "roc_5m_pct": None,
        "atr5_usd": None,
        "atr5_ratio": None,
        "range_position_pct": None,
        "session_points": 0.0,
        "comms_points": 0.0,
        "comms_from_flag": 0.0,
        "comms_from_sentiment": 0.0,
        "sentiment_score": None,
        "components": {},
    }

    if df_1m.empty or df_5m.empty:
        return 0, "LOW", metrics

    price = safe_float(df_1m["close"].iloc[-1])
    if price is None or price <= 0:
        return 0, "LOW", metrics

    roc_thresholds = cfg.get("roc_thresholds_pct_30m", [1.2, 2.0, 3.2])
    roc_30_pct = 0.0
    speed_score = 0.0
    if len(df_1m) >= 31:
        price_30 = safe_float(df_1m["close"].iloc[-30])
        roc_30_pct = abs(_percent_change(price, price_30) or 0.0)
        if roc_thresholds:
            tiers = list(roc_thresholds) + [roc_thresholds[-1] + 1]
            if roc_30_pct >= tiers[2]:
                speed_score += 24
            elif roc_30_pct >= tiers[1]:
                speed_score += 16
            elif roc_30_pct >= tiers[0]:
                speed_score += 9

    roc_5_pct = 0.0
    if len(df_5m) >= 2:
        price_5 = safe_float(df_5m["close"].iloc[-2])
        roc_5_pct = abs(_percent_change(price, price_5) or 0.0)
        if roc_5_pct >= max(roc_thresholds[0] / 2 if roc_thresholds else 0.6, 0.4):
            speed_score += 6
    speed_score = min(32.0, speed_score)

    atr_period = get_atr_period("BTCUSD")
    atr5_series = atr(df_5m, atr_period)
    atr5_val = safe_float(atr5_series.iloc[-1]) if not atr5_series.empty else None
    atr5_med = None
    if not atr5_series.empty:
        tail = atr5_series.tail(48).to_numpy(dtype=float)
        if tail.size:
            atr5_med = float(np.nanmedian(tail))
    atr_ratio = None
    vol_score = 0.0
    atr_spike_ratio = float(cfg.get("atr_spike_ratio", 1.7) or 1.7)
    if atr5_val and atr5_med:
        atr_ratio = atr5_val / atr5_med if atr5_med else None
        if atr_ratio and atr_ratio >= atr_spike_ratio:
            vol_score += 18
        elif atr_ratio and atr_ratio >= max(atr_spike_ratio - 0.4, 1.2):
            vol_score += 10

    range_score = 0.0
    range_position = None
    range_breakout_pct = float(cfg.get("range_breakout_pct", 1.0) or 1.0)
    if len(df_5m) >= 288:
        window = df_5m.tail(288)
        high = safe_float(window["high"].max())
        low = safe_float(window["low"].min())
        if high and low and high > low:
            range_position = (price - low) / (high - low) * 100.0
            top_pct = (high - price) / price * 100.0
            bottom_pct = (price - low) / price * 100.0
            if top_pct <= range_breakout_pct:
                range_score += 10
            if bottom_pct <= range_breakout_pct:
                range_score += 10

    var_ratio = None
    if len(df_1m) >= 120 and len(df_5m) >= 24:
        short_var = np.var(np.diff(df_1m["close"].iloc[-120:].to_numpy(dtype=float)))
        long_var = np.var(np.diff(df_5m["close"].iloc[-24:].to_numpy(dtype=float)))
        if np.isfinite(short_var) and np.isfinite(long_var) and long_var > 0:
            var_ratio = float(short_var / long_var)
            if var_ratio >= 1.8:
                vol_score += 8
            elif var_ratio >= 1.4:
                vol_score += 4
    vol_score = min(24.0, vol_score)

    base_comms = max(0.0, min(10.0, float(news_flag or 0)))
    sentiment_norm, sentiment_points = _sentiment_points_btcusd(sentiment_signal, cfg)
    applied_sentiment = 0.0
    if sentiment_points:
        applied_sentiment = max(-base_comms, min(sentiment_points, 10.0 - base_comms))
    comms = max(0.0, min(12.0, base_comms + applied_sentiment))

    total_score = speed_score + vol_score + range_score + comms
    crypto_score = int(round(max(0.0, min(100.0, total_score))))
    band = intervention_band(crypto_score, config.get("irs_bands", {}))

    metrics.update(
        {
            "price": price,
            "roc_30m_pct": roc_30_pct,
            "roc_5m_pct": roc_5_pct,
            "atr5_usd": atr5_val,
            "atr5_ratio": atr_ratio,
            "range_position_pct": range_position,
            "variance_ratio": var_ratio,
            "session_points": 0.0,
            "comms_points": comms,
            "comms_from_flag": base_comms,
            "comms_from_sentiment": applied_sentiment,
            "sentiment_score": sentiment_norm,
            "components": {
                "speed": speed_score,
                "volatility": vol_score,
                "range": range_score,
                "comms": comms,
            },
        }
    )

    return crypto_score, band, metrics


def load_intervention_config(outdir: str) -> Dict[str, Any]:
    base_cfg = deepcopy(INTERVENTION_WATCH_DEFAULT.get("BTCUSD", {}))
    path = os.path.join(outdir, INTERVENTION_CONFIG_FILENAME)
    override = load_json(path)
    if isinstance(override, dict):
        candidate = override
        if "BTCUSD" in candidate and isinstance(candidate["BTCUSD"], dict):
            candidate = candidate["BTCUSD"]
        if "crypto_watch" in candidate and isinstance(candidate["crypto_watch"], dict):
            candidate = candidate["crypto_watch"]
        if isinstance(candidate, dict):
            base_cfg = deep_update(deepcopy(base_cfg), candidate)
    return base_cfg


def load_intervention_news_flag(outdir: str, config: Dict[str, Any]) -> int:
    cfg = config.get("crypto_watch", config)
    base_flag = cfg.get("news_flag", 0)
    path = os.path.join(outdir, INTERVENTION_NEWS_FILENAME)
    data = load_json(path)
    if isinstance(data, dict):
        candidate = data.get("news_flag")
        if candidate is not None:
            base_flag = candidate
    try:
        flag = int(base_flag)
    except (TypeError, ValueError):
        flag = 0
    return max(0, min(10, flag))


def load_intervention_state(outdir: str) -> Dict[str, Any]:
    path = os.path.join(outdir, INTERVENTION_STATE_FILENAME)
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def update_intervention_state(outdir: str, band: str, now_utc: datetime) -> str:
    state = load_intervention_state(outdir)
    since = state.get("since_utc") if state.get("band") == band else None
    if not since:
        since = to_utc_iso(now_utc)
    save_json(
        os.path.join(outdir, INTERVENTION_STATE_FILENAME),
        {"band": band, "since_utc": since},
    )
    return since


def build_intervention_reasons(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    roc = safe_float(metrics.get("roc_30m_pct"))
    if roc is not None and roc >= 3.0:
        reasons.append(f"30m change {roc:.1f}%")
    elif roc is not None and roc >= 1.5:
        reasons.append(f"30m impulse {roc:.1f}%")

    roc5 = safe_float(metrics.get("roc_5m_pct"))
    if roc5 is not None and roc5 >= 1.0:
        reasons.append(f"5m burst {roc5:.1f}%")

    atr_ratio = safe_float(metrics.get("atr5_ratio"))
    if atr_ratio and atr_ratio >= 1.7:
        reasons.append(f"ATR spike {atr_ratio:.1f}×")
    elif atr_ratio and atr_ratio >= 1.3:
        reasons.append(f"ATR elevated {atr_ratio:.1f}×")

    variance_ratio = safe_float(metrics.get("variance_ratio"))
    if variance_ratio and variance_ratio >= 1.8:
        reasons.append(f"Variance-ratio {variance_ratio:.2f}")

    range_position = safe_float(metrics.get("range_position_pct"))
    if range_position is not None:
        if range_position >= 95:
            reasons.append("Pressing daily high")
        elif range_position <= 5:
            reasons.append("Testing daily low")

    comms_flag = safe_float(metrics.get("comms_from_flag"))
    if comms_flag:
        reasons.append(f"Manual news flag +{int(round(comms_flag))}")
    sentiment_pts = safe_float(metrics.get("comms_from_sentiment"))
    if sentiment_pts:
        sign = "+" if sentiment_pts > 0 else ""
        reasons.append(f"Sentiment {sign}{sentiment_pts:.1f}")

    return reasons


def build_intervention_policy(band: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    policy: List[str] = [
        "increase_spread_buffer_when_volatility>=1.5×",
        "discord_alert_cooldown:10m",
    ]
    actions = config.get("crypto_watch", {}).get("actions", {})

    if band in {"HIGH", "EXTREME"}:
        high_actions = actions.get("HIGH", {})
        if high_actions.get("tighten_sl"):
            ts = high_actions["tighten_sl"]
            policy.append(
                "tighten_sl>=max({:.1f}×ATR5m,{:.2f}%)".format(
                    float(ts.get("atr5_mult", 0.4)), float(ts.get("min_pct", 0.35))
                )
            )
        if high_actions.get("require_retest"):
            policy.append("require_retest_before_breakout_entries")
        p_add = high_actions.get("p_score_add", 0)
        if p_add:
            policy.append(f"p_score_long_add:+{int(p_add)}")

    if band == "EXTREME":
        extreme_actions = actions.get("EXTREME", {})
        if extreme_actions.get("pause_breakouts"):
            policy.append("pause_breakout_entries")
        limit_mult = extreme_actions.get("limit_order_atr_buffer", 0.6)
        policy.append(f"limit_orders_buffer≥{float(limit_mult):.1f}×ATR5m")
        policy.append("scale_out_on_1.5×ATR adverse move")

    return policy


def save_intervention_asset_summary(outdir: str, risk_summary: Dict[str, Any]) -> None:
    payload = {
        "asset": "BTCUSD",
        "generated_utc": nowiso(),
        "crypto_risk": risk_summary,
    }
    save_json(os.path.join(outdir, INTERVENTION_SUMMARY_FILENAME), payload)


def diagnostics_payload(tf_meta: Dict[str, Dict[str, Any]],
                        source_files: Dict[str, Optional[str]],
                        latency_flags: List[str]) -> Dict[str, Any]:
    return {
        "timeframes": tf_meta,
        "source_files": source_files,
        "latency_flags": list(latency_flags),
        "refresh_tips": list(REFRESH_TIPS),
    }


def _latency_guard_status(
    asset: str,
    latency_seconds: Dict[str, Optional[int]],
    guard_config: Dict[str, Dict[str, Any]],
    *,
    profile: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    asset_key = str(asset or "").upper()
    cfg = guard_config.get(asset_key) or guard_config.get("DEFAULT")
    profile_limit = None
    if profile:
        profile_limit = LATENCY_GUARD_PROFILE_LIMIT_SECONDS.get(profile)
    if not isinstance(cfg, dict):
        cfg = {}
    limit_raw = cfg.get("latency_k1m_sec_max")
    if limit_raw is None and profile_limit is not None:
        limit_raw = profile_limit
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return None
    if profile_limit is not None and limit < int(profile_limit):
        limit = int(profile_limit)
    if limit <= 0:
        return None
    age_raw = latency_seconds.get("k1m")
    try:
        age_seconds = int(age_raw) if age_raw is not None else None
    except (TypeError, ValueError):
        age_seconds = None
    if age_seconds is None:
        return None
    if age_seconds > limit:
        return {
            "asset": asset_key,
            "feed": "k1m",
            "age_seconds": age_seconds,
            "limit_seconds": limit,
        }
    return None


def _log_latency_guard_recovery(
    asset: str,
    outdir: str,
    latency_guard_state: Optional[Dict[str, Any]],
    analysis_now: datetime,
    profile: Optional[str],
    entry_thresholds_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(latency_guard_state, dict) or not latency_guard_state.get("active"):
        return None
    ts_format = "%Y-%m-%d %H:%M:%S"
    recovered_utc = analysis_now.strftime(ts_format)
    recovered_cet = analysis_now.astimezone(LOCAL_TZ).strftime(ts_format)
    recovery_payload = {
        "asset": asset,
        "feed": latency_guard_state.get("feed", "k1m"),
        "recovered_utc": recovered_utc,
        "recovered_cet": recovered_cet,
        "previous_triggered_utc": latency_guard_state.get("triggered_utc"),
        "previous_triggered_cet": latency_guard_state.get("triggered_cet"),
        "limit_seconds": latency_guard_state.get("limit_seconds"),
        "profile": profile,
    }
    if isinstance(entry_thresholds_meta, dict):
        entry_thresholds_meta["latency_guard_recovery"] = recovery_payload
    LOGGER.info("Latency guard feloldva", extra=recovery_payload)
    new_state = dict(latency_guard_state)
    new_state["active"] = False
    new_state["recovered_utc"] = recovered_utc
    new_state["recovered_cet"] = recovered_cet
    save_latency_guard_state(outdir, new_state)
    return recovery_payload


def _normalize_blockers(blockers: Any) -> List[str]:
    if isinstance(blockers, list):
        return [str(item) for item in blockers if item]
    if isinstance(blockers, (set, tuple)):
        return [str(item) for item in blockers if item]
    if blockers:
        return [str(blockers)]
    return []


def _log_precision_gate_summary(
    asset: str,
    precision_gate_snapshot: Dict[str, Any],
    blockers_snapshot: Any,
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    blockers_list = _normalize_blockers(blockers_snapshot)
    logger.info(
        "Precision kapu összegzés",
        extra={
            "asset": asset,
            **precision_gate_snapshot,
            "precision_block_reason": ",".join(blockers_list),
        },
    )


def _emit_precision_gate_log(
    asset: str,
    gate_name: str,
    decision: bool,
    reason_code: str,
    *,
    order_flow_metrics: Dict[str, Any],
    tick_order_flow: Dict[str, Any],
    latency_seconds: Dict[str, Optional[int]],
    precision_plan: Optional[Dict[str, Any]] = None,
    logger: logging.Logger = LOGGER,
    timestamp: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    now_utc = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ts_format = "%Y-%m-%d %H:%M:%S"
    latency_payload = latency_seconds if isinstance(latency_seconds, dict) else {}
    order_flow = order_flow_metrics if isinstance(order_flow_metrics, dict) else {}
    tick_flow = tick_order_flow if isinstance(tick_order_flow, dict) else {}
    age_raw = latency_payload.get("k1m")
    try:
        age_seconds = int(age_raw) if age_raw is not None else None
    except (TypeError, ValueError):
        age_seconds = None
    window_minutes = tick_flow.get("window_minutes")
    if window_minutes is None:
        window_minutes = OFI_Z_LOOKBACK if OFI_Z_LOOKBACK > 0 else None
    ofi_present = False
    try:
        ofi_present = np.isfinite(float(order_flow.get("imbalance_z")))
    except Exception:
        ofi_present = False
    flow_strength = None
    trigger_state = None
    trigger_ready = None
    precision_score = None
    if precision_plan:
        flow_strength = precision_plan.get("order_flow_strength")
        trigger_state = precision_plan.get("trigger_state")
        trigger_ready = precision_plan.get("trigger_ready")
        precision_score = precision_plan.get("score")
    payload: Dict[str, Any] = {
        "asset": asset,
        "gate": gate_name,
        "decision": bool(decision),
        "reason_code": reason_code,
        "ofi_present": bool(ofi_present),
        "ofi_age_seconds": age_seconds,
        "ofi_window_minutes": window_minutes,
        "ofi_conf": flow_strength,
        "ofi_source": tick_flow.get("source")
        or order_flow.get("status"),
        "timestamp_utc": now_utc.strftime(ts_format),
        "timestamp_cet": now_utc.astimezone(LOCAL_TZ).strftime(ts_format),
        "timestamp_bud": now_utc.astimezone(LOCAL_TZ).strftime(ts_format),
    }
    if trigger_state is not None:
        payload["precision_trigger_state"] = trigger_state
    if trigger_ready is not None:
        payload["precision_trigger_ready"] = bool(trigger_ready)
    if precision_score is not None:
        payload["precision_score"] = precision_score

    if (
        str(asset or "").upper() in {"NVDA", "XAGUSD"}
        and gate_name == "precision_flow"
        and not bool(decision)
    ):
        missing_components: Dict[str, Any] = {}
        if isinstance(precision_plan, dict):
            blockers_raw = precision_plan.get("order_flow_blockers")
            blockers: List[str] = []
            if isinstance(blockers_raw, list):
                blockers = [str(item) for item in blockers_raw if item]
            elif blockers_raw:
                blockers = [str(blockers_raw)]
            if blockers:
                missing_components["blockers"] = blockers

            settings = precision_plan.get("order_flow_settings") or {}

            def _safe_int(value: Any) -> Optional[int]:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

            def _safe_float(value: Any) -> Optional[float]:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            required_signals: Optional[int] = None
            signals_required_raw = _safe_int(settings.get("min_signals"))
            if signals_required_raw is not None:
                required_signals = max(1, signals_required_raw)

            signals_actual = _safe_int(precision_plan.get("order_flow_signals"))
            if required_signals is not None:
                if signals_required_raw is not None and signals_required_raw <= 0:
                    missing_signals = (signals_actual or 0) <= 0
                else:
                    missing_signals = (
                        signals_actual is None or signals_actual < required_signals
                    )
                if missing_signals:
                    missing_components["signals"] = {
                        "actual": signals_actual,
                        "required": required_signals,
                    }

            strength_floor = _safe_float(settings.get("strength_floor"))
            strength_actual = _safe_float(precision_plan.get("order_flow_strength"))
            if (
                strength_floor is not None
                and (strength_actual is None or strength_actual < strength_floor)
            ):
                missing_components["strength"] = {
                    "actual": strength_actual,
                    "required": strength_floor,
                }

            status_raw = str(precision_plan.get("order_flow_status") or "").strip()
            if status_raw and status_raw.lower() not in {"ok", "ready", "available"}:
                missing_components["status"] = status_raw

            if precision_plan.get("order_flow_stalled"):
                missing_components["stalled"] = True

        if missing_components:
            payload["missing_precision_components"] = missing_components
    if extra:
        payload.update(extra)
    logger.info("Precision kapu állapot", extra=payload)


def _handle_spot_realtime_staleness(
    asset: str,
    rt_meta: Dict[str, Any],
    now_utc: datetime,
    notifier: Callable[..., Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(rt_meta, dict):
        return result
    age_raw = rt_meta.get("age_seconds")
    try:
        age_seconds = int(age_raw) if age_raw is not None else None
    except (TypeError, ValueError):
        age_seconds = None
    result["age_seconds"] = age_seconds
    if age_seconds is None:
        return result
    try:
        threshold = int(rt_meta.get("max_age_seconds") or 600)
    except (TypeError, ValueError):
        threshold = 600
    if age_seconds <= threshold:
        return result
    ts_format = "%Y-%m-%d %H:%M:%S"
    payload = {
        "asset": asset,
        "feed": "spot_realtime",
        "age_seconds": age_seconds,
        "threshold_seconds": threshold,
        "timestamp_utc": now_utc.strftime(ts_format),
        "timestamp_cet": now_utc.astimezone(LOCAL_TZ).strftime(ts_format),
    }
    try:
        notifier(
            asset,
            "spot_realtime",
            f"Realtime spot frissítés szükséges — {age_seconds} másodperces késés",
            metadata=payload,
        )
        k1m_payload = dict(payload, feed="k1m")
        notifier(
            asset,
            "k1m",
            "1m gyertya frissítés szükséges a realtime spot késése miatt",
            metadata=k1m_payload,
        )
        result["notified"] = True
        result["refresh_requested"] = True
        result.setdefault("retry_after_seconds", 60)
        logger.info(
            "Realtime spot frissítés jelzés elküldve",
            extra=dict(payload, action="refresh_request"),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        result["notified"] = False
        result["refresh_requested"] = False
        result["error"] = str(exc)
        logger.warning("Realtime spot frissítés jelzés hibára futott: %s", exc)
    return result


def should_enforce_stale_frame(asset: str,
                               frame: str,
                               session_meta: Optional[Dict[str, Any]]) -> bool:
    meta = session_meta or {}
    asset_key = (asset or "").upper()
    if not (
        meta.get("open")
        or meta.get("within_monitor_window")
        or meta.get("entry_open")
    ):
        return False
    if not meta.get("entry_open"):
        status = str(meta.get("status") or "").lower()
        if status in {"open_entry_limited", "closed_out_of_hours", "closed_weekend"}:
            return False
        if not meta.get("within_entry_window") and not meta.get("within_window"):
            return False
    return True


def classify_critical_staleness(
    asset: str,
    stale_timeframes: Dict[str, bool],
    latency_seconds: Dict[str, Optional[int]],
    session_meta: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, bool], List[str]]:
    """Return critical stale flags and human-readable reasons for gating."""

    critical_flags: Dict[str, bool] = {}
    reasons: List[str] = []
    for key, description in CRITICAL_STALE_FRAMES.items():
        enforce = should_enforce_stale_frame(asset, key, session_meta)
        is_critical = bool(stale_timeframes.get(key) and enforce)
        critical_flags[key] = is_critical
        if not is_critical:
            continue
        latency = latency_seconds.get(key)
        if latency is not None:
            minutes = latency // 60
            reasons.append(f"{description} — {minutes} perc késés")
        else:
            reasons.append(description)
    return critical_flags, reasons


def compute_data_integrity_status(
    spot_stale: bool,
    critical_flags: Dict[str, bool],
    latency_seconds: Dict[str, Optional[int]],
) -> Tuple[bool, bool, bool, Dict[str, str]]:
    """Derive core vs precision data health and human readable reasons."""

    reason_by_frame: Dict[str, str] = {}
    for key, description in CRITICAL_STALE_FRAMES.items():
        if not critical_flags.get(key):
            continue
        latency = latency_seconds.get(key)
        if latency is not None:
            minutes = latency // 60
            reason_by_frame[key] = f"{description} — {minutes} perc késés"
        else:
            reason_by_frame[key] = description

    core_data_ok = (not spot_stale) and not critical_flags.get("k5m", False)
    precision_data_ok = core_data_ok and not critical_flags.get("k1m", False)
    precision_disabled_due_to_data_gap = core_data_ok and not precision_data_ok
    return core_data_ok, precision_data_ok, precision_disabled_due_to_data_gap, reason_by_frame


# Probability stack helpers -------------------------------------------------


def ensure_probability_metadata(
    metadata: Optional[Dict[str, Any]],
    default_source: str = "sklearn",
) -> Dict[str, Any]:
    """Return a serialisable metadata payload with a guaranteed ``source`` field."""

    normalised: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        normalised = {key: value for key, value in metadata.items() if value is not None}

    if not normalised.get("source"):
        normalised["source"] = default_source

    return normalised


def _ml_feature_snapshot_present(asset: str, base_dir: Optional[Path] = None) -> bool:
    asset_root = Path(base_dir or PUBLIC_DIR) / str(asset or "").upper()
    feature_dir = asset_root / ML_FEATURE_SNAPSHOT_DIRNAME
    if not feature_dir.exists() or not feature_dir.is_dir():
        return False
    try:
        return any(feature_dir.iterdir())
    except OSError:
        return False


def _probability_stack_snapshot_path(
    asset: str, base_dir: Optional[Path] = None
) -> Path:
    root = Path(base_dir or PUBLIC_DIR)
    return (root / asset.upper() / PROB_STACK_SNAPSHOT_FILENAME).resolve()


def _probability_stack_export_path(asset: str, base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir or PUBLIC_DIR)
    return (root / asset.upper() / PROB_STACK_EXPORT_FILENAME).resolve()


def _load_probability_stack_snapshot(
    asset: str,
    *,
    base_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    stale_minutes: int = PROB_STACK_GAP_STALE_MINUTES,
) -> Optional[Dict[str, Any]]:
    path = _probability_stack_snapshot_path(asset, base_dir)
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    ts_raw = payload.get("retrieved_at_utc") or payload.get("timestamp")
    ts = _parse_utc_timestamp(ts_raw) if ts_raw else None
    if ts is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    clock = now or datetime.now(timezone.utc)
    age_minutes = (clock - ts).total_seconds() / 60.0
    if age_minutes > stale_minutes:
        return None
    payload["_snapshot_path"] = str(path)
    payload["_snapshot_age_minutes"] = age_minutes
    return payload


def _load_probability_stack_export(
    asset: str,
    *,
    base_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    stale_minutes: int = PROB_STACK_GAP_STALE_MINUTES,
) -> Optional[Dict[str, Any]]:
    path = _probability_stack_export_path(asset, base_dir)
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    ts_raw = payload.get("retrieved_at_utc") or payload.get("timestamp")
    ts = _parse_utc_timestamp(ts_raw) if ts_raw else None
    if ts is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    clock = now or datetime.now(timezone.utc)
    age_minutes = (clock - ts).total_seconds() / 60.0
    if age_minutes > stale_minutes:
        return None
    payload["_snapshot_path"] = str(path)
    payload["_snapshot_age_minutes"] = age_minutes
    payload.setdefault("status", payload.get("status") or "snapshot")
    return payload


def _store_probability_stack_snapshot(
    asset: str,
    metadata: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    path = _probability_stack_snapshot_path(asset, base_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata)
        ts_val = timestamp or datetime.now(timezone.utc)
        if not payload.get("retrieved_at_utc"):
            payload["retrieved_at_utc"] = ts_val.isoformat()
        save_json(str(path), payload)
        export_path = _probability_stack_export_path(asset, base_dir)
        save_json(str(export_path), payload)
    except Exception:
        LOGGER.debug("probability_stack_snapshot_store_failed", exc_info=True)


def _probability_stack_missing(metadata: Dict[str, Any]) -> bool:
    if not metadata:
        return True
    status = str(metadata.get("status") or "").lower()
    if status in {"halt", "halted", "halted_feed", "missing", "unavailable"}:
        return True
    return False


def _probability_stack_saveworthy(metadata: Dict[str, Any]) -> bool:
    if not metadata:
        return False
    status = str(metadata.get("status") or "").lower()
    if status in {"data_gap", "analysis_error", "unavailable", "halt", "halted"}:
        return False
    return True


def _apply_probability_stack_gap_guard(
    asset: str,
    metadata: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    base_dir: Optional[Path] = None,
    logger: logging.Logger = LOGGER,
) -> Dict[str, Any]:
    if str(asset or "").upper() != "BTCUSD":
        return metadata
    if os.getenv(PROB_STACK_GAP_ENV_DISABLE):
        return metadata
    clock = now or datetime.now(timezone.utc)
    feature_snapshot_ok = _ml_feature_snapshot_present(asset, base_dir=base_dir)
    if _probability_stack_missing(metadata):
        export_snapshot = _load_probability_stack_export(
            asset,
            base_dir=base_dir,
            now=clock,
            stale_minutes=PROB_STACK_GAP_STALE_MINUTES,
        )
        if export_snapshot:
            payload = dict(export_snapshot)
            payload.pop("_snapshot_path", None)
            payload.pop("_snapshot_age_minutes", None)
            metadata = ensure_probability_metadata(payload)
            metadata.setdefault("status", "stale_snapshot")
            metadata.setdefault("gap_fallback", True)
            metadata["snapshot_age_minutes"] = export_snapshot.get("_snapshot_age_minutes")
            metadata["snapshot_path"] = export_snapshot.get("_snapshot_path")
            logger.warning(
                "prob_stack_gap",
                extra={
                    "asset": asset,
                    "age_minutes": export_snapshot.get("_snapshot_age_minutes"),
                    "snapshot_path": export_snapshot.get("_snapshot_path"),
                },
            )
    if _probability_stack_missing(metadata):
        snapshot = _load_probability_stack_snapshot(
            asset,
            base_dir=base_dir,
            now=clock,
            stale_minutes=PROB_STACK_GAP_STALE_MINUTES,
        )
        if snapshot:
            payload = dict(snapshot)
            payload.pop("_snapshot_path", None)
            payload.pop("_snapshot_age_minutes", None)
            metadata = ensure_probability_metadata(payload)
            metadata.setdefault("status", "stale_snapshot")
            metadata.setdefault("gap_fallback", True)
            metadata["snapshot_age_minutes"] = snapshot.get("_snapshot_age_minutes")
            metadata["snapshot_path"] = snapshot.get("_snapshot_path")
            logger.warning(
                "prob_stack_gap",
                extra={
                    "asset": asset,
                    "age_minutes": snapshot.get("_snapshot_age_minutes"),
                    "snapshot_path": snapshot.get("_snapshot_path"),
                },
            )
    if _probability_stack_missing(metadata) and not feature_snapshot_ok:
        metadata = ensure_probability_metadata(metadata)
        metadata.setdefault("source", "fallback")
        metadata.setdefault("status", "fallback")
        metadata.setdefault("fallback", {})
        metadata.setdefault("unavailable_reason", "feature_snapshot_missing")
        if isinstance(metadata.get("fallback"), dict):
            metadata["fallback"].setdefault("reason", "feature_snapshot_missing")
            metadata["fallback"].setdefault("action", "no_entry")
            metadata["fallback"].setdefault("detail", "BTCUSD feature snapshot missing")

    if _probability_stack_saveworthy(metadata):
        _store_probability_stack_snapshot(asset, metadata, base_dir=base_dir, timestamp=clock)
    return metadata


def build_data_gap_signal(
    asset: str,
    spot_price: Any,
    spot_utc: str,
    spot_retrieved: str,
    leverage: float,
    reasons: List[str],
    display_spot: Optional[float],
    diagnostics: Dict[str, Any],
    session_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session_payload: Dict[str, Any] = {
        "open": None,
        "within_window": None,
        "weekday_ok": None,
        "status": "unavailable",
        "status_note": "Hiányzó adat",
    }
    if session_meta:
        session_payload = dict(session_meta)

    gates_mode = "data_gap"
    required: List[str] = ["data_integrity"]
    missing: List[str] = ["data_integrity"]
    signal = "no entry"
    reasons_payload = list(reasons)

    if session_meta:
        open_now = bool(session_meta.get("open"))
        monitor_ok = bool(session_meta.get("within_monitor_window"))
        entry_open = bool(session_meta.get("entry_open"))
        if not open_now and not monitor_ok and not entry_open:
            gates_mode = "session_closed"
            required = ["session"]
            missing = ["session"]
            signal = "market closed"
            status_note = session_meta.get("status_note")
            if status_note and status_note not in reasons_payload:
                reasons_payload.insert(0, status_note)

    detail_text = "; ".join(
        str(reason) for reason in reasons_payload if isinstance(reason, str)
    )
    probability_stack = ensure_probability_metadata(
        {
            "source": "sklearn",
            "status": "data_gap",
            "unavailable_reason": "data_gap",
        }
    )
    if detail_text:
        probability_stack["detail"] = detail_text
    placeholder_meta = ML_PROBABILITY_PLACEHOLDER_INFO.get(asset.upper())
    if isinstance(placeholder_meta, dict) and placeholder_meta:
        probability_stack.setdefault("placeholder_model", {})
        probability_stack["placeholder_model"].update(
            {
                key: value
                for key, value in placeholder_meta.items()
                if key not in {"asset"} and value is not None
            }
        )

    return {
        "asset": asset,
        "ok": False,
        "retrieved_at_utc": nowiso(),
        "source": "Twelve Data (lokális JSON)",
        "spot": {
            "price": display_spot if display_spot is not None else spot_price,
            "utc": spot_utc,
            "retrieved_at_utc": spot_retrieved,
        },
        "signal": signal,
        "probability": 0,
        "probability_model": None,
        "probability_model_raw": None,
        "probability_calibrated": None,
        "probability_threshold": None,
        "probability_model_source": "sklearn",
        "probability_stack": probability_stack,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "rr": None,
        "leverage": leverage,
        "gates": {
            "mode": gates_mode,
            "required": required,
            "missing": missing,
        },
        "session_info": session_payload,
        "diagnostics": diagnostics,
        "reasons": reasons_payload,
    }


def build_analysis_error_signal(asset: str, error: Exception) -> Dict[str, Any]:
    """Return a fallback signal payload when ``analyze`` raises an exception."""

    outdir = os.path.join(PUBLIC_DIR, asset)
    spot = load_json(os.path.join(outdir, "spot.json")) or {}
    if not isinstance(spot, dict):
        spot = {}

    raw_price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
    display_price = safe_float(raw_price)
    spot_utc = str(spot.get("utc") or spot.get("timestamp") or "-")
    retrieved = str(spot.get("retrieved_at_utc") or spot.get("retrieved") or nowiso())
    message = f"analysis error: {error}".strip()

    probability_stack = ensure_probability_metadata(
        {
            "source": "sklearn",
            "status": "analysis_error",
            "unavailable_reason": "analysis_error",
            "detail": message[:280],
        }
    )

    diagnostics = {
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        }
    }

    return {
        "asset": asset,
        "ok": False,
        "retrieved_at_utc": nowiso(),
        "source": "analysis.py",
        "spot": {
            "price": display_price,
            "utc": spot_utc,
            "retrieved_at_utc": retrieved,
        },
        "signal": "error",
        "probability": 0,
        "probability_model": None,
        "probability_model_raw": None,
        "probability_calibrated": None,
        "probability_threshold": None,
        "probability_model_source": "sklearn",
        "probability_stack": probability_stack,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "rr": None,
        "leverage": LEVERAGE.get(asset, 1.0),
        "gates": {
            "mode": "analysis_error",
            "required": [],
            "missing": ["analysis"],
        },
        "diagnostics": diagnostics,
        "reasons": [message[:140]] if message else ["analysis error"],
    }

def save_json(path: str, obj: Any) -> None:
    target_path = Path(path)
    os.makedirs(target_path.parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=target_path.parent
    ) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, target_path)


def build_status_snapshot(summary: Dict[str, Any], public_dir: Path) -> Dict[str, Any]:
    """Persist a simplified ``status.json`` derived from the analysis summary.

    Struktúra (kimeneti mezők):
    - ok: bool
    - status: "ok" vagy "error"
    - generated_utc: ISO 8601 UTC
    - assets: {ASSET: {ok, signal, (latency_seconds), (expected_latency_seconds), (notes)}}
    - notes: opcionális reset/hiba üzenetek listája

    Tudatosan nem írunk fel nem használt mezőket (pl. ``td_base``), hogy a
    monitoring JSON kompakt és zajmentes maradjon.
    """

    assets = summary.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("analysis summary contains no assets")

    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    summary_ok = summary.get("ok") if isinstance(summary.get("ok"), bool) else None
    asset_health: list[bool] = []
    status_assets: Dict[str, Dict[str, Any]] = {}

    for asset, payload in assets.items():
        if not isinstance(payload, dict):
            continue

        asset_ok = payload.get("ok")
        if isinstance(asset_ok, bool):
            asset_health.append(asset_ok)

        probability_stack = payload.get("probability_stack")
        latency_seconds = None
        expected_latency = None
        if isinstance(probability_stack, dict):
            latency_seconds = probability_stack.get("latest_latency_seconds") or probability_stack.get(
                "latency_seconds"
            )
            expected_latency = probability_stack.get("expected_latency_seconds")

        asset_notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []

        status_assets[asset] = {
            "ok": asset_ok is not False,
            "signal": payload.get("signal") or payload.get("decision") or payload.get("state") or "no entry",
        }

        if latency_seconds is not None:
            status_assets[asset]["latency_seconds"] = latency_seconds
        if expected_latency is not None:
            status_assets[asset]["expected_latency_seconds"] = expected_latency
        if asset_notes:
            status_assets[asset]["notes"] = asset_notes

    if not status_assets:
        raise ValueError("analysis summary assets missing")

    if summary_ok is None and asset_health:
        summary_ok = all(asset_health)

    overall_ok = bool(summary_ok)

    generated_utc = normalize_generated_utc(
        summary.get("generated_utc"), field="summary.generated_utc"
    )
    normalized_notes = []
    for idx, note in enumerate(notes):
        if isinstance(note, dict) and "reset_utc" in note:
            note = dict(note)
            note["reset_utc"] = normalize_generated_utc(
                note.get("reset_utc"), field=f"notes[{idx}].reset_utc"
            )
        normalized_notes.append(note)

    status_payload = {
        "ok": overall_ok,
        "status": "ok" if overall_ok else "error",
        "generated_utc": generated_utc,
        "assets": status_assets,
        "notes": normalized_notes,
    }

    save_json(Path(public_dir) / "status.json", status_payload)
    return status_payload


def detect_analysis_revision() -> Optional[Dict[str, Any]]:
    """Return git metadata for the analysis build, if available."""

    script_dir = Path(__file__).resolve().parent
    try:
        repo_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=script_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    if not repo_root:
        return None

    metadata: Dict[str, Any] = {}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    if not commit:
        return None

    metadata["commit"] = commit

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        branch = ""
    if branch and branch != "HEAD":
        metadata["branch"] = branch

    try:
        describe = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        describe = ""
    if describe and describe != commit:
        metadata["describe"] = describe

    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .splitlines()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        status = []

    if os.getenv("CI"):
        status = []

    relevant_changes = []
    for line in status:
        entry = line.strip()
        if not entry:
            continue
        try:
            _, path = entry.split(None, 1)
        except ValueError:
            path = entry
        if path.startswith("public/"):
            continue
        relevant_changes.append(entry)

    if relevant_changes and not os.getenv("CI"):
        metadata["dirty"] = True
    elif os.getenv("CI"):
        metadata["dirty"] = False

    return metadata

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_signal_stability_config() -> Dict[str, Any]:
    config = settings.load_config().get("signal_stability") or {}
    return config if isinstance(config, dict) else {}


def _resolve_asset_value(config_map: Any, asset: str, default: int) -> int:
    if isinstance(config_map, dict):
        try:
            return int(config_map.get(asset, config_map.get("default", default)))
        except (TypeError, ValueError):
            return int(default)
    return int(default)


@lru_cache(maxsize=4)
def _load_manual_positions_from_file(
    positions_path: str, treat_missing_file_as_flat: bool
) -> Dict[str, Any]:
    return position_tracker.load_positions(positions_path, treat_missing_file_as_flat)


def _manual_position_state(
    asset: str,
    stability_config: Dict[str, Any],
    now_dt: datetime,
    manual_positions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tracking_cfg = stability_config.get("manual_position_tracking") or {}
    if manual_positions is None:
        positions_path = tracking_cfg.get("positions_file") or "public/_manual_positions.json"
        treat_missing = bool(tracking_cfg.get("treat_missing_file_as_flat", False))
        manual_positions = _load_manual_positions_from_file(
            positions_path, treat_missing
        )
    manual_positions = manual_positions if isinstance(manual_positions, dict) else {}
    return position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt)


def _extract_tracked_levels(
    asset: str,
    manual_state: Dict[str, Any],
    manual_positions: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return normalized levels for the manually tracked position (if any)."""

    source = manual_state.get("position") if isinstance(manual_state, dict) else None
    if not isinstance(source, dict) and isinstance(manual_positions, dict):
        candidate = manual_positions.get(asset)
        source = candidate if isinstance(candidate, dict) else None

    if not isinstance(source, dict):
        return {}

    return {
        key: source.get(key)
        for key in ("entry", "sl", "tp2", "opened_at_utc", "side")
        if source.get(key) is not None
    }


def _format_manual_position_note(
    asset: str, manual_state: Dict[str, Any], tracked_levels: Dict[str, Any]
) -> Optional[str]:
    if not manual_state.get("has_position"):
        return None

    side = (manual_state.get("side") or tracked_levels.get("side") or "").lower()
    side_txt = "long" if side in {"buy", "long"} else "short" if side in {"sell", "short"} else "open"

    opened_at = tracked_levels.get("opened_at_utc") or manual_state.get("opened_at_utc")
    entry = tracked_levels.get("entry") or manual_state.get("entry")
    sl = tracked_levels.get("sl") or manual_state.get("sl")
    tp2 = tracked_levels.get("tp2") or manual_state.get("tp2")

    details: List[str] = []
    if opened_at:
        details.append(f"opened_at: {opened_at}")
    if entry is not None:
        details.append(f"Entry: {entry}")
    if sl is not None:
        details.append(f"SL: {sl}")
    if tp2 is not None:
        details.append(f"TP2: {tp2}")

    detail_suffix = " (" + ", ".join(details) + ")" if details else ""
    return f"Pozíciómenedzsment: aktív {side_txt} pozíció{detail_suffix}"


def _extract_spot_price(payload: Dict[str, Any]) -> Optional[float]:
    """Best-effort spot price lookup from the analysis payload."""

    for key in ("spot_price", "current_price", "price"):
        try:
            value = payload.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    spot_block = payload.get("spot") if isinstance(payload, dict) else None
    if isinstance(spot_block, dict):
        for key in ("price", "price_usd"):
            try:
                value = spot_block.get(key)
            except Exception:
                value = None
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _resolve_post_exit_cooldown_minutes(
    config: Dict[str, Any], asset: str, default: int = 20
) -> int:
    cooldown_map = (config.get("post_exit_cooldown_minutes") or {})
    return _resolve_asset_value(cooldown_map, asset, default)


def _resolve_setup_grade(signal_data: Dict[str, Any], decision: str) -> Optional[str]:
    if not isinstance(signal_data, dict):
        return None

    raw_grade = signal_data.get("setup_grade")
    if isinstance(raw_grade, str) and raw_grade.strip().upper() in {"A", "B", "C", "X", "-"}:
        return raw_grade.strip().upper()

    classification = signal_data.get("setup_classification")
    if isinstance(classification, str):
        text = classification.strip().upper()
        if text.startswith("A SETUP"):
            return "A"
        if text.startswith("B SETUP"):
            return "B"
        if text.startswith("C SETUP"):
            return "C"

    try:
        p_score = float(signal_data.get("probability_raw", 0) or 0)
    except Exception:
        p_score = 0.0

    gates_for_setup = signal_data.get("gates") or {}
    missing = gates_for_setup.get("missing", []) if isinstance(gates_for_setup, dict) else []
    is_active_signal = (decision or "").lower() in {"buy", "sell"}
    if p_score >= 80 and is_active_signal:
        return "A"

    soft_blockers = {"atr", "bias", "regime", "choppy"}
    is_soft_blocked = bool(missing) and all(m in soft_blockers for m in missing)
    if p_score >= 30:
        if is_active_signal or is_soft_blocked:
            return "B"

    if p_score >= 25:
        return "C"
    return None


def _extract_trade_levels(signal_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(signal_data, dict):
        return None, None, None

    trade_block = signal_data.get("trade") or {}
    levels_block = signal_data.get("levels") or {}

    entry = signal_data.get("entry")
    if entry is None and isinstance(trade_block, dict):
        entry = trade_block.get("entry")
    if entry is None and isinstance(levels_block, dict):
        entry = levels_block.get("entry")

    sl = signal_data.get("sl")
    if sl is None and isinstance(trade_block, dict):
        sl = trade_block.get("sl")
    if sl is None and isinstance(levels_block, dict):
        sl = levels_block.get("sl")

    tp2 = signal_data.get("tp2")
    if tp2 is None and isinstance(trade_block, dict):
        tp2 = trade_block.get("tp2")
    if tp2 is None and isinstance(levels_block, dict):
        tp2 = levels_block.get("tp2")

    try:
        entry = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        entry = None
    try:
        sl = float(sl) if sl is not None else None
    except (TypeError, ValueError):
        sl = None
    try:
        tp2 = float(tp2) if tp2 is not None else None
    except (TypeError, ValueError):
        tp2 = None

    return entry, sl, tp2


def _load_signal_state(path: Path, history_window: int) -> Dict[str, Any]:
    state = load_json(str(path)) or {}
    if not isinstance(state, dict):
        state = {}
    entry_history = [
        item
        for item in state.get("entry_direction_history", [])
        if item in {"buy", "sell"}
    ]
    state.setdefault("last_notified_intent", None)
    state.setdefault("last_notified_side", None)
    state.setdefault("last_notified_at_utc", None)
    state.setdefault("hard_exit_cooldown_until_utc", None)
    state.setdefault("hard_exit_side", None)
    state["entry_direction_history"] = entry_history[-history_window:]
    return state


def _save_signal_state(path: Path, state: Dict[str, Any]) -> None:
    save_json(str(path), state)


def _update_entry_history(state: Dict[str, Any], direction: Optional[str], window: int) -> None:
    if direction not in {"buy", "sell"}:
        return
    history = state.get("entry_direction_history") or []
    history.append(direction)
    state["entry_direction_history"] = history[-window:]


def _trailing_streak(entries: Sequence[str], target: Optional[str]) -> int:
    if target not in {"buy", "sell"}:
        return 0
    streak = 0
    for item in reversed(list(entries)):
        if item == target:
            streak += 1
        else:
            break
    return streak


def apply_signal_stability_layer(
    asset: str,
    payload: Dict[str, Any],
    *,
    decision: str,
    action_plan: Optional[Dict[str, Any]],
    exit_signal: Optional[Dict[str, Any]],
    gates_missing: Sequence[str],
    analysis_timestamp: str,
    outdir: Path,
    stability_config: Optional[Dict[str, Any]] = None,
    manual_positions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = stability_config or _resolve_signal_stability_config()
    stability_enabled = bool(config.get("enabled", False))
    now_dt = parse_utc_timestamp(analysis_timestamp) or datetime.now(timezone.utc)
    tracking_cfg = config.get("manual_position_tracking") or {}
    positions_path = tracking_cfg.get("positions_file") or "public/_manual_positions.json"
    treat_missing = bool(tracking_cfg.get("treat_missing_file_as_flat", False))
    if manual_positions is None:
        manual_positions = _load_manual_positions_from_file(positions_path, treat_missing)
    manual_positions = manual_positions if isinstance(manual_positions, dict) else {}

    manual_state = position_tracker.compute_state(
        asset, tracking_cfg, manual_positions, now_dt
    )
    cooldown_minutes = _resolve_post_exit_cooldown_minutes(
        tracking_cfg, asset, default=20
    )

    if manual_state.get("tracking_enabled") and manual_state.get("has_position"):
        spot_price = _extract_spot_price(payload)
        changed, reason, manual_positions = position_tracker.check_close_by_levels(
            asset,
            manual_positions,
            spot_price,
            now_dt,
            cooldown_minutes,
        )
        if changed:
            position_tracker.save_positions_atomic(positions_path, manual_positions)
            _load_manual_positions_from_file.cache_clear()
            manual_state = position_tracker.compute_state(
                asset, tracking_cfg, manual_positions, now_dt
            )
            LOGGER.debug(
                "CLOSE state transition %s reason=%s cooldown_until=%s",
                asset,
                reason,
                manual_state.get("cooldown_until_utc"),
            )

    tracked_levels = _extract_tracked_levels(asset, manual_state, manual_positions)
    direction_map = {"buy": "buy", "sell": "sell"}
    exit_direction_map = {"long": "buy", "short": "sell"}
    entry_side = direction_map.get(str(decision or "").lower())
    exit_side = exit_direction_map.get(
        str((exit_signal or {}).get("direction") or "").lower()
    )
    setup_grade = _resolve_setup_grade(payload, decision)

    intent = "standby"
    actionable = False
    if exit_signal and exit_signal.get("state") == "hard_exit":
        intent = "hard_exit"
        actionable = manual_state["has_position"] or not manual_state["tracking_enabled"]
    elif action_plan and action_plan.get("status") == "manage_position":
        intent = "manage_position"
        actionable = manual_state["has_position"] or not manual_state["tracking_enabled"]
    elif entry_side in {"buy", "sell"} and not list(gates_missing):
        intent = "entry"
        actionable = manual_state["is_flat"] or not manual_state["tracking_enabled"]

    notify: Dict[str, Any] = {
        "should_notify": actionable,
        "reason": None,
        "cooldown_until_utc": None,
    }

    if not actionable:
        if intent in {"manage_position", "hard_exit"} and manual_state["tracking_enabled"]:
            notify["reason"] = "no_open_position_tracked"
        elif intent == "entry" and manual_state["tracking_enabled"] and manual_state["has_position"]:
            notify["reason"] = "position_already_open"
        elif intent == "entry" and manual_state["tracking_enabled"] and manual_state.get("cooldown_active"):
            notify["reason"] = "cooldown_active"
            notify["cooldown_until_utc"] = manual_state.get("cooldown_until_utc")
        else:
            notify["reason"] = "non_actionable"
    else:
        notify["reason"] = "actionable"

    if stability_enabled:
        flip_cfg = config.get("flip_flop_guard") or {}
        min_bars_map = flip_cfg.get("min_bars_same_direction_before_flip") or {}
        history_window = max(
            1,
            max(
                [
                    int(value)
                    for value in min_bars_map.values()
                    if isinstance(value, (int, float))
                ]
                or [1]
            ),
        )
        state_path = outdir / "signal_state.json"
        state = _load_signal_state(state_path, history_window)
        _update_entry_history(state, entry_side, history_window)

        if actionable and intent == "entry":
            cooldown_cfg = (config.get("cooldowns") or {}).get(
                "min_minutes_between_entry_notifications", {}
            )
            min_between = _resolve_asset_value(cooldown_cfg, asset, 0)
            last_intent = state.get("last_notified_intent")
            last_ts = parse_utc_timestamp(state.get("last_notified_at_utc"))
            if (
                last_intent == "entry"
                and last_ts
                and now_dt - last_ts < timedelta(minutes=min_between)
            ):
                notify["should_notify"] = False
                notify["reason"] = "entry_cooldown_active"
                notify["cooldown_until_utc"] = to_utc_iso(
                    last_ts + timedelta(minutes=min_between)
                )

            hard_exit_cd = parse_utc_timestamp(state.get("hard_exit_cooldown_until_utc"))
            hard_exit_side = state.get("hard_exit_side")
            if hard_exit_cd and now_dt < hard_exit_cd:
                if not hard_exit_side or (entry_side and entry_side != hard_exit_side):
                    notify["should_notify"] = False
                    notify["reason"] = "hard_exit_cooldown_active"
                    notify["cooldown_until_utc"] = to_utc_iso(hard_exit_cd)

            if notify["should_notify"] and flip_cfg.get("enabled", False):
                last_side = state.get("last_notified_side")
                required = _resolve_asset_value(min_bars_map, asset, history_window)
                streak = _trailing_streak(state.get("entry_direction_history", []), entry_side)
                if last_side in {"buy", "sell"} and entry_side and entry_side != last_side:
                    if streak < required:
                        notify["should_notify"] = False
                        notify["reason"] = "flip_flop_guard"

        if notify["should_notify"] and config.get("only_notify_on_state_change", False):
            last_intent = state.get("last_notified_intent")
            last_side = state.get("last_notified_side")
            if intent == last_intent and (intent != "entry" or entry_side == last_side):
                notify["should_notify"] = False
                notify["reason"] = "no_state_change"

        if notify["should_notify"]:
            state["last_notified_intent"] = intent
            state["last_notified_side"] = entry_side or exit_side
            state["last_notified_at_utc"] = to_utc_iso(now_dt)
            if intent == "hard_exit":
                cd_cfg = (config.get("cooldowns") or {}).get(
                    "min_minutes_after_hard_exit", {}
                )
                min_after_exit = _resolve_asset_value(cd_cfg, asset, 0)
                state["hard_exit_side"] = exit_side or entry_side
                state["hard_exit_cooldown_until_utc"] = to_utc_iso(
                    now_dt + timedelta(minutes=min_after_exit)
                )
        _save_signal_state(state_path, state)

    positions_changed = False
    entry_opened = False
    if (
        manual_state.get("tracking_enabled")
        and intent == "hard_exit"
        and manual_state.get("has_position")
    ):
        manual_positions = position_tracker.close_position(
            asset,
            reason="hard_exit",
            closed_at_utc=to_utc_iso(now_dt),
            cooldown_minutes=cooldown_minutes,
            positions=manual_positions,
        )
        positions_changed = True
        LOGGER.debug(
            "CLOSE state transition %s reason=%s cooldown_until=%s",
            asset,
            "hard_exit",
            manual_positions.get(asset, {}).get("cooldown_until_utc")
            if isinstance(manual_positions, dict)
            else None,
        )

    open_conditions_met = (
        manual_state.get("tracking_enabled")
        and manual_state.get("is_flat")
        and intent == "entry"
        and notify.get("should_notify")
        and entry_side in {"buy", "sell"}
    )
    if open_conditions_met:
        if setup_grade in {"A", "B"}:
            entry_level, sl_level, tp2_level = _extract_trade_levels(payload)
            position_tracker.log_audit_event(
                "entry open attempt",
                event="OPEN_ATTEMPT",
                asset=asset,
                intent=intent,
                decision=decision,
                entry_side=entry_side,
                setup_grade=setup_grade,
                actionable=bool(actionable),
                stable=bool(stability_enabled),
                gates_missing=list(gates_missing),
                notify_should_notify=bool(notify.get("should_notify")),
                notify_reason=notify.get("reason"),
                cooldown_until_utc=notify.get("cooldown_until_utc"),
                manual_tracking_enabled=manual_state.get("tracking_enabled"),
                manual_has_position=manual_state.get("has_position"),
                manual_cooldown_active=manual_state.get("cooldown_active"),
                entry_level=entry_level,
                sl=sl_level,
                tp2=tp2_level,
            )
            manual_positions = position_tracker.open_position(
                asset,
                side="long" if entry_side == "buy" else "short",
                entry=entry_level,
                sl=sl_level,
                tp2=tp2_level,
                opened_at_utc=to_utc_iso(now_dt),
                positions=manual_positions,
            )
            positions_changed = True
            entry_opened = True
            LOGGER.debug(
                "OPEN state transition %s %s entry=%s sl=%s tp2=%s opened_at=%s",
                asset,
                entry_side,
                entry_level,
                sl_level,
                tp2_level,
                to_utc_iso(now_dt),
            )
    elif intent == "entry":
        suppression_reason = notify.get("reason") or "notified_blocked"
        if not manual_state.get("tracking_enabled"):
            suppression_reason = "tracking_disabled"
        elif manual_state.get("has_position"):
            suppression_reason = "position_already_open"
        elif manual_state.get("cooldown_active"):
            suppression_reason = "cooldown_active"
        elif entry_side not in {"buy", "sell"}:
            suppression_reason = "invalid_entry_side"
        elif setup_grade not in {"A", "B"}:
            suppression_reason = "setup_grade_filtered"
        position_tracker.log_audit_event(
            "entry suppressed",
            event="ENTRY_SUPPRESSED",
            asset=asset,
            intent=intent,
            decision=decision,
            entry_side=entry_side,
            setup_grade=setup_grade,
            actionable=bool(actionable),
            stable=bool(stability_enabled),
            gates_missing=list(gates_missing),
            notify_should_notify=bool(notify.get("should_notify")),
            notify_reason=notify.get("reason"),
            cooldown_until_utc=notify.get("cooldown_until_utc")
            or manual_state.get("cooldown_until_utc"),
            manual_tracking_enabled=manual_state.get("tracking_enabled"),
            manual_has_position=manual_state.get("has_position"),
            manual_cooldown_active=manual_state.get("cooldown_active"),
            suppression_reason=suppression_reason,
        )

    if positions_changed:
        position_tracker.save_positions_atomic(positions_path, manual_positions)
        _load_manual_positions_from_file.cache_clear()
        manual_state = position_tracker.compute_state(
            asset, tracking_cfg, manual_positions, now_dt
        )
        tracked_levels = _extract_tracked_levels(asset, manual_state, manual_positions)

        if entry_opened:
            entry_level, sl_level, tp2_level = _extract_trade_levels(payload)
            position_tracker.log_audit_event(
                "entry open committed",
                event="OPEN_COMMIT",
                asset=asset,
                intent=intent,
                decision=decision,
                entry_side=entry_side,
                setup_grade=setup_grade,
                entry=entry_level,
                sl=sl_level,
                tp2=tp2_level,
                positions_file=positions_path,
            )

    payload["position_state"] = {
        "tracking_enabled": manual_state.get("tracking_enabled"),
        "side": manual_state.get("side"),
        "has_position": manual_state.get("has_position"),
        "cooldown_active": manual_state.get("cooldown_active"),
        "cooldown_until_utc": manual_state.get("cooldown_until_utc"),
        "opened_at_utc": manual_state.get("opened_at_utc"),
        "entry": manual_state.get("entry"),
        "sl": manual_state.get("sl"),
        "tp2": manual_state.get("tp2"),
    }
    if tracked_levels:
        payload["tracked_levels"] = tracked_levels

    manual_note = _format_manual_position_note(asset, manual_state, tracked_levels)
    if manual_note:
        payload["position_management"] = manual_note
        reasons = payload.get("reasons") if isinstance(payload, dict) else None
        if isinstance(reasons, list):
            if manual_note not in reasons:
                reasons.append(manual_note)
        elif isinstance(payload, dict):
            payload["reasons"] = [manual_note]

    payload["intent"] = intent
    payload["actionable"] = bool(actionable)
    payload["notify"] = notify
    return payload


def _load_heartbeat_timestamp(path: Path) -> Optional[datetime]:
    try:
        payload = load_json(str(path)) or {}
    except Exception:
        payload = {}
    ts_raw = None
    if isinstance(payload, dict):
        ts_raw = payload.get("last_update_utc") or payload.get("timestamp")
    ts = parse_utc_timestamp(ts_raw) if ts_raw else None
    if ts:
        return ts
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, timezone.utc)


def load_latency_profile(outdir: str) -> Dict[str, Any]:
    profile_path = os.path.join(outdir, LATENCY_PROFILE_FILENAME)
    data = load_json(profile_path) or {}
    if not isinstance(data, dict):
        return {}
    return data


def update_latency_profile(outdir: str, latency_seconds: Optional[int]) -> None:
    if latency_seconds is None:
        return
    profile_path = os.path.join(outdir, LATENCY_PROFILE_FILENAME)
    data = load_latency_profile(outdir)
    ema_delay = float(data.get("ema_delay", latency_seconds))
    alpha = data.get("alpha", 0.15)
    ema_delay = (1 - alpha) * ema_delay + alpha * float(latency_seconds)
    payload = {"ema_delay": ema_delay, "alpha": alpha, "samples": int(data.get("samples", 0)) + 1}
    save_json(profile_path, payload)


def load_latency_guard_state(outdir: str) -> Dict[str, Any]:
    path = os.path.join(outdir, LATENCY_GUARD_STATE_FILENAME)
    data = load_json(path) or {}
    return data if isinstance(data, dict) else {}


def save_latency_guard_state(outdir: str, state: Dict[str, Any]) -> None:
    try:
        save_json(os.path.join(outdir, LATENCY_GUARD_STATE_FILENAME), state)
    except Exception:
        LOGGER.debug("latency_guard_state_save_failed", exc_info=True)

_EMPTY_KLINE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _empty_klines_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_KLINE_COLUMNS)


def as_df_klines(raw: Any) -> pd.DataFrame:
    if not raw:
        return _empty_klines_df()

    arr = raw if isinstance(raw, list) else (raw.get("values") or [])
    if not arr:
        return _empty_klines_df()

    records = [entry for entry in arr if isinstance(entry, dict)]
    if not records:
        return _empty_klines_df()

    schema_map: List[Tuple[str, Dict[str, str]]] = [
        ("datetime", {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}),
        ("t", {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}),
    ]

    timestamp_key: Optional[str] = None
    value_keys: Dict[str, str] = {}
    for ts_key, mapping in schema_map:
        if any(ts_key in rec for rec in records):
            timestamp_key = ts_key
            value_keys = mapping
            break

    if not timestamp_key:
        return _empty_klines_df()

    df = pd.DataFrame.from_records(records)

    time_values = pd.to_datetime(df.get(timestamp_key), errors="coerce", utc=True)
    if time_values.isna().all():
        return _empty_klines_df()

    open_series = pd.to_numeric(df.get(value_keys["open"]), errors="coerce")
    high_series = pd.to_numeric(df.get(value_keys["high"]), errors="coerce")
    low_series = pd.to_numeric(df.get(value_keys["low"]), errors="coerce")
    close_series = pd.to_numeric(df.get(value_keys["close"]), errors="coerce")
    volume_source = df.get(value_keys.get("volume", "volume"))
    volume_series = pd.to_numeric(volume_source, errors="coerce") if volume_source is not None else pd.Series(0.0, index=df.index)
    volume_series = volume_series.fillna(0.0)

    valid_mask = (
        time_values.notna()
        & open_series.notna()
        & high_series.notna()
        & low_series.notna()
        & close_series.notna()
    )

    if not valid_mask.any():
        return _empty_klines_df()

    filtered_index = pd.DatetimeIndex(time_values[valid_mask].to_numpy(), name="time")
    result = pd.DataFrame(
        {
            "open": open_series[valid_mask].to_numpy(dtype=float),
            "high": high_series[valid_mask].to_numpy(dtype=float),
            "low": low_series[valid_mask].to_numpy(dtype=float),
            "close": close_series[valid_mask].to_numpy(dtype=float),
            "volume": volume_series[valid_mask].to_numpy(dtype=float),
        },
        index=filtered_index,
    )

    return result.sort_index()


def load_tick_order_flow(asset: str, outdir: str) -> Dict[str, Any]:
    path = os.path.join(outdir, "order_flow_ticks.json")
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    metrics: Dict[str, Any] = {}
    for key in ("delta", "aggressor_ratio", "volume_buy", "volume_sell", "imbalance", "pressure"):
        value = data.get(key)
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    window = data.get("window_minutes")
    if isinstance(window, (int, float)):
        metrics["window_minutes"] = int(window)
    metrics["source"] = path
    return metrics
  
# --- TA: EMA/RSI/ATR, swingek, sweep, BOS, Fib zóna ----------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.bfill().fillna(50.0)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def volume_ratio(df: pd.DataFrame, recent: int, baseline: int) -> Optional[float]:
    if df.empty or "volume" not in df.columns:
        return None
    vol = df["volume"].astype(float)
    if len(vol) < recent + baseline:
        return None
    recent_slice = vol.iloc[-recent:]
    baseline_slice = vol.iloc[-(recent + baseline):-recent]
    baseline_mean = baseline_slice.mean()
    if not np.isfinite(baseline_mean) or baseline_mean <= 0:
        return None
    recent_mean = recent_slice.mean()
    if not np.isfinite(recent_mean):
        return None
    return float(recent_mean / baseline_mean)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = low.diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr_series = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr_series
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr_series
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (plus_di - minus_di).abs() / di_sum * 100.0
    adx_series = dx.ewm(span=period, adjust=False).mean()
    return adx_series.bfill().dropna()


def latest_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    series = compute_adx(df, period=period)
    if series.empty:
        return None
    value = series.iloc[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bucket_for_minute(minute: int, buckets: List[Dict[str, Any]]) -> str:
    for spec in buckets:
        try:
            start = int(spec.get("start_minute", 0))
            end = int(spec.get("end_minute", 1440))
        except (TypeError, ValueError):
            start, end = 0, 1440
        name = str(spec.get("name") or "mid")
        if start <= minute < end:
            return name
    if buckets:
        return str(buckets[-1].get("name") or "mid")
    return "all"


def compute_rel_atr_percentile(
    asset: str,
    now: datetime,
    df: pd.DataFrame,
    atr_series: pd.Series,
    adx_value: Optional[float] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    details: Dict[str, Any] = {"bucket": None, "samples": 0}
    if df.empty or atr_series.empty:
        return None, details
    minute_now = _min_of_day(now.hour, now.minute)
    buckets = ATR_PERCENTILE_BUCKETS
    if asset == "BTCUSD":
        bucket_name = _btc_time_of_day_bucket(minute_now)
    else:
        bucket_name = _bucket_for_minute(minute_now, buckets)
    details["bucket"] = bucket_name
    percentile_target: Optional[float] = None
    if asset == "BTCUSD":
        profile = _btc_active_profile()
        percentiles = BTC_ATR_PCT_TOD.get(profile) or {}
        percentile_target = percentiles.get(bucket_name, percentiles.get("mid"))
        details["profile"] = profile
    else:
        for spec in buckets:
            if str(spec.get("name")) == bucket_name:
                try:
                    percentile_target = float(spec.get("percentile"))
                except (TypeError, ValueError):
                    percentile_target = None
                break
    if percentile_target is None:
        percentile_target = 0.5
    overlap_cfg = ATR_PERCENTILE_OVERLAP
    try:
        overlap_assets = set(overlap_cfg.get("assets") or [])
    except Exception:
        overlap_assets = set()
    if asset in overlap_assets:
        try:
            overlap_start = int(overlap_cfg.get("start_minute", 0))
            overlap_end = int(overlap_cfg.get("end_minute", 0))
            overlap_pct = float(overlap_cfg.get("percentile", percentile_target))
        except (TypeError, ValueError):
            overlap_start = overlap_end = 0
            overlap_pct = percentile_target
        if overlap_start <= minute_now <= overlap_end:
            percentile_target = min(percentile_target, overlap_pct)
    if adx_value is not None and ATR_PERCENTILE_RANGE_ADX and adx_value < ATR_PERCENTILE_RANGE_ADX:
        percentile_target = max(percentile_target, ATR_PERCENTILE_RANGE_FLOOR or percentile_target)
    percentile_target = max(0.0, min(1.0, percentile_target))
    try:
        close_series = df["close"].astype(float)
    except Exception:
        return None, details
    rel_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
    if rel_series.empty:
        return None, details
    if ATR_PERCENTILE_LOOKBACK_DAYS > 0:
        cutoff = now - timedelta(days=ATR_PERCENTILE_LOOKBACK_DAYS)
        rel_series = rel_series.loc[rel_series.index >= cutoff]
    if rel_series.empty:
        return None, details
    if asset == "BTCUSD":
        bucket_labels = [
            _btc_time_of_day_bucket(_min_of_day(ts.hour, ts.minute)) for ts in rel_series.index
        ]
    else:
        bucket_labels = [
            _bucket_for_minute(_min_of_day(ts.hour, ts.minute), buckets) for ts in rel_series.index
        ]
    rel_by_bucket = rel_series.iloc[[i for i, name in enumerate(bucket_labels) if name == bucket_name]]
    if rel_by_bucket.empty:
        rel_by_bucket = rel_series
    details["samples"] = int(rel_by_bucket.size)
    if rel_by_bucket.empty:
        return None, details
    try:
        value = float(np.nanpercentile(rel_by_bucket.values, percentile_target * 100.0))
    except Exception:
        return None, details
    details["percentile"] = percentile_target
    details["threshold"] = value
    return value, details


def compute_ofi_zscore(k1m: pd.DataFrame, window: int) -> Optional[float]:
    if window <= 0 or k1m.empty or "close" not in k1m.columns or "volume" not in k1m.columns:
        return None
    prices = k1m["close"].astype(float).copy()
    volumes = k1m["volume"].astype(float).copy()
    if len(prices) < window + 5:
        return None
    price_delta = prices.diff().fillna(0.0)
    signed_volume = np.sign(price_delta) * volumes
    rolling_sum = signed_volume.rolling(5, min_periods=3).sum()
    rolling_vol = volumes.rolling(5, min_periods=3).sum()
    imbalance = (rolling_sum / rolling_vol.replace(0.0, np.nan)).dropna()
    if imbalance.empty:
        return None
    imbalance = imbalance.tail(window)
    if imbalance.size < 10:
        return None
    current = imbalance.iloc[-1]
    mean = imbalance.mean()
    std = imbalance.std()
    if not np.isfinite(std) or std <= 1e-6:
        return None
    return float((current - mean) / std)


def compute_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty or "close" not in df.columns:
        return None
    price = df["close"].astype(float)
    volume = df.get("volume")
    if volume is None:
        volume = pd.Series(np.ones(len(price)), index=price.index)
    else:
        volume = volume.astype(float).replace(0.0, np.nan)
    typical_price = price
    cumsum_price = (typical_price * volume.fillna(0.0)).cumsum()
    cumsum_volume = volume.ffill().bfill().cumsum()
    vwap_series = cumsum_price / cumsum_volume.replace(0.0, np.nan)
    return vwap_series.dropna()


def evaluate_vwap_confluence(
    asset: str,
    direction: Optional[str],
    regime: str,
    price: Optional[float],
    atr_abs: Optional[float],
    k1m: pd.DataFrame,
    ofi_z: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"trend_pullback": False, "mean_revert": False}
    if price is None or atr_abs is None or atr_abs <= 0:
        return result
    vwap_series = compute_vwap(k1m)
    if vwap_series is None or vwap_series.empty:
        return result
    current_vwap = vwap_series.iloc[-1]
    band_trend = atr_abs * VWAP_TREND_BAND
    band_mean = atr_abs * VWAP_MEAN_REVERT_BAND
    delta = float(price - current_vwap)
    result["vwap"] = current_vwap
    result["distance"] = delta
    if abs(delta) <= band_trend and regime == "trend":
        if direction == "long" and delta <= 0:
            result["trend_pullback"] = True
        elif direction == "short" and delta >= 0:
            result["trend_pullback"] = True
    weakening_flow = ofi_z is not None and ofi_z <= OFI_Z_WEAKENING
    if abs(delta) >= band_mean and regime == "range" and weakening_flow:
        if direction == "long" and delta < 0:
            result["mean_revert"] = True
        elif direction == "short" and delta > 0:
            result["mean_revert"] = True
    return result


def _news_settings_for_asset(asset: str) -> Tuple[int, int, float]:
    settings = NEWS_ASSET_SETTINGS.get(asset, {})
    lockout = NEWS_LOCKOUT_MINUTES_DEFAULT
    stabilisation = NEWS_STABILISATION_MINUTES_DEFAULT
    severity = NEWS_SEVERITY_THRESHOLD_DEFAULT
    if isinstance(settings, dict):
        try:
            lockout = int(settings.get("lockout_minutes", lockout) or 0)
        except (TypeError, ValueError):
            lockout = NEWS_LOCKOUT_MINUTES_DEFAULT
        try:
            stabilisation = int(settings.get("stabilisation_minutes", stabilisation) or 0)
        except (TypeError, ValueError):
            stabilisation = NEWS_STABILISATION_MINUTES_DEFAULT
        try:
            severity = float(settings.get("severity_threshold", severity) or severity)
        except (TypeError, ValueError):
            severity = NEWS_SEVERITY_THRESHOLD_DEFAULT
    return lockout, stabilisation, severity


def load_calendar_events(asset: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for name in NEWS_CALENDAR_FILES:
        local_path = os.path.join(PUBLIC_DIR, asset, name)
        shared_path = os.path.join(PUBLIC_DIR, name)
        for candidate in (local_path, shared_path):
            data = load_json(candidate)
            if isinstance(data, dict):
                raw_events = data.get("events") or []
                if isinstance(raw_events, list):
                    for evt in raw_events:
                        if isinstance(evt, dict):
                            evt = dict(evt)
                            evt.setdefault("source", candidate)
                            events.append(evt)
            elif isinstance(data, list):
                for evt in data:
                    if isinstance(evt, dict):
                        evt = dict(evt)
                        evt.setdefault("source", candidate)
                        events.append(evt)
    return events


def _evaluate_macro_lockout(asset: str, now: datetime) -> Tuple[bool, Optional[str]]:
    if not asset:
        return False, None
    windows = _load_macro_lockout_windows()
    if not windows:
        return False, None
    asset_windows = windows.get(asset.upper())
    if not asset_windows:
        return False, None
    now_utc = now.astimezone(timezone.utc)
    for window in asset_windows:
        start = window.get("start")
        end = window.get("end")
        if start is None or end is None:
            continue
        if start <= now_utc <= end:
            label = window.get("label") or "Macro event"
            provider = window.get("provider")
            release = window.get("release")
            timing: Optional[str] = None
            if isinstance(release, datetime):
                if now_utc < release:
                    timing = "pre-release"
                elif now_utc > release:
                    timing = "post-release"
                else:
                    timing = "at release"
            reason_parts = ["Macro lockout"]
            if provider:
                reason_parts.append(f"{provider}")
            reason_parts.append(str(label))
            if timing:
                reason_parts.append(f"({timing})")
            reason = ": ".join([reason_parts[0], " ".join(reason_parts[1:])]) if len(reason_parts) > 1 else reason_parts[0]
            return True, reason
    return False, None


def evaluate_news_lockout(asset: str, now: datetime) -> Tuple[bool, Optional[str]]:
    macro_lockout, macro_reason = _evaluate_macro_lockout(asset, now)
    if macro_lockout:
        return True, macro_reason
    lockout_minutes, stabilisation_minutes, severity_threshold = _news_settings_for_asset(asset)
    if lockout_minutes <= 0 and stabilisation_minutes <= 0:
        return False, None
    events = load_calendar_events(asset)
    if not events:
        return False, None
    lockout = False
    reason: Optional[str] = None
    for event in events:
        ts = event.get("time") or event.get("utc") or event.get("datetime")
        if not ts:
            continue
        try:
            event_time = pd.to_datetime(ts).to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            continue
        severity = event.get("severity") or event.get("impact")
        try:
            severity_val = float(severity)
        except (TypeError, ValueError):
            severity_val = 1.0
        if severity_val < severity_threshold:
            continue
        delta_minutes = (now - event_time).total_seconds() / 60.0
        if abs(delta_minutes) <= lockout_minutes:
            lockout = True
            reason = event.get("title") or event.get("name") or "High impact news"
            break
        if 0 < delta_minutes < stabilisation_minutes:
            lockout = True
            reason = (event.get("title") or event.get("name") or "High impact news") + " — stabilizáció"
            break
    return lockout, reason


def load_funding_snapshot(asset: str) -> Optional[float]:
    path_candidates = [
        os.path.join(PUBLIC_DIR, asset, "funding_rate.json"),
        os.path.join(PUBLIC_DIR, asset, "funding.json"),
    ]
    for path in path_candidates:
        data = load_json(path)
        if isinstance(data, dict):
            for key in ("funding_rate", "value", "rate", "last"):
                if key in data:
                    try:
                        return float(data[key])
                    except (TypeError, ValueError):
                        continue
    return None
def compute_order_flow_metrics(
    k1m: pd.DataFrame,
    k5m: pd.DataFrame,
    tick_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "imbalance": None,
        "pressure": None,
        "delta_volume": None,
        "aggressor_ratio": None,
        "imbalance_z": None,
        "status": "unavailable",
    }
    if tick_metrics:
        for key in ("imbalance", "pressure", "delta", "delta_volume", "aggressor_ratio"):
            if key in tick_metrics and tick_metrics.get(key) is not None:
                mapped_key = "delta_volume" if key in {"delta", "delta_volume"} else key
                try:
                    metrics[mapped_key] = float(tick_metrics[key])
                except (TypeError, ValueError):
                    continue
    if tick_metrics and metrics.get("imbalance") is not None and metrics.get("pressure") is not None:
        metrics["status"] = "tick_only"
        return metrics

    if k1m.empty or "volume" not in k1m.columns or len(k1m) < ORDER_FLOW_LOOKBACK_MIN:
        metrics["status"] = "volume_unavailable"
        return metrics

    recent = k1m.tail(ORDER_FLOW_LOOKBACK_MIN).copy()
    price_delta = recent["close"].diff().fillna(0.0)
    signed_volume = np.sign(price_delta) * recent["volume"].fillna(0.0)
    buy_vol = signed_volume[signed_volume > 0].sum()
    sell_vol = -signed_volume[signed_volume < 0].sum()
    total = buy_vol + sell_vol
    if total > 0:
        metrics["imbalance"] = float((buy_vol - sell_vol) / total)

    pressure = price_delta.rolling(5).mean().iloc[-1]
    volume_avg = recent["volume"].rolling(5).mean().iloc[-1]
    if np.isfinite(pressure) and np.isfinite(volume_avg) and volume_avg > 0:
        metrics["pressure"] = float(pressure * volume_avg / max(total, 1e-9))

    delta_sum = signed_volume.sum()
    metrics["delta_volume"] = float(delta_sum) if np.isfinite(delta_sum) else None

    if not k5m.empty and "volume" in k5m.columns:
        ref = k5m.tail(12)
        if not ref.empty:
            rel = recent["volume"].sum() / max(ref["volume"].sum(), 1e-9)
            metrics["pressure"] = float(rel * (metrics["pressure"] or 0.0))

    if OFI_Z_LOOKBACK > 0:
        z_score = compute_ofi_zscore(k1m, OFI_Z_LOOKBACK)
        if z_score is not None and np.isfinite(z_score):
            metrics["imbalance_z"] = float(z_score)

    metrics["status"] = "ok"
    return metrics


@lru_cache(maxsize=16)
def _load_reference_dataframe(reference: str) -> pd.DataFrame:
    base = Path(PUBLIC_DIR)
    ref_key = reference.upper()
    candidates = [
        base / ref_key / "klines_5m.json",
        base / ref_key / "klines_1m.json",
        base / "correlation" / f"{ref_key}_klines_5m.json",
        base / "correlation" / f"{ref_key}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = load_json(str(path))
        if not data:
            continue
        df = as_df_klines(data)
        if df.empty or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        if close.empty:
            continue
        if len(close) >= 10:
            deltas = close.index.to_series().diff().dropna()
            if not deltas.empty and deltas.median() <= pd.Timedelta(minutes=1, seconds=30):
                close = close.resample("5T").last().dropna()
        return close.to_frame(name="close")
    return pd.DataFrame(columns=["close"])


def _auto_smt_penalty(asset: str, k5m: pd.DataFrame) -> Tuple[int, Optional[str]]:
    cfg = SMT_AUTO_CONFIG.get(asset.upper())
    if not cfg:
        return 0, None
    if k5m.empty or "close" not in k5m.columns:
        return 0, None
    reference = cfg.get("reference")
    if not reference:
        return 0, None
    ref_df = _load_reference_dataframe(str(reference))
    if ref_df.empty or "close" not in ref_df.columns:
        return 0, None

    asset_returns = k5m["close"].astype(float).pct_change()
    ref_returns = ref_df["close"].astype(float).pct_change()
    combined = pd.concat(
        [asset_returns.rename("asset"), ref_returns.rename("reference")],
        axis=1,
        join="inner",
    ).dropna()
    if combined.empty:
        return 0, None
    window = int(cfg.get("window", max(SMT_REQUIRED_BARS * 3, 6)))
    combined = combined.tail(window)
    threshold = abs(float(cfg.get("threshold", 0.0) or 0.0))
    min_bars = max(int(cfg.get("min_bars", SMT_REQUIRED_BARS)), 1)
    relationship = str(cfg.get("relationship", "direct")).lower()

    diverging = 0
    considered = 0
    last_significant_direction = None
    for _, row in combined.iterrows():
        a_ret = float(row["asset"])
        r_ret = float(row["reference"])
        if not np.isfinite(a_ret) or not np.isfinite(r_ret):
            continue
        if abs(a_ret) < threshold and abs(r_ret) < threshold:
            continue
        considered += 1
        sign_asset = np.sign(a_ret)
        sign_ref = np.sign(r_ret)
        if relationship == "inverse":
            diverge_now = sign_asset == sign_ref and sign_asset != 0 and sign_ref != 0
        else:
            diverge_now = sign_asset != sign_ref and sign_asset != 0 and sign_ref != 0
        if diverge_now:
            diverging += 1
            last_significant_direction = "bullish" if sign_asset > 0 else "bearish"
    if considered >= min_bars and diverging >= min_bars:
        direction = last_significant_direction or "divergence"
        note = f"SMT divergencia auto ({reference}, {direction})"
        return SMT_PENALTY_VALUE, note
    return 0, None


def find_swings(df: pd.DataFrame, lb: int = 2) -> pd.DataFrame:
    if df.empty: return df
    hi = df["high"]; lo = df["low"]
    swing_hi = (hi.shift(lb) == hi.rolling(lb*2+1, center=True).max())
    swing_lo = (lo.shift(lb) == lo.rolling(lb*2+1, center=True).min())
    out = df.copy()
    out["swing_hi"] = swing_hi.fillna(False)
    out["swing_lo"] = swing_lo.fillna(False)
    return out

def last_swing_levels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if df.empty or ("swing_hi" not in df.columns): return None, None
    hi = df[df["swing_hi"]].tail(1)["high"].values
    lo = df[df["swing_lo"]].tail(1)["low"].values
    return (float(hi[0]) if len(hi) else None, float(lo[0]) if len(lo) else None)

def detect_sweep(df: pd.DataFrame, lookback: int = 24) -> Dict[str, bool]:
    out = {"sweep_high": False, "sweep_low": False}
    if len(df) < lookback + 2: return out
    ref = df.iloc[-(lookback+1):-1]
    last = df.iloc[-1]
    prev_max, prev_min = ref["high"].max(), ref["low"].min()
    if last["high"] > prev_max and last["close"] < prev_max: out["sweep_high"] = True
    if last["low"]  < prev_min and last["close"] > prev_min: out["sweep_low"]  = True
    return out

def detect_bos(df: pd.DataFrame, direction: str) -> bool:
    if direction not in ("long", "short"):
        return False
    sw = find_swings(df, lb=2)
    hi, lo = last_swing_levels(sw.iloc[:-1])
    if direction == "long" and hi is not None:
        return sw["high"].iloc[-1] > hi
    if direction == "short" and lo is not None:
        return sw["low"].iloc[-1] < lo
    return False

def broke_structure(df: pd.DataFrame, direction: str, lookback: Optional[int] = None) -> bool:
    """Egyszerű szerkezeti törés: utolsó high/low áttöri az előző N bar csúcsát/alját."""
    lb = lookback or DEFAULT_BOS_LOOKBACK
    if df.empty or len(df) < lb + 2:
        return False
    ref = df.iloc[-(lb + 1) : -1]
    last = df.iloc[-1]
    if direction == "long":
        return last["high"] > ref["high"].max()
    if direction == "short":
        return last["low"] < ref["low"].min()
    return False


def retest_level(df: pd.DataFrame, direction: str, lookback: Optional[int] = None) -> bool:
    if df.empty or len(df) < 2:
        return False
    lb = lookback or DEFAULT_BOS_LOOKBACK
    if len(df) < lb + 1:
        ref = df.iloc[:-1]
    else:
        ref = df.iloc[-(lb + 1) : -1]
    if ref.empty:
        return False
    last = df.iloc[-1]
    if direction == "long":
        level = ref["high"].max()
        if not np.isfinite(level):
            return False
        return last["low"] <= level <= last["high"]
    if direction == "short":
        level = ref["low"].min()
        if not np.isfinite(level):
            return False
        return last["low"] <= level <= last["high"]
    return False


def structure_break_with_retest(
    df: pd.DataFrame, direction: str, lookback: Optional[int] = None
) -> bool:
    if direction not in ("long", "short"):
        return False
    lb = lookback or DEFAULT_BOS_LOOKBACK
    if not broke_structure(df, direction, lb):
        return False
    return retest_level(df, direction, lb)


def micro_bos_with_retest(
    k1m: pd.DataFrame, k5m: pd.DataFrame, direction: str, lookback: Optional[int] = None
) -> bool:
    if direction not in ("long", "short"):
        return False
    if k1m.empty or len(k1m) < 10:
        return False
    if not detect_bos(k1m, direction):
        return False
    return retest_level(k5m, direction, lookback or DEFAULT_BOS_LOOKBACK)


def _recent_liquidity_levels(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 30,
) -> List[float]:
    if df.empty:
        return []
    window = df.tail(max(lookback + 5, 10))
    swings = find_swings(window, lb=1)
    levels: List[float] = []
    if direction == "buy":
        lows = swings[swings.get("swing_lo", False)]
        levels = lows["low"].dropna().astype(float).tolist()
    elif direction == "sell":
        highs = swings[swings.get("swing_hi", False)]
        levels = highs["high"].dropna().astype(float).tolist()
    return levels[-5:]


def _confidence_from_score(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def compute_precision_entry(
    asset: str,
    direction: str,
    k1m: pd.DataFrame,
    k5m: pd.DataFrame,
    price_now: Optional[float],
    atr5: Optional[float],
    order_flow_metrics: Dict[str, Any],
    score_threshold: float = PRECISION_SCORE_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "asset": asset,
        "direction": direction,
        "score": 0.0,
        "confidence": "low",
        "entry": None,
        "entry_window": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "risk": None,
        "liquidity_levels": [],
        "factors": [],
        "score_threshold": score_threshold,
        "score_ready": False,
        "trigger_state": "standby",
        "trigger_ready": False,
        "trigger_levels": None,
        "ready_ts": None,
        "order_flow_ready": False,
        "order_flow_strength": None,
        "microstructure_score": None,
        "trigger_progress": 0.0,
        "trigger_confidence": 0.0,
        "trigger_reasons": [],
    }

    metrics_status = str(order_flow_metrics.get("status") or "unavailable")
    flow_optional = metrics_status in {"volume_unavailable", "unavailable", "stale"}
    plan["order_flow_status"] = metrics_status
    plan["order_flow_optional"] = flow_optional

    flow_rules = get_precision_flow_rules(asset)
    imbalance_threshold = float(
        flow_rules.get("imbalance_threshold", ORDER_FLOW_IMBALANCE_TH)
    )
    pressure_threshold = float(
        flow_rules.get("pressure_threshold", ORDER_FLOW_PRESSURE_TH)
    )
    imbalance_margin = float(
        flow_rules.get("imbalance_margin", PRECISION_FLOW_IMBALANCE_MARGIN)
    )
    pressure_margin = float(
        flow_rules.get("pressure_margin", PRECISION_FLOW_PRESSURE_MARGIN)
    )
    min_flow_signals = max(0, int(flow_rules.get("min_signals", 1)))
    strength_floor = max(
        0.0, float(flow_rules.get("strength_floor", PRECISION_FLOW_STRENGTH_BASE))
    )
    block_ratio = float(flow_rules.get("block_ratio", 0.0))

    plan["order_flow_settings"] = {
        "imbalance_threshold": round(imbalance_threshold, 4),
        "pressure_threshold": round(pressure_threshold, 4),
        "imbalance_margin": round(imbalance_margin, 3),
        "pressure_margin": round(pressure_margin, 3),
        "min_signals": min_flow_signals,
        "strength_floor": round(strength_floor, 2),
        "block_ratio": round(block_ratio, 4),
    }

    if direction not in {"buy", "sell"}:
        plan["factors"].append("direction unsupported")
        return plan
    if price_now is None or not np.isfinite(price_now):
        plan["factors"].append("price unavailable")
        return plan
    if k1m.empty:
        plan["factors"].append("1m dataset empty")
        return plan

    base_price = float(price_now)
    atr1m_series = atr(k1m, period=14)
    atr1m = float(atr1m_series.iloc[-1]) if not atr1m_series.empty else None
    liquidity_levels = _recent_liquidity_levels(k1m, "buy" if direction == "buy" else "sell")
    plan["liquidity_levels"] = liquidity_levels
    if not liquidity_levels:
        plan["factors"].append("no nearby liquidity levels detected")

    score = 40.0
    factors: List[str] = []

    imb = order_flow_metrics.get("imbalance")
    flow_signal_count = 0
    flow_strength = 0.0
    flow_reasons: List[str] = []
    flow_blockers: List[str] = []    
    if imb is not None:
        strong_threshold = imbalance_threshold * imbalance_margin
        base_threshold = max(imbalance_threshold, 1e-9)
        strong_threshold = max(strong_threshold, base_threshold)
        if direction == "buy" and imb > imbalance_threshold:
            score += 12.0
            flow_strength += min(1.0, float(imb) / base_threshold)
            factors.append(f"order flow imbalance +{imb:.2f}")
            flow_reasons.append(f"imbalance {imb:.2f}")
            if imb >= strong_threshold:
                flow_signal_count += 1
        elif direction == "sell" and imb < -imbalance_threshold:
            score += 12.0
            flow_strength += min(1.0, abs(float(imb)) / base_threshold)
            factors.append(f"order flow imbalance {imb:.2f}")
            flow_reasons.append(f"imbalance {imb:.2f}")
            if imb <= -strong_threshold:
                flow_signal_count += 1
        else:
            if direction == "buy" and imb <= -strong_threshold:
                flow_blockers.append(f"imbalance {imb:.2f}")
            elif direction == "sell" and imb >= strong_threshold:
                flow_blockers.append(f"imbalance {imb:.2f}")

    pressure = order_flow_metrics.get("pressure")
    if pressure is not None:
        pressure_strong = pressure_threshold * pressure_margin
        pressure_base = max(pressure_threshold, 1e-9)
        pressure_strong = max(pressure_strong, pressure_base)
        if direction == "buy" and pressure > pressure_threshold:
            score += 10.0
            flow_strength += min(1.0, float(pressure) / pressure_base)
            factors.append(f"order flow pressure +{pressure:.2f}")
            flow_reasons.append(f"pressure {pressure:.2f}")
            if pressure >= pressure_strong:
                flow_signal_count += 1
        elif direction == "sell" and pressure < -pressure_threshold:
            score += 10.0
            flow_strength += min(1.0, abs(float(pressure)) / pressure_base)
            factors.append(f"order flow pressure {pressure:.2f}")
            flow_reasons.append(f"pressure {pressure:.2f}")
            if pressure <= -pressure_strong:
                flow_signal_count += 1
        else:
            if direction == "buy" and pressure <= -pressure_strong:
                flow_blockers.append(f"pressure {pressure:.2f}")
            elif direction == "sell" and pressure >= pressure_strong:
                flow_blockers.append(f"pressure {pressure:.2f}")

    delta_volume = order_flow_metrics.get("delta_volume")
    if delta_volume is not None:
        try:
            dv = float(delta_volume)
            if direction == "buy" and dv > 0:
                flow_strength += min(0.5, abs(dv))
                flow_reasons.append(f"delta +{dv:.1f}")
                flow_signal_count += 1
            elif direction == "sell" and dv < 0:
                flow_strength += min(0.5, abs(dv))
                flow_reasons.append(f"delta {dv:.1f}")
                flow_signal_count += 1
            elif direction == "buy" and dv < 0:
                flow_blockers.append(f"delta {dv:.1f}")
            elif direction == "sell" and dv > 0:
                flow_blockers.append(f"delta +{dv:.1f}")
        except (TypeError, ValueError):
            pass

    if flow_signal_count > 0:
        signal_metric = flow_strength / float(flow_signal_count)
    else:
        signal_metric = flow_strength

    if flow_strength > 0 or flow_signal_count > 0:
        plan["order_flow_strength"] = round(min(2.0, signal_metric), 2)
    plan["order_flow_signals"] = flow_signal_count

    meets_count = flow_signal_count >= max(1, min_flow_signals)
    if min_flow_signals <= 0:
        meets_count = flow_signal_count > 0
    meets_strength = flow_strength > 0 and signal_metric >= strength_floor
    if meets_strength and not meets_count:
        flow_reasons.append(
            f"flow strength override ≥ {strength_floor:.2f}"
        )

    flow_ready = (meets_count or meets_strength) and not flow_blockers
    if flow_optional:
        flow_ready = True
        flow_blockers = []
        flow_reasons.append("order flow optional (volume unavailable)")

    def _abs_or_none(value: Any) -> Optional[float]:
        try:
            return abs(float(value))
        except (TypeError, ValueError):
            return None

    metrics_near_zero = True
    for candidate, eps in (
        (imb, PRECISION_FLOW_STALLED_EPS),
        (pressure, PRECISION_FLOW_STALLED_EPS),
    ):
        candidate_abs = _abs_or_none(candidate)
        if candidate_abs is not None and candidate_abs > eps:
            metrics_near_zero = False
            break
    if metrics_near_zero:
        delta_abs = _abs_or_none(delta_volume)
        if delta_abs is not None and delta_abs > PRECISION_FLOW_STALLED_DELTA_EPS:
            metrics_near_zero = False

    stalled_flow = (
        not flow_optional
        and metrics_near_zero
        and flow_signal_count == 0
        and (_abs_or_none(flow_strength) or 0.0) <= PRECISION_FLOW_STALLED_EPS
        and not flow_blockers
    )

    plan["order_flow_ready"] = flow_ready
    if flow_blockers:
        plan["order_flow_blockers"] = flow_blockers
    else:
        plan["order_flow_blockers"] = []
    if flow_reasons:
        plan["trigger_reasons"].extend(flow_reasons)
    plan["order_flow_stalled"] = stalled_flow
      
    if micro_bos_with_retest(k1m, k5m, direction):
        score += 12.0
        factors.append("micro BOS + retest confirmed")

    if atr1m is not None and atr1m > 0:
        score += 6.0
        factors.append(f"ATR1m {atr1m:.4f}")

    if atr5 is not None and np.isfinite(atr5) and atr5 > 0:
        score += 6.0
        factors.append(f"ATR5m {float(atr5):.4f}")

    entry_level: Optional[float] = None
    if liquidity_levels:
        liquidity_levels_sorted = sorted(
            liquidity_levels,
            key=lambda lvl: abs(base_price - float(lvl)),
        )
        entry_level = float(liquidity_levels_sorted[0])
        distance = abs(base_price - entry_level)
        tolerance = max(atr1m or 0.0, (float(atr5) if atr5 else 0.0) * 0.6)
        if tolerance > 0 and distance <= tolerance * 1.2:
            score += 10.0
            factors.append("price near liquidity pocket")
    else:
        entry_level = base_price

    if direction == "buy" and entry_level > base_price:
        entry_level = base_price
    if direction == "sell" and entry_level < base_price:
        entry_level = base_price

    score = max(0.0, min(100.0, score))
    plan["microstructure_score"] = score
    plan["score"] = round(score, 2)
    plan["confidence"] = _confidence_from_score(score)
    plan["factors"].extend(factors)
    plan["entry"] = entry_level

    risk_buffer_candidates = [
        value
        for value in [atr1m, (float(atr5) if atr5 else None)]
        if value is not None and np.isfinite(value) and value > 0
    ]
    if not risk_buffer_candidates:
        risk_buffer = abs(base_price * 0.001)
    else:
        risk_buffer = max(risk_buffer_candidates) * 1.2

    if direction == "buy":
        stop_loss = entry_level - risk_buffer if entry_level is not None else None
        tp1 = entry_level + risk_buffer * TP1_R if entry_level is not None else None
        tp2 = entry_level + risk_buffer * TP2_R if entry_level is not None else None
        entry_window = (
            entry_level - risk_buffer * 0.4,
            entry_level + risk_buffer * 0.2,
        ) if entry_level is not None else None
    else:
        stop_loss = entry_level + risk_buffer if entry_level is not None else None
        tp1 = entry_level - risk_buffer * TP1_R if entry_level is not None else None
        tp2 = entry_level - risk_buffer * TP2_R if entry_level is not None else None
        entry_window = (
            entry_level - risk_buffer * 0.2,
            entry_level + risk_buffer * 0.4,
        ) if entry_level is not None else None

    plan["risk"] = risk_buffer if np.isfinite(risk_buffer) else None
    plan["stop_loss"] = stop_loss
    plan["take_profit_1"] = tp1
    plan["take_profit_2"] = tp2
    plan["entry_window"] = entry_window

    if entry_window is not None:
        plan["entry_window"] = tuple(float(x) for x in entry_window)
    if stop_loss is not None and not np.isfinite(stop_loss):
        plan["stop_loss"] = None
    if tp1 is not None and not np.isfinite(tp1):
        plan["take_profit_1"] = None
    if tp2 is not None and not np.isfinite(tp2):
        plan["take_profit_2"] = None

    trigger_levels: Dict[str, float] = {}
    window_tuple = plan.get("entry_window")
    if window_tuple is not None:
        window_lo, window_hi = window_tuple
        trigger_levels["window_low"] = float(window_lo)
        trigger_levels["window_high"] = float(window_hi)
        if entry_level is not None:
            trigger_levels["fire"] = float(entry_level)
        if direction == "buy":
            trigger_levels["arm"] = float(window_hi)
            trigger_levels["disarm"] = float(window_lo)
        else:
            trigger_levels["arm"] = float(window_lo)
            trigger_levels["disarm"] = float(window_hi)
    elif entry_level is not None:
        trigger_levels["fire"] = float(entry_level)
    if trigger_levels:
        plan["trigger_levels"] = trigger_levels

    trigger_state = "standby"
    trigger_progress = 0.0
    if plan["score_ready"]:
        trigger_state = "ready"
        trigger_progress = 0.35
        if plan["order_flow_ready"]:
            trigger_state = "arming"
            trigger_progress = 0.6

    price_val: Optional[float] = None
    if price_now is not None and np.isfinite(price_now):
        price_val = float(price_now)

    risk_val = None
    try:
        risk_raw = plan.get("risk")
        if risk_raw is not None:
            risk_val = float(risk_raw)
    except (TypeError, ValueError):
        risk_val = None
    tolerance = (risk_val or 0.0) * PRECISION_TRIGGER_NEAR_MULT

    window_tuple = plan.get("entry_window")
    if price_val is not None and window_tuple is not None:
        window_lo, window_hi = float(window_tuple[0]), float(window_tuple[1])
        if direction == "buy":
            pivot = float(entry_level) if entry_level is not None else window_lo
            if price_val <= pivot:
                trigger_state = "fire"
                trigger_progress = 1.0
                plan["trigger_reasons"].append("price hit entry")
            elif price_val <= window_hi:
                trigger_state = "arming"
                trigger_progress = max(trigger_progress, 0.85)
                plan["trigger_reasons"].append("price inside window")
            elif tolerance and price_val <= window_hi + tolerance:
                trigger_progress = max(trigger_progress, 0.7)
                plan["trigger_reasons"].append("price near window")
        else:
            pivot = float(entry_level) if entry_level is not None else window_hi
            if price_val >= pivot:
                trigger_state = "fire"
                trigger_progress = 1.0
                plan["trigger_reasons"].append("price hit entry")
            elif price_val >= window_lo:
                trigger_state = "arming"
                trigger_progress = max(trigger_progress, 0.85)
                plan["trigger_reasons"].append("price inside window")
            elif tolerance and price_val >= window_lo - tolerance:
                trigger_progress = max(trigger_progress, 0.7)
                plan["trigger_reasons"].append("price near window")
    elif price_val is not None and entry_level is not None:
        pivot = float(entry_level)
        if direction == "buy" and price_val <= pivot:
            trigger_state = "fire"
            trigger_progress = 1.0
            plan["trigger_reasons"].append("price hit entry")
        elif direction == "sell" and price_val >= pivot:
            trigger_state = "fire"
            trigger_progress = 1.0
            plan["trigger_reasons"].append("price hit entry")

    if (
        asset.upper() == "BTCUSD"
        and not flow_ready
        and stalled_flow
        and trigger_state == "fire"
    ):
        flow_ready = True
        plan["order_flow_ready"] = True
        plan["order_flow_stalled"] = True
        plan["order_flow_fallback"] = "stalled_flow_trigger_fire"
        plan["trigger_reasons"].append("order flow stalled override")

    plan["trigger_state"] = trigger_state
    plan["trigger_ready"] = trigger_state in {"arming", "fire"}
    plan["trigger_progress"] = round(min(max(trigger_progress, 0.0), 1.0), 3)

    score_conf = min(1.0, score / 100.0) if score else 0.0
    flow_strength = plan.get("order_flow_strength") or 0.0
    try:
        flow_conf = min(1.0, float(flow_strength))
    except (TypeError, ValueError):
        flow_conf = 0.0
    plan["trigger_confidence"] = round(
        min(1.0, (score_conf + flow_conf + plan["trigger_progress"]) / 3.0), 3
    )

    if plan["trigger_reasons"]:
        plan["trigger_reasons"] = list(dict.fromkeys(plan["trigger_reasons"]))

    ready_ts: Optional[str] = None
    try:
        idx = k1m.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            tail = idx[-min(3, len(idx)):].sort_values()
            median_pos = len(tail) // 2
            candidate = pd.Timestamp(tail[median_pos])
            if candidate.tzinfo is None:
                candidate = candidate.tz_localize(timezone.utc)
            else:
                candidate = candidate.tz_convert(timezone.utc)
            ready_ts = (
                candidate.floor("s").isoformat().replace("+00:00", "Z")
            )
    except Exception:
        ready_ts = None
    plan["ready_ts"] = ready_ts

    try:
        plan_score = float(plan.get("score") or 0.0)
    except (TypeError, ValueError):
        plan_score = 0.0
    plan["score_ready"] = plan_score >= score_threshold

    if plan["confidence"] == "low":
        plan["factors"].append("precision confidence low")

    return plan


def ema_cross_recent(short: pd.Series, long: pd.Series, bars: int = MOMENTUM_BARS, direction: str = "long") -> bool:
    if short.empty or long.empty or len(short) < bars + 2 or len(long) < bars + 2:
        return False
    short = short.dropna()
    long = long.dropna()
    if len(short) < bars + 2 or len(long) < bars + 2:
        return False
    for i in range(1, bars + 1):
        idx_now = -i
        idx_prev = -i - 1
        try:
            s_now = short.iloc[idx_now]
            s_prev = short.iloc[idx_prev]
            l_now = long.iloc[idx_now]
            l_prev = long.iloc[idx_prev]
        except IndexError:
            continue
        if not (np.isfinite(s_now) and np.isfinite(s_prev) and np.isfinite(l_now) and np.isfinite(l_prev)):
            continue
        if direction == "long" and s_prev <= l_prev and s_now > l_now:
            return True
        if direction == "short" and s_prev >= l_prev and s_now < l_now:
            return True
    return False

def smt_penalty(asset: str, k5m: pd.DataFrame) -> Tuple[int, Optional[str]]:
    auto_penalty, auto_note = _auto_smt_penalty(asset, k5m)
    if auto_penalty:
        return auto_penalty, auto_note

    primary = os.path.join(PUBLIC_DIR, asset, "smt.json")
    data = load_json(primary)
    if not data:
        alt = os.path.join(PUBLIC_DIR, "correlation", f"{asset.upper()}_smt.json")
        data = load_json(alt)
    if not data:
        return 0, None
    diverging = bool(data.get("divergence"))
    consecutive = int(data.get("consecutive_bars") or data.get("consecutive_5m_bars") or 0)
    if diverging and consecutive >= SMT_REQUIRED_BARS:
        pair = data.get("pair") or data.get("reference") or "pair"
        direction = data.get("direction") or "divergence"
        note = f"SMT divergencia ({pair}, {direction})"
        return SMT_PENALTY_VALUE, note
    return 0, None

def fib_zone_ok(move_hi, move_lo, price_now,
                low=0.618, high=0.886,
                tol_abs=0.0, tol_frac=0.02) -> bool:
    if move_hi is None or move_lo is None or move_hi == move_lo:
        return False
    length = move_hi - move_lo
    if length == 0:
        return False
    z1_long  = move_lo + low  * length
    z2_long  = move_lo + high * length
    z1_short = move_hi - high * length
    z2_short = move_hi - low  * length
    tol = max(float(tol_abs), abs(length) * float(tol_frac))
    in_long  = min(z1_long,  z2_long ) - tol <= price_now <= max(z1_long,  z2_long ) + tol
    in_short = min(z1_short, z2_short) - tol <= price_now <= max(z1_short, z2_short) + tol
    return in_long or in_short

def bias_from_emas(df: pd.DataFrame) -> str:
    if df.empty: return "neutral"
    c = df["close"]
    e9, e21, e50, e200 = ema(c,9).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1], ema(c,200).iloc[-1]
    last = c.iloc[-1]
    if last > e200 and e50 > e200 and e9 > e21:  return "long"
    if last < e200 and e50 < e200 and e9 < e21:  return "short"
    return "neutral"

def ema_slope_ok(
    df_1h: pd.DataFrame,
    period: int = EMA_SLOPE_PERIOD,
    lookback: int = EMA_SLOPE_LOOKBACK,
    th: float = EMA_SLOPE_TH,
) -> Tuple[bool, float, float]:
    """EMA21 relatív meredekség 1h-n: abs(ema_now - ema_prev)/price_now >= th."""

    if df_1h.empty or len(df_1h) < period + lookback + 1:
        return False, 0.0, 0.0

    c = df_1h["close"]
    e = ema(c, period)
    ema_now = float(e.iloc[-1])
    ema_prev = float(e.iloc[-1 - lookback])
    price_now = float(c.iloc[-1])
    slope_signed = (ema_now - ema_prev) / max(1e-9, price_now)
    rel = abs(slope_signed)
    return (rel >= th), rel, slope_signed


class RegimeClassifier:
    def __init__(
        self,
        adx_period: int = 14,
        ema_period: int = EMA_SLOPE_PERIOD,
        ema_lookback: int = EMA_SLOPE_LOOKBACK,
        ema_threshold: float = EMA_SLOPE_TH,
        adx_trend_threshold: float = ADX_TREND_MIN,
        adx_range_threshold: float = ADX_RANGE_MAX,
    ) -> None:
        self.adx_period = adx_period
        self.ema_period = ema_period
        self.ema_lookback = ema_lookback
        self.ema_threshold = ema_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold

    def classify(self, k5m: pd.DataFrame, k1h: pd.DataFrame) -> Dict[str, Any]:
        """Return regime label + raw metrics based on 5m ADX and 1h EMA slope."""

        if k5m is None or k1h is None or k5m.empty or k1h.empty:
            return {"label": "CHOPPY", "adx": None, "ema_slope": None, "ema_slope_signed": None}

        adx_value = latest_adx(k5m, period=self.adx_period)
        slope_ok, slope_abs, slope_signed = ema_slope_ok(
            k1h,
            period=self.ema_period,
            lookback=self.ema_lookback,
            th=self.ema_threshold,
        )

        label = "CHOPPY"
        if adx_value is not None and np.isfinite(adx_value):
            if adx_value >= self.adx_trend_threshold and slope_abs >= self.ema_threshold:
                label = "TRENDING"
            elif adx_value < self.adx_range_threshold:
                label = "RANGING"
            elif slope_ok:
                label = "CHOPPY"

        return {
            "label": label,
            "adx": None if adx_value is None else float(adx_value),
            "ema_slope": float(slope_abs) if slope_abs is not None else None,
            "ema_slope_signed": float(slope_signed) if slope_signed is not None else None,
        }


def compute_dynamic_tp_profile(
    asset: str,
    atr_series: pd.Series,
    rel_atr: float,
    price: Optional[float],
) -> Dict[str, Any]:
    history = atr_series.dropna().tail(720)
    if history.empty:
        return {
            "core": {"tp1": TP1_R, "tp2": TP2_R, "rr": CORE_RR_MIN.get(asset, CORE_RR_MIN["default"])},
            "momentum": {
                "tp1": TP1_R_MOMENTUM,
                "tp2": TP2_R_MOMENTUM,
                "rr": MOMENTUM_RR_MIN.get(asset, MOMENTUM_RR_MIN["default"]),
            },
            "regime": "neutral",
        }

    current = history.iloc[-1]
    perc20 = history.quantile(0.2)
    perc50 = history.quantile(0.5)
    perc80 = history.quantile(0.8)

    regime = "normal"
    tp1_core = TP1_R
    tp2_core = TP2_R
    rr_core = CORE_RR_MIN.get(asset, CORE_RR_MIN["default"])
    tp1_mom = TP1_R_MOMENTUM
    tp2_mom = TP2_R_MOMENTUM
    rr_mom = MOMENTUM_RR_MIN.get(asset, MOMENTUM_RR_MIN["default"])

    if asset == "GOLD_CFD":
        tp1_core = max(tp1_core, 2.2)
        tp2_core = max(tp2_core, 3.6)
        rr_core = max(rr_core, 2.1)
        tp1_mom = max(tp1_mom, 1.9)
        tp2_mom = max(tp2_mom, 3.0)
        rr_mom = max(rr_mom, 1.75)

    if asset == "NVDA":
        low_rel = float(NVDA_RR_BANDS.get("low_rel_atr") or 0.0)
        high_rel = float(NVDA_RR_BANDS.get("high_rel_atr") or 0.0)
        rr_low = float(NVDA_RR_BANDS.get("rr_low") or rr_core)
        rr_high_core = float(NVDA_RR_BANDS.get("rr_high_core") or rr_core)
        rr_high_mom = float(NVDA_RR_BANDS.get("rr_high_momentum") or rr_mom)
        if np.isfinite(rel_atr):
            if high_rel > 0 and rel_atr >= high_rel:
                rr_core = max(rr_core, rr_high_core)
                rr_mom = max(rr_mom, rr_high_mom)
                tp1_core = max(tp1_core, 1.9)
                tp2_core = max(tp2_core, rr_core + 0.6)
                tp1_mom = max(tp1_mom, 1.6)
                tp2_mom = max(tp2_mom, rr_mom + 0.4)
            elif low_rel > 0 and rel_atr <= low_rel:
                rr_core = max(rr_low, min(rr_core, rr_low))
                rr_mom = max(rr_low, min(rr_mom, rr_low))
                tp1_core = max(tp1_core * 0.95, 1.2)
                tp2_core = max(tp2_core * 0.95, tp1_core + 0.3)
                tp1_mom = max(tp1_mom * 0.95, 1.1)
                tp2_mom = max(tp2_mom * 0.95, tp1_mom + 0.25)

    if current <= perc20:
        regime = "low_vol"
        tp1_core = max(1.2, TP1_R * 0.8)
        tp2_core = max(1.8, TP2_R * 0.8)
        rr_core = max(1.6, rr_core * 0.85)
        tp1_mom = max(1.1, TP1_R_MOMENTUM * 0.85)
        tp2_mom = max(1.6, TP2_R_MOMENTUM * 0.85)
        rr_mom = max(1.3, rr_mom * 0.85)
    elif current >= perc80:
        regime = "high_vol"
        tp1_core = max(tp1_core, TP1_R * 1.15)
        tp2_core = max(tp2_core, TP2_R * 1.2)
        rr_core = max(rr_core, rr_core * 1.05)
        tp1_mom = max(tp1_mom, TP1_R_MOMENTUM * 1.1)
        tp2_mom = max(tp2_mom, TP2_R_MOMENTUM * 1.15)
        rr_mom = max(rr_mom, rr_mom * 1.05)
        if asset == "GOLD_CFD":
            tp1_core = max(tp1_core, 2.4)
            tp2_core = max(tp2_core, 3.8)
            rr_core = max(rr_core, 2.25)
            tp1_mom = max(tp1_mom, 2.0)
            tp2_mom = max(tp2_mom, 3.2)
            rr_mom = max(rr_mom, 1.9)

    if np.isfinite(rel_atr) and price:
        if rel_atr < ATR_LOW_TH_ASSET.get(asset, ATR_LOW_TH_DEFAULT) * 0.8:
            regime = "compressed"
            tp1_core = max(tp1_core * 0.9, 1.1)
            tp2_core = max(tp2_core * 0.9, tp1_core + 0.3)

    return {
        "core": {"tp1": float(tp1_core), "tp2": float(tp2_core), "rr": float(rr_core)},
        "momentum": {"tp1": float(tp1_mom), "tp2": float(tp2_mom), "rr": float(rr_mom)},
        "regime": regime,
        "reference": {
            "current": float(current),
            "q20": float(perc20),
            "q50": float(perc50),
            "q80": float(perc80),
        },
    }


def trading_day_bounds(now: datetime, tz: ZoneInfo = MARKET_TIMEZONE) -> Tuple[datetime, datetime]:
    """Return the UTC bounds of the trading day anchored to ``tz``."""

    local_now = now.astimezone(tz)
    start_local = datetime(local_now.year, local_now.month, local_now.day, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def compute_intraday_profile(
    asset: str,
    k1m: pd.DataFrame,
    price_now: Optional[float],
    atr5: Optional[float],
    now: datetime,
    tz: ZoneInfo = MARKET_TIMEZONE,
) -> Dict[str, Any]:
    """Analyse the current trading day's range structure for intraday context."""

    profile: Dict[str, Any] = {
        "asset": asset,
        "timestamp": to_utc_iso(now),
        "day_start_utc": None,
        "day_open": None,
        "day_high": None,
        "day_low": None,
        "day_range": None,
        "range_vs_atr": None,
        "range_position": None,
        "range_bias": "unknown",
        "range_state": "unknown",
        "range_compression": False,
        "range_expansion": False,
        "range_exhaustion_long": False,
        "range_exhaustion_short": False,
        "range_guard": {"long": False, "short": False},
        "opening_range_high": None,
        "opening_range_low": None,
        "opening_break": "none",
        "opening_range_minutes": OPENING_RANGE_MINUTES,
        "opening_drive_ratio": None,
        "previous_close": None,
        "opening_gap": None,
        "opening_gap_pct": None,
        "opening_gap_direction": "flat",
        "price_reference": safe_float(price_now),
        "elapsed_minutes": None,
        "notes": [],
    }

    if k1m.empty or not isinstance(k1m.index, pd.DatetimeIndex):
        profile["notes"].append("Intraday 1m idősor nem érhető el.")
        return profile

    df = k1m.copy()
    try:
        if df.index.tz is None:
            df = df.tz_localize(timezone.utc)
        else:
            df = df.tz_convert(timezone.utc)
    except Exception:
        df.index = pd.to_datetime(df.index, utc=True)

    day_start_utc, _ = trading_day_bounds(now, tz)
    profile["day_start_utc"] = to_utc_iso(day_start_utc)

    mask = (df.index >= day_start_utc) & (df.index <= now)
    intraday = df.loc[mask]
    if intraday.empty:
        intraday = df.tail(24 * 60)
    if intraday.empty:
        profile["notes"].append("Intraday ablak üres – nincs napi range.")
        return profile

    day_open = safe_float(intraday["open"].iloc[0])
    day_high = safe_float(intraday["high"].max())
    day_low = safe_float(intraday["low"].min())
    last_close = safe_float(intraday["close"].iloc[-1])
    prev_close = None
    prev_slice = df.loc[df.index < day_start_utc]
    if not prev_slice.empty:
        prev_close = safe_float(prev_slice["close"].iloc[-1])
    price_ref = profile["price_reference"]
    if price_ref is None:
        price_ref = last_close
    profile["price_reference"] = price_ref
    profile["previous_close"] = prev_close

    day_range: Optional[float] = None
    if day_high is not None and day_low is not None:
        try:
            rng = float(day_high) - float(day_low)
            if np.isfinite(rng) and rng >= 0:
                day_range = rng
        except (TypeError, ValueError):
            day_range = None

    atr_val = safe_float(atr5) if atr5 is not None else None
    range_vs_atr: Optional[float] = None
    if day_range is not None and atr_val is not None and atr_val > 0:
        range_vs_atr = float(day_range) / float(atr_val)

    range_position: Optional[float] = None
    if (
        day_range is not None
        and day_range > 0
        and price_ref is not None
        and day_low is not None
    ):
        try:
            ratio = (float(price_ref) - float(day_low)) / float(day_range)
            range_position = min(1.0, max(0.0, ratio))
        except (TypeError, ValueError, ZeroDivisionError):
            range_position = None

    opening_end = min(now, day_start_utc + timedelta(minutes=OPENING_RANGE_MINUTES))
    opening_slice = intraday.loc[(intraday.index >= day_start_utc) & (intraday.index <= opening_end)]
    opening_high = safe_float(opening_slice["high"].max()) if not opening_slice.empty else None
    opening_low = safe_float(opening_slice["low"].min()) if not opening_slice.empty else None

    opening_range = None
    if opening_high is not None and opening_low is not None:
        try:
            diff = float(opening_high) - float(opening_low)
            if np.isfinite(diff) and diff >= 0:
                opening_range = diff
        except (TypeError, ValueError):
            opening_range = None

    opening_break = "none"
    tol_up = 1.0002
    tol_down = 0.9998
    if opening_high is not None and day_high is not None and day_high > opening_high * tol_up:
        opening_break = "up"
    if opening_low is not None and day_low is not None and day_low < opening_low * tol_down:
        opening_break = "down" if opening_break == "none" else "both"

    opening_gap = None
    opening_gap_pct = None
    opening_gap_direction = "flat"
    if day_open is not None and prev_close is not None and prev_close != 0:
        try:
            opening_gap = float(day_open) - float(prev_close)
            if np.isfinite(opening_gap):
                opening_gap_pct = opening_gap / float(prev_close) * 100.0
                if opening_gap > 0:
                    opening_gap_direction = "up"
                elif opening_gap < 0:
                    opening_gap_direction = "down"
        except (TypeError, ValueError, ZeroDivisionError):
            opening_gap = None
            opening_gap_pct = None
            opening_gap_direction = "flat"

    opening_drive_ratio = None
    if opening_range and day_range:
        try:
            opening_drive_ratio = min(1.0, max(0.0, opening_range / day_range))
        except (TypeError, ValueError, ZeroDivisionError):
            opening_drive_ratio = None

    range_bias = "unknown"
    if range_position is not None:
        if range_position >= INTRADAY_BALANCE_HIGH:
            range_bias = "upper"
        elif range_position <= INTRADAY_BALANCE_LOW:
            range_bias = "lower"
        else:
            range_bias = "balanced"

    range_state = "normal"
    range_compression = bool(
        range_vs_atr is not None and range_vs_atr <= INTRADAY_COMPRESSION_TH
    )
    range_expansion = bool(
        range_vs_atr is not None and range_vs_atr >= INTRADAY_EXPANSION_TH
    )
    if range_compression:
        range_state = "compression"
    elif range_expansion:
        range_state = "expansion"

    exhaustion_long = False
    exhaustion_short = False
    if range_position is not None:
        lower_threshold = 1.0 - INTRADAY_EXHAUSTION_PCT
        meets_atr = range_vs_atr is None or range_vs_atr >= INTRADAY_ATR_EXHAUSTION
        if range_position >= INTRADAY_EXHAUSTION_PCT and meets_atr:
            exhaustion_long = True
        if range_position <= lower_threshold and meets_atr:
            exhaustion_short = True

    notes: List[str] = []
    if exhaustion_long:
        notes.append("Intraday range felső része telített — ne üldözd a csúcsot.")
    if exhaustion_short:
        notes.append("Intraday range alsó része telített — csak visszapattanásra lépj.")
    if range_compression:
        notes.append("Napi range az ATR 45%-a alatt — valószínű oldalazás.")
    if range_expansion and range_vs_atr is not None and range_vs_atr >= 1.5:
        notes.append("Range expanzió >1.5×ATR — agresszív menedzsment indokolt.")
    if opening_break in {"up", "down", "both"}:
        notes.append(f"Opening range áttörés iránya: {opening_break}.")

    elapsed_minutes = None
    try:
        elapsed = (now - day_start_utc).total_seconds() / 60.0
        if elapsed >= 0:
            elapsed_minutes = int(elapsed)
    except Exception:
        elapsed_minutes = None

    profile.update(
        {
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "day_range": day_range,
            "range_vs_atr": range_vs_atr,
            "range_position": range_position,
            "range_bias": range_bias,
            "range_state": range_state,
            "range_compression": range_compression,
            "range_expansion": range_expansion,
            "range_exhaustion_long": exhaustion_long,
            "range_exhaustion_short": exhaustion_short,
            "range_guard": {"long": exhaustion_long, "short": exhaustion_short},
            "opening_range_high": opening_high,
            "opening_range_low": opening_low,
            "opening_break": opening_break,
            "opening_drive_ratio": opening_drive_ratio,
            "opening_gap": opening_gap,
            "opening_gap_pct": opening_gap_pct,
            "opening_gap_direction": opening_gap_direction,
            "elapsed_minutes": elapsed_minutes,
            "notes": notes,
        }
    )

    return profile

# --- Override bucket helpers --------------------------------------------------


def _ensure_override_bucket(entry_thresholds_meta: Dict[str, Any], key: str) -> Dict[str, Any]:
    bucket = entry_thresholds_meta.get(key)
    if isinstance(bucket, dict):
        return bucket
    new_bucket: Dict[str, Any] = {}
    entry_thresholds_meta[key] = new_bucket
    return new_bucket


def _initialize_asset_overrides(
    entry_thresholds_meta: Dict[str, Any], asset: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    xag_overrides: Dict[str, Any] = {}
    usoil_overrides: Dict[str, Any] = {}
    eurusd_overrides: Dict[str, Any] = {}
    if asset == "XAGUSD":
        xag_overrides = _ensure_override_bucket(entry_thresholds_meta, "xag_overrides")
    if asset == "USOIL":
        usoil_overrides = _ensure_override_bucket(entry_thresholds_meta, "usoil_overrides")
    if asset == "EURUSD":
        eurusd_overrides = _ensure_override_bucket(entry_thresholds_meta, "eurusd_overrides")
    return xag_overrides, usoil_overrides, eurusd_overrides


# ------------------------------ elemzés egy eszközre ---------------------------

def analyze(asset: str) -> Dict[str, Any]:
    outdir = os.path.join(PUBLIC_DIR, asset)
    os.makedirs(outdir, exist_ok=True)

    # 1) Bemenetek
    latency_profile = load_latency_profile(outdir)
    latency_guard_state = load_latency_guard_state(outdir)
    avg_delay: float = float(latency_profile.get("ema_delay", 0.0) or 0.0)
    spot = load_json(os.path.join(outdir, "spot.json")) or {}
    spot_realtime = load_json(os.path.join(outdir, "spot_realtime.json")) or {}
    funding_rate_snapshot = load_funding_snapshot(asset)
    spot_meta = spot if isinstance(spot, dict) else {}
    realtime_stats = spot_realtime.get("statistics") if isinstance(spot_realtime, dict) else {}
    k1m_raw = load_json(os.path.join(outdir, "klines_1m.json"))
    k5m_raw = load_json(os.path.join(outdir, "klines_5m.json"))
    k1h_raw = load_json(os.path.join(outdir, "klines_1h.json"))
    k4h_raw = load_json(os.path.join(outdir, "klines_4h.json"))

    def _ensure_meta(obj: Optional[Any]) -> Dict[str, Any]:
        return obj if isinstance(obj, dict) else {}

    k1m_meta = _ensure_meta(load_json(os.path.join(outdir, "klines_1m_meta.json")))
    k5m_meta = _ensure_meta(load_json(os.path.join(outdir, "klines_5m_meta.json")))
    k1h_meta = _ensure_meta(load_json(os.path.join(outdir, "klines_1h_meta.json")))
    k4h_meta = _ensure_meta(load_json(os.path.join(outdir, "klines_4h_meta.json")))

    k1m, k5m, k1h, k4h = as_df_klines(k1m_raw), as_df_klines(k5m_raw), as_df_klines(k1h_raw), as_df_klines(k4h_raw)

    spot_price = None
    spot_price_reference: Optional[float] = None
    spot_utc = "-"
    spot_retrieved = "-"
    display_spot: Optional[float] = None
    realtime_used = False
    realtime_reason: Optional[str] = None
    realtime_confidence: float = 1.0
    realtime_transport = str(spot_realtime.get("transport") or "http").lower() if isinstance(spot_realtime, dict) else "http"
    asset_entry_profile = _entry_threshold_profile_name(asset)
    schedule_cfg = getattr(settings, "_ENTRY_PROFILE_SCHEDULE", {})
    profile_resolution: Dict[str, Any] = {
        "asset": asset,
        "entry_profile": asset_entry_profile,
        "global_default_profile": os.getenv("ENTRY_THRESHOLD_PROFILE_NAME") or None,
        "schedule_enabled": bool(schedule_cfg),
        "schedule_bucket": None,
        "schedule_selected_profile": None,
        "resolution_notes": [],
    }
    entry_thresholds_meta: Dict[str, Any] = {"profile": asset_entry_profile}
    entry_gate_context_hu: Dict[str, Any] = {}
    dynamic_logic_cfg_raw = settings.DYNAMIC_LOGIC if isinstance(settings.DYNAMIC_LOGIC, dict) else {}
    dynamic_logic_cfg, dynamic_logic_warnings = validate_dynamic_logic_config(
        dynamic_logic_cfg_raw, logger=LOGGER
    )
    if dynamic_logic_warnings:
        entry_thresholds_meta["dynamic_logic_validation"] = {
            "warnings": dynamic_logic_warnings
        }
    p_score_min_base = get_p_score_min(asset)
    p_score_min_local = p_score_min_base
    xag_overrides, usoil_overrides, eurusd_overrides = _initialize_asset_overrides(
        entry_thresholds_meta, asset
    )
    btc_profile_name = _btc_active_profile() if asset == "BTCUSD" else None
    if btc_profile_name:
        entry_thresholds_meta["btc_profile"] = btc_profile_name
    btc_atr_floor_ratio: Optional[float] = None
    btc_atr_floor_passed = False
    now = datetime.now(timezone.utc)
    schedule_bucket: Optional[str] = None
    schedule_profile: Optional[str] = None
    try:
        schedule_bucket = settings.time_of_day_bucket(asset, now)
    except Exception:
        schedule_bucket = None
    resolve_fn = getattr(settings, "_resolve_scheduled_profile", None)
    if callable(resolve_fn):
        try:
            schedule_profile = resolve_fn(asset, now)
        except Exception:
            schedule_profile = None
    if schedule_bucket:
        profile_resolution["schedule_bucket"] = schedule_bucket
    if schedule_profile:
        profile_resolution["schedule_selected_profile"] = schedule_profile
    if schedule_bucket and schedule_profile:
        profile_resolution["resolution_notes"].append(
            f"profile via schedule bucket {schedule_bucket} -> {schedule_profile}"
        )
    asset_leverage = LEVERAGE.get(asset, 1.0)
    spot_max_age = int(SPOT_MAX_AGE_SECONDS.get(asset, SPOT_MAX_AGE_SECONDS["default"]))
    if spot:
        spot_price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
        spot_utc = spot.get("utc") or spot.get("timestamp") or "-"
        spot_retrieved = spot.get("retrieved_at_utc") or spot.get("retrieved") or "-"
        try:
            spot_price_reference = float(spot_price) if spot_price is not None else None
        except (TypeError, ValueError):
            spot_price_reference = None

    spot_ts = parse_utc_timestamp(spot.get("retrieved_at_utc") or spot.get("retrieved") or spot.get("utc")) if isinstance(spot, dict) else None
    spot_latency: Optional[float] = None
    if spot_ts:
        spot_latency = (now - spot_ts).total_seconds()
    freshness_limit = spot_max_age
    staleness_max_age = freshness_limit
    # Allow per-asset overrides embedded in the spot snapshot metadata while keeping
    # the configured ``spot_max_age_seconds`` as the primary guardrail. This prevents
    # overly aggressive data-gap triggering when the upstream feed delivers crypto
    # prices with a slower cadence (e.g., >60s), as observed with BTCUSD.
    if isinstance(spot, dict) and spot.get("freshness_limit_seconds"):
        try:
            freshness_limit = int(spot["freshness_limit_seconds"])
        except (TypeError, ValueError):
            LOGGER.debug("Invalid freshness_limit_seconds in spot payload", exc_info=True)
    if spot_latency is not None and spot_latency > freshness_limit:
        reason = f"Data Stale (Lat: {int(spot_latency)} sec)"
        diagnostics = {
            "freshness_guard": {
                "latency_seconds": spot_latency,
                "limit_seconds": freshness_limit,
            }
        }
        return build_data_gap_signal(
            asset,
            spot_price_reference,
            spot_utc,
            spot_retrieved,
            asset_leverage,
            [reason],
            display_spot,
            diagnostics,
        )

    rt_price = None
    rt_utc = None
    if spot_realtime:
        rt_price = spot_realtime.get("price") if spot_realtime.get("price") is not None else spot_realtime.get("price_usd")
        rt_utc = spot_realtime.get("utc") or spot_realtime.get("timestamp") or spot_realtime.get("retrieved_at_utc")

    realtime_ttl = max(0, SPOT_REALTIME_TTL_SECONDS)
    realtime_expired_reason: Optional[str] = None
    realtime_age: Optional[float] = None
    if spot_realtime:
        if rt_ts := parse_utc_timestamp(rt_utc):
            realtime_age = (now - rt_ts).total_seconds()
            if realtime_ttl and realtime_age > realtime_ttl:
                realtime_expired_reason = "ttl_expired"
        elif rt_price is None:
            realtime_expired_reason = "missing_timestamp"
        if not realtime_expired_reason and spot_realtime.get("forced") and not spot_realtime.get("ok"):
            realtime_expired_reason = spot_realtime.get("stale_reason") or "forced_snapshot_stale"
            if spot_realtime.get("utc"):
                rt_ts = parse_utc_timestamp(spot_realtime.get("utc"))
                if rt_ts:
                    realtime_age = (now - rt_ts).total_seconds()
    if realtime_expired_reason:
        expire_meta = {
            "used": False,
            "reason": realtime_expired_reason,
            "max_age_seconds": float(realtime_ttl or 0),
        }
        if realtime_age is not None and realtime_age >= 0:
            expire_meta["age_seconds"] = float(realtime_age)
        if isinstance(spot_realtime, dict) and spot_realtime.get("retrieved_at_utc"):
            expire_meta["retrieved_at_utc"] = spot_realtime.get("retrieved_at_utc")
        expire_meta = _normalize_realtime_meta(
            expire_meta,
            max_age_seconds=realtime_ttl or spot_max_age,
            now_utc=now,
        )
        entry_thresholds_meta["spot_realtime_expired"] = expire_meta
        refresh_state = _handle_spot_realtime_staleness(
            asset,
            expire_meta,
            now,
            record_latency_alert,
            LOGGER,
        )
        if refresh_state:
            entry_thresholds_meta["spot_realtime_refresh"] = refresh_state
        realtime_path = os.path.join(outdir, "spot_realtime.json")
        try:
            os.remove(realtime_path)
        except FileNotFoundError:
            pass
        LOGGER.warning(
            "Realtime spot snapshot törölve a TTL miatt",
            extra={
                "asset": asset,
                "reason": realtime_expired_reason,
                "age_seconds": expire_meta.get("age_seconds"),
                "ttl_seconds": realtime_ttl,
            },
        )
        spot_realtime = {}
        rt_price = None
        rt_utc = None

    spot_ts_existing = parse_utc_timestamp(spot_utc)
    rt_ts = parse_utc_timestamp(rt_utc)
    use_realtime = False
    rt_decision_meta: Dict[str, Any] = {}
    if rt_price is not None or rt_ts is not None:
        use_realtime, rt_decision_meta = _should_use_realtime_spot(
            rt_price, rt_ts, spot_ts_existing, now, spot_max_age
        )
    if use_realtime:
        spot_price = rt_price
        display_spot = safe_float(rt_price)
        if rt_ts:
            spot_utc = to_utc_iso(rt_ts)
        if spot_realtime.get("retrieved_at_utc"):
            spot_retrieved = spot_realtime.get("retrieved_at_utc")
        realtime_used = True
        realtime_reason = "Realtime spot feed override"
    elif rt_decision_meta:
        rt_meta_payload = dict(rt_decision_meta, used=False)
        rt_meta_payload.setdefault("max_age_seconds", spot_max_age)
        rt_meta_payload = _normalize_realtime_meta(
            rt_meta_payload, max_age_seconds=spot_max_age, now_utc=now
        )
        entry_thresholds_meta["spot_realtime_ignored"] = rt_meta_payload
        refresh_state = _handle_spot_realtime_staleness(
            asset,
            rt_meta_payload,
            now,
            record_latency_alert,
            LOGGER,
        )
        if refresh_state:
            entry_thresholds_meta["spot_realtime_refresh"] = refresh_state
    session_ok_flag, session_meta = session_state(asset, now=now)
    LOGGER.info(
        "Session/entry profil",
        extra={
            "asset": asset,
            "session_profil": session_meta.get("status_profile") if isinstance(session_meta, dict) else None,
            "entry_kuszob_profil": asset_entry_profile,
        },
    )
    if asset == "BTCUSD":
        news_lockout_active = False
        news_reason = None
    else:
        news_lockout_active, news_reason = evaluate_news_lockout(asset, now)
    entry_thresholds_meta["news_lockout"] = bool(news_lockout_active)
    if news_lockout_active:
        session_ok_flag = False
        if isinstance(session_meta, dict):
            session_meta.setdefault("news_lockout", True)
            notes_list = session_meta.setdefault("notes", []) if isinstance(session_meta.get("notes"), list) else session_meta.setdefault("notes", [])
            if news_reason and news_reason not in notes_list:
                notes_list.append(f"News lockout: {news_reason}")

    spot_ts_primary = parse_utc_timestamp(spot_utc)
    spot_ts_fallback = parse_utc_timestamp(spot_retrieved)
    spot_ts = spot_ts_primary or spot_ts_fallback
    spot_latency_sec: Optional[int] = None
    spot_stale_reason: Optional[str] = None
    spot_max_age_limit = spot_max_age
    relaxed_spot_reason: Optional[str] = None
    if spot_ts:
        delta = now - spot_ts
        if delta.total_seconds() < 0:
            spot_latency_sec = 0
        else:
            spot_latency_sec = int(delta.total_seconds())
        if spot_latency_sec is not None and spot_latency_sec > spot_max_age_limit:
            age_min = spot_latency_sec // 60
            limit_min = spot_max_age_limit // 60 if spot_max_age_limit else 0
            spot_stale_reason = f"Spot data stale: {age_min} min behind (limit {limit_min} min)"
    elif spot_price is not None:
        spot_stale_reason = "Spot timestamp missing"
    else:
        spot_stale_reason = "Spot data missing"

    if spot_stale_reason and not should_enforce_stale_frame(asset, "spot", session_meta):
        relaxed_spot_reason = spot_stale_reason
        spot_stale_reason = None

    update_latency_profile(outdir, spot_latency_sec)
    latency_profile = load_latency_profile(outdir)
    avg_delay = float(latency_profile.get("ema_delay", spot_latency_sec or avg_delay) or 0.0)
    if realtime_used:
        base_conf = 0.75 if avg_delay else 0.8
        if realtime_transport == "websocket":
            base_conf = 0.9 if avg_delay else 0.95
        if avg_delay and spot_latency_sec is not None:
            base_conf = spot_latency_sec / max(avg_delay, 1.0)
        realtime_confidence = float(min(1.0, max(0.2, base_conf)))
    else:
        if avg_delay and spot_latency_sec is not None and spot_latency_sec > 0:
            ratio = avg_delay / float(spot_latency_sec)
            realtime_confidence = float(min(1.0, max(0.2, ratio)))
        else:
            realtime_confidence = 1.0

    if isinstance(realtime_stats, dict):
        latency_avg = realtime_stats.get("latency_avg")
        samples = realtime_stats.get("samples")
        latency_limit = 90.0 if realtime_transport == "websocket" else 120.0
        if isinstance(latency_avg, (int, float)) and latency_avg >= 0:
            realtime_confidence = float(
                min(realtime_confidence, max(0.2, 1.0 - float(latency_avg) / latency_limit))
            )
        if realtime_transport == "websocket" and isinstance(samples, (int, float)) and samples >= 5:
            realtime_confidence = float(min(1.0, realtime_confidence + 0.05))

    display_spot = safe_float(spot_price)
    k1m_closed = ensure_closed_candles(k1m, now)
    k5m_closed = ensure_closed_candles(k5m, now)
    k1h_closed = ensure_closed_candles(k1h, now)
    k4h_closed = ensure_closed_candles(k4h, now)

    tick_order_flow = load_tick_order_flow(asset, outdir)
    order_flow_metrics = compute_order_flow_metrics(k1m_closed, k5m_closed, tick_order_flow)

    order_flow_imbalance: Optional[float] = None
    order_flow_pressure: Optional[float] = None

    if isinstance(order_flow_metrics, dict):
        order_flow_imbalance = order_flow_metrics.get("imbalance")
        order_flow_pressure = order_flow_metrics.get("pressure")

    flow_data_available = False
    try:
        flow_data_available = bool(
            order_flow_imbalance is not None
            or (order_flow_pressure is not None and abs(float(order_flow_pressure)) > 0.0)
        )
    except (TypeError, ValueError):
        flow_data_available = order_flow_imbalance is not None

    ofi_zscore = None
    if isinstance(order_flow_metrics, dict):
        ofi_zscore = order_flow_metrics.get("imbalance_z")
        try:
            if ofi_zscore is not None and not np.isfinite(ofi_zscore):
                ofi_zscore = None
        except Exception:
            ofi_zscore = None
        if asset == "BTCUSD":
            lookback = int(BTC_OFI_Z.get("lookback_bars") or 0)
            if lookback > 0:
                btc_ofi_val = compute_ofi_zscore(k1m_closed, lookback)
                if btc_ofi_val is not None and np.isfinite(btc_ofi_val):
                    ofi_zscore = float(btc_ofi_val)
                    order_flow_metrics["imbalance_z"] = ofi_zscore
            entry_thresholds_meta["btc_ofi_thresholds"] = {
                "trigger": BTC_OFI_Z["trigger"],
                "strong": BTC_OFI_Z["strong"],
                "weakening": BTC_OFI_Z["weakening"],
            }
    if asset == "BTCUSD":
        momentum_state = globals().setdefault("_BTC_MOMENTUM_RUNTIME", {})
        if ofi_zscore is not None and np.isfinite(ofi_zscore):
            momentum_state["ofi_z"] = float(ofi_zscore)
        elif isinstance(momentum_state, dict):
            momentum_state.pop("ofi_z", None)
    ofi_available = ofi_zscore is not None

    if asset == "BTCUSD" and ofi_zscore is not None:
        state = "neutral"
        if ofi_zscore >= BTC_OFI_Z["strong"]:
            state = "strong"
        elif ofi_zscore <= BTC_OFI_Z["weakening"]:
            state = "weakening"
        entry_thresholds_meta["btc_ofi_state"] = {
            "z": float(ofi_zscore),
            "state": state,
        }

    anchor_state = current_anchor_state()
    anchor_record = anchor_state.get(asset.upper()) if isinstance(anchor_state, dict) else None
    anchor_bias = None
    anchor_timestamp = None
    anchor_price_state: Optional[float] = None
    anchor_prev_p: Optional[float] = None
    anchor_drift_state: Optional[str] = None
    anchor_drift_score: Optional[float] = None
    if isinstance(anchor_record, dict):
        side_raw = (anchor_record.get("side") or "").lower()
        if side_raw == "buy":
            anchor_bias = "long"
        elif side_raw == "sell":
            anchor_bias = "short"
        anchor_timestamp = anchor_record.get("timestamp")
        anchor_price_state = safe_float(anchor_record.get("price"))
        anchor_prev_p = safe_float(anchor_record.get("p_score"))
        anchor_drift_state = anchor_record.get("drift_state")
        anchor_drift_score = safe_float(anchor_record.get("drift_score"))

    last5_close: Optional[float] = None
    last5_closed_ts = df_last_timestamp(k5m_closed)
    if not k5m_closed.empty:
        try:
            last5_close = float(k5m_closed["close"].iloc[-1])
            if not np.isfinite(last5_close):
                last5_close = None
        except Exception:
            last5_close = None

    spot_issue_initial: Optional[str] = relaxed_spot_reason
    if spot_price is None:
        spot_issue_initial = "Spot price missing"
    elif spot_stale_reason:
        spot_issue_initial = spot_stale_reason

    spot_source = str(spot_meta.get("source") or ("quote_realtime" if realtime_used else "quote"))
    spot_fallback_used = bool(spot_meta.get("fallback_used"))
    spot_latency_notes: List[str] = []
    fallback_latency: Optional[int] = spot_latency_sec if spot_fallback_used else None
    if (spot_price is None or spot_stale_reason) and last5_close is not None:
        spot_fallback_used = True
        spot_source = "kline_5m_close"
        spot_price = last5_close
        display_spot = safe_float(last5_close)
        if last5_closed_ts:
            spot_utc = to_utc_iso(last5_closed_ts)
            spot_ts = last5_closed_ts
        spot_retrieved = to_utc_iso(now)
        if spot_ts:
            delta = now - spot_ts
            fallback_latency = 0 if delta.total_seconds() < 0 else int(delta.total_seconds())
            spot_latency_sec = fallback_latency
        else:
            spot_latency_sec = None
        if fallback_latency is not None and fallback_latency > spot_max_age:
            limit_min = spot_max_age // 60
            delay_min = fallback_latency // 60
            spot_stale_reason = (
                f"Spot fallback stale: {delay_min} min behind (limit {limit_min} min)"
            )
        elif fallback_latency is not None:
            spot_stale_reason = None
        else:
            spot_stale_reason = spot_stale_reason or "Spot fallback timestamp missing"

    if isinstance(realtime_stats, dict):
        guard_candidates = (
            display_spot,
            spot_price_reference,
            last5_close,
            safe_float(rt_price) if rt_price is not None else None,
        )
        guarded_stats = _guard_realtime_price_stats(asset, realtime_stats, guard_candidates)
        if guarded_stats is not realtime_stats:
            realtime_stats = guarded_stats
            guard_meta = realtime_stats.get("price_guard")
            if isinstance(guard_meta, dict) and guard_meta.get("adjusted"):
                adjusted_keys = ", ".join(str(key) for key in guard_meta.get("adjusted", []))
                note = "spot: realtime stat outlier guard aktiv"
                if adjusted_keys:
                    note += f" ({adjusted_keys})"
                spot_latency_notes.append(note)

    if spot_fallback_used:
        if spot_issue_initial:
            note = f"spot: {spot_issue_initial.lower()} — 5m zárt gyertya árát használjuk"
        else:
            note = "spot: 5m zárt gyertya árát használjuk"
        if fallback_latency is not None:
            note += f" ({fallback_latency // 60} perc késés)"
        spot_latency_notes.append(note)
        realtime_confidence = min(realtime_confidence, 0.4)
    elif relaxed_spot_reason:
        spot_latency_notes.append(
            f"spot: {relaxed_spot_reason.lower()} — session zárva miatt engedve"
        )
    elif realtime_used and realtime_reason:
        spot_latency_notes.append("spot: realtime feed aktív")
    elif isinstance(spot_realtime, dict) and spot_realtime.get("forced"):
        force_note = "spot: realtime mintavétel kényszerítve"
        if spot_realtime.get("force_reason"):
            force_note += f" ({spot_realtime.get('force_reason')})"
        spot_latency_notes.append(force_note)

    spread_abs = _resolve_spread_for_entry(
        spot_realtime,
        spot_price_reference,
        safe_float(rt_price) if rt_price is not None else None,
        use_realtime,
    )

    analysis_now = now

    intervention_summary: Optional[Dict[str, Any]] = None
    intervention_band: Optional[str] = None
    sentiment_signal = None
    sentiment_applied_points: Optional[float] = None
    sentiment_normalized: Optional[float] = None
    if asset == "BTCUSD":
        intervention_config = load_intervention_config(outdir)
        news_flag = load_intervention_news_flag(outdir, intervention_config)
        raw_sentiment = load_sentiment(asset, Path(outdir))
        sentiment_signal = None
        if isinstance(raw_sentiment, SentimentSignal):
            sentiment_signal = raw_sentiment
        elif isinstance(raw_sentiment, (list, tuple)) and raw_sentiment:
            candidate = raw_sentiment[0]
            try:
                if isinstance(candidate, SentimentSignal):
                    sentiment_signal = candidate
                else:
                    sentiment_signal = SentimentSignal(*raw_sentiment)
            except Exception:
                sentiment_signal = None
            finally:
                LOGGER.warning(
                    "btc_sentiment_tuple", extra={"asset": asset, "raw_type": type(raw_sentiment).__name__}
                )
        else:
            sentiment_signal = None
        irs_value, irs_band, irs_metrics = compute_btcusd_intraday_watch(
            k1m_closed,
            k5m_closed,
            analysis_now,
            intervention_config,
            news_flag=news_flag,
            sentiment_signal=sentiment_signal,
        )
        since_utc = update_intervention_state(outdir, irs_band, analysis_now)
        iw_reasons = build_intervention_reasons(irs_metrics, intervention_config)
        policy = build_intervention_policy(irs_band, irs_metrics, intervention_config)
        score_breakdown = {
            "speed": irs_metrics.get("components", {}).get("speed", 0.0),
            "volatility": irs_metrics.get("components", {}).get("volatility", 0.0),
            "range": irs_metrics.get("components", {}).get("range", 0.0),
            "comms": irs_metrics.get("components", {}).get("comms", 0.0),
        }
        intervention_summary = {
            "irs": irs_value,
            "band": irs_band,
            "since_utc": since_utc,
            "updated_utc": to_utc_iso(analysis_now),
            "reasons": iw_reasons,
            "policy": policy,
            "metrics": irs_metrics,
            "score_breakdown": score_breakdown,
            "news_flag": news_flag,
        }
        sentiment_applied_points = irs_metrics.get("comms_from_sentiment", 0.0)
        sentiment_normalized = irs_metrics.get("sentiment_score")
        intervention_band = irs_band
        if sentiment_signal:
            intervention_summary["sentiment"] = {
                "score": sentiment_signal.score,
                "bias": sentiment_signal.bias,
                "headline": sentiment_signal.headline,
                "expires_at": (
                    sentiment_signal.expires_at.isoformat()
                    if sentiment_signal.expires_at
                    else None
                ),
                "severity": sentiment_signal.effective_severity,
                "applied_points": sentiment_applied_points,
                "normalized_score": irs_metrics.get("sentiment_score"),
            }
        save_intervention_asset_summary(outdir, intervention_summary)

    expected_delays = {"k1m": 180, "k5m": 300, "k1h": 3600, "k4h": 4*3600}
    tf_meta: Dict[str, Dict[str, Any]] = {}
    latency_flags: List[str] = []
    latency_flags.extend(spot_latency_notes)
    stale_timeframes: Dict[str, bool] = {key: False for key in expected_delays}
    latency_by_frame: Dict[str, Optional[int]] = {key: None for key in expected_delays}
    if spot_meta.get("freshness_violation") and spot_latency_sec is not None:
        limit_sec = spot_meta.get("freshness_limit_seconds") or spot_max_age
        limit_min = int(limit_sec // 60) if limit_sec else 0
        delay_min = spot_latency_sec // 60
        msg = f"spot: trading fetch latency {delay_min} perc (limit {limit_min} perc)"
        if msg not in latency_flags:
            latency_flags.append(msg)
    timeframe_meta_lookup = {
        "k1m": k1m_meta,
        "k5m": k5m_meta,
        "k1h": k1h_meta,
        "k4h": k4h_meta,
    }
    tf_inputs = [
        ("k1m", k1m, k1m_closed, os.path.join(outdir, "klines_1m.json")),
        ("k5m", k5m, k5m_closed, os.path.join(outdir, "klines_5m.json")),
        ("k1h", k1h, k1h_closed, os.path.join(outdir, "klines_1h.json")),
        ("k4h", k4h, k4h_closed, os.path.join(outdir, "klines_4h.json")),
    ]
    latency_guard_seconds: Dict[str, Optional[int]] = dict(latency_by_frame)
    for key, df_full, df_closed, path in tf_inputs:
        meta_info = timeframe_meta_lookup.get(key, {})
        last_raw = df_last_timestamp(df_full)
        last_closed = df_last_timestamp(df_closed)
        latency_sec: Optional[int] = None
        if last_closed:
            latency_sec = int((now - last_closed).total_seconds())
            expected = expected_delays.get(key, 0)
            if expected and latency_sec > expected + 240:
                latency_flags.append(f"{key}: utolsó zárt gyertya {latency_sec//60} perc késésben van")
            tol = TF_STALE_TOLERANCE.get(key, 0)
            if expected and tol and latency_sec > expected + tol:
                stale_timeframes[key] = True
                delay_min = latency_sec // 60
                flag_msg = f"{key}: jelzés korlátozva {delay_min} perc késés miatt"
                if flag_msg not in latency_flags:
                    latency_flags.append(flag_msg)
        latency_by_frame[key] = latency_sec
        guard_latency = latency_sec
        retrieved_iso = None
        if isinstance(meta_info, dict):
            retrieved_iso = (
                meta_info.get("retrieved_at_utc")
                or meta_info.get("retrieved_at")
                or meta_info.get("retrieved")
            )
        guard_adjustment: Optional[int] = None
        retrieved_ts = parse_utc_timestamp(retrieved_iso) if retrieved_iso else None
        if guard_latency is not None and retrieved_ts:
            adjustment = int(max((now - retrieved_ts).total_seconds(), 0.0))
            guard_adjustment = adjustment
            guard_latency = max(0, guard_latency - adjustment)
        latency_guard_seconds[key] = guard_latency
        tf_meta[key] = {
            "last_raw_utc": to_utc_iso(last_raw) if last_raw else None,
            "last_closed_utc": to_utc_iso(last_closed) if last_closed else None,
            "latency_seconds": latency_sec,
            "expected_max_delay_seconds": expected_delays.get(key),
            "source_mtime_utc": file_mtime(path),
            "stale_for_signals": stale_timeframes.get(key, False),
            "freshness_limit_seconds": meta_info.get("freshness_limit_seconds"),
            "freshness_violation": bool(meta_info.get("freshness_violation")),
            "freshness_retries": meta_info.get("freshness_retries"),
            "used_symbol": meta_info.get("used_symbol"),
            "used_exchange": meta_info.get("used_exchange"),
        }
        if retrieved_ts:
            tf_meta[key]["retrieved_at_utc"] = to_utc_iso(retrieved_ts)
        elif retrieved_iso:
            tf_meta[key]["retrieved_at_utc"] = retrieved_iso
        if guard_latency is not None:
            tf_meta[key]["latency_guard_seconds"] = guard_latency
        if guard_adjustment is not None:
            tf_meta[key]["latency_guard_adjustment_seconds"] = guard_adjustment
        if meta_info.get("fallback_previous_payload"):
            tf_meta[key]["fallback_previous_payload"] = True
        fallback_reason = meta_info.get("fallback_reason")
        if fallback_reason:
            tf_meta[key]["fallback_reason"] = fallback_reason
            note_msg = f"{key}: {fallback_reason}"
            if note_msg not in latency_flags:
                latency_flags.append(note_msg)
        error_message = meta_info.get("error") or meta_info.get("message")
        if error_message:
            tf_meta[key]["error"] = error_message
        error_code = meta_info.get("error_code")
        if error_code is not None:
            tf_meta[key]["error_code"] = error_code

    tf_meta["spot"] = {
        "last_raw_utc": to_utc_iso(spot_ts) if spot_ts else None,
        "latency_seconds": spot_latency_sec,
        "expected_max_delay_seconds": spot_max_age,
        "source_mtime_utc": file_mtime(os.path.join(outdir, "spot.json")),
        "source": spot_source,
        "fallback_used": spot_fallback_used,
        "realtime_override": realtime_used,
        "freshness_limit_seconds": spot_meta.get("freshness_limit_seconds") or spot_max_age,
        "freshness_retries": spot_meta.get("freshness_retries"),
        "freshness_violation": bool(spot_stale_reason),
        "freshness_violation_meta": bool(spot_meta.get("freshness_violation")),
        "used_symbol": spot_meta.get("used_symbol"),
        "used_exchange": spot_meta.get("used_exchange"),
    }
    if spot_issue_initial:
        tf_meta["spot"]["original_issue"] = spot_issue_initial
    if isinstance(spot_realtime, dict):
        if spot_realtime.get("forced"):
            tf_meta["spot"]["realtime_forced"] = True
        if spot_realtime.get("force_reason"):
            tf_meta["spot"]["realtime_force_reason"] = spot_realtime.get("force_reason")
    if spot_stale_reason and spot_latency_sec is not None and spot_latency_sec > spot_max_age:
        if spot_latency_sec >= 3600:
            age_hours = spot_latency_sec // 3600
            latency_flags.append(f"spot: utolsó adat {age_hours} óra késésben van")
        else:
            latency_flags.append(f"spot: utolsó adat {spot_latency_sec//60} perc késésben van")

    source_files = {
        "spot.json": file_mtime(os.path.join(outdir, "spot.json")),
        "klines_1m.json": tf_meta["k1m"].get("source_mtime_utc"),
        "klines_5m.json": tf_meta["k5m"].get("source_mtime_utc"),
        "klines_1h.json": tf_meta["k1h"].get("source_mtime_utc"),
        "klines_4h.json": tf_meta["k4h"].get("source_mtime_utc"),
    }

    reasons: List[str] = []
    gate_skips: List[str] = []

    guard_status = _latency_guard_status(
        asset,
        latency_guard_seconds,
        DATA_LATENCY_GUARD,
        profile=asset_entry_profile,
    )
    latency_penalty = 0.0
    latency_relax_meta: Optional[Dict[str, Any]] = None
    latency_relax_cfg = (
        dynamic_logic_cfg.get("latency_relaxation") if isinstance(dynamic_logic_cfg, dict) else {}
    )

    strict_limit_seconds: Optional[float] = None
    if isinstance(latency_relax_cfg, dict):
        profiles_cfg = latency_relax_cfg.get("profiles")
        strict_profile = profiles_cfg.get("strict") if isinstance(profiles_cfg, dict) else None
        strict_limit_seconds = (
            strict_profile.get("limit") if isinstance(strict_profile, dict) else None
        )

    latency_penalty, latency_relax_meta, guard_status = apply_latency_relaxation(
        asset,
        guard_status,
        latency_relax_cfg,
        profile=asset_entry_profile,
        latency_seconds=latency_guard_seconds.get("k1m"),
        strict_limit_seconds=strict_limit_seconds,
    )
    if latency_relax_meta:
        entry_thresholds_meta["latency_relaxation"] = latency_relax_meta
        if latency_relax_meta.get("mode") == "penalized":
            try:
                age_minutes = int((latency_relax_meta.get("age_seconds") or 0) // 60)
            except Exception:
                age_minutes = None
            relax_note = "Latency guard lazítva"
            if latency_relax_meta.get("profile"):
                relax_note += f" ({latency_relax_meta['profile']})"
            if age_minutes is not None:
                relax_note += f" — {age_minutes} perc késés"
            if latency_penalty:
                relax_note += f" ({-latency_penalty:+.1f})"
            latency_flags.append(relax_note)
            reasons.append(
                "Relaxed latency guard: belépés engedélyezve kiterjesztett késleltetéssel"
                + (f" (−{latency_penalty:.1f} P-score)" if latency_penalty else "")
            )
            entry_thresholds_meta["latency_relaxation_used"] = True

    if guard_status:
        k1m_meta = tf_meta.setdefault("k1m", {})
        k1m_meta["latency_guard_triggered"] = True
        k1m_meta["latency_guard_age_seconds"] = guard_status["age_seconds"]
        k1m_meta["latency_guard_limit_seconds"] = guard_status["limit_seconds"]
        guard_minutes = guard_status["age_seconds"] // 60
        guard_limit_minutes = guard_status["limit_seconds"] // 60
        guard_note = (
            f"{guard_status['feed']}: latency guard {guard_minutes} perc késés (limit {guard_limit_minutes} perc)"
        )
        if guard_note not in latency_flags:
            latency_flags.append(guard_note)
        ts_format = "%Y-%m-%d %H:%M:%S"
        triggered_utc = analysis_now.strftime(ts_format)
        triggered_cet = analysis_now.astimezone(LOCAL_TZ).strftime(ts_format)
        guard_meta = dict(guard_status)
        guard_meta.update({
            "triggered_utc": triggered_utc,
            "triggered_cet": triggered_cet,
            "mode": "block_trade",
            "profile": asset_entry_profile,
        })
        entry_thresholds_meta["latency_guard"] = guard_meta
        log_payload = dict(guard_meta, asset=asset)
        log_payload["latency_guard"] = guard_meta
        log_payload["action"] = guard_meta["mode"]
        LOGGER.warning("Latency guard aktiválva", extra=log_payload)
        save_latency_guard_state(outdir, dict(guard_meta, active=True))
        try:
            record_latency_alert(
                asset,
                guard_status["feed"],
                f"{asset} {guard_status['feed']} késés {guard_minutes} perc — latency guard aktív",
                metadata=guard_meta,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Latency guard értesítés hibás: %s", exc)
        guard_reasons = [
            "Critical data latency — belépés tiltva",
            guard_note,
        ]
        diag_payload = diagnostics_payload(tf_meta, source_files, latency_flags)
        msg = build_data_gap_signal(
            asset,
            spot_price,
            spot_utc,
            spot_retrieved,
            LEVERAGE.get(asset, 2.0),
            guard_reasons,
            display_spot,
            diag_payload,
            session_meta=session_meta,
        )
        if realtime_used and realtime_reason:
            msg.setdefault("reasons", []).append(realtime_reason)
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg
    else:
        _log_latency_guard_recovery(
            asset,
            outdir,
            latency_guard_state,
            analysis_now,
            asset_entry_profile,
            entry_thresholds_meta,
        )

    diag_factory = lambda: diagnostics_payload(tf_meta, source_files, latency_flags)

    critical_flags, critical_reasons = classify_critical_staleness(
        asset,
        stale_timeframes,
        latency_by_frame,
        session_meta,
    )
    for key, flag in critical_flags.items():
        if key in tf_meta:
            tf_meta[key]["critical_stale"] = flag

    spot_critical = bool(spot_stale_reason or spot_price is None)
    (
        core_data_ok,
        precision_data_ok,
        precision_disabled_due_to_data_gap,
        critical_reason_map,
    ) = compute_data_integrity_status(spot_critical, critical_flags, latency_by_frame)

    core_data_gap = not core_data_ok
    other_critical_blocks = critical_flags.get("k1h") or critical_flags.get("k4h")
    if core_data_gap or other_critical_blocks:
        reasons_payload = ["Critical data latency — belépés tiltva"]
        if spot_critical and spot_stale_reason:
            reasons_payload.append(spot_stale_reason)
        for key in ("k5m", "k1h", "k4h", "k1m"):
            if critical_flags.get(key):
                reason = critical_reason_map.get(key)
                if reason:
                    reasons_payload.append(reason)
        msg = build_data_gap_signal(
            asset,
            spot_price,
            spot_utc,
            spot_retrieved,
            LEVERAGE.get(asset, 2.0),
            reasons_payload,
            display_spot,
            diag_factory(),
            session_meta=session_meta,
        )
        if realtime_used and realtime_reason:
            msg.setdefault("reasons", []).append(realtime_reason)
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    data_gap_reasons: List[str] = []
    if spot_price is None:
        data_gap_reasons.append("Missing spot price")
    elif spot_stale_reason:
        data_gap_reasons.append(spot_stale_reason)

    if k5m.empty or k1h.empty or k4h.empty:
        missing_frames = []
        if k5m.empty:
            missing_frames.append("k5m")
        if k1h.empty:
            missing_frames.append("k1h")
        if k4h.empty:
            missing_frames.append("k4h")
        data_gap_reasons.append("Missing candles: " + ", ".join(missing_frames))

    if data_gap_reasons:
        msg = build_data_gap_signal(
            asset,
            spot_price,
            spot_utc,
            spot_retrieved,
            LEVERAGE.get(asset, 2.0),
            data_gap_reasons,
            display_spot,
            diag_factory(),
            session_meta=session_meta,
        )
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    gates_mode_override: Optional[str] = None
    precision_data_gap_reason: Optional[str] = None
    if precision_disabled_due_to_data_gap:
        flow_data_available = False
        ofi_available = False
        precision_data_gap_reason = (
            "k1m stale — precision/momentum/flow modulok letiltva, core (spot+k5m) fut tovább"
        )
        gates_mode_override = "core_only"
        if precision_data_gap_reason not in reasons:
            reasons.append(precision_data_gap_reason)

    required_closed = {
        "k5m": k5m_closed,
        "k1h": k1h_closed,
        "k4h": k4h_closed,
    }
    missing_closed = [name for name, df in required_closed.items() if df.empty]
    if missing_closed:
        msg = build_data_gap_signal(
            asset,
            spot_price,
            spot_utc,
            spot_retrieved,
            LEVERAGE.get(asset, 2.0),
            [
                "Insufficient closed data ({} missing)".format(
                    ", ".join(sorted(missing_closed))
                )
            ],
            display_spot,
            diag_factory(),
            session_meta=session_meta,
        )
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    if last5_close is None:
        msg = build_data_gap_signal(
            asset,
            spot_price,
            spot_utc,
            spot_retrieved,
            LEVERAGE.get(asset, 2.0),
            ["Insufficient closed data (5m close)"],
            display_spot,
            diag_factory(),
            session_meta=session_meta,
        )
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    price_for_calc = spot_price if spot_price is not None else last5_close
    if price_for_calc is None:
        price_for_calc = last5_close

    if display_spot is None and price_for_calc is not None and np.isfinite(price_for_calc):
        display_spot = price_for_calc

    # reasons, gate_skips already initialized above
    # 2) Bias 4H→1H (zárt 1h/4h)
    raw_bias4h = bias_from_emas(k4h_closed)
    raw_bias1h = bias_from_emas(k1h_closed)
    bias4h = "neutral" if stale_timeframes.get("k4h") else raw_bias4h
    bias1h = "neutral" if stale_timeframes.get("k1h") else raw_bias1h
    trend_bias = (
        "long"
        if (bias4h == "long" and bias1h != "short")
        else (
            "short"
            if (bias4h == "short" and bias1h != "long")
            else "neutral"
        )
    )

    if asset in {"EURUSD", "BTCUSD", "NVDA"}:
        if bias4h != bias1h or bias1h not in {"long", "short"}:
            trend_bias = "neutral"

    slope_threshold = EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT)
    adx_trend_threshold = ADX_TREND_MIN
    if asset == "BTCUSD":
        adx_trend_threshold = max(float(adx_trend_threshold), BTC_ADX_TREND_MIN)
    classifier = RegimeClassifier(
        ema_threshold=slope_threshold,
        adx_trend_threshold=adx_trend_threshold,
        adx_range_threshold=ADX_RANGE_MAX,
    )
    regime_snapshot = classifier.classify(k5m_closed, k1h_closed)
    try:
        regime_adx = float(regime_snapshot.get("adx"))
    except (TypeError, ValueError):
        regime_adx = None
    regime_val = float(regime_snapshot.get("ema_slope") or 0.0)
    regime_slope_signed = float(regime_snapshot.get("ema_slope_signed") or 0.0)
    regime_ok = regime_val >= slope_threshold
    slope_sign_ok = True
    desired_bias = trend_bias if trend_bias in {"long", "short"} else bias1h
    if stale_timeframes.get("k1h"):
        regime_ok = False
        slope_sign_ok = False
    if asset in EMA_SLOPE_SIGN_ENFORCED and desired_bias in {"long", "short"}:
        if desired_bias == "long":
            slope_sign_ok = regime_slope_signed >= slope_threshold
        else:
            slope_sign_ok = regime_slope_signed <= -slope_threshold
        regime_ok = regime_ok and slope_sign_ok
    # 3) HTF sweep (zárt 1h/4h)
    sw1h = detect_sweep(k1h_closed, 24); sw4h = detect_sweep(k4h_closed, 24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]
    if stale_timeframes.get("k1h") or stale_timeframes.get("k4h"):
        swept = False

    # 4) 5M BOS a trend irányába (zárt 5m)
    bos5m_long = detect_bos(k5m_closed, "long")
    bos5m_short = detect_bos(k5m_closed, "short")
    if stale_timeframes.get("k5m"):
        bos5m_long = bos5m_short = False

    bos1h_long = detect_bos(k1h_closed, "long")
    bos1h_short = detect_bos(k1h_closed, "short")

    # 5) ATR szűrő (relatív) — a stabil árhoz viszonyítjuk (zárt 5m)
    atr_period = get_atr_period(asset)
    atr_series_5 = atr(k5m_closed, period=atr_period)
    atr5 = atr_series_5.iloc[-1]
    atr5_mean_recent = (
        float(atr_series_5.tail(48).mean()) if not atr_series_5.empty else None
    )
    rel_atr = float(atr5 / price_for_calc) if (atr5 and price_for_calc) else float("nan")
    bid = safe_float(spot.get("bid")) if isinstance(spot, dict) else None
    ask = safe_float(spot.get("ask")) if isinstance(spot, dict) else None
    spread_abs = (float(ask) - float(bid)) if (ask is not None and bid is not None) else None
    if spread_abs is not None and atr5 is not None and np.isfinite(float(atr5)) and float(atr5) > 0:
        spread_ratio = spread_abs / float(atr5)
        if spread_ratio > 0.2:
            reason = f"High Spread ({spread_ratio * 100:.1f}% of ATR)"
            diagnostics = {
                "spread_guard": {
                    "spread": spread_abs,
                    "atr5": float(atr5),
                    "spread_ratio": spread_ratio,
                }
            }
            return build_data_gap_signal(
                asset,
                spot_price_reference,
                spot_utc,
                spot_retrieved,
                asset_leverage,
                [reason],
                display_spot,
                diagnostics,
            )
    adx_value = regime_snapshot.get("adx")
    if adx_value is not None and not np.isfinite(adx_value):
        adx_value = None
    entry_thresholds_meta["adx_value"] = adx_value
    try:
        entry_thresholds_meta["adx_trend_threshold"] = float(adx_trend_threshold)
    except (TypeError, ValueError):
        entry_thresholds_meta["adx_trend_threshold"] = None
    regime_label = str(regime_snapshot.get("label") or "").lower()
    adx_regime = "balanced"
    if regime_label == "trending":
        adx_regime = "trend"
    elif regime_label == "ranging":
        adx_regime = "range"
    entry_thresholds_meta["adx_regime_initial"] = adx_regime
    eurusd_range_mode = False
    eurusd_range_meta: Dict[str, Any] = {}
    eurusd_range_signal: Optional[str] = None
    eurusd_range_levels: Dict[str, Any] = {}
    eurusd_momentum_trigger = False
    if asset == "EURUSD":
        adx_for_eurusd = regime_adx if regime_adx is not None else adx_value
        bias_flat = bias1h == "neutral" and bias4h == "neutral"
        if adx_for_eurusd is not None and adx_for_eurusd < 15 and bias_flat:
            eurusd_range_mode = True
            eurusd_range_meta = {
                "adx": float(adx_for_eurusd),
                "bias1h": bias1h,
                "bias4h": bias4h,
            }
        atr_spike = (
            atr5 is not None
            and atr5_mean_recent is not None
            and np.isfinite(float(atr5_mean_recent))
            and float(atr5_mean_recent) > 0
            and float(atr5) > 3.0 * float(atr5_mean_recent)
        )
        adx_momentum = adx_for_eurusd is not None and adx_for_eurusd > 25
        eurusd_momentum_trigger = bool(atr_spike and adx_momentum)
        if eurusd_momentum_trigger:
            eurusd_range_meta.setdefault("momentum_drive", {})
            eurusd_range_meta["momentum_drive"] = {
                "atr5": float(atr5) if atr5 is not None else None,
                "atr5_avg": float(atr5_mean_recent)
                if atr5_mean_recent is not None
                else None,
                "adx": float(adx_for_eurusd) if adx_for_eurusd is not None else None,
            }
        swings = find_swings(k5m_closed, lb=3)
        range_high, range_low = last_swing_levels(swings)
        eurusd_range_levels = {"range_high": range_high, "range_low": range_low}
        range_width_pips: Optional[float] = None
        if range_high is not None and range_low is not None and range_high > range_low:
            range_width_pips = (range_high - range_low) / EURUSD_PIP
            eurusd_range_levels["range_width_pips"] = range_width_pips
        price_now = price_for_calc
        near_low = False
        near_high = False
        tolerance = 0.0008
        if price_now is not None and np.isfinite(price_now):
            if range_low is not None:
                near_low = abs(price_now - range_low) <= tolerance
            if range_high is not None:
                near_high = abs(price_now - range_high) <= tolerance
        if eurusd_range_mode:
            if range_width_pips is not None:
                eurusd_range_meta["range_width_pips"] = range_width_pips
            if near_low and np.isfinite(rsi14_val) and rsi14_val < 30 and bos5m_long:
                eurusd_range_signal = "buy"
            elif near_high and np.isfinite(rsi14_val) and rsi14_val > 70 and bos5m_short:
                eurusd_range_signal = "sell"
            if eurusd_range_signal:
                eurusd_range_meta["signal"] = eurusd_range_signal
                eurusd_range_meta["rsi14"] = rsi14_val if np.isfinite(rsi14_val) else None
                eurusd_range_meta["bos"] = {
                    "long": bool(bos5m_long),
                    "short": bool(bos5m_short),
                }
        if eurusd_range_meta:
            entry_thresholds_meta["eurusd_range_mode"] = eurusd_range_meta
    eurusd_range_mode = False
    eurusd_range_meta: Dict[str, Any] = {}
    eurusd_range_signal: Optional[str] = None
    eurusd_range_levels: Dict[str, Any] = {}
    eurusd_momentum_trigger = False
    if asset == "BTCUSD":
        btc_profile_active = btc_profile_name or _btc_active_profile()
        if adx_value is not None:
            rr_override, rr_meta = btc_rr_min_with_adx(btc_profile_active, asset, float(adx_value))
        else:
            rr_override, rr_meta = (None, {})
        if rr_override is not None:
            btc_rr_band_meta = {"profile": btc_profile_active, "rr_min": float(rr_override), **rr_meta}
            meta_mode = rr_meta.get("mode")
            if meta_mode in {"trend", "range"}:
                adx_regime = meta_mode
            entry_thresholds_meta["btc_rr_adx"] = btc_rr_band_meta.copy()
    atr_profile_multiplier = get_atr_threshold_multiplier(asset)
    atr_threshold = atr_low_threshold(asset)
    btc_profile_section: Dict[str, Any] = {}
    if asset == "BTCUSD" and isinstance(BTC_PROFILE_CONFIG, dict):
        btc_profile_section = dict(BTC_PROFILE_CONFIG)

    atr_ratio_threshold_map = {}
    if isinstance(SETTINGS, dict):
        atr_ratio_threshold_map = SETTINGS.get("atr_low_threshold", {}) or {}
    atr_ratio_threshold_value = None
    if isinstance(atr_ratio_threshold_map, dict):
        atr_ratio_threshold_value = atr_ratio_threshold_map.get(asset)

    if asset == "BTCUSD":
        atr_floor_usd = None
        floor_cfg = btc_profile_section.get("atr_floor_usd") if btc_profile_section else None
        if floor_cfg is not None:
            try:
                atr_floor_usd = float(floor_cfg)
            except (TypeError, ValueError):
                atr_floor_usd = None
        if atr_floor_usd is None:
            profile_key = btc_profile_name or "baseline"
            atr_floor_usd = BTC_ATR_FLOOR_USD.get(profile_key, BTC_ATR_FLOOR_USD.get("baseline"))
        if atr_floor_usd is not None:
            entry_thresholds_meta["btc_atr_floor_usd"] = float(atr_floor_usd)
        if atr_floor_usd is not None and price_for_calc:
            try:
                btc_atr_floor_ratio = float(atr_floor_usd) / float(price_for_calc)
            except (TypeError, ValueError, ZeroDivisionError):
                btc_atr_floor_ratio = None
        if btc_atr_floor_ratio is not None and np.isfinite(btc_atr_floor_ratio):
            atr_threshold = max(float(atr_threshold), btc_atr_floor_ratio)
            btc_atr_floor_passed = bool(not np.isnan(rel_atr) and rel_atr >= btc_atr_floor_ratio)
            entry_thresholds_meta["btc_atr_floor_ratio"] = btc_atr_floor_ratio
            entry_thresholds_meta["btc_atr_floor_passed"] = btc_atr_floor_passed
        if atr_ratio_threshold_value is not None:
            entry_thresholds_meta["btc_atr_ratio_threshold"] = (
                None if atr_ratio_threshold_value in (None, "null") else atr_ratio_threshold_value
            )
        btc_profile = btc_profile_name or "baseline"
        try:
            atr5_abs = float(atr5)
        except (TypeError, ValueError):
            atr5_abs = float("nan")
        btc_gate_threshold_usd = btc_atr_gate_threshold(btc_profile, asset, analysis_now)
        entry_thresholds_meta["btc_atr_gate_threshold_usd"] = btc_gate_threshold_usd
        entry_thresholds_meta["btc_atr_value_usd"] = atr5_abs if np.isfinite(atr5_abs) else None
        btc_atr_gate_passed = btc_atr_gate_ok(btc_profile, asset, atr5_abs, analysis_now)
        entry_thresholds_meta["btc_atr_gate_ok"] = btc_atr_gate_passed
    else:
        btc_atr_gate_passed = True
    entry_thresholds_meta["atr_multiplier"] = atr_profile_multiplier
    if atr_profile_multiplier not in {None, 0.0}:
        try:
            entry_thresholds_meta["atr_threshold_base"] = atr_threshold / atr_profile_multiplier
        except Exception:
            entry_thresholds_meta["atr_threshold_base"] = atr_threshold
    else:
        entry_thresholds_meta["atr_threshold_base"] = atr_threshold
    entry_thresholds_meta["atr_threshold_initial"] = atr_threshold
    intraday_relax_factor = INTRADAY_ATR_RELAX.get(asset)
    if intraday_relax_factor not in (None, 0.0) and intraday_relax_factor != 1.0:
        try:
            atr_threshold = float(atr_threshold) * float(intraday_relax_factor)
        except Exception:
            atr_threshold = atr_threshold * intraday_relax_factor  # best effort
        entry_thresholds_meta["atr_intraday_relax"] = float(intraday_relax_factor)
        entry_thresholds_meta["atr_threshold_intraday"] = atr_threshold
    atr_percentile_value, atr_percentile_meta = compute_rel_atr_percentile(
        asset,
        analysis_now,
        k5m_closed,
        atr_series_5,
        adx_value,
    )
    if atr_percentile_meta:
        entry_thresholds_meta["atr_percentile_meta"] = atr_percentile_meta
    if atr_percentile_value is not None and np.isfinite(atr_percentile_value):
        atr_threshold = max(atr_threshold, float(atr_percentile_value))
        entry_thresholds_meta["atr_threshold_percentile"] = float(atr_percentile_value)
    volatility_overlay: Dict[str, Any] = {}
    atr_overlay_min: Optional[float] = None
    atr_overlay_gate = True
    try:
        volatility_overlay = load_volatility_overlay(asset, Path(outdir), k1m_closed)
    except Exception:
        volatility_overlay = {}
    overlay_ratio = volatility_overlay.get("implied_realised_ratio")
    overlay_regime = (volatility_overlay.get("regime") or "").lower()
    if overlay_ratio is not None:
        try:
            overlay_ratio = float(overlay_ratio)
        except Exception:
            overlay_ratio = None
    if overlay_ratio is not None and overlay_ratio > 1.35:
        # Implied volatility is significantly above realised volatility –
        # demand a healthier intraday ATR and widen TP projections.
        atr_threshold *= 1.05
        if np.isnan(rel_atr) or rel_atr < atr_threshold:
            atr_overlay_gate = False
    if overlay_regime in {"suppressed", "compressed", "discount"}:
        # realised vol is muted – require momentum confirmation later.
        if np.isnan(rel_atr) or rel_atr < atr_threshold * 1.05:
            atr_overlay_gate = False
    if overlay_regime in {"implied_elevated", "implied_extreme"}:
        atr_threshold *= 0.95
    spread_gate_ok = True
    spread_ratio = None
    spread_limit: Optional[float] = None
    try:
        spread_limit = float(get_spread_max_atr_pct(asset))
    except Exception:
        spread_limit = Non  
    if spread_limit and spread_abs is not None and atr5 is not None:
        try:
            spread_ratio = float(spread_abs) / float(atr5) if float(atr5) > 0 else None
        except Exception:
            spread_ratio = None
        if spread_ratio is not None and np.isfinite(spread_ratio):
            entry_thresholds_meta["spread_ratio"] = spread_ratio
            entry_thresholds_meta["spread_limit_atr"] = float(spread_limit)
            if spread_ratio > float(spread_limit):
                spread_gate_ok = False
    else:
        entry_thresholds_meta["spread_ratio"] = spread_ratio
    atr_soft_penalty = 0.0
    atr_soft_meta: Dict[str, Any] = {}
    atr_soft_cfg: Dict[str, Any] = {}
    if isinstance(dynamic_logic_cfg, dict):
        soft_gates_cfg = dynamic_logic_cfg.get("soft_gates")
        if isinstance(soft_gates_cfg, dict):
            atr_soft_cfg = soft_gates_cfg.get("atr") if isinstance(soft_gates_cfg.get("atr"), dict) else {}
    asset_soft_overrides = atr_soft_cfg.get("asset_overrides") if isinstance(atr_soft_cfg, dict) else None
    if isinstance(asset_soft_overrides, dict):
        override_cfg = asset_soft_overrides.get(asset)
        if isinstance(override_cfg, dict):
            atr_soft_cfg = {**atr_soft_cfg, **override_cfg}
    volatility_manager = VolatilityManager(atr_soft_cfg)
    atr_gate_result = volatility_manager.evaluate(rel_atr, atr_threshold)
    base_ratio_ok = bool(atr_gate_result.get("ok"))
    atr_soft_penalty = float(atr_gate_result.get("penalty") or 0.0)
    atr_soft_meta = atr_gate_result.get("meta") or {}
    if atr_gate_result.get("warning"):
        atr_soft_meta["warning"] = atr_gate_result["warning"]
    if atr_soft_meta:
        entry_thresholds_meta["atr_soft_gate"] = atr_soft_meta
    if atr_soft_meta.get("mode") == "soft_pass":
        soft_gate_reason = "ATR Soft Gate: belépés engedélyezve toleranciával"
        if atr_soft_penalty:
            soft_gate_reason += f" (−{atr_soft_penalty:.1f} P-score)"
        reasons.append(soft_gate_reason)
        entry_thresholds_meta["atr_soft_gate_used"] = True

    atr_ratio_ok = base_ratio_ok
    if asset == "BTCUSD":
        if atr_ratio_threshold_value in (None, "null"):
            atr_ratio_ok = base_ratio_ok
        elif isinstance(atr_ratio_threshold_value, (int, float)):
            try:
                rel_atr_value = float(rel_atr)
            except (TypeError, ValueError):
                rel_atr_value = float("nan")
            atr_ratio_ok = bool(
                not np.isnan(rel_atr_value) and rel_atr_value >= float(atr_ratio_threshold_value)
            )
            atr_ratio_ok = atr_ratio_ok and base_ratio_ok
        else:
            atr_ratio_ok = base_ratio_ok

    atr_abs_min_config = get_atr_abs_min(asset)
    atr_abs_override_applied = False
    if asset in ATR_ABS_MIN_OVERRIDE:
        atr_abs_override_applied = True
        atr_abs_min_config = ATR_ABS_MIN_OVERRIDE[asset]
    atr_abs_min = atr_abs_min_config
    atr_abs_ok = True
    if atr_abs_min is not None:
        try:
            atr_abs_ok = float(atr5) >= atr_abs_min
        except Exception:
            atr_abs_ok = False

    xag_atr_floor_triggered = False
    if asset == "XAGUSD" and XAGUSD_ATR_5M_FLOOR_ENABLED:
        try:
            xag_atr_floor_triggered = float(atr5) < float(XAGUSD_ATR_5M_FLOOR)
        except Exception:
            xag_atr_floor_triggered = False

    atr_ok = bool(atr_ratio_ok and atr_abs_ok)
    if xag_atr_floor_triggered:
        atr_ok = False
    entry_thresholds_meta["atr_ratio_ok"] = bool(atr_ratio_ok)
    if not spread_gate_ok:
        atr_ok = False
    if not atr_overlay_gate:
        atr_ok = False
    if not btc_atr_gate_passed:
        atr_ok = False
    if stale_timeframes.get("k5m"):
        atr_ok = False

    entry_thresholds_meta["atr_threshold_effective"] = atr_threshold
    entry_thresholds_meta["spread_gate_ok"] = spread_gate_ok
    cost_model = settings.ASSET_COST_MODEL.get(asset) or settings.DEFAULT_COST_MODEL
    risk_cap_pct = get_max_risk_pct(asset)
    if asset == "EURUSD" and eurusd_range_mode:
        risk_cap_pct = 1.0 if risk_cap_pct is None else min(risk_cap_pct, 1.0)
    risk_guard_meta = {
        "profile": asset_entry_profile,
        "max_risk_pct": float(risk_cap_pct) if risk_cap_pct is not None else None,
        "spread_gate_ok": bool(spread_gate_ok),
        "allowed": bool(spread_gate_ok),
        "cost_model": cost_model,
    }
    if asset == "XAGUSD":
        entry_thresholds_meta["xag_atr_5m_floor"] = {
            "enabled": bool(XAGUSD_ATR_5M_FLOOR_ENABLED),
            "floor": float(XAGUSD_ATR_5M_FLOOR),
            "triggered": bool(xag_atr_floor_triggered),
        }
        if xag_atr_floor_triggered:
            reasons.append(
                "XAGUSD 5m ATR floor alatt — új belépés blokkolva"
            )
    entry_thresholds_meta["atr_abs_min_used"] = (
        float(atr_abs_min) if atr_abs_min is not None else None
    )
    entry_thresholds_meta["atr_abs_override_applied"] = bool(
        atr_abs_override_applied
    )
    entry_thresholds_meta["risk_guard"] = risk_guard_meta
    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        entry_gate_context_hu.update(
            {
                "atr_kapu": {
                    "ok": bool(atr_ok),
                    "rel_atr": float(rel_atr) if rel_atr is not None else None,
                    "threshold": float(atr_threshold) if atr_threshold is not None else None,
                    "abs_ok": bool(atr_abs_ok),
                    "overlay_ok": bool(atr_overlay_gate),
                    "btc_atr_gate": bool(btc_atr_gate_passed),
                    **_gate_timestamp_fields(analysis_now),
                },
                "spread_kapu": {
                    "ok": bool(spread_gate_ok),
                    "spread_ratio_atr": float(spread_ratio) if spread_ratio is not None else None,
                    "spread_limit_atr": float(spread_limit) if spread_limit is not None else None,
                    **_gate_timestamp_fields(analysis_now),
                },
                "kockazat_guard": {
                    "max_risk_pct": risk_guard_meta.get("max_risk_pct"),
                    "engedi": bool(spread_gate_ok),
                    "cost_model": cost_model,
                    **_gate_timestamp_fields(analysis_now),
                },
            }
        )
    momentum_vol_ratio = volume_ratio(k5m_closed, MOMENTUM_VOLUME_RECENT, MOMENTUM_VOLUME_BASE)
    dynamic_tp_profile = compute_dynamic_tp_profile(
        asset,
        atr_series_5,
        float(rel_atr) if not np.isnan(rel_atr) else float("nan"),
        price_for_calc,
    )
    volatility_adjustments: List[str] = []
    if volatility_overlay:
        try:
            core_profile = dynamic_tp_profile.setdefault("core", {})
            mom_profile = dynamic_tp_profile.setdefault("momentum", {})
            if overlay_ratio is not None and overlay_ratio > 1.35:
                core_profile["tp1"] = float(core_profile.get("tp1", TP1_R) * 1.08)
                core_profile["tp2"] = float(core_profile.get("tp2", TP2_R) * 1.1)
                mom_profile["tp1"] = float(mom_profile.get("tp1", TP1_R_MOMENTUM) * 1.05)
                mom_profile["tp2"] = float(mom_profile.get("tp2", TP2_R_MOMENTUM) * 1.08)
                volatility_adjustments.append("Implied/realised vol prémium miatt TP szélesítés")
            if overlay_regime in {"suppressed", "compressed"}:
                core_profile["rr"] = float(core_profile.get("rr", CORE_RR_MIN.get(asset, MIN_R_CORE)) * 1.05)
                mom_profile["rr"] = float(mom_profile.get("rr", MOMENTUM_RR_MIN.get(asset, MIN_R_MOMENTUM)) * 1.05)
                volatility_adjustments.append("Alacsony realised vol → magasabb RR igény")
        except Exception:
            pass

    intraday_price_ref = display_spot if display_spot is not None else last5_close
    atr5_for_profile = None
    try:
        if atr5 is not None and np.isfinite(float(atr5)):
            atr5_for_profile = float(atr5)
    except Exception:
        atr5_for_profile = None
    intraday_profile = compute_intraday_profile(
        asset,
        k1m_closed,
        intraday_price_ref,
        atr5_for_profile,
        analysis_now,
    )

    realtime_jump_flag = False
    if (
        realtime_used
        and last5_close is not None
        and atr5 is not None
        and np.isfinite(float(atr5))
        and display_spot is not None
    ):
        jump = abs(display_spot - last5_close)
        if jump > REALTIME_JUMP_MULT * float(atr5):
            realtime_jump_flag = True

    # 6) Fib zóna (0.618–0.886) 1H swingekre, ATR(1h) alapú tűréssel — zárt 1h
    k1h_sw = find_swings(k1h_closed, lb=2)
    move_hi, move_lo = last_swing_levels(k1h_sw)
    atr1h_val = float(atr(k1h_closed).iloc[-1]) if not k1h_closed.empty else float("nan")
    atr1h = atr1h_val if np.isfinite(atr1h_val) else None
    nvda_daily_atr: Optional[float] = None
    nvda_daily_atr_rel: Optional[float] = None
    if asset == "NVDA":
        nvda_daily_candidates: List[float] = []
        if atr1h is not None and atr1h > 0 and NVDA_DAILY_ATR_MULTIPLIER > 0:
            nvda_daily_candidates.append(float(atr1h) * NVDA_DAILY_ATR_MULTIPLIER)
        if atr5 is not None and atr5 > 0 and NVDA_DAILY_ATR_MULTIPLIER > 0:
            nvda_daily_candidates.append(float(atr5) * NVDA_DAILY_ATR_MULTIPLIER * 12.0)
        if nvda_daily_candidates:
            nvda_daily_atr = max(nvda_daily_candidates)
            if price_for_calc:
                try:
                    nvda_daily_atr_rel = nvda_daily_atr / float(price_for_calc)
                except Exception:
                    nvda_daily_atr_rel = None
    atr_half = (atr1h * 0.5) if (atr1h is not None) else None
    invalid_buffer_candidates: List[float] = []
    min_buffer = ACTIVE_INVALID_BUFFER_ABS.get(asset)
    if min_buffer is not None:
        invalid_buffer_candidates.append(float(min_buffer))
    if atr_half is not None:
        invalid_buffer_candidates.append(float(atr_half))
    invalid_buffer = max(invalid_buffer_candidates) if invalid_buffer_candidates else None
    invalid_level_sell = (
        float(move_hi + invalid_buffer)
        if (move_hi is not None and invalid_buffer is not None)
        else None
    )
    invalid_level_buy = (
        float(move_lo - invalid_buffer)
        if (move_lo is not None and invalid_buffer is not None)
        else None
    )
    atr1h_tol = atr1h_val if np.isfinite(atr1h_val) else 0.0
    fib_tol = get_fib_tolerance(asset, profile=asset_entry_profile)
    fib_ok = fib_zone_ok(
        move_hi, move_lo, price_for_calc,
        low=0.618, high=0.886,
        tol_abs=atr1h_tol * 0.75,   # SZÉLESÍTVE: ±0.75×ATR(1h)
        tol_frac=fib_tol
    )
    entry_thresholds_meta["fib_tolerance_fraction"] = float(fib_tol)
    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        profil_context = {
            "profil_kuszobok": {
                "profil": asset_entry_profile,
                "p_score_min": float(p_score_min_local)
                if p_score_min_local is not None
                else None,
                "p_score_min_base": float(p_score_min_base)
                if p_score_min_base is not None
                else None,
                "atr_kuszob_rel": float(atr_threshold)
                if atr_threshold is not None
                else None,
                "atr_abs_min": float(atr_abs_min) if atr_abs_min is not None else None,
                "fib_tures_frac": float(fib_tol),
                "fib_tures_atr": float(atr1h_tol * 0.75)
                if np.isfinite(atr1h_val)
                else None,
                **_gate_timestamp_fields(analysis_now),
            }
        }
        entry_gate_context_hu.update(profil_context)

    # 6/b) Kiegészítő likviditás kontextus (1h EMA21 közelség + szerkezeti retest)
    ema21_1h = float(ema(k1h_closed["close"], 21).iloc[-1]) if not k1h_closed.empty else float("nan")
    ema21_dist_ok = (
        np.isfinite(ema21_1h)
        and np.isfinite(atr1h_val)
        and abs(price_for_calc - ema21_1h) <= max(atr1h_val, 0.0008 * price_for_calc)
    )

    last_close_1h = float(k1h_closed["close"].iloc[-1]) if not k1h_closed.empty else float("nan")
    ema21_relation = "unknown"
    if np.isfinite(last_close_1h) and np.isfinite(ema21_1h):
        tol = max(
            (atr1h_val * 0.1) if np.isfinite(atr1h_val) else 0.0,
            abs(ema21_1h) * 0.0001,
            1e-5,
        )
        diff = last_close_1h - ema21_1h
        if diff > tol:
            ema21_relation = "above"
        elif diff < -tol:
            ema21_relation = "below"
        else:
            ema21_relation = "at"
    if stale_timeframes.get("k1h"):
        ema21_dist_ok = False
        ema21_relation = "stale"

    bos_lookback = get_bos_lookback(asset)
    session_window_payload = SESSION_WINDOWS_UTC.get(
        asset, SESSION_WINDOWS_UTC.get(asset.upper(), {})
    )

    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        fib_zone_payload = {
            "ok": bool(fib_ok),
            "sav": [0.618, 0.886],
            "tures_frac": fib_tol,
            "tures_atr": float(atr1h_tol * 0.75) if np.isfinite(atr1h_val) else None,
        }
        entry_gate_context_hu.update(
            {
                "bos_visszatekintes": bos_lookback,
                "fib_zona": {**fib_zone_payload, **_gate_timestamp_fields(analysis_now)},
                "session_ablak_utc": session_window_payload,
                "p_score_profil": asset_entry_profile,
            }
        )
    struct_retest_long = structure_break_with_retest(k5m_closed, "long", bos_lookback)
    struct_retest_short = structure_break_with_retest(k5m_closed, "short", bos_lookback)
    if stale_timeframes.get("k5m"):
        struct_retest_long = struct_retest_short = False

    micro_bos_long = micro_bos_with_retest(k1m_closed, k5m_closed, "long", bos_lookback)
    micro_bos_short = micro_bos_with_retest(k1m_closed, k5m_closed, "short", bos_lookback)
    if stale_timeframes.get("k1m") or stale_timeframes.get("k5m"):
        micro_bos_long = micro_bos_short = False

    ema_cross_long = ema_cross_short = False
    ema9_5m: Optional[pd.Series] = None
    ema21_5m: Optional[pd.Series] = None
    if asset in ENABLE_MOMENTUM_ASSETS and not k5m_closed.empty:
        ema9_5m = ema(k5m_closed["close"], 9)
        ema21_5m = ema(k5m_closed["close"], 21)
        cross_bars = 7 if asset == "NVDA" else MOMENTUM_BARS
        ema_cross_long = ema_cross_recent(ema9_5m, ema21_5m, bars=cross_bars, direction="long")
        ema_cross_short = ema_cross_recent(ema9_5m, ema21_5m, bars=cross_bars, direction="short")
    if asset == "BTCUSD":
        momentum_state = globals().setdefault("_BTC_MOMENTUM_RUNTIME", {})
        momentum_state["ema_cross"] = {"long": bool(ema_cross_long), "short": bool(ema_cross_short)}

    nvda_cross_long = nvda_cross_short = False
    if asset == "NVDA":
        nvda_cross_long = ema_cross_long
        nvda_cross_short = ema_cross_short

    effective_bias = trend_bias
    bias_override_used = False
    bias_override_reason: Optional[str] = None
    intraday_bias_gate_meta: Optional[Dict[str, Any]] = None
    bias_gate_notes: List[str] = []
    mean_reversion_bias: Optional[str] = None
    rsi14_val: float = float("nan")
    try:
        rsi_series = rsi(k5m_closed["close"].astype(float), 14)
        rsi14_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")
    except Exception:
        rsi14_val = float("nan")
    if regime_label == "choppy":
        if np.isfinite(rsi14_val):
            if rsi14_val < 30:
                mean_reversion_bias = "long"
            elif rsi14_val > 70:
                mean_reversion_bias = "short"
        if mean_reversion_bias:
            reasons.append(
                f"Mean reversion jelzés CHOPPY rezsimben (RSI14 {rsi14_val:.1f} → {mean_reversion_bias})"
            )

    gold_reversal_meta: Dict[str, Any] = {}
    gold_reversal_hits = 0
    gold_reversal_reasons: List[str] = []
    xag_reversal_active = False
    xag_reversal_side: Optional[str] = None
    xag_reversal_meta: Dict[str, Any] = {}
    if asset == "GOLD_CFD":
        rsi_extreme = np.isfinite(rsi14_val) and (rsi14_val < 30 or rsi14_val > 70)
        if rsi_extreme:
            gold_reversal_hits += 1
            gold_reversal_reasons.append(f"RSI14 szélsőséges ({rsi14_val:.1f})")

        daily_range: Optional[float] = None
        if not k1h_closed.empty:
            last_day = k1h_closed.tail(24)
            if not last_day.empty:
                try:
                    daily_range = float(last_day["high"].max() - last_day["low"].min())
                except Exception:
                    daily_range = None
        if daily_range is not None and atr1h is not None and atr1h > 0:
            if daily_range >= 1.2 * float(atr1h):
                gold_reversal_hits += 1
                gold_reversal_reasons.append("Napi range ≥ 1.2×ATR")
            gold_reversal_meta["daily_range"] = daily_range

        fib_touch = bool(fib_ok)
        if fib_touch:
            gold_reversal_hits += 1
            gold_reversal_reasons.append("Kulcsszint érintve (Fib/pivot)")

        bos_against_trend = False
        if trend_bias == "long" and (bos1h_short or bos5m_short):
            bos_against_trend = True
        elif trend_bias == "short" and (bos1h_long or bos5m_long):
            bos_against_trend = True
        if bos_against_trend:
            gold_reversal_hits += 1
            gold_reversal_reasons.append("BOS a trend ellen")

        adx_value_safe: Optional[float] = None
        if adx_value is not None and np.isfinite(adx_value):
            adx_value_safe = float(adx_value)
        bias_conflict = trend_bias in {"long", "short"} and effective_bias in {"long", "short"} and trend_bias != effective_bias
        adx_weak = adx_value_safe is not None and adx_value_safe < 18.0
        gold_reversal_meta.update(
            {
                "rsi": rsi14_val if np.isfinite(rsi14_val) else None,
                "hits": gold_reversal_hits,
                "adx_value": adx_value_safe,
                "bias_conflict": bool(bias_conflict),
                "rr_target": [1.0, 1.2],
                "sl_pct": [0.005, 0.008],
            }
        )
        if gold_reversal_hits >= 2 and (adx_weak or bias_conflict):
            gold_reversal_meta["active"] = True
            gold_reversal_meta["reasons"] = gold_reversal_reasons.copy()
            reasons.append("GOLD reversal mód engedélyezve — legalább 2 jelzés aktív")
            dynamic_tp_profile["reversal"] = {"tp1": 1.0, "tp2": 1.2, "rr": 1.1}
            entry_thresholds_meta["gold_reversal"] = gold_reversal_meta
        elif gold_reversal_reasons:
            gold_reversal_meta["active"] = False
            gold_reversal_meta["reasons"] = gold_reversal_reasons.copy()
            entry_thresholds_meta["gold_reversal"] = gold_reversal_meta

    if asset == "XAGUSD":
        price_now = safe_float(last5_close)
        recent_window = k5m_closed.tail(48)
        swing_hi = swing_lo = None
        if not k5m_closed.empty:
            swings = find_swings(k5m_closed, lb=2)
            swing_hi, swing_lo = last_swing_levels(swings)

        pct_up_move = pct_down_move = None
        if price_now and not recent_window.empty:
            try:
                min_close = float(recent_window["close"].min())
                max_close = float(recent_window["close"].max())
                if min_close > 0:
                    pct_up_move = (price_now - min_close) / min_close * 100.0
                if max_close > 0:
                    pct_down_move = (price_now - max_close) / max_close * 100.0
            except Exception:
                pct_up_move = pct_down_move = None

        def _near_level(level: Optional[float]) -> bool:
            if price_now is None or level is None:
                return False
            try:
                return abs(price_now - float(level)) / float(price_now) <= 0.002
            except Exception:
                return False

        def _wick_reversal(df: pd.DataFrame, side: str) -> bool:
            if df.empty:
                return False
            last = df.iloc[-1]
            try:
                high = float(last["high"])
                low = float(last["low"])
                open_ = float(last["open"])
                close_ = float(last["close"])
                volume = float(last.get("volume", 0.0))
            except Exception:
                return False
            range_val = high - low
            if range_val <= 0:
                return False
            body = abs(close_ - open_)
            upper_wick = high - max(open_, close_)
            lower_wick = min(open_, close_) - low
            vol_window = df["volume"].tail(20) if "volume" in df else None
            vol_ok = True
            if vol_window is not None and not vol_window.empty:
                try:
                    vol_ok = volume >= 1.2 * float(vol_window.median())
                except Exception:
                    vol_ok = True
            if side == "short":
                return upper_wick >= 0.4 * range_val and upper_wick > body and vol_ok
            if side == "long":
                return lower_wick >= 0.4 * range_val and lower_wick > body and vol_ok
            return False

        rsi_extreme_short = np.isfinite(rsi14_val) and rsi14_val >= 75
        rsi_extreme_long = np.isfinite(rsi14_val) and rsi14_val <= 25
        level_resistance = any(
            _near_level(level)
            for level in (
                intraday_profile.get("day_high") if isinstance(intraday_profile, dict) else None,
                swing_hi,
            )
        )
        level_support = any(
            _near_level(level)
            for level in (
                intraday_profile.get("day_low") if isinstance(intraday_profile, dict) else None,
                swing_lo,
            )
        )

        pattern_short = _wick_reversal(k5m_closed, "short")
        pattern_long = _wick_reversal(k5m_closed, "long")
        bos_short = bool(micro_bos_short)
        bos_long = bool(micro_bos_long)

        xag_reversal_meta = {
            "rsi": rsi14_val if np.isfinite(rsi14_val) else None,
            "pct_up_move": pct_up_move,
            "pct_down_move": pct_down_move,
            "level_resistance": level_resistance,
            "level_support": level_support,
            "pattern_short": pattern_short,
            "pattern_long": pattern_long,
            "bos_short": bos_short,
            "bos_long": bos_long,
        }

        short_conditions = all(
            (
                rsi_extreme_short,
                pct_up_move is not None and pct_up_move >= 1.5,
                level_resistance,
                pattern_short,
                bos_short,
            )
        )
        long_conditions = all(
            (
                rsi_extreme_long,
                pct_down_move is not None and pct_down_move <= -1.5,
                level_support,
                pattern_long,
                bos_long,
            )
        )

        if short_conditions:
            xag_reversal_active = True
            xag_reversal_side = "sell"
            xag_reversal_meta["direction"] = "short"
        elif long_conditions:
            xag_reversal_active = True
            xag_reversal_side = "buy"
            xag_reversal_meta["direction"] = "long"

        if xag_reversal_active:
            xag_reversal_meta["active"] = True
            xag_reversal_meta["tp_targets"] = [1.0, 1.5]
            xag_reversal_meta["risk_cap_pct"] = 1.0
            reasons.append("XAGUSD reversal setup aktív — kockázat korlátozva 1% alá")
            dynamic_tp_profile["reversal"] = {"tp1": 1.0, "tp2": 1.5, "rr": 1.0}
        elif xag_reversal_meta:
            xag_reversal_meta["active"] = False
        if xag_reversal_meta:
            entry_thresholds_meta["xag_reversal"] = xag_reversal_meta

    if trend_bias == "neutral" and bias1h in ("long", "short"):
        override_dir = bias1h
        bos_support = bos5m_long if override_dir == "long" else bos5m_short
        struct_support = struct_retest_long if override_dir == "long" else struct_retest_short
        micro_support = micro_bos_long if override_dir == "long" else micro_bos_short
        atr_push = bool(
            atr_ok and not np.isnan(rel_atr)
            and rel_atr >= max(atr_threshold * 1.2, MOMENTUM_ATR_REL)
        )
        if regime_ok and (bos_support or struct_support or (micro_support and atr_push)):
            effective_bias = override_dir
            bias_override_used = True
            bias_override_reason = f"Bias override: 1h trend {override_dir} + momentum támogatás"

    if (
        asset == "NVDA"
        and effective_bias == "neutral"
        and bias1h in {"long", "short"}
        and bias4h == bias1h
        and regime_ok
    ):
        effective_bias = bias1h
        bias_override_used = True
        bias_override_reason = "Bias override: NVDA 1h trend cash-session megerősítés"

    if mean_reversion_bias:
        if effective_bias != mean_reversion_bias:
            bias_override_used = True
            bias_override_reason = "Mean reversion bias override"
        effective_bias = mean_reversion_bias

    btc_bias_cfg = _btc_profile_section("bias_relax") if asset == "BTCUSD" else {}
    btc_momentum_cfg = _btc_profile_section("momentum_override") if asset == "BTCUSD" else {}
    btc_rr_cfg = _btc_profile_section("rr") if asset == "BTCUSD" else {}
    if asset == "BTCUSD":
        momentum_state = globals().setdefault("_BTC_MOMENTUM_RUNTIME", {})
        momentum_state["cfg"] = dict(btc_momentum_cfg)
        if btc_profile_name:
            momentum_state["profile"] = btc_profile_name
            limit_map = globals().setdefault("_BTC_NO_CHASE_LIMITS", {})
            raw_limit = btc_momentum_cfg.get("no_chase_r")
            if raw_limit is not None:
                try:
                    limit_map[btc_profile_name] = float(raw_limit)
                except (TypeError, ValueError):
                    limit_map[btc_profile_name] = BTC_NO_CHASE_R.get(
                        btc_profile_name, BTC_NO_CHASE_R.get("baseline", 0.25)
                    )
            elif btc_profile_name not in limit_map:
                limit_map[btc_profile_name] = BTC_NO_CHASE_R.get(
                    btc_profile_name, BTC_NO_CHASE_R.get("baseline", 0.25)
                )

    if effective_bias == "neutral":
        relax_cfg = INTRADAY_BIAS_RELAX.get(asset)
        if relax_cfg:
            requirement_labels = {
                "micro_bos_long": "1m micro BOS long",
                "micro_bos_short": "1m micro BOS short",
                "struct_retest_long": "5m retest long",
                "struct_retest_short": "5m retest short",
                "bos5m_long": "5m BOS long",
                "bos5m_short": "5m BOS short",
                "atr_ok": "ATR gate",
                "atr_strong": "Magas ATR megerősítés",
                "momentum_volume": "Momentum volume ráta",
                "nvda_cross_long": "EMA9×21 long",
                "nvda_cross_short": "EMA9×21 short",
            }

            def describe_requirement(token: str) -> str:
                return requirement_labels.get(token, token)

            def requirement_ok(token: str, direction: str) -> bool:
                if token == "atr_ok":
                    return bool(atr_ok)
                if token == "atr_strong":
                    if np.isnan(rel_atr):
                        return False
                    strong_th = max(atr_threshold, MOMENTUM_ATR_REL) * 1.1
                    try:
                        return float(rel_atr) >= float(strong_th)
                    except Exception:
                        return False
                if token == "momentum_volume":
                    ratio_ok = (
                        momentum_vol_ratio is not None
                        and momentum_vol_ratio >= MOMENTUM_VOLUME_RATIO_TH
                    )
                    if asset == "BTCUSD" and btc_bias_cfg:
                        ofi_th = float(btc_bias_cfg.get("ofi_sub_threshold", 0.0) or 0.0)
                        if ofi_th > 0 and ofi_zscore is not None:
                            if direction == "long":
                                ratio_ok = ratio_ok or ofi_zscore >= ofi_th
                            elif direction == "short":
                                ratio_ok = ratio_ok or ofi_zscore <= -ofi_th
                    return ratio_ok
                if token == "micro_bos_long":
                    return bool(micro_bos_long and not micro_bos_short)
                if token == "micro_bos_short":
                    return bool(micro_bos_short and not micro_bos_long)
                if token == "struct_retest_long":
                    return bool(struct_retest_long and not struct_retest_short)
                if token == "struct_retest_short":
                    return bool(struct_retest_short and not struct_retest_long)
                if token == "bos5m_long":
                    return bool(bos5m_long and not bos5m_short)
                if token == "bos5m_short":
                    return bool(bos5m_short and not bos5m_long)
                if token == "nvda_cross_long":
                    return bool(nvda_cross_long)
                if token == "nvda_cross_short":
                    return bool(nvda_cross_short)
                return False

            intraday_bias_gate_meta = {
                "configured": True,
                "allow_neutral": bool(relax_cfg.get("allow_neutral")),
                "scenarios": [],
            }
            scenario_defs = relax_cfg.get("scenarios") or []
            satisfied_direction: Optional[str] = None
            selected_state: Optional[Dict[str, Any]] = None
            conflict = False
            allow_neutral = bool(relax_cfg.get("allow_neutral"))

            for scenario in scenario_defs:
                if not isinstance(scenario, dict):
                    continue
                direction = str(scenario.get("direction", "")).lower()
                if direction not in {"long", "short"}:
                    continue
                requires_raw = scenario.get("requires") or []
                requires = [str(req) for req in requires_raw if isinstance(req, str)]
                missing = [req for req in requires if not requirement_ok(req, direction)]
                state: Dict[str, Any] = {
                    "direction": direction,
                    "requires": requires,
                    "missing": missing,
                    "met": not missing,
                }
                label_value = scenario.get("label")
                if isinstance(label_value, str):
                    state["label"] = label_value
                if missing:
                    state["missing_pretty"] = [describe_requirement(req) for req in missing]
                else:
                    state["missing_pretty"] = []
                state["requires_pretty"] = [describe_requirement(req) for req in requires]
                intraday_bias_gate_meta["scenarios"].append(state)
                if allow_neutral and not missing:
                    if satisfied_direction is None:
                        satisfied_direction = direction
                        selected_state = state
                    elif satisfied_direction != direction:
                        conflict = True
            if allow_neutral and selected_state and not conflict:
                effective_bias = selected_state["direction"]
                bias_override_used = True
                label_text = selected_state.get("label")
                base_message = label_text or f"intraday {effective_bias} setup"
                bias_override_reason = f"Bias override: Intraday {base_message}"
                intraday_bias_gate_meta["selected"] = {
                    "direction": selected_state["direction"],
                    "label": label_text,
                    "requires": selected_state.get("requires", []),
                    "requires_pretty": selected_state.get("requires_pretty"),
                }
                intraday_bias_gate_meta["matched"] = True
            elif conflict:
                intraday_bias_gate_meta["conflict"] = True
                intraday_bias_gate_meta["matched"] = False
            elif intraday_bias_gate_meta["scenarios"]:
                intraday_bias_gate_meta["matched"] = False

            if (
                intraday_bias_gate_meta
                and effective_bias == "neutral"
                and intraday_bias_gate_meta.get("allow_neutral")
            ):
                if intraday_bias_gate_meta.get("conflict"):
                    msg = "Intraday bias gate: ellentétes long/short jelek → várakozás"
                    if msg not in bias_gate_notes:
                        bias_gate_notes.append(msg)
                else:
                    missing_msgs: List[str] = []
                    for state in intraday_bias_gate_meta.get("scenarios", []):
                        if state.get("met"):
                            continue
                        label_text = state.get("label") or f"{state.get('direction')} setup"
                        missing_pretty = state.get("missing_pretty") or []
                        if missing_pretty:
                            missing_msgs.append(
                                f"{label_text}: {', '.join(missing_pretty)}"
                            )
                    if missing_msgs:
                        msg = "Intraday bias gate feltételek hiányosak — " + " | ".join(missing_msgs)
                        if msg not in bias_gate_notes:
                            bias_gate_notes.append(msg)

    if effective_bias == "long":
        bos5m = bos5m_long
    elif effective_bias == "short":
        bos5m = bos5m_short
    else:
        bos5m = False

    micro_bos_active = (
        micro_bos_long if effective_bias == "long"
        else micro_bos_short if effective_bias == "short"
        else False
    )

    recent_break_long = broke_structure(k5m_closed, "long", bos_lookback)
    recent_break_short = broke_structure(k5m_closed, "short", bos_lookback)
    if stale_timeframes.get("k5m"):
        recent_break_long = recent_break_short = False

    structure_notes: List[str] = []
    structure_components: Dict[str, bool] = {"bos": False, "liquidity": False, "ofi": False}

    # 7) P-score — volatilitás-adaptív súlyozás
    P, reasons = 15.0, []
    if bias_gate_notes:
        reasons.extend(bias_gate_notes)
    if effective_bias != "neutral":
        bias_strength = 1.0 + (0.3 if bias_override_used else 0.0)
        bias_points = 15.0 * bias_strength
        P += bias_points
        if bias_override_used:
            label = bias_override_reason or f"Bias override: {effective_bias}"
        else:
            label = f"Bias(4H→1H)={effective_bias}"
        reasons.append(f"{label} (+{bias_points:.1f})")
    else:
        reasons.append("Bias neutrálsávban")

    if regime_ok:
        ema_ratio = abs(regime_slope_signed) / max(1e-9, slope_threshold)
        regime_points = 6.0 + 6.0 * min(2.5, ema_ratio)
        P += regime_points
        reasons.append(f"EMA21 slope {ema_ratio:.2f}× küszöb (+{regime_points:.1f})")
    else:
        reasons.append("Regime filter inaktív")
        P -= 4.0

    if swept:
        sweep_points = 10.0
        P += sweep_points
        reasons.append(f"HTF sweep ok (+{sweep_points:.1f})")

    struct_retest_active = (
        (effective_bias == "long" and struct_retest_long)
        or (effective_bias == "short" and struct_retest_short)
    )
    if bos5m:
        swing_points = 16.0
        P += swing_points
        reasons.append(f"5M BOS trendirányba (+{swing_points:.1f})")
    elif struct_retest_active:
        swing_points = 11.0
        P += swing_points
        reasons.append(f"5m szerkezeti törés + retest a trend irányába (+{swing_points:.1f})")
    elif asset == "NVDA" and (
        (effective_bias == "long" and nvda_cross_long)
        or (effective_bias == "short" and nvda_cross_short)
    ):
        swing_points = 13.0
        P += swing_points
        reasons.append(f"5m EMA9×21 momentum kereszt megerősítés (+{swing_points:.1f})")
    elif micro_bos_active:
        if atr_ok and not np.isnan(rel_atr) and rel_atr >= max(atr_threshold, MOMENTUM_ATR_REL):
            P += MICRO_BOS_P_BONUS
            reasons.append(f"Micro BOS megerősítés (1m szerkezet + magas ATR) (+{MICRO_BOS_P_BONUS})")
        else:
            reasons.append("1m BOS + 5m retest — várjuk a 5m megerősítést")

    if not spread_gate_ok:
        reasons.append("Spread gate: aktuális spread meghaladja az ATR arány limitet")

    minute_now = _min_of_day(analysis_now.hour, analysis_now.minute)
    time_penalty_total = 0.0
    time_penalty_notes: List[str] = []
    open_cfg = P_SCORE_TIME_WINDOWS.get("open") or {}
    try:
        open_window = int(open_cfg.get("window_minutes") or 0)
        open_penalty = float(open_cfg.get("penalty") or 0.0)
    except (TypeError, ValueError):
        open_window, open_penalty = 0, 0.0
    if open_window > 0 and minute_now <= open_window:
        time_penalty_total += open_penalty
        time_penalty_notes.append(f"Nyitási illikviditás hatás ({open_penalty:+.1f})")
    close_cfg = P_SCORE_TIME_WINDOWS.get("close") or {}
    try:
        close_window = int(close_cfg.get("window_minutes") or 0)
        close_penalty = float(close_cfg.get("penalty") or 0.0)
    except (TypeError, ValueError):
        close_window, close_penalty = 0, 0.0
    if close_window > 0 and minute_now >= max(0, 1440 - close_window):
        time_penalty_total += close_penalty
        time_penalty_notes.append(f"Zárási sáv likviditás büntetés ({close_penalty:+.1f})")
    overlap_cfg = P_SCORE_TIME_WINDOWS.get("overlap") or {}
    overlap_start = ATR_PERCENTILE_OVERLAP.get("start_minute") if ATR_PERCENTILE_OVERLAP else None
    overlap_end = ATR_PERCENTILE_OVERLAP.get("end_minute") if ATR_PERCENTILE_OVERLAP else None
    try:
        overlap_penalty = float(overlap_cfg.get("penalty") or 0.0)
        overlap_window = int(overlap_cfg.get("window_minutes") or 0)
    except (TypeError, ValueError):
        overlap_penalty = 0.0
        overlap_window = 0
    if overlap_start is not None and overlap_end is not None:
        if overlap_start <= minute_now <= overlap_end:
            time_penalty_total += overlap_penalty
            time_penalty_notes.append(f"London–NY overlap profil ({overlap_penalty:+.1f})")
    elif overlap_window > 0 and 720 - overlap_window <= minute_now <= 720 + overlap_window:
        time_penalty_total += overlap_penalty
        time_penalty_notes.append(f"Overlap ablak penalty ({overlap_penalty:+.1f})")
    if time_penalty_total:
        P += time_penalty_total
        reasons.append("; ".join(time_penalty_notes))

    if (
        not structure_components.get("bos")
        and structure_components.get("ofi")
        and P_SCORE_OFI_BONUS
    ):
        P += P_SCORE_OFI_BONUS
        reasons.append(f"OFI megerősítés (+{P_SCORE_OFI_BONUS:.1f})")

    atr_ratio = 0.0
    if not np.isnan(rel_atr) and atr_threshold > 0:
        atr_ratio = rel_atr / atr_threshold
    if atr_ok:
        atr_points = 5.0 + 6.0 * min(3.0, atr_ratio)
        P += atr_points
        reasons.append(f"ATR erősség {atr_ratio:.2f}× küszöb (+{atr_points:.1f})")
    else:
        reasons.append("ATR nincs rendben (relatív vagy abszolút küszöb)")
        P -= 8.0

    if volatility_adjustments:
        for note in volatility_adjustments:
            if note not in reasons:
                reasons.append(note)

    candidate_dir = (
        effective_bias
        if effective_bias in ("long", "short")
        else (bias1h if bias1h in ("long", "short") else None)
    )
    btc_momentum_override_active = False
    btc_momentum_override_rr: Optional[float] = None
    btc_momentum_override_desc: Optional[str] = None
    xag_momentum_override = False
    xag_atr_ratio: Optional[float] = None
    strong_momentum = False
    if candidate_dir == "long":
        strong_momentum = bool(
            bos5m_long
            or (struct_retest_long and not recent_break_short)
            or micro_bos_long
        )
        if asset == "NVDA":
            strong_momentum = strong_momentum or nvda_cross_long
    elif candidate_dir == "short":
        strong_momentum = bool(
            bos5m_short
            or (struct_retest_short and not recent_break_long)
            or micro_bos_short
        )
        if asset == "NVDA":
            strong_momentum = strong_momentum or nvda_cross_short

    if isinstance(intraday_profile, dict):
        range_position = intraday_profile.get("range_position")
        range_compression = bool(intraday_profile.get("range_compression"))
        range_expansion = bool(intraday_profile.get("range_expansion"))
        exhaustion_long = bool(intraday_profile.get("range_exhaustion_long"))
        exhaustion_short = bool(intraday_profile.get("range_exhaustion_short"))
        if effective_bias == "long":
            if exhaustion_long:
                P -= 7.0
                reasons.append("Intraday range felső harmada telített (−7.0)")
            elif range_position is not None and range_position <= INTRADAY_BALANCE_LOW:
                P += 4.0
                reasons.append("Ár az intraday range alsó zónájában (+4.0)")
        elif effective_bias == "short":
            if exhaustion_short:
                P -= 7.0
                reasons.append("Intraday range alsó harmada telített (−7.0)")
            elif range_position is not None and range_position >= INTRADAY_BALANCE_HIGH:
                P += 4.0
                reasons.append("Ár az intraday range felső zónájában (+4.0)")

        if range_compression and not strong_momentum:
            P -= 5.0
            reasons.append("Napi range <0.45×ATR — breakout előtt türelem (−5.0)")
        elif range_expansion and strong_momentum:
            P += 3.0
            reasons.append("Range expanzió támogatja a momentumot (+3.0)")

    if asset == "BTCUSD":
        volume_factor = momentum_vol_ratio if momentum_vol_ratio is not None else 0.0
        btc_range_drive = bool(
            isinstance(intraday_profile, dict) and intraday_profile.get("range_expansion")
        )
        ofi_th = float(btc_momentum_cfg.get("ofi_z", 0.0) or 0.0)
        cross_flag = False
        if candidate_dir == "long":
            cross_flag = bool(ema_cross_long)
        elif candidate_dir == "short":
            cross_flag = bool(ema_cross_short)
        ofi_condition = False
        if ofi_th > 0 and ofi_zscore is not None and candidate_dir in {"long", "short"}:
            if candidate_dir == "long":
                ofi_condition = ofi_zscore >= ofi_th
            else:
                ofi_condition = ofi_zscore <= -ofi_th
        atr_condition = bool(atr_ok and (btc_atr_floor_ratio is None or btc_atr_floor_passed))
        override_active = False
        override_rr = None
        override_desc = ""
        if candidate_dir in {"long", "short"} and btc_profile_name:
            override_active, override_rr, override_desc = btc_momentum_override(
                btc_profile_name, asset, candidate_dir, atr_condition
            )
        if override_active:
            btc_momentum_override_active = True
            btc_momentum_override_rr = override_rr
            btc_momentum_override_desc = override_desc or "BTC momentum override"
        entry_thresholds_meta["btc_atr_ratio"] = float(atr_ratio if np.isfinite(atr_ratio) else 0.0)
        entry_thresholds_meta["btc_volume_ratio"] = float(volume_factor)
        entry_thresholds_meta["btc_range_expansion"] = btc_range_drive
        entry_thresholds_meta["btc_momentum_cross"] = cross_flag
        entry_thresholds_meta["btc_momentum_ofi_ok"] = ofi_condition
        entry_thresholds_meta["btc_momentum_atr_ok"] = atr_condition
        momentum_state = globals().setdefault("_BTC_MOMENTUM_RUNTIME", {})
        if isinstance(momentum_state, dict):
            momentum_state["override"] = {
                "active": btc_momentum_override_active,
                "rr_min": btc_momentum_override_rr,
                "desc": btc_momentum_override_desc,
                "side": candidate_dir if btc_momentum_override_active else None,
            }
        if btc_momentum_override_active:
            override_note = (
                btc_momentum_override_desc
                if btc_momentum_override_desc
                else "BTC momentum override — EMA9×21 + OFI megerősítés"
            )
            if override_note not in reasons:
                reasons.append(override_note)
    if asset == "XAGUSD":
        xag_atr_ratio = float(atr_ratio) if atr_ratio > 0 else None
        if xag_atr_ratio is not None:
            xag_overrides["atr_ratio"] = xag_atr_ratio
        expansion_drive = bool(
            isinstance(intraday_profile, dict) and intraday_profile.get("range_expansion")
        )
        high_atr_drive = bool(np.isfinite(atr_ratio) and atr_ratio >= 1.45)
        xag_momentum_override = bool(strong_momentum and (high_atr_drive or expansion_drive))
        entry_thresholds_meta["xag_momentum_override"] = xag_momentum_override
        if expansion_drive:
            xag_overrides["range_expansion"] = True
        xag_overrides.setdefault(
            "preferred_timeframes",
            ["15m", "30m", "5m (scalp)"]
        )
        if xag_momentum_override:
            override_note = "XAG momentum override — stop tágítva erős trendnél"
            if override_note not in reasons:
                reasons.append(override_note)
            xag_overrides.setdefault("momentum_override", {})
            xag_overrides["momentum_override"]["atr_ratio"] = xag_atr_ratio
            xag_overrides["momentum_override"]["range_expansion"] = expansion_drive
    if asset == "USOIL" and atr1h is not None and atr1h > 0 and strong_momentum:
        momentum_buffer = float(atr1h) * USOIL_MOMENTUM_STOP_MULT
        if invalid_buffer is None or momentum_buffer > invalid_buffer:
            invalid_buffer = momentum_buffer
            usoil_overrides.setdefault("momentum_stop", {})
            usoil_overrides["momentum_stop"]["atr_multiplier"] = USOIL_MOMENTUM_STOP_MULT
            usoil_overrides["momentum_stop"]["buffer"] = momentum_buffer
            note = (
                f"USOIL: erős momentum — invalid zóna {USOIL_MOMENTUM_STOP_MULT:.1f}×ATR(1h)-re tágítva"
            )
            if note not in reasons:
                reasons.append(note)
    if asset == "XAGUSD" and xag_momentum_override:
        momentum_buffer = None
        atr_mult = 1.3
        if atr1h is not None and atr1h > 0:
            momentum_buffer = float(atr1h) * atr_mult
        elif atr5 is not None:
            try:
                momentum_buffer = float(atr5) * max(1.0, atr_mult * 0.5)
            except Exception:
                momentum_buffer = None
        if momentum_buffer and (invalid_buffer is None or momentum_buffer > invalid_buffer):
            invalid_buffer = momentum_buffer
            xag_overrides.setdefault("momentum_stop", {})
            xag_overrides["momentum_stop"]["atr_multiplier"] = atr_mult if atr1h is not None else None
            xag_overrides["momentum_stop"]["buffer"] = momentum_buffer
            note = "XAGUSD: momentum override — stop távolság 1.3×ATR-re tágítva"
            if note not in reasons:
                reasons.append(note)

    if fib_ok:
        fib_points = 18.0
        P += fib_points
        reasons.append(f"Fib zóna konfluencia (0.618–0.886) (+{fib_points:.1f})")
    elif ema21_dist_ok:
        ema_points = 10.0
        P += ema_points
        reasons.append(f"Ár 1h EMA21 zónában (+{ema_points:.1f})")

    of_imb = order_flow_metrics.get("imbalance")
    if of_imb is not None:
        if abs(of_imb) >= ORDER_FLOW_IMBALANCE_TH:
            boost = 8.0 * min(1.0, abs(of_imb))
            P += boost
            reasons.append(f"Order flow imbalance {of_imb:+.2f} (+{boost:.1f})")
        else:
            penalty = 5.0 * (ORDER_FLOW_IMBALANCE_TH - abs(of_imb)) / ORDER_FLOW_IMBALANCE_TH
            P -= penalty
            reasons.append(f"Order flow neutrális (−{penalty:.1f})")

    of_pressure = order_flow_metrics.get("pressure")
    if of_pressure is not None and effective_bias in {"long", "short"}:
        if effective_bias == "long" and of_pressure > ORDER_FLOW_PRESSURE_TH:
            P += 5.0
            reasons.append("Buy pressure támogatja a setupot (+5)")
        elif effective_bias == "short" and of_pressure < -ORDER_FLOW_PRESSURE_TH:
            P += 5.0
            reasons.append("Sell pressure támogatja a setupot (+5)")
        elif abs(of_pressure) > ORDER_FLOW_PRESSURE_TH:
            P -= 4.0
            reasons.append("Order flow ellentétes irányba tolódik (−4)")

    if ofi_zscore is not None and effective_bias in {"long", "short"} and OFI_Z_TRIGGER > 0:
        if effective_bias == "long" and ofi_zscore <= -OFI_Z_TRIGGER:
            penalty = min(8.0, (abs(ofi_zscore) - OFI_Z_TRIGGER + 1.0) * 3.0)
            P -= penalty
            reasons.append(f"OFI toxikus long irányra (−{penalty:.1f})")
        elif effective_bias == "short" and ofi_zscore >= OFI_Z_TRIGGER:
            penalty = min(8.0, (abs(ofi_zscore) - OFI_Z_TRIGGER + 1.0) * 3.0)
            P -= penalty
            reasons.append(f"OFI toxikus short irányra (−{penalty:.1f})")

    aggressor_ratio = order_flow_metrics.get("aggressor_ratio")
    if aggressor_ratio is not None and effective_bias in {"long", "short"}:
        if effective_bias == "long" and aggressor_ratio > 0.6:
            P += 4.0
            reasons.append("Aggresszív vevők dominálnak (+4)")
        elif effective_bias == "short" and aggressor_ratio < -0.6:
            P += 4.0
            reasons.append("Aggresszív eladók dominálnak (+4)")
        elif abs(aggressor_ratio) < 0.2:
            P -= 2.0
            reasons.append("Aggresszor arány neutrális (−2)")

    stale_penalty = 0.0
    if stale_timeframes.get("k5m"):
        stale_penalty += 22.0
        reasons.append("5m adat késés — momentum pontok csökkentve")
    if stale_timeframes.get("k1h") or stale_timeframes.get("k4h"):
        stale_penalty += 9.0
        reasons.append("HTF adat késés — trend pontok csökkentve")
    if stale_penalty:
        P = max(0.0, P - stale_penalty)

    if realtime_jump_flag:
        P -= 6.0
        reasons.append("Realtime ár >1.5×ATR eltérés — extra validáció szükséges")

    if realtime_confidence < 0.6:
        penalty = (0.6 - realtime_confidence) * 12.0
        P -= penalty
        reasons.append(f"Realtime megbízhatóság alacsony (−{penalty:.1f})")

    if anchor_drift_state == "deteriorating":
        P -= 5.0
        reasons.append("Anchor trail drift romlik (−5)")
    elif anchor_drift_state == "improving":
        P += 3.0
        reasons.append("Anchor trail drift javul (+3)")

    smt_pen, smt_reason = smt_penalty(asset, k5m_closed)
    if smt_pen and smt_reason:
        effective_smt_pen = smt_pen
        if asset == "EURUSD":
            effective_smt_pen = min(effective_smt_pen, 3)
        P -= effective_smt_pen
        reasons.append(f"SMT büntetés −{effective_smt_pen}% ({smt_reason})")

    if sentiment_signal and ENABLE_SENTIMENT_PROBABILITY:
        sentiment_points = 0.0
        if effective_bias == "long":
            sentiment_points = 8.0 * sentiment_signal.score
        elif effective_bias == "short":
            sentiment_points = -8.0 * sentiment_signal.score
        else:
            sentiment_points = 4.0 * sentiment_signal.score
        severity = sentiment_signal.effective_severity
        sentiment_points *= severity
        P += sentiment_points
        sentiment_msg = (
            f"News sentiment ({sentiment_signal.bias}) {sentiment_signal.score:+.2f}"
        )
        if severity not in {1.0, 1}:
            sentiment_msg += f" × severity {severity:.2f}"
        if sentiment_points >= 0:
            reasons.append(f"{sentiment_msg} (+{sentiment_points:.1f})")
        else:
            reasons.append(f"{sentiment_msg} ({sentiment_points:.1f})")
    elif sentiment_signal:
        reasons.append(
            "Makrohír sentiment jelen van, de a pontozás kikapcsolva (ENABLE_SENTIMENT_PROBABILITY=1 esetén aktiválható)."
        )

    P = max(0.0, min(100.0, P))

    base_p_score = P

    dynamic_score_engine = DynamicScoreEngine(
        dynamic_logic_cfg if isinstance(dynamic_logic_cfg, dict) else {},
        validate_config=False,
    )
    volatility_score_data = {
        "volatility_z": overlay_ratio,
        "regime": overlay_regime,
    }
    P, dynamic_score_notes, dynamic_score_meta = dynamic_score_engine.score(
        base_p_score,
        regime_snapshot,
        volatility_score_data,
        atr_soft_gate_penalty=atr_soft_penalty,
        latency_penalty=latency_penalty,
    )

    # Sanity clamp to ensure persisted P-score remains in [0, 100].
    P = max(0.0, min(100.0, P))

    if dynamic_score_notes:
        reasons.extend(dynamic_score_notes)
    if dynamic_score_meta:
        entry_thresholds_meta["dynamic_score_engine"] = dynamic_score_meta

    range_guard_ok = True
    range_guard_reason: Optional[str] = None
    if isinstance(intraday_profile, dict):
        exhaustion_long = bool(intraday_profile.get("range_exhaustion_long"))
        exhaustion_short = bool(intraday_profile.get("range_exhaustion_short"))
        allow_override = bool(
            intraday_profile.get("range_expansion") and strong_momentum
        )
        if effective_bias == "long" and exhaustion_long and not allow_override:
            range_guard_ok = False
            range_guard_reason = "Intraday range felső része telített — visszahúzódásra várunk."
        elif effective_bias == "short" and exhaustion_short and not allow_override:
            range_guard_ok = False
            range_guard_reason = "Intraday range alsó része telített — visszapattanásra várunk."
    if range_guard_reason and range_guard_reason not in reasons:
        reasons.append(range_guard_reason)

    # --- Kapuk (liquidity = Fib zóna VAGY sweep) + session + regime ---
    nvda_open_window = False
    nvda_close_window = False
    if asset == "NVDA":
        h, m = now_utctime_hm()
        minute = h * 60 + m
        cash_start = _min_of_day(13, 30)
        cash_end = _min_of_day(20, 0)
        in_cash_session = cash_start <= minute <= cash_end
        nvda_open_window = cash_start <= minute <= min(cash_end, cash_start + 60)
        nvda_close_window = max(cash_start, cash_end - 30) <= minute <= cash_end
        high_atr_for_extended_hours = (
            not np.isnan(rel_atr)
            and rel_atr >= NVDA_EXTENDED_ATR_REL
            and atr_abs_ok
        )
        if not in_cash_session and high_atr_for_extended_hours:
            session_ok_flag = True
            session_meta["entry_open"] = True
            session_meta["within_window"] = True
            session_meta.setdefault("notes", []).append(
                "Extended hours engedélyezve magas ATR miatt"
            )
        elif not in_cash_session and not high_atr_for_extended_hours:
            session_meta.setdefault("notes", []).append(
                "Cash sessionen kívül — ATR nem elég magas a kereskedéshez"
            )
        else:
            session_meta.setdefault("notes", [])
            if nvda_open_window:
                session_meta["notes"].append("NVDA nyitási lendület ablak aktív (0-60 perc)")
            if nvda_close_window:
                session_meta["notes"].append("NVDA zárás előtti 30 perc — VWAP konfluencia kiemelve")

        entry_thresholds_meta["nvda_daily_atr"] = nvda_daily_atr
        entry_thresholds_meta["nvda_daily_atr_rel"] = nvda_daily_atr_rel
        entry_thresholds_meta.setdefault("nvda_overrides", {})
        entry_thresholds_meta["nvda_overrides"].setdefault(
            "session_windows",
            {"open": nvda_open_window, "close": nvda_close_window},
        )

    entry_thresholds_meta["ofi_z"] = ofi_zscore

    atr_abs_for_vwap = None
    try:
        if atr5 is not None and np.isfinite(float(atr5)):
            atr_abs_for_vwap = float(atr5)
    except Exception:
        atr_abs_for_vwap = None
    vwap_confluence = evaluate_vwap_confluence(
        asset,
        effective_bias if effective_bias in {"long", "short"} else None,
        "trend" if adx_regime == "trend" else "range" if adx_regime == "range" else "balanced",
        price_for_calc if price_for_calc is not None else None,
        atr_abs_for_vwap,
        k1m_closed,
        ofi_zscore,
    )
    entry_thresholds_meta["vwap_confluence"] = vwap_confluence

    eurusd_price_above_vwap = False
    eurusd_price_below_vwap = False
    eurusd_vwap_break_long = False
    eurusd_vwap_break_short = False
    usoil_price_above_vwap = False
    usoil_price_below_vwap = False
    usoil_vwap_bias: str = "neutral"
    usoil_gap_break_long = False
    usoil_gap_break_short = False
    if asset == "EURUSD":
        vwap_distance_val = (
            safe_float(vwap_confluence.get("distance"))
            if isinstance(vwap_confluence, dict)
            else None
        )
        eurusd_price_above_vwap = vwap_distance_val is not None and vwap_distance_val >= 0
        eurusd_price_below_vwap = vwap_distance_val is not None and vwap_distance_val <= 0
        bos_alignment_long = bool(bos1h_long or bos5m_long or micro_bos_long)
        bos_alignment_short = bool(bos1h_short or bos5m_short or micro_bos_short)
        eurusd_vwap_break_long = bool(eurusd_price_above_vwap and bos_alignment_long)
        eurusd_vwap_break_short = bool(eurusd_price_below_vwap and bos_alignment_short)
        eurusd_overrides["vwap_alignment"] = {
            "distance": vwap_distance_val,
            "above": eurusd_price_above_vwap,
            "below": eurusd_price_below_vwap,
            "bos_long": bos_alignment_long,
            "bos_short": bos_alignment_short,
        }
    elif asset == "USOIL":
        vwap_distance_val = (
            safe_float(vwap_confluence.get("distance"))
            if isinstance(vwap_confluence, dict)
            else None
        )
        usoil_price_above_vwap = vwap_distance_val is not None and vwap_distance_val >= 0
        usoil_price_below_vwap = vwap_distance_val is not None and vwap_distance_val <= 0
        usoil_overrides["vwap_distance"] = vwap_distance_val
        vwap_bias_meta: Dict[str, Any] = {"lookback_minutes": 0}
        vwap_series = compute_vwap(k1m_closed)
        if vwap_series is not None and not vwap_series.empty:
            joined = (
                pd.DataFrame({"close": k1m_closed["close"], "vwap": vwap_series})
                .dropna()
            )
            if not joined.empty:
                lookback = min(USOIL_VWAP_BIAS_LOOKBACK, len(joined))
                recent = joined.tail(lookback)
                vwap_bias_meta["lookback_minutes"] = lookback
                if len(recent) >= max(60, lookback // 4):
                    above_ratio = float((recent["close"] > recent["vwap"]).mean())
                    below_ratio = float((recent["close"] < recent["vwap"]).mean())
                    vwap_bias_meta["above_ratio"] = above_ratio
                    vwap_bias_meta["below_ratio"] = below_ratio
                    if above_ratio >= USOIL_VWAP_BIAS_RATIO:
                        usoil_vwap_bias = "long"
                    elif below_ratio >= USOIL_VWAP_BIAS_RATIO:
                        usoil_vwap_bias = "short"
        vwap_bias_meta.setdefault("above_ratio", None)
        vwap_bias_meta.setdefault("below_ratio", None)
        vwap_bias_meta["bias"] = usoil_vwap_bias
        usoil_overrides["vwap_bias"] = vwap_bias_meta
        opening_gap_pct = safe_float(
            intraday_profile.get("opening_gap_pct") if isinstance(intraday_profile, dict) else None
        )
        opening_gap_direction = (
            (intraday_profile.get("opening_gap_direction") or "flat")
            if isinstance(intraday_profile, dict)
            else "flat"
        )
        gap_meta: Dict[str, Any] = {
            "pct": opening_gap_pct,
            "direction": opening_gap_direction,
        }
        if (
            opening_gap_pct is not None
            and abs(opening_gap_pct) >= USOIL_GAP_VWAP_MIN_PCT
            and vwap_distance_val is not None
        ):
            if opening_gap_pct > 0 and vwap_distance_val >= 0 and (bos1h_long or bos5m_long):
                usoil_gap_break_long = True
            elif opening_gap_pct < 0 and vwap_distance_val <= 0 and (bos1h_short or bos5m_short):
                usoil_gap_break_short = True
        gap_meta["gap_vwap_break_long"] = usoil_gap_break_long
        gap_meta["gap_vwap_break_short"] = usoil_gap_break_short
        usoil_overrides["gap_context"] = gap_meta

    if asset in {"EURUSD", "BTCUSD"}:
        liquidity_ok_base = bool(fib_ok)
    elif asset == "NVDA":
        liquidity_ok_base = bool(
            fib_ok
            or (effective_bias == "long" and struct_retest_long)
            or (effective_bias == "short" and struct_retest_short)
        )
    else:
        liquidity_ok_base = bool(
            fib_ok
            or swept
            or ema21_dist_ok
            or (effective_bias == "long" and struct_retest_long)
            or (effective_bias == "short" and struct_retest_short)
        )
    liquidity_relaxed = False
    liquidity_ok = (
        liquidity_ok_base
        or bool(vwap_confluence.get("trend_pullback"))
        or bool(vwap_confluence.get("mean_revert"))
    )
    if vwap_confluence.get("trend_pullback"):
        structure_notes.append("VWAP pullback konfluencia aktív — trend pullback engedve")
    if vwap_confluence.get("mean_revert"):
        structure_notes.append("VWAP túlnyúlás → mean reversion setup")
    directional_confirmation = False
    if candidate_dir == "long":
        directional_confirmation = bool(bos5m_long or micro_bos_long)
    elif candidate_dir == "short":
        directional_confirmation = bool(bos5m_short or micro_bos_short)
    if not liquidity_ok_base and strong_momentum:
        high_atr_push = bool(
            atr_ok and not np.isnan(rel_atr)
            and rel_atr >= max(atr_threshold * 1.3, MOMENTUM_ATR_REL)
        )
        nvda_momentum_liquidity = bool(
            asset == "NVDA"
            and nvda_daily_atr is not None
            and NVDA_DAILY_ATR_STRONG > 0
            and nvda_daily_atr >= NVDA_DAILY_ATR_STRONG
        )
        if (high_atr_push or nvda_momentum_liquidity) and directional_confirmation:
            liquidity_ok = True
            liquidity_relaxed = True
            if nvda_momentum_liquidity:
                note = "NVDA momentum override — napi ATR támogatja a likviditási lazítást"
                if note not in structure_notes:
                    structure_notes.append(note)

    if asset == "BTCUSD" and btc_momentum_override_active and not liquidity_ok:
        liquidity_ok = True
        liquidity_relaxed = True
        if "BTC momentum override — likviditási kapu lazítva" not in structure_notes:
            structure_notes.append("BTC momentum override — likviditási kapu lazítva")

    structure_components = {"bos": False, "liquidity": False, "ofi": False}
    if effective_bias == "long":
        bos_signal = bool(bos5m_long or micro_bos_long)
        if asset == "NVDA":
            bos_signal = bos_signal or nvda_cross_long
        structure_components["bos"] = bos_signal and not recent_break_short
    elif effective_bias == "short":
        bos_signal = bool(bos5m_short or micro_bos_short)
        if asset == "NVDA":
            bos_signal = bos_signal or nvda_cross_short
        structure_components["bos"] = bos_signal and not recent_break_long

    btc_structure_combos: Dict[str, Dict[str, Any]] = {}
    if asset == "BTCUSD":
        vwap_distance_base = safe_float(vwap_confluence.get("distance")) if isinstance(vwap_confluence, dict) else None
        price_above_vwap_base = vwap_distance_base is not None and vwap_distance_base >= 0
        price_below_vwap_base = vwap_distance_base is not None and vwap_distance_base <= 0
        vwap_trend_pullback = bool(vwap_confluence.get("trend_pullback"))
        structure_cfg = _btc_profile_section("structure")
        ofi_gate = float(structure_cfg.get("ofi_gate", 1.0) or 1.0)
        btc_structure_combos = {
            "long": {
                "micro": bool(micro_bos_long and not recent_break_short),
                "vwap": bool(price_above_vwap_base or vwap_trend_pullback),
                "ofi": bool(ofi_gate > 0 and ofi_zscore is not None and ofi_zscore >= ofi_gate),
                "price_above_vwap": price_above_vwap_base,
                "price_below_vwap": price_below_vwap_base,
            },
            "short": {
                "micro": bool(micro_bos_short and not recent_break_long),
                "vwap": bool(price_below_vwap_base or vwap_trend_pullback),
                "ofi": bool(ofi_gate > 0 and ofi_zscore is not None and ofi_zscore <= -ofi_gate),
                "price_above_vwap": price_above_vwap_base,
                "price_below_vwap": price_below_vwap_base,
            },
        }
        micro_runtime = globals().setdefault("_BTC_STRUCT_MICRO", {})
        vwap_runtime = globals().setdefault("_BTC_STRUCT_VWAP", {})
        if isinstance(micro_runtime, dict):
            micro_runtime["long"] = btc_structure_combos["long"].get("micro")
            micro_runtime["short"] = btc_structure_combos["short"].get("micro")
        if isinstance(vwap_runtime, dict):
            vwap_runtime["long"] = btc_structure_combos["long"].get("vwap")
            vwap_runtime["short"] = btc_structure_combos["short"].get("vwap")

    if asset == "EURUSD" and effective_bias in {"long", "short"}:
        if effective_bias == "long":
            higher_timeframe_break = bool(bos1h_long or bos5m_long)
            structure_components["bos"] = (
                eurusd_vwap_break_long and higher_timeframe_break and not recent_break_short
            )
            if structure_components["bos"]:
                structure_notes.append("EURUSD BOS + VWAP felett — trend folytatás megerősítve")
            elif higher_timeframe_break and not eurusd_price_above_vwap:
                structure_notes.append("EURUSD BOS aktív, de VWAP alatt — türelem")
            if eurusd_price_above_vwap or vwap_confluence.get("trend_pullback"):
                structure_components["liquidity"] = True
        elif effective_bias == "short":
            higher_timeframe_break = bool(bos1h_short or bos5m_short)
            structure_components["bos"] = (
                eurusd_vwap_break_short and higher_timeframe_break and not recent_break_long
            )
            if structure_components["bos"]:
                structure_notes.append("EURUSD BOS + VWAP alatt — bearish folytatás megerősítve")
            elif higher_timeframe_break and not eurusd_price_below_vwap:
                structure_notes.append("EURUSD BOS aktív, de VWAP felett — várakozás")
            if eurusd_price_below_vwap or vwap_confluence.get("trend_pullback"):
                structure_components["liquidity"] = True
        if ofi_zscore is not None and structure_components["bos"]:
            if effective_bias == "long" and ofi_zscore >= OFI_Z_TRIGGER:
                structure_notes.append("OFI long irányt támogatja — EURUSD konfluencia")
            elif effective_bias == "short" and ofi_zscore <= -OFI_Z_TRIGGER:
                structure_notes.append("OFI short irányt támogatja — EURUSD konfluencia")
    if asset == "USOIL" and effective_bias in {"long", "short"}:
        if usoil_vwap_bias == "long" and effective_bias == "long":
            structure_components["liquidity"] = True
            note = "USOIL: ár tartósan VWAP felett — long bias támogatva"
            if note not in structure_notes:
                structure_notes.append(note)
        elif usoil_vwap_bias == "short" and effective_bias == "short":
            structure_components["liquidity"] = True
            note = "USOIL: ár tartósan VWAP alatt — short bias támogatva"
            if note not in structure_notes:
                structure_notes.append(note)
        if effective_bias == "long":
            higher_break = bool(bos1h_long)
            if usoil_gap_break_long:
                structure_components["bos"] = True
                gap_note = "USOIL: Gap + VWAP törés — long setup prioritás"
                if gap_note not in structure_notes:
                    structure_notes.append(gap_note)
            elif not higher_break:
                structure_components["bos"] = False
                tighten_note = "USOIL: oldalazó szerkezet — 1h BOS nélkül nincs belépő"
                if tighten_note not in structure_notes:
                    structure_notes.append(tighten_note)
        elif effective_bias == "short":
            higher_break = bool(bos1h_short)
            if usoil_gap_break_short:
                structure_components["bos"] = True
                gap_note = "USOIL: Gap + VWAP törés — short setup prioritás"
                if gap_note not in structure_notes:
                    structure_notes.append(gap_note)
            elif not higher_break:
                structure_components["bos"] = False
                tighten_note = "USOIL: oldalazó szerkezet — 1h BOS nélkül nincs belépő"
                if tighten_note not in structure_notes:
                    structure_notes.append(tighten_note)

    elif asset == "GOLD_CFD" and effective_bias in {"long", "short"}:
        vwap_distance = safe_float(vwap_confluence.get("distance")) if isinstance(vwap_confluence, dict) else None
        price_above_vwap = vwap_distance is not None and vwap_distance >= 0
        price_below_vwap = vwap_distance is not None and vwap_distance <= 0
        if effective_bias == "long":
            higher_timeframe_break = bool(bos1h_long or (bos5m_long and micro_bos_long))
            structure_components["bos"] = higher_timeframe_break and price_above_vwap and not recent_break_short
            if higher_timeframe_break and price_above_vwap:
                structure_notes.append("H1 kitörés + VWAP felett — GOLD long konfluencia")
            elif higher_timeframe_break and not price_above_vwap:
                structure_notes.append("VWAP alatt — GOLD long breakout késleltetve")
        elif effective_bias == "short":
            higher_timeframe_break = bool(bos1h_short or (bos5m_short and micro_bos_short))
            structure_components["bos"] = higher_timeframe_break and price_below_vwap and not recent_break_long
            if higher_timeframe_break and price_below_vwap:
                structure_notes.append("H1 letörés + VWAP alatt — GOLD short konfluencia")
            elif higher_timeframe_break and not price_below_vwap:
                structure_notes.append("VWAP felett — GOLD short breakout várakozik")
        if effective_bias == "long" and price_above_vwap:
            structure_components["liquidity"] = True
        if effective_bias == "short" and price_below_vwap:
            structure_components["liquidity"] = True

    elif asset == "XAGUSD" and effective_bias in {"long", "short"}:
        vwap_distance = safe_float(vwap_confluence.get("distance")) if isinstance(vwap_confluence, dict) else None
        price_above_vwap = vwap_distance is not None and vwap_distance >= 0
        price_below_vwap = vwap_distance is not None and vwap_distance <= 0
        higher_break_long = bool(bos1h_long or bos5m_long or struct_retest_long)
        higher_break_short = bool(bos1h_short or bos5m_short or struct_retest_short)
        micro_long = bool(micro_bos_long and not micro_bos_short)
        micro_short = bool(micro_bos_short and not micro_bos_long)
        xag_structure_meta: Dict[str, Any] = {
            "price_above_vwap": price_above_vwap,
            "price_below_vwap": price_below_vwap,
            "higher_break_long": higher_break_long,
            "higher_break_short": higher_break_short,
            "micro_long": micro_long,
            "micro_short": micro_short,
        }
        xag_overrides.setdefault("structure", {}).update(xag_structure_meta)
        if effective_bias == "long":
            long_break = bool(higher_break_long and price_above_vwap and not recent_break_short)
            if not long_break and price_above_vwap and micro_long and (strong_momentum or xag_momentum_override):
                long_break = True
                structure_notes.append("XAG: mikro HL/HH + VWAP felett — gyors long reakció")
            elif not price_above_vwap:
                structure_notes.append("XAG: VWAP alatt — long setup türelmet igényel")
            structure_components["bos"] = long_break
            if price_above_vwap:
                structure_components["liquidity"] = True
        elif effective_bias == "short":
            short_break = bool(higher_break_short and price_below_vwap and not recent_break_long)
            if not short_break and price_below_vwap and micro_short and (strong_momentum or xag_momentum_override):
                short_break = True
                structure_notes.append("XAG: mikro LH/LL + VWAP alatt — gyors short reakció")
            elif not price_below_vwap:
                structure_notes.append("XAG: VWAP felett — short setup türelmet igényel")
            structure_components["bos"] = short_break
            if price_below_vwap:
                structure_components["liquidity"] = True
        if xag_momentum_override and structure_components.get("bos"):
            structure_components["liquidity"] = True
            momentum_note = "XAG momentum override — szerkezeti break elfogadva VWAP oldalán"
            if momentum_note not in structure_notes:
                structure_notes.append(momentum_note)

    elif asset == "BTCUSD":
        combo = btc_structure_combos.get(effective_bias) if effective_bias in {"long", "short"} else None
        price_above_vwap = bool(combo.get("price_above_vwap")) if combo else False
        price_below_vwap = bool(combo.get("price_below_vwap")) if combo else False
        side = effective_bias if effective_bias in {"long", "short"} else ""
        btc_trigger_meta: Dict[str, Any] = {"bos_ok": False, "vwap_ok": False, "ofi_ok": False, "ofi_z": None}
        try:
            btc_triggers_ok, btc_trigger_meta = btc_core_triggers_ok(asset, side)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("BTC core trigger evaluation failed: %s", exc)
            btc_triggers_ok = False
        micro_ok = bool(btc_trigger_meta.get("bos_ok"))
        vwap_ok = bool(btc_trigger_meta.get("vwap_ok"))
        ofi_ok = bool(btc_trigger_meta.get("ofi_ok"))
        satisfied_combo = int(micro_ok) + int(vwap_ok) + int(ofi_ok)
        if effective_bias == "long":
            if micro_ok:
                structure_notes.append("BTC mikro BOS long aktív")
            if vwap_ok and not price_above_vwap:
                structure_notes.append("BTC VWAP reclaim long — trend pullback engedve")
        elif effective_bias == "short":
            if micro_ok:
                structure_notes.append("BTC mikro BOS short aktív")
            if vwap_ok and not price_below_vwap:
                structure_notes.append("BTC VWAP reclaim short — trend pullback engedve")
        structure_components["bos"] = micro_ok
        structure_components["liquidity"] = vwap_ok
        structure_components["ofi"] = ofi_ok
        entry_thresholds_meta["btc_core_triggers"] = dict(btc_trigger_meta, hits=satisfied_combo)
        entry_thresholds_meta["btc_structure_combo"] = {
            "active_direction": effective_bias if combo else None,
            "active": {
                "micro": micro_ok,
                "vwap": vwap_ok,
                "ofi": ofi_ok,
                "count": satisfied_combo,
            },
            "combos": btc_structure_combos,
            "either_of_ok": bool(btc_triggers_ok),
        }

    if asset == "BTCUSD":
        structure_components["liquidity"] = bool(structure_components["liquidity"])
    else:
        structure_components["liquidity"] = bool(
            structure_components["liquidity"]
            or liquidity_ok
            or liquidity_ok_base
            or vwap_confluence.get("trend_pullback")
            or vwap_confluence.get("mean_revert")
        )
    if asset == "NVDA" and nvda_close_window:
        vwap_distance = safe_float(vwap_confluence.get("distance"))
        if vwap_distance is not None:
            if effective_bias == "long" and vwap_distance >= 0:
                structure_components["liquidity"] = True
                note = "NVDA: zárási VWAP felett stabilizáció — long likviditás"
                if note not in structure_notes:
                    structure_notes.append(note)
            elif effective_bias == "short" and vwap_distance <= 0:
                structure_components["liquidity"] = True
                note = "NVDA: zárási VWAP alatt stabilizáció — short likviditás"
                if note not in structure_notes:
                    structure_notes.append(note)
    if asset == "NVDA" and nvda_open_window and not structure_components["bos"]:
        if effective_bias == "long" and (bos5m_long or micro_bos_long or nvda_cross_long):
            structure_components["bos"] = True
            note = "NVDA: nyitási BOS/momentum elfogadva — early session bias"
            if note not in structure_notes:
                structure_notes.append(note)
        elif effective_bias == "short" and (bos5m_short or micro_bos_short or nvda_cross_short):
            structure_components["bos"] = True
            note = "NVDA: nyitási BOS/momentum elfogadva — early session bias"
            if note not in structure_notes:
                structure_notes.append(note)
    if ofi_zscore is not None and OFI_Z_TRIGGER > 0:
        if effective_bias == "long":
            structure_components["ofi"] = ofi_zscore >= OFI_Z_TRIGGER
        elif effective_bias == "short":
            structure_components["ofi"] = ofi_zscore <= -OFI_Z_TRIGGER
    if structure_components.get("ofi") and ofi_zscore is not None:
        structure_notes.append(f"OFI z-score {ofi_zscore:.2f} támogatja az irányt")
    structure_gate = sum(1 for flag in structure_components.values() if flag) >= 2
    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        entry_gate_context_hu["structure_kapu"] = {
            "ok": bool(structure_gate),
            "komponensek": {k: bool(v) for k, v in structure_components.items()},
            **_gate_timestamp_fields(analysis_now),
        }
    if asset == "BTCUSD" and btc_momentum_override_active and not structure_gate:
        if structure_components.get("bos") and structure_components.get("ofi"):
            structure_gate = True
        elif structure_components.get("liquidity") and structure_components.get("ofi"):
            structure_gate = True
    if asset == "BTCUSD" and not btc_momentum_override_active and not structure_gate:
        profile_lookup = btc_profile_name or _btc_active_profile()
        try:
            p_min = p_score_min_local  # type: ignore[name-defined]
        except NameError:
            p_min = BTC_P_SCORE_MIN.get(profile_lookup, BTC_P_SCORE_MIN["baseline"])
        p_min_val = None
        try:
            if p_min is not None:
                p_min_val = float(p_min)
        except (TypeError, ValueError):
            p_min_val = None
        P_val = None
        try:
            if P is not None:
                P_val = float(P)
        except (TypeError, ValueError):
            P_val = None
        atr_gate_th = None
        try:
            atr_gate_th = float(atr_threshold)  # type: ignore[name-defined]
        except (TypeError, ValueError, NameError):
            atr_gate_th = None
        rel_atr_val = None
        try:
            if rel_atr is not None:
                rel_atr_val = float(rel_atr)
        except (TypeError, ValueError):
            rel_atr_val = None
        good_pscore = (
            p_min_val is not None
            and P_val is not None
            and np.isfinite(p_min_val)
            and np.isfinite(P_val)
            and P_val >= (p_min_val + 6.0)
        )
        good_vola = (
            rel_atr_val is not None
            and atr_gate_th is not None
            and np.isfinite(rel_atr_val)
            and np.isfinite(atr_gate_th)
            and rel_atr_val >= (atr_gate_th * 1.15)
        )
        one_of_three = bool(micro_ok or vwap_ok or ofi_ok)
        if good_pscore and good_vola and one_of_three:
            structure_gate = True
            entry_thresholds_meta["core_relax_applied"] = True
            entry_thresholds_meta.setdefault("core_relax_context", {})
            entry_thresholds_meta["core_relax_context"].update(
                {
                    "p_score": P_val,
                    "p_score_min": p_min_val,
                    "rel_atr": rel_atr_val,
                    "atr_threshold": atr_gate_th,
                    "signals": {
                        "micro": micro_ok,
                        "vwap": vwap_ok,
                        "ofi": ofi_ok,
                    },
                }
            )
    entry_thresholds_meta["structure_components"] = structure_components
    if asset == "BTCUSD":
        entry_thresholds_meta["btc_momentum_override"] = btc_momentum_override_active
        if effective_bias == "neutral" and btc_structure_combos:
            combo_th = float(btc_bias_cfg.get("combo_threshold", 0.0) or 0.0)
            long_combo = btc_structure_combos.get("long") or {}
            short_combo = btc_structure_combos.get("short") or {}
            selected_bias: Optional[str] = None
            profile_lookup = btc_profile_name or _btc_active_profile()
            if profile_lookup == "relaxed":
                if long_combo.get("vwap") and long_combo.get("ofi"):
                    selected_bias = "long"
                elif short_combo.get("vwap") and short_combo.get("ofi"):
                    selected_bias = "short"
            elif profile_lookup == "suppressed":
                long_count = sum(1 for key in ("micro", "vwap", "ofi") if long_combo.get(key))
                short_count = sum(1 for key in ("micro", "vwap", "ofi") if short_combo.get(key))
                threshold = 2
                if combo_th >= 3:
                    threshold = 3
                if long_count >= threshold and long_count >= short_count:
                    selected_bias = "long"
                elif short_count >= threshold:
                    selected_bias = "short"
            if selected_bias:
                effective_bias = selected_bias
                bias_override_used = True
                bias_override_reason = f"Bias override: BTC intraday combo {selected_bias}"
                note = (
                    "BTC bias override — VWAP + OFI megerősítés"
                    if profile_lookup == "relaxed"
                    else "BTC bias override — 2/3 momentum kapu"
                )
                if note not in reasons:
                    reasons.append(note)

    p_score_min_base = get_p_score_min(asset)
    p_score_min_local = p_score_min_base
    if asset == "BTCUSD":
        profile = btc_profile_name or _btc_active_profile()
        override = BTC_P_SCORE_MIN.get(profile)
        if override is not None:
            p_score_min_base = float(override)
            p_score_min_local = float(override)
            entry_thresholds_meta["p_score_min_profile_override"] = {
                "profile": profile,
                "value": float(override),
            }
    entry_thresholds_meta["p_score_min_base"] = p_score_min_base
    eurusd_p_score_meta: Dict[str, Any] = {}
    if asset == "EURUSD":
        atr1h_pips = None
        if atr1h is not None and atr1h > 0:
            atr1h_pips = float(atr1h) / EURUSD_PIP
            eurusd_overrides.setdefault("atr1h_pips", atr1h_pips)
        if atr1h_pips is not None and atr1h_pips < EURUSD_ATR_PIPS_LOW:
            p_score_min_local += EURUSD_P_SCORE_LOW_VOL_ADD
            eurusd_p_score_meta["low_vol_add"] = EURUSD_P_SCORE_LOW_VOL_ADD
            reasons.append(
                "EURUSD: ATR(1h) alacsony — P-score küszöb +6 pontra emelve"
            )
        if (
            atr1h_pips is not None
            and atr1h_pips >= EURUSD_ATR_PIPS_HIGH
            and strong_momentum
        ):
            p_score_min_local += EURUSD_P_SCORE_MOMENTUM_ADD
            eurusd_p_score_meta["momentum_add"] = EURUSD_P_SCORE_MOMENTUM_ADD
            reasons.append(
                "EURUSD: Momentum override aktív — extra P-score követelmény (+2)"
            )
        if eurusd_p_score_meta:
            eurusd_overrides["p_score_adjustments"] = eurusd_p_score_meta
    if asset == "NVDA":
        nvda_p_score_meta: Dict[str, Any] = {}
        if nvda_daily_atr is not None:
            nvda_p_score_meta["daily_atr"] = nvda_daily_atr
            if NVDA_DAILY_ATR_MIN > 0 and nvda_daily_atr < NVDA_DAILY_ATR_MIN:
                p_score_min_local += NVDA_LOW_ATR_P_SCORE_ADD
                nvda_p_score_meta["low_atr_add"] = NVDA_LOW_ATR_P_SCORE_ADD
                reasons.append(
                    "NVDA: napi ATR becslés alacsony — P-score küszöb emelve"
                )
            elif (
                NVDA_DAILY_ATR_STRONG > 0
                and nvda_daily_atr >= NVDA_DAILY_ATR_STRONG
                and strong_momentum
            ):
                relax = float(NVDA_RR_BANDS.get("momentum_p_score_relax") or 0.0)
                if relax > 0:
                    p_score_min_local = max(0.0, p_score_min_local - relax)
                    nvda_p_score_meta["momentum_relax"] = relax
                    reasons.append(
                        "NVDA: magas napi ATR + momentum — P-score küszöb lazítva"
                    )
        if nvda_daily_atr_rel is not None:
            nvda_p_score_meta["daily_atr_rel"] = nvda_daily_atr_rel
        if nvda_p_score_meta:
            entry_thresholds_meta.setdefault("nvda_overrides", {})
            entry_thresholds_meta["nvda_overrides"]["p_score"] = nvda_p_score_meta
    if asset == "USOIL":
        usoil_p_score_meta: Dict[str, Any] = {}
        if atr1h is not None and atr1h > 0:
            usoil_p_score_meta["atr1h"] = float(atr1h)
            if atr1h < USOIL_ATR1H_BREAKOUT_MIN:
                p_score_min_local += USOIL_P_SCORE_LOW_ATR_ADD
                usoil_p_score_meta["low_atr_add"] = USOIL_P_SCORE_LOW_ATR_ADD
                reasons.append(
                    "USOIL: 1h ATR <1 USD — P-score küszöb emelve a zajos sáv miatt"
                )
        if isinstance(intraday_profile, dict):
            if intraday_profile.get("range_compression") and not strong_momentum:
                p_score_min_local += USOIL_P_SCORE_SIDEWAYS_ADD
                usoil_p_score_meta["sideways_add"] = USOIL_P_SCORE_SIDEWAYS_ADD
                reasons.append(
                    "USOIL: oldalazásban szigorított P-score — kevesebb fals jelzés"
                )
        if usoil_p_score_meta:
            usoil_overrides.setdefault("p_score_adjustments", {})
            usoil_overrides["p_score_adjustments"].update(usoil_p_score_meta)
    if asset == "BTCUSD" and intervention_band in {"HIGH", "EXTREME"} and effective_bias == "long":
        p_score_min_local += INTERVENTION_P_SCORE_ADD
        note = (
            f"Crypto Watch: BTCUSD long P-score küszöb +{INTERVENTION_P_SCORE_ADD} (IRS {intervention_summary['irs']} {intervention_band})"
            if intervention_summary
            else None
        )
        if note and note not in reasons:
            reasons.append(note)
        entry_thresholds_meta["p_score_min_intervention_add"] = INTERVENTION_P_SCORE_ADD
    entry_thresholds_meta["p_score_min_effective"] = p_score_min_local

    tp_net_threshold = tp_net_min_for(asset)
    tp_net_pct_display = f"{tp_net_threshold * 100:.2f}".rstrip("0").rstrip(".")
    tp_net_label = f"tp1_net>=+{tp_net_pct_display}%"
    if tp_net_threshold > 0.2:
        profile_resolution["resolution_notes"].append(
            "WARNING: tp_net_min unusually high; check units"
        )

    risk_template_values = get_risk_template(asset)
    core_rr_min = float(risk_template_values.get("core_rr_min", CORE_RR_MIN.get(asset, CORE_RR_MIN["default"])))
    momentum_rr_min = float(risk_template_values.get("momentum_rr_min", MOMENTUM_RR_MIN.get(asset, MOMENTUM_RR_MIN["default"])))
    if asset == "BTCUSD":
        profile = btc_profile_name or _btc_active_profile()
        core_rr_min = max(core_rr_min, BTC_RR_MIN_TREND.get(profile, core_rr_min))
        momentum_rr_min = max(momentum_rr_min, BTC_MOMENTUM_RR_MIN.get(profile, momentum_rr_min))
    position_size_scale = 1.0
    funding_dir_filter: Optional[str] = None
    funding_reason: Optional[str] = None
    range_time_stop_plan: Optional[Dict[str, Any]] = None
    range_giveback_ratio: Optional[float] = None
    btc_rr_band_meta: Dict[str, Any] = {}
    nvda_small_breakout = False
    if asset == "XAGUSD" and xag_reversal_active:
        risk_cap_pct = 1.0 if risk_cap_pct is None else min(risk_cap_pct, 1.0)
        risk_guard_meta["max_risk_pct"] = risk_cap_pct
        entry_thresholds_meta["risk_guard"] = risk_guard_meta
    if asset == "NVDA":
        nvda_rr_meta: Dict[str, Any] = {}
        low_rel = float(NVDA_RR_BANDS.get("low_rel_atr") or 0.0)
        high_rel = float(NVDA_RR_BANDS.get("high_rel_atr") or 0.0)
        rr_low = float(NVDA_RR_BANDS.get("rr_low") or core_rr_min)
        rr_high_core = float(NVDA_RR_BANDS.get("rr_high_core") or core_rr_min)
        rr_high_mom = float(NVDA_RR_BANDS.get("rr_high_momentum") or momentum_rr_min)
        if not np.isnan(rel_atr):
            if high_rel > 0 and rel_atr >= high_rel:
                core_rr_min = max(core_rr_min, rr_high_core)
                momentum_rr_min = max(momentum_rr_min, rr_high_mom)
                nvda_rr_meta["regime"] = "high_vol"
            elif low_rel > 0 and rel_atr <= low_rel:
                core_rr_min = max(rr_low, min(core_rr_min, rr_low))
                momentum_rr_min = max(rr_low, min(momentum_rr_min, rr_low))
                nvda_rr_meta["regime"] = "compressed"
        if effective_bias == "long":
            nvda_small_breakout = bool(
                not bos5m_long
                and (micro_bos_long or nvda_cross_long)
                and not recent_break_short
            )
        elif effective_bias == "short":
            nvda_small_breakout = bool(
                not bos5m_short
                and (micro_bos_short or nvda_cross_short)
                and not recent_break_long
            )
        if nvda_small_breakout:
            core_rr_min = min(core_rr_min, rr_low)
            momentum_rr_min = min(momentum_rr_min, rr_low)
            nvda_rr_meta["breakout_mode"] = True
        if nvda_rr_meta:
            entry_thresholds_meta.setdefault("nvda_overrides", {})
            entry_thresholds_meta["nvda_overrides"]["rr"] = nvda_rr_meta
    if asset == "XAGUSD":
        xag_rr_meta: Dict[str, Any] = {}
        if xag_atr_ratio is not None:
            xag_rr_meta["atr_ratio"] = xag_atr_ratio
        if xag_momentum_override:
            core_rr_min = min(core_rr_min, 1.45)
            momentum_rr_min = min(momentum_rr_min, 1.3)
            xag_rr_meta["regime"] = "momentum"
            msg = "XAGUSD: momentum kitörés — RR cél 1:1.45-re szűkítve"
            if msg not in reasons:
                reasons.append(msg)
        elif xag_atr_ratio is not None and xag_atr_ratio <= 1.1:
            core_rr_min = max(core_rr_min, 1.7)
            momentum_rr_min = max(momentum_rr_min, 1.5)
            xag_rr_meta["regime"] = "consolidation"
            msg = "XAGUSD: visszafogott ATR — RR cél 1:1.7-re emelve"
            if msg not in reasons:
                reasons.append(msg)
        else:
            xag_rr_meta["regime"] = "balanced"
        xag_rr_meta["core_rr_min_effective"] = core_rr_min
        xag_rr_meta["momentum_rr_min_effective"] = momentum_rr_min
        xag_overrides.setdefault("rr", {}).update(xag_rr_meta)
    if asset == "USOIL" and atr1h is not None and atr1h > 0:
        usoil_rr_meta: Dict[str, Any] = {
            "atr1h": float(atr1h),
            "core_rr_min": core_rr_min,
            "momentum_rr_min": momentum_rr_min,
        }
        if atr1h >= USOIL_ATR1H_PARABOLIC:
            core_rr_min = min(core_rr_min, 1.4)
            momentum_rr_min = min(momentum_rr_min, 1.2)
            usoil_rr_meta["regime"] = "parabolic"
            msg = "USOIL: parabolikus lendület — RR cél 1:1.4 környékére szűkítve"
            if msg not in reasons:
                reasons.append(msg)
        elif atr1h >= USOIL_ATR1H_TARGET:
            core_rr_min = max(core_rr_min, 2.0)
            momentum_rr_min = max(momentum_rr_min, 1.7)
            usoil_rr_meta["regime"] = "swing"
            msg = "USOIL: 1h ATR ≥1.5 USD — RR cél 1:2-re emelve"
            if msg not in reasons:
                reasons.append(msg)
        else:
            core_rr_min = min(core_rr_min, 1.5)
            momentum_rr_min = min(momentum_rr_min, 1.3)
            usoil_rr_meta["regime"] = "muted"
            msg = "USOIL: csökkenő volatilitás — RR cél 1:1.5-re igazítva"
            if msg not in reasons:
                reasons.append(msg)
        usoil_rr_meta["core_rr_min_effective"] = core_rr_min
        usoil_rr_meta["momentum_rr_min_effective"] = momentum_rr_min
        usoil_overrides["rr_adjustments"] = usoil_rr_meta
    if asset == "BTCUSD" and btc_rr_band_meta:
        rr_override = btc_rr_band_meta.get("rr_min")
        mode = btc_rr_band_meta.get("mode")
        if rr_override is not None:
            rr_override_val = float(rr_override)
            if mode == "trend":
                core_rr_min = max(core_rr_min, rr_override_val)
            elif mode == "range":
                core_rr_min = min(core_rr_min, rr_override_val)
        if mode == "range":
            size_scale = btc_rr_band_meta.get("size_scale")
            if size_scale:
                try:
                    position_size_scale = min(position_size_scale, float(size_scale))
                except (TypeError, ValueError):
                    pass
            time_stop = btc_rr_band_meta.get("time_stop")
            be_trigger = btc_rr_band_meta.get("be_trigger_r")
            if time_stop:
                try:
                    time_stop_val = int(time_stop)
                except (TypeError, ValueError):
                    time_stop_val = None
                if time_stop_val and time_stop_val > 0:
                    if range_time_stop_plan is None:
                        range_time_stop_plan = {"timeout": time_stop_val}
                    else:
                        existing_timeout = range_time_stop_plan.get("timeout")
                        if existing_timeout is None or time_stop_val < int(existing_timeout):
                            range_time_stop_plan["timeout"] = time_stop_val
            if be_trigger:
                try:
                    be_trigger_val = float(be_trigger)
                except (TypeError, ValueError):
                    be_trigger_val = None
                if be_trigger_val and be_trigger_val > 0:
                    if range_time_stop_plan is None:
                        range_time_stop_plan = {"breakeven_trigger": be_trigger_val}
                    else:
                        existing_be = range_time_stop_plan.get("breakeven_trigger")
                        if existing_be is None or be_trigger_val < float(existing_be):
                            range_time_stop_plan["breakeven_trigger"] = be_trigger_val
        if btc_rr_band_meta:
            applied_meta: Dict[str, Any] = {"applied_rr_min": core_rr_min}
            if mode == "range":
                applied_meta["applied_position_scale"] = position_size_scale
                if range_time_stop_plan:
                    applied_meta["applied_timeout"] = range_time_stop_plan.get("timeout")
                    applied_meta["applied_be_trigger"] = range_time_stop_plan.get("breakeven_trigger")
            entry_thresholds_meta.setdefault("btc_rr_adx", {}).update(applied_meta)
    if adx_regime == "trend":
        core_rr_min = max(core_rr_min, ADX_TREND_CORE_RR)
        momentum_rr_min = max(momentum_rr_min, ADX_TREND_MOM_RR)
        if asset == "BTCUSD" and btc_rr_cfg:
            trend_core = btc_rr_cfg.get("trend_core")
            trend_mom = btc_rr_cfg.get("trend_momentum")
            if trend_core is not None:
                core_rr_min = max(core_rr_min, float(trend_core))
            if trend_mom is not None:
                momentum_rr_min = max(momentum_rr_min, float(trend_mom))
    elif adx_regime == "range":
        if ADX_RANGE_CORE_RR:
            core_rr_min = min(core_rr_min, ADX_RANGE_CORE_RR)
        if ADX_RANGE_MOM_RR:
            momentum_rr_min = min(momentum_rr_min, ADX_RANGE_MOM_RR)
        if ADX_RANGE_SIZE_SCALE and ADX_RANGE_SIZE_SCALE > 0:
            position_size_scale = min(position_size_scale, ADX_RANGE_SIZE_SCALE)
        if ADX_RANGE_TIME_STOP > 0 and ADX_RANGE_BE_TRIGGER > 0:
            range_time_stop_plan = {
                "timeout": ADX_RANGE_TIME_STOP,
                "breakeven_trigger": ADX_RANGE_BE_TRIGGER,
            }
        if ADX_RANGE_GIVEBACK > 0:
            range_giveback_ratio = float(ADX_RANGE_GIVEBACK)
        if asset == "BTCUSD" and btc_rr_cfg:
            range_core = btc_rr_cfg.get("range_core")
            range_mom = btc_rr_cfg.get("range_momentum")
            if range_core is not None:
                core_rr_min = float(range_core)
            if range_mom is not None:
                momentum_rr_min = float(range_mom)
            size_scale = btc_rr_cfg.get("range_size_scale")
            if size_scale:
                position_size_scale = min(position_size_scale, float(size_scale))
            time_stop = btc_rr_cfg.get("range_time_stop")
            breakeven = btc_rr_cfg.get("range_breakeven")
            if time_stop and breakeven:
                range_time_stop_plan = {
                    "timeout": int(time_stop),
                    "breakeven_trigger": float(breakeven),
                }
            giveback = btc_rr_cfg.get("range_giveback")
            if giveback is not None:
                try:
                    range_giveback_ratio = float(giveback)
                except (TypeError, ValueError):
                    pass
        if range_giveback_ratio is not None and np.isfinite(range_giveback_ratio):
            if range_time_stop_plan is None:
                range_time_stop_plan = {}
            range_time_stop_plan["giveback_ratio"] = float(range_giveback_ratio)
    if range_giveback_ratio is not None and np.isfinite(range_giveback_ratio):
        entry_thresholds_meta["range_giveback_ratio"] = float(range_giveback_ratio)

    if asset == "EURUSD" and atr1h is not None and atr1h > 0:
        atr1h_pips = float(atr1h) / EURUSD_PIP
        eurusd_overrides.setdefault("atr1h_pips", atr1h_pips)
        eurusd_rr_bias: Optional[str] = None
        core_profile = dynamic_tp_profile.setdefault("core", {})
        mom_profile = dynamic_tp_profile.setdefault("momentum", {})
        if atr1h_pips >= EURUSD_ATR_PIPS_HIGH:
            eurusd_rr_bias = "high_vol"
            core_rr_min = max(core_rr_min, 2.0)
            momentum_rr_min = max(momentum_rr_min, 1.6)
            core_profile["tp1"] = float(max(core_profile.get("tp1", TP1_R), 1.6))
            core_profile["tp2"] = float(max(core_profile.get("tp2", TP2_R), 2.4))
            mom_profile["tp1"] = float(max(mom_profile.get("tp1", TP1_R_MOMENTUM), 1.4))
            mom_profile["tp2"] = float(max(mom_profile.get("tp2", TP2_R_MOMENTUM), 2.0))
            reasons.append("EURUSD: Magas volatilitás → RR célok 1:2 környékére szélesítve")
        elif atr1h_pips <= EURUSD_ATR_PIPS_LOW:
            eurusd_rr_bias = "low_vol"
            core_rr_min = max(core_rr_min, 1.75)
            momentum_rr_min = max(momentum_rr_min, 1.45)
            core_profile["tp1"] = float(max(core_profile.get("tp1", TP1_R), 1.3))
            core_profile["tp2"] = float(max(core_profile.get("tp2", TP2_R), 1.9))
            mom_profile["tp1"] = float(max(mom_profile.get("tp1", TP1_R_MOMENTUM), 1.1))
            mom_profile["tp2"] = float(max(mom_profile.get("tp2", TP2_R_MOMENTUM), 1.6))
            reasons.append("EURUSD: Visszafogott ATR — célok szűkítve, kockázat kontroll alatt")
        if eurusd_rr_bias:
            eurusd_overrides["rr_bias"] = {
                "mode": eurusd_rr_bias,
                "atr1h_pips": atr1h_pips,
            }
    if asset == "EURUSD":
        momentum_rr_min = min(momentum_rr_min, 1.2)

    entry_thresholds_meta["adx_regime"] = adx_regime

    funding_rules = FUNDING_RATE_RULES.get(asset, {})
    funding_value: Optional[float] = None
    if funding_rules and funding_rate_snapshot is not None:
        try:
            funding_value = float(funding_rate_snapshot)
        except (TypeError, ValueError):
            funding_value = None
        if funding_value is not None:
            pos_ext = float(funding_rules.get("positive_extreme") or 0.0)
            neg_ext = float(funding_rules.get("negative_extreme") or 0.0)
            moderate = abs(float(funding_rules.get("moderate_band") or 0.0))
            size_scale = float(funding_rules.get("size_scale") or 1.0)
            if funding_value >= pos_ext and pos_ext:
                funding_dir_filter = "short"
                funding_reason = f"Funding +{funding_value:.3f} → csak short"
            elif funding_value <= neg_ext and neg_ext:
                funding_dir_filter = "long"
                funding_reason = f"Funding {funding_value:.3f} → csak long"
            elif moderate and abs(funding_value) >= moderate and size_scale > 0:
                position_size_scale = min(position_size_scale, size_scale)
                funding_reason = f"Funding {funding_value:.3f} → pozícióméret skálázás"
    if funding_reason:
        structure_notes.append(funding_reason)

    if asset == "GOLD_CFD":
        gold_atr_ratio: Optional[float] = None
        if atr_threshold and atr_threshold > 0 and not np.isnan(rel_atr):
            try:
                gold_atr_ratio = float(rel_atr) / float(atr_threshold)
            except Exception:
                gold_atr_ratio = None
        if gold_atr_ratio is not None:
            if gold_atr_ratio >= 1.8:
                position_size_scale = min(position_size_scale, 0.45)
            elif gold_atr_ratio >= 1.4:
                position_size_scale = min(position_size_scale, 0.55)
            elif gold_atr_ratio >= 1.1:
                position_size_scale = min(position_size_scale, 0.7)
            else:
                position_size_scale = min(position_size_scale, 0.85)
        else:
            position_size_scale = min(position_size_scale, 0.8)
        entry_thresholds_meta["gold_atr_ratio"] = gold_atr_ratio

    if asset == "XAGUSD":
        xag_scale_meta: Dict[str, Any] = {}
        if xag_atr_ratio is not None:
            xag_scale_meta["atr_ratio"] = xag_atr_ratio
        if xag_atr_ratio is not None:
            if xag_atr_ratio >= 1.7:
                target_scale = 0.55
            elif xag_atr_ratio >= 1.4:
                target_scale = 0.62
            else:
                target_scale = 0.7
        else:
            target_scale = 0.68
        if xag_momentum_override:
            target_scale = min(target_scale, 0.55)
            if "XAG momentum override — pozícióméret csökkentve" not in reasons:
                reasons.append("XAG momentum override — pozícióméret csökkentve")
        if target_scale < position_size_scale:
            position_size_scale = target_scale
            msg = f"XAGUSD: ATR-alapú pozícióskálázás ×{target_scale:.2f}"
            if msg not in reasons:
                reasons.append(msg)
        xag_scale_meta["applied"] = position_size_scale
        xag_overrides.setdefault("position_scale", {}).update(xag_scale_meta)

    if asset == "BTCUSD":
        btc_position_meta: Dict[str, Any] = {}
        btc_active_profile = btc_profile_name or _btc_active_profile()
        if btc_active_profile in {"relaxed", "intraday"}:
            profile_cap = 0.62 if btc_active_profile == "relaxed" else 0.6
            if profile_cap < position_size_scale:
                position_size_scale = profile_cap
                btc_position_meta["profile_cap"] = profile_cap
                btc_position_meta["profile"] = btc_active_profile
        if btc_momentum_override_active:
            position_size_scale = min(position_size_scale, 0.55)
            if "BTC momentum override — pozícióméret csökkentve" not in reasons:
                reasons.append("BTC momentum override — pozícióméret csökkentve")
            btc_position_meta["momentum_override_cap"] = 0.55
        else:
            position_size_scale = min(position_size_scale, 0.75)
        entry_thresholds_meta["btc_position_scale"] = position_size_scale
        if btc_position_meta:
            entry_thresholds_meta["btc_position_meta"] = btc_position_meta

    if asset == "USOIL":
        minute_now = analysis_now.hour * 60 + analysis_now.minute
        session_label = "asia"
        session_scale = 0.7
        if 12 * 60 <= minute_now < 21 * 60:
            session_label = "us"
            session_scale = 1.0
        elif 6 * 60 <= minute_now < 12 * 60:
            session_label = "europe"
            session_scale = 0.85
        elif minute_now >= 21 * 60:
            session_label = "overnight"
            session_scale = 0.75
        scale_meta: Dict[str, Any] = {
            "session": session_label,
            "session_scale": session_scale,
        }
        if atr1h is not None and atr1h > 0:
            scale_meta["atr1h"] = float(atr1h)
        target_scale = min(position_size_scale, session_scale)
        if atr1h is not None and atr1h < USOIL_ATR1H_BREAKOUT_MIN:
            target_scale = min(target_scale, 0.75)
            scale_meta["low_vol_scale"] = 0.75
        if target_scale < position_size_scale:
            position_size_scale = target_scale
            label_map = {
                "us": "US nyitás", "europe": "európai délelőtt", "asia": "ázsiai sáv", "overnight": "nyitás előtti sáv"
            }
            session_note = label_map.get(session_label, session_label)
            msg = f"USOIL: {session_note} — pozícióméret ×{target_scale:.2f}"
            if msg not in reasons:
                reasons.append(msg)
        scale_meta["applied"] = position_size_scale
        usoil_overrides["position_scale"] = scale_meta
    if asset == "EURUSD":
        atr_for_position: Optional[float] = None
        if atr1h is not None and atr1h > 0:
            atr_for_position = float(atr1h)
        elif atr5 is not None:
            try:
                atr_candidate = float(atr5)
            except Exception:
                atr_candidate = 0.0
            if atr_candidate > 0:
                atr_for_position = atr_candidate * 12.0  # durva 5m→1h fel-skálázás
        if atr_for_position is not None and atr_for_position > 0:
            raw_scale = EURUSD_ATR_REF / atr_for_position
            if raw_scale >= 1.0:
                scale_target = 1.0
            else:
                scale_target = max(
                    EURUSD_POSITION_SCALE_MIN, raw_scale / EURUSD_ATR_POSITION_MULT
                )
            eurusd_overrides.setdefault("position_scale", {})
            eurusd_overrides["position_scale"] = {
                "atr_basis": atr_for_position,
                "raw_scale": raw_scale,
                "applied": scale_target,
            }
            if scale_target < position_size_scale:
                position_size_scale = scale_target
                msg = (
                    f"EURUSD: ATR-alapú pozícióskálázás ×{scale_target:.2f} — volatilitás kontroll"
                )
                if msg not in reasons:
                    reasons.append(msg)

    if asset == "NVDA":
        nvda_scale_meta: Dict[str, Any] = {}
        base_scale = float(NVDA_POSITION_SCALE.get("base", 1.0) or 1.0)
        low_scale = float(NVDA_POSITION_SCALE.get("low_daily_atr", base_scale) or base_scale)
        high_scale = float(NVDA_POSITION_SCALE.get("high_daily_atr", 1.0) or 1.0)
        target_scale = position_size_scale
        if nvda_daily_atr is not None:
            if NVDA_DAILY_ATR_MIN > 0 and nvda_daily_atr < NVDA_DAILY_ATR_MIN:
                target_scale = min(target_scale, low_scale)
                nvda_scale_meta["regime"] = "low_daily_atr"
            elif NVDA_DAILY_ATR_STRONG > 0 and nvda_daily_atr >= NVDA_DAILY_ATR_STRONG:
                target_scale = min(target_scale, high_scale)
                nvda_scale_meta["regime"] = "high_daily_atr"
            else:
                target_scale = min(target_scale, base_scale)
                nvda_scale_meta["regime"] = "neutral"
        else:
            target_scale = min(target_scale, base_scale)
        if target_scale < position_size_scale:
            position_size_scale = target_scale
            msg = f"NVDA: ATR-alapú pozícióskálázás ×{position_size_scale:.2f}"
            if msg not in reasons:
                reasons.append(msg)
        if nvda_scale_meta:
            nvda_scale_meta["applied"] = position_size_scale
            entry_thresholds_meta.setdefault("nvda_overrides", {})
            entry_thresholds_meta["nvda_overrides"]["position_scale"] = nvda_scale_meta

    for note in structure_notes:
        if note not in reasons:
            reasons.append(note)

    entry_thresholds_meta["position_size_scale"] = position_size_scale
    if funding_dir_filter:
        entry_thresholds_meta["funding_filter"] = funding_dir_filter
    if funding_value is not None:
        entry_thresholds_meta["funding_rate"] = funding_value

    if asset == "BTCUSD":
        core_profile = dynamic_tp_profile.setdefault(
            "core",
            {
                "tp1": TP1_R,
                "tp2": TP2_R,
                "rr": BTC_RR_MIN_TREND.get(btc_profile_name or _btc_active_profile(), MIN_R_CORE),
            },
        )
        mom_profile = dynamic_tp_profile.setdefault(
            "momentum",
            {
                "tp1": TP1_R_MOMENTUM,
                "tp2": TP2_R_MOMENTUM,
                "rr": BTC_MOMENTUM_RR_MIN.get(
                    btc_profile_name or _btc_active_profile(), MIN_R_MOMENTUM
                ),
            },
        )
        base_core_rr = btc_rr_cfg.get("trend_core")
        if base_core_rr is not None:
            core_profile["rr"] = float(base_core_rr)
            core_rr_min = min(core_rr_min, float(base_core_rr))
        base_mom_rr = btc_rr_cfg.get("trend_momentum")
        if base_mom_rr is not None:
            mom_profile["rr"] = float(base_mom_rr)
            momentum_rr_min = min(momentum_rr_min, float(base_mom_rr))
        if btc_momentum_override_active:
            rr_override = btc_momentum_cfg.get("rr_min")
            if rr_override is not None:
                rr_override_val = float(rr_override)
                mom_profile["rr"] = rr_override_val
                momentum_rr_min = min(momentum_rr_min, rr_override_val)

    core_tp1_mult = dynamic_tp_profile["core"]["tp1"]
    core_tp2_mult = dynamic_tp_profile["core"]["tp2"]
    core_rr_min = max(core_rr_min, dynamic_tp_profile["core"]["rr"])
    mom_tp1_mult = dynamic_tp_profile["momentum"]["tp1"]
    mom_tp2_mult = dynamic_tp_profile["momentum"]["tp2"]
    momentum_rr_min = max(momentum_rr_min, dynamic_tp_profile["momentum"]["rr"])

    low_atr_meta: Dict[str, Any] = {}
    low_atr_tp_override: Optional[float] = None
    low_atr_cfg = get_low_atr_override(asset)
    low_atr_floor = safe_float(low_atr_cfg.get("floor"))
    if low_atr_floor is not None and not np.isnan(rel_atr) and rel_atr < low_atr_floor:
        low_atr_meta["floor"] = low_atr_floor
        low_atr_meta["rel_atr"] = float(rel_atr)
        tp_override = safe_float(low_atr_cfg.get("tp_min_pct"))
        if tp_override is not None:
            low_atr_tp_override = tp_override
            low_atr_meta["tp_min_pct"] = tp_override
        rr_override = safe_float(low_atr_cfg.get("rr_required"))
        if rr_override is not None:
            core_rr_min = min(core_rr_min, rr_override)
            momentum_rr_min = min(momentum_rr_min, rr_override)
            low_atr_meta["rr_required"] = rr_override
        msg = (
            f"ATR-floor override: rel_atr {rel_atr:.4f} < {low_atr_floor:.4f} "
            "→ adaptív RR/TP küszöb"
        )
        if msg not in reasons:
            reasons.append(msg)

    rr_gate_meta: Dict[str, Any] = {
        "core_rr_min_base": core_rr_min,
        "momentum_rr_min_base": momentum_rr_min,
    }
    low_vol_relax = False
    try:
        if settings.RR_RELAX_ENABLED and atr_threshold is not None and not np.isnan(rel_atr):
            low_vol_relax = float(rel_atr) < float(atr_threshold) * settings.RR_RELAX_ATR_RATIO_TRIGGER
    except Exception:
        low_vol_relax = False
    if settings.RR_RELAX_ENABLED and low_vol_relax:
        relaxed_core = min(core_rr_min, settings.RR_RELAX_RANGE_CORE)
        relaxed_momentum = min(momentum_rr_min, settings.RR_RELAX_RANGE_MOMENTUM)
        if relaxed_core < core_rr_min or relaxed_momentum < momentum_rr_min:
            rr_gate_meta["relaxed"] = True
            rr_gate_meta["relax_reason"] = "alacsony vol / range momentum"
            core_rr_min = relaxed_core
            momentum_rr_min = relaxed_momentum
    if low_atr_meta:
        rr_gate_meta["low_atr_override"] = {
            **low_atr_meta,
            "core_rr_min_effective": core_rr_min,
            "momentum_rr_min_effective": momentum_rr_min,
        }
    rr_gate_meta["core_rr_min"] = core_rr_min
    rr_gate_meta["momentum_rr_min"] = momentum_rr_min
    entry_thresholds_meta["rr_min_core"] = core_rr_min
    entry_thresholds_meta["rr_min_momentum"] = momentum_rr_min
    entry_thresholds_meta["rr_meta"] = rr_gate_meta
    if low_atr_meta:
        entry_thresholds_meta["low_atr_override"] = dict(low_atr_meta)
        if low_atr_tp_override is not None:
            entry_thresholds_meta["low_atr_override"]["tp_min_pct_override"] = low_atr_tp_override

    range_guard_label = "intraday_range_guard"
    structure_label = "structure(2of3)"

    gate_extra_context = {} if os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE) else dict(entry_gate_context_hu)
    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        entry_gate_context_hu["rr_kapuk"] = {
            "core_rr_min": float(rr_gate_meta.get("core_rr_min")) if rr_gate_meta.get("core_rr_min") is not None else None,
            "momentum_rr_min": float(rr_gate_meta.get("momentum_rr_min")) if rr_gate_meta.get("momentum_rr_min") is not None else None,
            "relaxalt": bool(rr_gate_meta.get("relaxed")),
            "indok": rr_gate_meta.get("relax_reason"),
            **_gate_timestamp_fields(analysis_now),
        }
        gate_extra_context.update(entry_gate_context_hu)

    def _with_gate_context(extra: Dict[str, Any]) -> Dict[str, Any]:
        if gate_extra_context:
            merged = dict(gate_extra_context)
            merged.update(extra)
            return merged
        return extra

    data_integrity_ok = core_data_ok and not critical_flags.get("k1h", False) and not critical_flags.get("k4h", False)
    soft_structure_gate = bool(structure_gate or (strong_momentum and not structure_gate))

    core_required = [
        "session",
        "spread_guard",
        "data_integrity",
        "risk_reward",
        structure_label,
        range_guard_label,
        "atr",
        f"rr_math>={core_rr_min:.1f}",
        "tp_min_profit",
        "min_stoploss",
        tp_net_label,
    ]

    intraday_relax_active = bool(intraday_profile) and is_intraday_relax_enabled(asset)
    intraday_relaxed_guards: List[str] = []
    intraday_relax_scale = get_intraday_relax_size_scale(asset)

    conds_core = {
        "session": bool(session_ok_flag),
        "spread_guard": bool(spread_gate_ok),
        "data_integrity": bool(data_integrity_ok),
        "regime": bool(regime_ok),
        "bias": effective_bias in ("long", "short"),
        structure_label: bool(soft_structure_gate),
        "atr": bool(atr_ok),
        range_guard_label: range_guard_ok,
    }
    if funding_dir_filter:
        core_required.append("funding_alignment")
        if effective_bias in ("long", "short"):
            conds_core["funding_alignment"] = effective_bias == funding_dir_filter
        else:
            conds_core["funding_alignment"] = False

    critical_missing: List[str] = []
    soft_flags: List[str] = []
    intraday_relaxable_guards = {"data_integrity", range_guard_label}
    for name, ok in conds_core.items():
        category = classify_gate_failure(name)
        if ok:
            continue
        if (
            intraday_relax_active
            and category == "critical"
            and name in intraday_relaxable_guards
        ):
            if name not in intraday_relaxed_guards:
                intraday_relaxed_guards.append(name)
                position_size_scale *= intraday_relax_scale
                reasons.append(
                    f"Intraday relax: {name} hiány kockázatcsökkentéssel engedve (×{intraday_relax_scale:.2f})"
                )
            continue
        if category == "critical":
            critical_missing.append(name)
        else:
            soft_flags.append(name)

    if intraday_relaxed_guards:
        entry_thresholds_meta["intraday_relaxed_guards"] = list(intraday_relaxed_guards)
        entry_thresholds_meta["intraday_relax_size_scale"] = intraday_relax_scale

    if not data_integrity_ok:
        reasons.append("Data integrity gate: adatfrissítés >2 perc vagy hiányzik")
    if not spread_gate_ok:
        reasons.append("Spread gate: aktuális spread meghaladja az ATR arány limitet")

    if not regime_ok and "regime" in conds_core:
        P -= 10.0
        reasons.append("Regime kapu: CHOPPY miatt −10 P-score, pozícióskálázás csökkentve")
    if not atr_ok:
        P -= 5.0
        position_size_scale *= 0.5
        reasons.append("ATR soft gate: alacsony volatilitás −5 P-score, méret felezve")
    if not conds_core.get("bias", True):
        P -= 15.0
        reasons.append("Bias eltérés: H1 trend ütközik az M5 belépéssel (−15 P-score)")
    if not structure_gate and soft_structure_gate:
        reasons.append("Strukturális kapu lazítva: momentum erős, BOS hiány engedve")

    P = max(0.0, min(100.0, P))

    size_multiplier = calculate_position_size_multiplier(regime_label, atr_ok, P)
    position_size_scale *= size_multiplier
    position_scale_floor = POSITION_SIZE_SCALE_FLOOR_BY_ASSET.get(asset, 0.25)
    floor = position_scale_floor
    if position_size_scale < floor:
        critical_missing.append("position_sizing_floor")
        reasons.append(
            f"Position size scale {position_size_scale:.4f} below floor {floor:.2f} — entry blocked"
        )

    base_core_ok = not critical_missing
    can_enter_core = (P >= p_score_min_local) and base_core_ok
    missing_core = list(critical_missing)
    if P < p_score_min_local:
        if float(p_score_min_local).is_integer():
            p_score_label = str(int(round(p_score_min_local)))
        else:
            p_score_label = f"{p_score_min_local:.1f}"
        missing_core.append(f"P_score>={p_score_label}")
    _emit_precision_gate_log(
        asset,
        "can_enter_core",
        bool(can_enter_core),
        "core_ok" if can_enter_core else "core_blockers",
        order_flow_metrics=order_flow_metrics,
        tick_order_flow=tick_order_flow,
        latency_seconds=latency_by_frame,
        precision_plan=None,
        timestamp=analysis_now,
        extra=_with_gate_context({
            "missing": list(missing_core),
            "p_score": P,
            "p_score_min": p_score_min_local,
            "base_core_ok": bool(base_core_ok),
        }),
    )
    if liquidity_relaxed:
        reasons.append("Likviditási kapu lazítva erős momentum miatt")

    # --- Momentum feltételek (override) — kriptókra (zárt 5m-ből) ---
    momentum_used = False
    mom_dir: Optional[str] = None
    mom_atr_ok: Optional[bool] = None
    momentum_trigger_ok: Optional[bool] = None
    mom_required = [
        "session",
        "regime",
        "bias",
        "momentum_trigger",
        structure_label,
        "atr",
        f"rr_math>={momentum_rr_min:.1f}",
        "tp_min_profit",
        "min_stoploss",
        tp_net_label,
        range_guard_label,
    ]
    if funding_dir_filter:
        mom_required.append("funding_alignment")
    missing_mom: List[str] = []
    mom_trigger_desc: Optional[str] = None

    momentum_liquidity_ok = True
    momentum_adx_ok = True
    xag_momentum_regime_bypass = asset == "XAGUSD" and P >= 30.0
    eurusd_momentum_regime_bypass = (
        asset == "EURUSD" and eurusd_momentum_trigger and P >= p_score_min_local
    )
    liquidity_data_available = momentum_vol_ratio is not None
    if precision_disabled_due_to_data_gap:
        mom_dir = None
        momentum_trigger_ok = False
        missing_mom = []
        momentum_liquidity_ok = False
        mom_trigger_desc = precision_data_gap_reason or "precision data gap"
    elif asset in ENABLE_MOMENTUM_ASSETS:
        direction = effective_bias if effective_bias in {"long", "short"} else None
        if direction is None and asset == "EURUSD" and eurusd_momentum_regime_bypass:
            direction = desired_bias if desired_bias in {"long", "short"} else None
        if not session_ok_flag:
            missing_mom.append("session")
        regime_gate_for_momentum = bool(
            regime_ok or xag_momentum_regime_bypass or eurusd_momentum_regime_bypass
        )
        if not regime_gate_for_momentum:
            missing_mom.append("regime")
        if direction is None:
            if eurusd_momentum_regime_bypass:
                direction = trend_bias if trend_bias in {"long", "short"} else direction
            if direction is None:
                missing_mom.append("bias")
        if funding_dir_filter and direction in {"long", "short"}:
            if direction != funding_dir_filter:
                missing_mom.append("funding_alignment")
        if not range_guard_ok:
            missing_mom.append(range_guard_label)

        if asset == "BTCUSD":
            mom_atr_ok = bool(atr_ok and (btc_atr_floor_ratio is None or btc_atr_floor_passed))
            ofi_th = float(btc_momentum_cfg.get("ofi_z", 0.0) or 0.0)
            cross_flag = False
            ofi_confirm = False
            if direction == "long":
                cross_flag = bool(ema_cross_long)
                ofi_confirm = ofi_available and ofi_th > 0 and ofi_zscore >= ofi_th
            elif direction == "short":
                cross_flag = bool(ema_cross_short)
                ofi_confirm = ofi_available and ofi_th > 0 and ofi_zscore <= -ofi_th
            momentum_state = globals().get("_BTC_MOMENTUM_RUNTIME", {})
            override_state: Dict[str, Any] = {}
            if isinstance(momentum_state, dict):
                raw_override = momentum_state.get("override")
                if isinstance(raw_override, dict):
                    override_state = raw_override
            override_side = override_state.get("side")
            override_active = (
                bool(override_state.get("active"))
                and direction in {"long", "short"}
                and override_side == direction
            )
            if (
                override_active
                and session_ok_flag
                and regime_ok
                and mom_atr_ok
            ):
                mom_dir = "buy" if direction == "long" else "sell"
                override_desc = override_state.get("desc")
                mom_trigger_desc = (
                    override_desc if isinstance(override_desc, str) and override_desc else "EMA9×21 momentum cross"
                )
                rr_override_val = override_state.get("rr_min")
                if rr_override_val is not None:
                    try:
                        momentum_rr_min = min(momentum_rr_min, float(rr_override_val))
                    except (TypeError, ValueError):
                        pass
                missing_mom = [
                    item
                    for item in missing_mom
                    if item not in {"liquidity", "ofi", "momentum_trigger"}
                ]
                if funding_dir_filter and (
                    (funding_dir_filter == "long" and mom_dir != "buy")
                    or (funding_dir_filter == "short" and mom_dir != "sell")
                ):
                    mom_dir = None
                    if "funding_alignment" not in missing_mom:
                        missing_mom.append("funding_alignment")
            elif (
                direction in {"long", "short"}
                and session_ok_flag
                and (regime_ok or xag_momentum_regime_bypass)
                and mom_atr_ok
                and cross_flag
                and ofi_confirm
            ):
                mom_dir = "buy" if direction == "long" else "sell"
                mom_trigger_desc = "EMA9×21 momentum cross"
                momentum_trigger_ok = True
                missing_mom = [item for item in missing_mom if item not in {"liquidity", "ofi"}]
                if asset == "XAGUSD" and xag_momentum_regime_bypass:
                    entry_thresholds_meta["xag_momentum_bias_bypass"] = True
                if funding_dir_filter and (
                    (funding_dir_filter == "long" and mom_dir != "buy")
                    or (funding_dir_filter == "short" and mom_dir != "sell")
                ):
                    mom_dir = None
                    if "funding_alignment" not in missing_mom:
                        missing_mom.append("funding_alignment")
            else:
                if not mom_atr_ok and "atr" not in missing_mom:
                    missing_mom.append("atr")
                if not cross_flag and "momentum_trigger" not in missing_mom:
                    missing_mom.append("momentum_trigger")
                if not ofi_confirm and "ofi" not in missing_mom:
                    missing_mom.append("ofi")
        else:
            if not liquidity_data_available:
                momentum_liquidity_ok = True
                if "Liquidity gate skipped (no volume proxy available)" not in reasons:
                    reasons.append("Liquidity gate skipped (no volume proxy available)")
                if "liquidity_no_data" not in gate_skips:
                    gate_skips.append("liquidity_no_data")
            elif momentum_vol_ratio < MOMENTUM_VOLUME_RATIO_TH:
                momentum_liquidity_ok = False
                missing_mom.append("liquidity")

            if flow_data_available:
                if order_flow_imbalance is not None and abs(order_flow_imbalance) < ORDER_FLOW_IMBALANCE_TH:
                    momentum_liquidity_ok = False
                    missing_mom.append("order_flow")
                if order_flow_pressure is not None and direction is not None:
                    if direction == "long" and order_flow_pressure < ORDER_FLOW_PRESSURE_TH:
                        momentum_liquidity_ok = False
                        missing_mom.append("order_flow_pressure")
                    elif direction == "short" and order_flow_pressure > -ORDER_FLOW_PRESSURE_TH:
                        momentum_liquidity_ok = False
                        missing_mom.append("order_flow_pressure")
            else:
                if "Order-flow gates skipped (no tick/imbalance data)" not in reasons:
                    reasons.append("Order-flow gates skipped (no tick/imbalance data)")
                if "orderflow_no_data" not in gate_skips:
                    gate_skips.append("orderflow_no_data")
            if asset == "GOLD_CFD":
                momentum_adx_ok = regime_adx is not None and regime_adx >= 20.0
                if not momentum_adx_ok:
                    missing_mom.append("adx")
            ofi_confirm = True
            if ofi_available and OFI_Z_TRIGGER > 0 and direction in {"long", "short"}:
                if direction == "long":
                    ofi_confirm = ofi_zscore >= OFI_Z_TRIGGER
                else:
                    ofi_confirm = ofi_zscore <= -OFI_Z_TRIGGER
            if not ofi_confirm:
                if ofi_available:
                    missing_mom.append("ofi")
                elif "Order-flow gates skipped (no tick/imbalance data)" not in reasons:
                    reasons.append("Order-flow gates skipped (no tick/imbalance data)")
                if not ofi_available and "ofi_no_data" not in gate_skips:
                    gate_skips.append("ofi_no_data")
            elif not ofi_available and "Order-flow gates skipped (no tick/imbalance data)" not in reasons:
                reasons.append("Order-flow gates skipped (no tick/imbalance data)")
                if "ofi_no_data" not in gate_skips:
                    gate_skips.append("ofi_no_data")

            if direction is not None:
                cross_flag = False
                mom_atr_ok = False
                if asset == "NVDA":
                    mom_atr_ok = not np.isnan(rel_atr) and rel_atr >= NVDA_MOMENTUM_ATR_REL and atr_abs_ok
                    cross_flag = nvda_cross_long if direction == "long" else nvda_cross_short
                else:
                    mom_atr_ok = bool(atr_ok and not np.isnan(rel_atr))
                    cross_flag = ema_cross_long if direction == "long" else ema_cross_short
                if (
                    session_ok_flag
                    and regime_ok
                    and momentum_adx_ok
                    and mom_atr_ok
                    and cross_flag
                    and momentum_liquidity_ok
                    and ofi_confirm
                ):
                    mom_dir = "buy" if direction == "long" else "sell"
                    mom_trigger_desc = "EMA9×21 momentum cross"
                    momentum_trigger_ok = True
                    missing_mom = []
                    if funding_dir_filter and ((funding_dir_filter == "long" and mom_dir != "buy") or (funding_dir_filter == "short" and mom_dir != "sell")):
                        mom_dir = None
                        missing_mom.append("funding_alignment")
                else:
                    if not mom_atr_ok:
                        missing_mom.append("atr")
                    if not cross_flag:
                        missing_mom.append("momentum_trigger")
                    if not momentum_liquidity_ok:
                        missing_mom.append("liquidity")

    if not os.getenv(ENTRY_GATE_EXTRA_LOGS_DISABLE):
        momentum_context: Dict[str, Any] = {
            "momentum_ok": bool(mom_dir),
            "momentum_direction": mom_dir,
            "momentum_reason": mom_trigger_desc
            or ("missing:" + ",".join(sorted(dict.fromkeys(missing_mom))) if missing_mom else ""),
            "momentum_missing": list(dict.fromkeys(missing_mom)),
            "momentum_regime_ok": bool(regime_ok),
            "momentum_structure_ok": bool(structure_gate),
            "momentum_atr_ok": mom_atr_ok if mom_atr_ok is not None else False,
            "momentum_liquidity_ok": bool(momentum_liquidity_ok),
            "momentum_trigger_ok": bool(momentum_trigger_ok),
            "momentum_ofi_z": safe_float(ofi_zscore),
            **_gate_timestamp_fields(analysis_now),
        }
        if asset in {"NVDA", "USOIL", "XAGUSD", "GOLD_CFD"}:
            momentum_context["momentum_asset_class"] = "energy_equity" if asset in {"USOIL"} else "equity_metal"
        entry_gate_context_hu["momentum_kapu"] = momentum_context
        gate_extra_context.update(entry_gate_context_hu)   
               
    # 8) Döntés + szintek (RR/TP matek) — core vagy momentum
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)
    mode = "core"
    missing = list(missing_core)
    required_list: List[str] = list(core_required)
    min_stoploss_ok = True
    tp1_net_pct_value: Optional[float] = None
    momentum_trailing_plan: Optional[Dict[str, Any]] = None
    last_computed_risk: Optional[float] = None
    current_tp1_mult = core_tp1_mult
    current_tp2_mult = core_tp2_mult
    btc_level_checks_state: Dict[str, Any] = {}
    precision_plan: Optional[Dict[str, Any]] = None
    precision_ready_for_entry = False
    precision_flow_ready = False
    precision_trigger_ready = False
    precision_flow_gate_needed = False
    precision_trigger_gate_needed = False
    precision_trigger_state: Optional[str] = None
    precision_override_active = False
    session_entry_open = bool(session_meta.get("entry_open"))
    precision_threshold_value = get_precision_score_threshold(asset)
    if float(precision_threshold_value).is_integer():
        precision_threshold_label = str(int(round(precision_threshold_value)))
    else:
        precision_threshold_label = f"{precision_threshold_value:.1f}"
    precision_gate_label = f"precision_score>={precision_threshold_label}"
    precision_flow_gate_label = "precision_flow_alignment"
    precision_trigger_gate_label = "precision_trigger_sync"
    precision_timeouts = get_precision_timeouts(asset)

    sl_buffer_defaults = get_sl_buffer_config(asset)
    if not sl_buffer_defaults:
        sl_buffer_defaults = {"atr_mult": 0.2, "abs_min": 0.0005}
    tp_min_abs_default = get_tp_min_abs_value(asset)
    profile_slippage_limit = get_max_slippage_r(asset)
    rr_required_effective: Optional[float] = None
    tp_min_profit_pct: Optional[float] = None
    min_stoploss_pct = MIN_STOPLOSS_PCT

    def compute_slippage_state(decision_side: str, limit_r: Optional[float]) -> Optional[Dict[str, float]]:
        if (
            limit_r is None
            or last_computed_risk is None
            or price_for_calc is None
            or last5_close is None
        ):
            return None
        slip = (
            max(0.0, price_for_calc - last5_close)
            if decision_side == "buy"
            else max(0.0, last5_close - price_for_calc)
        )
        allowed = float(limit_r) * float(last_computed_risk)
        slip_r = slip / max(1e-9, float(last_computed_risk))
        return {
            "slip": float(slip),
            "allowed": float(allowed),
            "limit_r": float(limit_r),
            "slip_r": float(slip_r),
            "entry_price": float(price_for_calc),
            "trigger_price": float(last5_close),
            "risk_abs": float(last_computed_risk),
        }

    if (
        asset == "BTCUSD"
        and BTC_PROFILE_CONFIG.get("range_guard_requires_override")
        and btc_momentum_override_active
    ):
        range_guard_ok = True
        if range_guard_label in missing_mom:
            missing_mom = [item for item in missing_mom if item != range_guard_label]

    def compute_levels(decision_side: str, rr_required: float, tp1_mult: float = TP1_R, tp2_mult: float = TP2_R):
        nonlocal entry, sl, tp1, tp2, rr, missing, min_stoploss_ok, tp1_net_pct_value, last_computed_risk, current_tp1_mult, current_tp2_mult, rr_required_effective, tp_min_profit_pct
        current_tp1_mult = tp1_mult
        current_tp2_mult = tp2_mult
        atr5_val  = float(atr5 or 0.0)
        rr_required_effective = rr_required

        buf_rule = dict(sl_buffer_defaults) if isinstance(sl_buffer_defaults, dict) else {}
        if asset == "BTCUSD":
            sl_cfg = BTC_PROFILE_CONFIG.get("sl_buffer")
            if isinstance(sl_cfg, dict):
                for key, value in sl_cfg.items():
                    try:
                        buf_rule[key] = float(value)
                    except (TypeError, ValueError):
                        continue
            profile = btc_profile_name or _btc_active_profile()

            def _safe_float(value: Any, default: float) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            atr_mult_default = BTC_SL_ATR_MULT.get(profile, buf_rule.get("atr_mult", 0.2))
            abs_min_default = BTC_SL_ABS_MIN
            buf_rule["atr_mult"] = _safe_float(buf_rule.get("atr_mult"), float(atr_mult_default))
            buf_rule["abs_min"] = max(
                abs_min_default,
                _safe_float(buf_rule.get("abs_min"), float(abs_min_default)),
            )
        buf = max(
            float(buf_rule.get("atr_mult", 0.2)) * atr5_val,
            float(buf_rule.get("abs_min", tp_min_abs_default)),
        )
      
        k5_sw = find_swings(k5m_closed, lb=2)
        hi5, lo5 = last_swing_levels(k5_sw)

        entry = price_for_calc
        if decision_side == "buy":
            base_sl = lo5 if lo5 is not None else (entry - atr5_val)
            sl = base_sl - buf
            risk = entry - sl
            if risk < 0:
                sl = entry - buf
                risk = entry - sl
            tp1 = entry + tp1_mult * risk
            tp2 = entry + tp2_mult * risk
            tp1_dist = tp1 - entry
            ok_math = (sl < entry < tp1 <= tp2)
        else:
            base_sl = hi5 if hi5 is not None else (entry + atr5_val)
            sl = base_sl + buf
            risk = sl - entry
            if risk < 0:
                sl = entry + buf
                risk = sl - entry
            tp1 = entry - tp1_mult * risk
            tp2 = entry - tp2_mult * risk
            tp1_dist = entry - tp1
            ok_math = (tp2 <= tp1 < entry < sl)

        if asset == "GOLD_CFD" and atr5_val > 0:
            desired_min = 2.0 * atr5_val
            desired_max = (3.0 if strong_momentum else 2.4) * atr5_val
            target_risk = risk
            if risk < desired_min:
                target_risk = desired_min
            elif risk > desired_max:
                target_risk = desired_max
            if not np.isclose(target_risk, risk):
                if decision_side == "buy":
                    sl = entry - target_risk
                    risk = entry - sl
                    tp1 = entry + tp1_mult * risk
                    tp2 = entry + tp2_mult * risk
                    tp1_dist = tp1 - entry
                    ok_math = (sl < entry < tp1 <= tp2)
                else:
                    sl = entry + target_risk
                    risk = sl - entry
                    tp1 = entry - tp1_mult * risk
                    tp2 = entry - tp2_mult * risk
                    tp1_dist = entry - tp1
                    ok_math = (tp2 <= tp1 < entry < sl)
                if atr5_val > 0:
                    risk_ratio = target_risk / atr5_val
                    adjust_note = f"GOLD stop lazítás: ATR alapú kockázat ×{risk_ratio:.2f}"
                    if adjust_note not in reasons:
                        reasons.append(adjust_note)

        if asset == "NVDA":
            atr_basis = None
            if atr1h is not None and atr1h > 0:
                atr_basis = float(atr1h)
            elif atr5_val > 0:
                atr_basis = float(atr5_val) * 12.0
            if atr_basis and NVDA_STOP_ATR_MIN > 0:
                desired_min = NVDA_STOP_ATR_MIN * atr_basis
                desired_max = (
                    NVDA_STOP_ATR_MAX * atr_basis
                    if NVDA_STOP_ATR_MAX and NVDA_STOP_ATR_MAX >= NVDA_STOP_ATR_MIN
                    else desired_min * 1.5
                )
                target_risk = min(max(risk, desired_min), desired_max)
                if not np.isclose(target_risk, risk):
                    if decision_side == "buy":
                        sl = entry - target_risk
                        risk = entry - sl
                        tp1 = entry + tp1_mult * risk
                        tp2 = entry + tp2_mult * risk
                        tp1_dist = tp1 - entry
                        ok_math = (sl < entry < tp1 <= tp2)
                    else:
                        sl = entry + target_risk
                        risk = sl - entry
                        tp1 = entry - tp1_mult * risk
                        tp2 = entry - tp2_mult * risk
                        tp1_dist = entry - tp1
                        ok_math = (tp2 <= tp1 < entry < sl)
                    ratio = target_risk / atr_basis if atr_basis else 0.0
                    adjust_note = f"NVDA stop igazítás: kockázat ×{ratio:.2f} ATR"
                    if adjust_note not in reasons:
                        reasons.append(adjust_note)

        if asset == "BTCUSD" and atr5_val > 0:
            lower_mult = 0.9 if btc_momentum_override_active else 1.0
            upper_mult = 1.6 if btc_momentum_override_active else 2.2
            desired_min = lower_mult * atr5_val
            desired_max = upper_mult * atr5_val
            target_risk = risk
            if risk < desired_min:
                target_risk = desired_min
            elif risk > desired_max:
                target_risk = desired_max
            if not np.isclose(target_risk, risk):
                if decision_side == "buy":
                    sl = entry - target_risk
                    risk = entry - sl
                    tp1 = entry + tp1_mult * risk
                    tp2 = entry + tp2_mult * risk
                    tp1_dist = tp1 - entry
                    ok_math = (sl < entry < tp1 <= tp2)
                else:
                    sl = entry + target_risk
                    risk = sl - entry
                    tp1 = entry - tp1_mult * risk
                    tp2 = entry - tp2_mult * risk
                    tp1_dist = entry - tp1
                    ok_math = (tp2 <= tp1 < entry < sl)
                risk_ratio = target_risk / atr5_val if atr5_val else 0.0
                adjust_note = f"BTC ATR stop igazítás: kockázat ×{risk_ratio:.2f}"
                if adjust_note not in reasons:
                    reasons.append(adjust_note)

        risk_min = max(
            MIN_RISK_ABS.get(asset, MIN_RISK_ABS["default"]),
            entry * MIN_STOPLOSS_PCT,
            buf,
        )
        if risk < risk_min:
            if decision_side == "buy":
                sl = entry - risk_min
                risk = entry - sl
            else:
                sl = entry + risk_min
                risk = sl - entry

        risk = max(risk, 1e-6)
        min_stop_distance = entry * MIN_STOPLOSS_PCT
        if asset == "EURUSD":
            min_stop_distance = 10 * EURUSD_PIP
        min_stoploss_ok_local = risk >= min_stop_distance - 1e-9
        if not min_stoploss_ok_local:
            min_stoploss_ok = False

        if decision_side == "buy":
            tp1 = entry + tp1_mult * risk
            tp2 = entry + tp2_mult * risk
            rr  = (tp2 - entry) / risk
            tp1_dist = tp1 - entry
            ok_math = ok_math and (sl < entry < tp1 <= tp2)
            gross_pct = tp1_dist / entry
        else:
            tp1 = entry - tp1_mult * risk
            tp2 = entry - tp2_mult * risk
            rr  = (entry - tp2) / risk
            tp1_dist = entry - tp1
            ok_math = ok_math and (tp2 <= tp1 < entry < sl)
            gross_pct = tp1_dist / entry

        btc_level_blockers: List[str] = []
        if asset == "BTCUSD" and entry is not None and sl is not None and tp1 is not None:
            profile_used = btc_profile_name or _btc_active_profile()
            mode_label = mode or "core"
            entry_val = float(entry)
            sl_val = float(sl)
            tp1_val = float(tp1)
            tp2_val = float(tp2) if tp2 is not None else None
            risk_val = float(risk)
            info_payload: Dict[str, Any] = {
                "profile": profile_used,
                "mode": mode_label,
                "side": decision_side,
                "entry": entry_val,
                "sl": sl_val,
                "tp1": tp1_val,
                "tp2": tp2_val,
                "risk_abs": risk_val,
                "tp1_mult": float(tp1_mult),
                "tp2_mult": float(tp2_mult),
                "rr_required": float(rr_required),
            }
            btc_sl_tp_checks(
                profile_used,
                asset,
                float(atr5_val),
                entry_val,
                sl_val,
                tp1_val,
                float(rr_required),
                info_payload,
                btc_level_blockers,
            )
            info_payload["tp1_distance"] = abs(tp1_val - entry_val)
            info_payload["sl_distance"] = abs(entry_val - sl_val)
            info_payload["rr_tp2"] = float(rr) if rr is not None else None
            if btc_level_blockers:
                for blocker in btc_level_blockers:
                    if blocker not in missing:
                        missing.append(blocker)
            state_key = mode_label
            state_payload = btc_level_checks_state.setdefault(state_key, {})
            state_payload.update(info_payload)
            state_payload["blockers"] = list(btc_level_blockers)

        rel_atr_local = float(rel_atr) if not np.isnan(rel_atr) else float("nan")
        high_vol = (not np.isnan(rel_atr_local)) and (rel_atr_local >= ATR_VOL_HIGH_REL)
        cost_mult = COST_MULT_HIGH_VOL if high_vol else COST_MULT_DEFAULT
        tp_min_base = tp_min_pct_for(asset, rel_atr_local, session_ok_flag)
        tp_min_pct = tp_min_base
        if low_atr_tp_override is not None:
            tp_min_pct = min(tp_min_base, low_atr_tp_override)
        if low_atr_meta:
            entry_thresholds_meta.setdefault("low_atr_override", {}).update(
                {
                    "tp_min_pct_effective": tp_min_pct,
                    "rr_required_effective": rr_required,
                }
            )
        tp_min_profit_pct = tp_min_pct
        overnight_days = estimate_overnight_days(asset, analysis_now)
        cost_round_pct, overnight_pct = compute_cost_components(asset, entry, overnight_days)
        total_cost_pct = cost_mult * cost_round_pct + overnight_pct
        net_pct = gross_pct - total_cost_pct
        tp1_net_pct_value = net_pct

        atr5_min_mult = ATR5_MIN_MULT_ASSET.get(asset, ATR5_MIN_MULT)

        min_profit_abs = max(
            tp_min_abs_default,
            tp_min_pct * entry,
            (cost_mult * cost_round_pct + overnight_pct) * entry,
            atr5_min_mult * atr5_val,
        )

        if (not ok_math) or (rr is None) or (rr < rr_required) or (tp1_dist < min_profit_abs) or (net_pct < tp_net_threshold):
            if rr is None or rr < rr_required:
                missing.append(f"rr_math>={rr_required:.1f}")
            if tp1_dist < min_profit_abs:
                missing.append("tp_min_profit")
            if net_pct < tp_net_threshold:
                missing.append(tp_net_label)
            if not min_stoploss_ok_local:
                missing.append("min_stoploss")
            return False
        last_computed_risk = risk
        return True

    reversal_mode_used = False
    if asset == "EURUSD" and eurusd_range_signal in {"buy", "sell"}:
        mode = "range_reversal"
        decision = eurusd_range_signal
        required_list = ["session", "data_integrity", "spread_guard", "reversal_signal"]
        missing = []
        if not session_ok_flag:
            missing.append("session")
        if not data_integrity_ok:
            missing.append("data_integrity")
        if not spread_gate_ok:
            missing.append("spread_guard")
        if decision in {"buy", "sell"} and not missing:
            band_width_pips = eurusd_range_levels.get("range_width_pips")
            if band_width_pips is None or not np.isfinite(band_width_pips):
                band_width_pips = 35.0
            band_width_pips = min(max(band_width_pips, 30.0), 40.0)
            band_width = band_width_pips * EURUSD_PIP
            anchor_level = eurusd_range_levels.get("range_low") if decision == "buy" else eurusd_range_levels.get("range_high")
            if anchor_level is None:
                anchor_level = price_for_calc
            stop_buffer = 0.0012
            if decision == "buy":
                sl = anchor_level - stop_buffer
                entry = price_for_calc
                tp1 = anchor_level + band_width
            else:
                sl = anchor_level + stop_buffer
                entry = price_for_calc
                tp1 = anchor_level - band_width
            tp2 = tp1
            if entry is not None and sl is not None and tp1 is not None:
                risk = abs(entry - sl)
                min_stoploss_ok = risk >= 10 * EURUSD_PIP - 1e-9
                rr = abs(tp1 - entry) / risk if risk > 0 else None
                if not min_stoploss_ok or rr is None or rr <= 0:
                    missing.append("min_stoploss")
            else:
                missing.append("reversal_signal")
            if not missing:
                reversal_mode_used = True
                reasons.append("EURUSD range/reversal setup aktiválva — ML küszöb lazítva")
                entry_thresholds_meta["eurusd_range_trade"] = {
                    "band_pips": band_width_pips,
                    "stop_buffer": stop_buffer,
                    "signal": eurusd_range_signal,
                }
        if missing:
            decision = "no entry"
        else:
            mode = "reversal"

    if asset == "XAGUSD" and xag_reversal_active and xag_reversal_side in {"buy", "sell"}:
        mode = "reversal"
        decision = xag_reversal_side
        required_list = ["session", "data_integrity", "spread_guard", "reversal_signal"]
        missing = []
        if not session_ok_flag:
            missing.append("session")
        if not data_integrity_ok:
            missing.append("data_integrity")
        if not spread_gate_ok:
            missing.append("spread_guard")
        if xag_atr_floor_triggered:
            missing.append("atr_floor")
        if decision in {"buy", "sell"} and not missing:
            position_size_scale *= 0.8
            if not compute_levels(decision, 1.0, tp1_mult=1.0, tp2_mult=1.5):
                decision = "no entry"
            else:
                reversal_mode_used = True
        else:
            decision = "no entry"

    if (
        not can_enter_core
        and not reversal_mode_used
        and _nvda_precision_override_ready(
            asset,
            precision_plan=precision_plan,
            precision_threshold=precision_threshold_value,
            spread_gate_ok=spread_gate_ok,
            session_entry_open=session_entry_open,
            risk_guard_allowed=bool(risk_guard_meta.get("allowed")),
            base_core_ok=base_core_ok,
        )
    ):
        decision = "buy" if effective_bias == "long" else "sell" if effective_bias == "short" else "no entry"
        if decision in {"buy", "sell"}:
            required_list = list(core_required)
            missing = [item for item in missing if not str(item).startswith("P_score>=")]
            position_size_scale *= 0.4
            entry_thresholds_meta["precision_override"] = True
            if compute_levels(decision, core_rr_min, core_tp1_mult, core_tp2_mult):
                precision_override_active = True
                override_note = "Precision override: reduced size belépő engedve"
                if override_note not in reasons:
                    reasons.append(override_note)
            else:
                decision = "no entry"

    if can_enter_core and not reversal_mode_used:
        if effective_bias == "long":
            decision = "buy"
        elif effective_bias == "short":
            decision = "sell"
        else:
            decision = "no entry"
        mode = "core"
        required_list = list(core_required)
        if decision in ("buy", "sell"):
            if not compute_levels(decision, core_rr_min, core_tp1_mult, core_tp2_mult):
                decision = "no entry"
            else:
                if asset == "BTCUSD":
                    profile_for_no_chase = btc_profile_name or _btc_active_profile()
                    core_slip_info: Dict[str, Any] = {}
                    if (
                        entry is not None
                        and sl is not None
                        and last5_close is not None
                    ):
                        try:
                            entry_price = float(entry)
                            trigger_price = float(last5_close)
                            sl_price = float(sl)
                        except (TypeError, ValueError):
                            entry_price = trigger_price = sl_price = None
                        if (
                            entry_price is not None
                            and trigger_price is not None
                            and sl_price is not None
                        ):
                            slip_limit = BTC_NO_CHASE_R.get(
                                profile_for_no_chase, BTC_NO_CHASE_R["baseline"]
                            )
                            limit_map = globals().get("_BTC_NO_CHASE_LIMITS", {})
                            if (
                                isinstance(limit_map, dict)
                                and profile_for_no_chase in limit_map
                            ):
                                try:
                                    slip_limit = float(limit_map[profile_for_no_chase])
                                except (TypeError, ValueError):
                                    slip_limit = BTC_NO_CHASE_R.get(
                                        profile_for_no_chase, BTC_NO_CHASE_R["baseline"]
                                    )
                            slip_abs = abs(entry_price - trigger_price)
                            denom = max(1e-9, abs(trigger_price - sl_price))
                            slip_r = slip_abs / denom
                            core_slip_info = {
                                "slip": float(slip_abs),
                                "slip_r": float(slip_r),
                                "limit_r": float(slip_limit),
                                "entry": float(entry_price),
                                "trigger": float(trigger_price),
                                "sl": float(sl_price),
                                "evaluated": True,
                            }
                            violation = slip_r > slip_limit
                            core_slip_info["violated"] = violation
                            entry_thresholds_meta["core_no_chase"] = core_slip_info
                            if violation:
                                msg = (
                                    "Core no-chase szabály: aktuális ár kedvezőtlenebb "
                                    f"mint {slip_limit:.2f}R"
                                )
                                if msg not in reasons:
                                    reasons.append(msg)
                                if "no_chase_core" not in missing:
                                    missing.append("no_chase_core")
                                append_blocker("no_chase_core")
                                decision = "no entry"
                                entry = sl = tp1 = tp2 = rr = None
                    else:
                        entry_thresholds_meta.setdefault("core_no_chase", {}).setdefault(
                            "evaluated", False
                        )
                if decision in ("buy", "sell") and profile_slippage_limit is not None:
                    slip_state = compute_slippage_state(decision, profile_slippage_limit)
                    if slip_state:
                        entry_thresholds_meta.setdefault("slippage_guard", {})["core"] = slip_state
                        if slip_state["slip"] > slip_state["allowed"] + 1e-9:
                            if "slippage_guard" not in missing:
                                missing.append("slippage_guard")
                            slip_reason = (
                                f"Belépési csúszás {slip_state['slip_r']:.2f}R meghaladja a limitet"
                                f" ({profile_slippage_limit:.2f}R)"
                            )
                            if slip_reason not in reasons:
                                reasons.append(slip_reason)
                            decision = "no entry"
                            entry = sl = tp1 = tp2 = rr = None
                if decision in ("buy", "sell") and tp1_net_pct_value is not None:
                    msg_net = f"TP1 nettó profit ≈ {tp1_net_pct_value*100:.2f}%"
                    if msg_net not in reasons:
                        reasons.append(msg_net)
    elif not reversal_mode_used:
        if mom_dir is not None:
            mode = "momentum"
            required_list = list(mom_required)
            missing = []
            momentum_used = True
            decision = mom_dir
            if not compute_levels(decision, momentum_rr_min, mom_tp1_mult, mom_tp2_mult):
                decision = "no entry"
            else:
                no_chase_violation = False
                slip_info: Dict[str, Any] = {}
                if (
                    decision in ("buy", "sell")
                    and last_computed_risk is not None
                    and price_for_calc is not None
                    and last5_close is not None
                    and sl is not None
                ):
                    slip_limit_r = (
                        BTC_NO_CHASE_R.get("baseline", 0.25)
                        if asset == "BTCUSD"
                        else 0.2
                    )
                    if asset != "BTCUSD" and profile_slippage_limit is not None:
                        try:
                            slip_limit_r = float(profile_slippage_limit)
                        except (TypeError, ValueError):
                            slip_limit_r = slip_limit_r
                    profile_for_no_chase = btc_profile_name or _btc_active_profile()
                    no_chase_limit_r = slip_limit_r
                    risk_no_chase = risk_template_values.get("no_chase_r")
                    if risk_no_chase is not None:
                        try:
                            no_chase_limit_r = float(risk_no_chase)
                        except (TypeError, ValueError):
                            no_chase_limit_r = slip_limit_r
                    if asset == "BTCUSD":
                        no_chase_cfg = None
                        if isinstance(BTC_PROFILE_CONFIG, dict):
                            no_chase_cfg = BTC_PROFILE_CONFIG.get("no_chase_r")
                        if no_chase_cfg is None:
                            profile_cfg = BTC_PROFILE_OVERRIDES.get(profile_for_no_chase)
                            if isinstance(profile_cfg, dict):
                                no_chase_cfg = profile_cfg.get("no_chase_r")
                        if no_chase_cfg is not None:
                            try:
                                no_chase_limit_r = float(no_chase_cfg)
                            except (TypeError, ValueError):
                                no_chase_limit_r = BTC_NO_CHASE_R.get(
                                    profile_for_no_chase, BTC_NO_CHASE_R.get("baseline", slip_limit_r)
                                )
                        else:
                            no_chase_limit_r = BTC_NO_CHASE_R.get(
                                profile_for_no_chase, BTC_NO_CHASE_R.get("baseline", slip_limit_r)
                            )
                        slip_limit_r = no_chase_limit_r
                        if btc_momentum_cfg:
                            max_slip_raw = btc_momentum_cfg.get("max_slippage_r")
                            if max_slip_raw is not None:
                                try:
                                    slip_limit_r = float(max_slip_raw)
                                except (TypeError, ValueError):
                                    slip_limit_r = max(slip_limit_r, no_chase_limit_r)
                    else:
                        slip_limit_r = min(slip_limit_r, no_chase_limit_r)
                    allowed_slip = slip_limit_r * last_computed_risk
                    if decision == "buy":
                        slip = max(0.0, price_for_calc - last5_close)
                    else:
                        slip = max(0.0, last5_close - price_for_calc)
                    entry_price = float(price_for_calc)
                    trigger_price = float(last5_close)
                    sl_price = float(sl)
                    slip_r = abs(entry_price - trigger_price) / max(1e-9, abs(trigger_price - sl_price))
                    limit_used = slip_limit_r
                    if asset == "BTCUSD":
                        limit_map = globals().get("_BTC_NO_CHASE_LIMITS", {})
                        if isinstance(limit_map, dict) and profile_for_no_chase in limit_map:
                            try:
                                limit_used = float(limit_map[profile_for_no_chase])
                            except (TypeError, ValueError):
                                limit_used = no_chase_limit_r
                        else:
                            limit_used = no_chase_limit_r
                        no_chase_violation = btc_no_chase_violated(
                            profile_for_no_chase,
                            entry_price,
                            trigger_price,
                            sl_price,
                        )
                    else:
                        no_chase_violation = slip > allowed_slip + 1e-9
                    slip_info = {
                        "slip": float(slip),
                        "allowed": float(allowed_slip),
                        "limit_r": float(limit_used if asset == "BTCUSD" else slip_limit_r),
                        "slip_r": float(slip_r),
                    }
                if slip_info:
                    entry_thresholds_meta.setdefault("slippage_guard", {})["momentum"] = {
                        "slip": float(slip),
                        "allowed": float(allowed_slip),
                        "limit_r": float(slip_limit_r),
                        "slip_r": float(slip_r),
                    }
                    if asset == "BTCUSD":
                        slip_info["no_chase_r"] = {
                            "slip_r": float(slip_info.get("slip_r", slip_r)),
                            "limit_r": float(limit_used if asset == "BTCUSD" else slip_limit_r),
                        }
                    slip_info["violated"] = no_chase_violation
                    entry_thresholds_meta["momentum_no_chase"] = slip_info
                if no_chase_violation:
                    reasons.append(
                        f"Momentum no-chase szabály: aktuális ár kedvezőtlenebb mint {slip_info['limit_r']:.2f}R"
                    )
                    decision = "no entry"
                    momentum_used = False
                    missing = ["no_chase"]
                else:
                    reason_msg = "Momentum override"
                    if mom_trigger_desc:
                        reason_msg += f" ({mom_trigger_desc})"
                    reasons.append(reason_msg)
                    reasons.append("Momentum: rész-realizálás javasolt 2.0R-n és trailing SL aktiválása")
                    if momentum_vol_ratio is not None:
                        reasons.append(
                            f"Momentum volume ratio ≈ {momentum_vol_ratio:.2f}"
                        )
                    if last_computed_risk is not None and entry is not None:
                        if decision == "buy":
                            trail_price = entry + last_computed_risk * MOMENTUM_TRAIL_LOCK
                        else:
                            trail_price = entry - last_computed_risk * MOMENTUM_TRAIL_LOCK
                        momentum_trailing_plan = {
                            "activation_rr": MOMENTUM_TRAIL_TRIGGER_R,
                            "trail_price": trail_price,
                            "lock_ratio": MOMENTUM_TRAIL_LOCK,
                        }
                    P = max(P, 75)
                    if tp1_net_pct_value is not None:
                        msg_net = f"TP1 nettó profit ≈ {tp1_net_pct_value*100:.2f}%"
                        if msg_net not in reasons:
                            reasons.append(msg_net)
        precision_direction: Optional[str] = None
    execution_playbook: List[Dict[str, Any]] = []
    if decision in ("buy", "sell"):
        precision_direction = decision
    elif effective_bias == "long":
        precision_direction = "buy"
    elif effective_bias == "short":
        precision_direction = "sell"

    momentum_diagnostics = {
        "evaluated": bool(asset in ENABLE_MOMENTUM_ASSETS),
        "used_for_decision": bool(momentum_used),
        "direction_result": mom_dir,
        "required": list(mom_required),
        "missing": list(dict.fromkeys(missing_mom)),
        "liquidity_data_available": bool(liquidity_data_available),
        "flow_data_available": bool(flow_data_available),
        "ofi_available": bool(ofi_available) if "ofi_available" in locals() else None,
        "momentum_trigger_ok": bool(momentum_trigger_ok) if momentum_trigger_ok is not None else None,
        "momentum_atr_ok": bool(mom_atr_ok) if mom_atr_ok is not None else None,
        "trigger_desc": mom_trigger_desc if "mom_trigger_desc" in locals() else None,
        "precision_data_gap_mode": bool(precision_disabled_due_to_data_gap),
    }

    if precision_direction and not precision_disabled_due_to_data_gap:
        atr5_value = float(atr5) if atr5 is not None and np.isfinite(float(atr5)) else None
        precision_plan = compute_precision_entry(
            asset,
            precision_direction,
            k1m_closed,
            k5m_closed,
            price_for_calc,
            atr5_value,
            order_flow_metrics,
            score_threshold=precision_threshold_value,
        )

    if precision_disabled_due_to_data_gap:
        precision_trigger_state = "disabled"

    if precision_plan:
        precision_plan.setdefault("score_threshold", precision_threshold_value)
        precision_plan.setdefault("ready_timeout_minutes", precision_timeouts.get("ready"))
        precision_plan.setdefault("arming_timeout_minutes", precision_timeouts.get("arming"))
        precision_profile_for_state = precision_plan.get("profile") or precision_plan.get(
            "profile_name"
        )
        if not precision_profile_for_state:
            precision_profile_for_state = asset_entry_profile
        precision_state = str(precision_plan.get("state") or "none")
        try:
            precision_score_val = float(precision_plan.get("score") or 0.0)
        except (TypeError, ValueError):
            precision_score_val = 0.0
        context_block = precision_plan.setdefault("context", {})
        if isinstance(context_block, dict) and intraday_profile:
            context_block.setdefault(
                "intraday_profile",
                {
                    "range_state": intraday_profile.get("range_state"),
                    "range_position": intraday_profile.get("range_position"),
                    "range_vs_atr": intraday_profile.get("range_vs_atr"),
                },
            )
        precision_ready_for_entry = precision_score_val >= precision_threshold_value
        precision_plan["score_ready"] = bool(precision_plan.get("score_ready") or precision_ready_for_entry)
        precision_trigger_state = str(precision_plan.get("trigger_state") or "standby")
        precision_plan["trigger_state"] = precision_trigger_state
        precision_flow_ready = bool(precision_plan.get("order_flow_ready"))
        if not precision_flow_ready and precision_plan.get("order_flow_optional"):
            precision_flow_ready = True
            precision_plan["order_flow_ready"] = True
        precision_trigger_ready = bool(
            precision_plan.get("trigger_ready")
            or precision_trigger_state in {"arming", "fire"}
        )
        precision_plan["trigger_ready"] = precision_trigger_ready
        if precision_plan.get("trigger_levels") is None:
            precision_plan["trigger_levels"] = {}
        _emit_precision_gate_log(
            asset,
            "precision_score",
            precision_ready_for_entry,
            "score_ready" if precision_ready_for_entry else "score_below",
            order_flow_metrics=order_flow_metrics,
            tick_order_flow=tick_order_flow,
            latency_seconds=latency_by_frame,
            precision_plan=precision_plan,
            timestamp=analysis_now,
            extra=_with_gate_context({
                "score": precision_score_val,
                "threshold": precision_threshold_value,
            }),
        )
        if precision_gate_label not in required_list:
            required_list.append(precision_gate_label)
        if not precision_ready_for_entry and precision_gate_label not in missing:
            missing.append(precision_gate_label)

        # Only track downstream precision gates once the score threshold is met –
        # otherwise suppressed-vol regimes would spam redundant blockers.
        precision_flow_gate_needed = precision_ready_for_entry and flow_data_available
        precision_trigger_gate_needed = precision_ready_for_entry

        if precision_ready_for_entry and not flow_data_available:
            if "Order-flow gates skipped (no tick/imbalance data)" not in reasons:
                reasons.append("Order-flow gates skipped (no tick/imbalance data)")

        if precision_flow_gate_needed:
            if precision_flow_gate_label not in required_list:
                required_list.append(precision_flow_gate_label)
            if not precision_flow_ready and precision_flow_gate_label not in missing:
                missing.append(precision_flow_gate_label)
        else:
            missing = [item for item in missing if item != precision_flow_gate_label]
            if precision_flow_gate_label in required_list:
                required_list = [
                    item for item in required_list if item != precision_flow_gate_label
                ]

        if precision_trigger_gate_needed:
            if precision_trigger_gate_label not in required_list:
                required_list.append(precision_trigger_gate_label)
            if not precision_trigger_ready and precision_trigger_gate_label not in missing:
                missing.append(precision_trigger_gate_label)
        else:
            missing = [item for item in missing if item != precision_trigger_gate_label]
            if precision_trigger_gate_label in required_list:
                required_list = [
                    item for item in required_list if item != precision_trigger_gate_label
                ]

        if precision_flow_ready:
            missing = [item for item in missing if item != precision_flow_gate_label]
        if precision_trigger_ready:
            missing = [item for item in missing if item != precision_trigger_gate_label]
        _emit_precision_gate_log(
            asset,
            "precision_flow",
            precision_flow_ready,
            "flow_ready" if precision_flow_ready else "flow_blocked",
            order_flow_metrics=order_flow_metrics,
            tick_order_flow=tick_order_flow,
            latency_seconds=latency_by_frame,
            precision_plan=precision_plan,
            timestamp=analysis_now,
            extra=_with_gate_context(
                {
                    "blockers": precision_plan.get("order_flow_blockers"),
                    "precision_profile": precision_profile_for_state,
                    "precision_threshold": precision_threshold_value,
                }
            ),
        )
        _emit_precision_gate_log(
            asset,
            "precision_trigger",
            precision_trigger_ready,
            "trigger_ready" if precision_trigger_ready else f"trigger_{precision_trigger_state}",
            order_flow_metrics=order_flow_metrics,
            tick_order_flow=tick_order_flow,
            latency_seconds=latency_by_frame,
            precision_plan=precision_plan,
            timestamp=analysis_now,
            extra=_with_gate_context(
                {
                    "trigger_state": precision_trigger_state,
                    "precision_profile": precision_profile_for_state,
                    "precision_threshold": precision_threshold_value,
                }
            ),
        )
        blockers = precision_plan.get("order_flow_blockers")
        blockers_snapshot = list(blockers) if isinstance(blockers, list) else blockers
        precision_gate_snapshot = {
            "score_ready": precision_ready_for_entry,
            "score": precision_score_val,
            "threshold": precision_threshold_value,
            "flow_ready": precision_flow_ready,
            "trigger_ready": precision_trigger_ready,
            "trigger_state": precision_trigger_state,
            "flow_blockers": blockers_snapshot,
            "profile": precision_profile_for_state,
        }
        entry_thresholds_meta["precision_gate_state"] = precision_gate_snapshot

    if precision_plan:
        if not precision_ready_for_entry:
            if decision in ("buy", "sell"):
                score_note = (
                    f"Precision score {precision_score_val:.2f} < {precision_threshold_label}"
                )
                if score_note not in reasons:
                    reasons.append(score_note)
                decision = "no entry"
                entry = sl = tp1 = tp2 = rr = None
        else:
            other_missing = [
                item
                for item in missing
                if item not in {precision_gate_label, precision_flow_gate_label, precision_trigger_gate_label}
            ]
            if decision in ("buy", "sell"):
                # csak akkor blokkoljon a flow, ha a flow-kapu ténylegesen aktív/szükséges
                if precision_flow_gate_needed and (not precision_flow_ready):
                    flow_note = "Precision belépő: order flow megerősítésre vár"
                    if flow_note not in reasons:
                        reasons.append(flow_note)
                    decision = "precision_ready"
                    entry = sl = tp1 = tp2 = rr = None
                elif precision_trigger_state != "fire":
                    arming_note = "Precision belépő: trigger ablak aktív"
                    if arming_note not in reasons:
                        reasons.append(arming_note)
                    decision = "precision_arming"
                    entry = sl = tp1 = tp2 = rr = None
            if (
                decision == "no entry"
                and precision_trigger_state in {"arming", "fire"}
                and precision_flow_ready
                and not other_missing
            ):
                decision = "precision_arming"
                entry = sl = tp1 = tp2 = rr = None
                armed_note = "Precision trigger kész → limit figyelés"
                if armed_note not in reasons:
                    reasons.append(armed_note)
        _emit_precision_gate_log(
            asset,
            "precision_state",
            decision in {"buy", "sell", "precision_ready", "precision_arming"},
            f"state_{decision}",
            order_flow_metrics=order_flow_metrics,
            tick_order_flow=tick_order_flow,
            latency_seconds=latency_by_frame,
            precision_plan=precision_plan,
            timestamp=analysis_now,
            extra=_with_gate_context({
                "decision": decision,
                "missing": list(missing),
                "precision_profile": precision_profile_for_state,
                "precision_threshold": precision_threshold_value,
                "precision_state": precision_state,
            }),
        )

    def _fmt_price(value: Optional[float]) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        v = float(value)
        if abs(v) >= 1000:
            decimals = 1
        elif abs(v) >= 100:
            decimals = 2
        elif abs(v) >= 10:
            decimals = 3
        else:
            decimals = 4
        return f"{v:.{decimals}f}"

    if (
        decision in ("buy", "sell")
        and precision_plan
        and precision_plan.get("entry_window")
        and precision_plan.get("stop_loss") is not None
    ):
        window = precision_plan["entry_window"]
        stop_loss = precision_plan.get("stop_loss")
        note = (
            f"Precision plan: entry {_fmt_price(window[0])}–{_fmt_price(window[1])}, SL {_fmt_price(stop_loss)}"
        )
        if note not in reasons:
            reasons.append(note)
        if precision_plan.get("confidence") in {"high", "medium"}:
            score_note = f"Precision confidence {precision_plan['confidence']} ({precision_plan['score']})"
            if score_note not in reasons:
                reasons.append(score_note)

    if decision in ("buy", "sell") and entry is not None and sl is not None:
        direction_label = "Long" if decision == "buy" else "Short"
        execution_playbook.append(
            {
                "step": "entry",
                "description": f"{direction_label} belépő {entry:.5f} környékén",
                "risk_abs": last_computed_risk,
                "confidence": realtime_confidence,
            }
        )
        if position_size_scale < 1.0:
            execution_playbook.append(
                {
                    "step": "position_scale",
                    "description": f"Pozícióméret skálázás ×{position_size_scale:.2f}",
                    "scale": position_size_scale,
                }
            )
        if precision_plan:
            window = precision_plan.get("entry_window")
            if (
                isinstance(window, (list, tuple))
                and len(window) == 2
                and all(isinstance(v, (int, float)) for v in window if v is not None)
            ):
                execution_playbook.append(
                    {
                        "step": "scalp_window",
                        "description": f"Precision belépő zóna {window[0]:.5f}–{window[1]:.5f}",
                        "confidence": precision_plan.get("confidence"),
                    }
                )
        if tp1 is not None:
            execution_playbook.append(
                {
                    "step": "tp1",
                    "description": f"50% pozíció zárása TP1-n {tp1:.5f}",
                    "rr": current_tp1_mult,
                }
            )
        if tp2 is not None:
            execution_playbook.append(
                {
                    "step": "tp2",
                    "description": f"Fennmaradó pozíció célár TP2 {tp2:.5f}",
                    "rr": current_tp2_mult,
                }
            )
        if momentum_trailing_plan:
            execution_playbook.append(
                {
                    "step": "trailing",
                    "description": "Momentum trail aktiválás a jelzett R-szinten",
                    "trigger_rr": momentum_trailing_plan.get("activation_rr"),
                    "lock_ratio": momentum_trailing_plan.get("lock_ratio"),
                }
            )
        if range_time_stop_plan:
            timeout_val = range_time_stop_plan.get("timeout")
            breakeven_val = range_time_stop_plan.get("breakeven_trigger")
            giveback_val = range_time_stop_plan.get("giveback_ratio")
            description = "Time-stop menedzsment aktív"
            if timeout_val and breakeven_val is not None:
                description = (
                    f"Time-stop {int(timeout_val)} percig, ha nem érjük el "
                    f"{float(breakeven_val):.2f}R-t"
                )
            if giveback_val is not None:
                try:
                    giveback_pct = float(giveback_val) * 100.0
                    description += f", profit giveback {giveback_pct:.0f}%"
                except (TypeError, ValueError):
                    giveback_val = None
            step_payload: Dict[str, Any] = {"step": "time_stop", "description": description}
            if timeout_val:
                step_payload["timeout_minutes"] = int(timeout_val)
            if breakeven_val is not None:
                step_payload["breakeven_trigger_r"] = float(breakeven_val)
            if giveback_val is not None:
                step_payload["giveback_ratio"] = float(giveback_val)
            execution_playbook.append(step_payload)

    if precision_plan:
        precision_state = "none"
        score_value = safe_float(precision_plan.get("score")) or 0.0
        trigger_state_value = str(precision_plan.get("trigger_state") or "")
        trigger_ready_flag = bool(precision_plan.get("trigger_ready")) or trigger_state_value in {
            "arming",
            "fire",
        }
        profile_for_precision: Optional[str] = None
        if asset == "BTCUSD":
            if btc_profile_name and btc_profile_name in BTC_PRECISION_MIN:
                profile_for_precision = btc_profile_name
            else:
                profile_for_precision = _btc_active_profile()
        elif btc_profile_name:
            profile_for_precision = btc_profile_name
        precision_state = btc_precision_state(
            profile_for_precision or "baseline",
            asset,
            score_value,
            trigger_ready_flag,
        )
        if precision_state == "none" and decision in {"precision_ready", "precision_arming"}:
            precision_state = decision

        window_payload: Optional[List[float]] = None
        window = precision_plan.get("entry_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            try:
                window_payload = [float(window[0]), float(window[1])]
            except (TypeError, ValueError):
                window_payload = None
        if precision_state in {"precision_ready", "precision_arming"}:
            timeout_key = (
                "ready_timeout_minutes"
                if precision_state == "precision_ready"
                else "arming_timeout_minutes"
            )
            ttl_val = precision_plan.get(timeout_key)
            fallback_key = "ready" if precision_state == "precision_ready" else "arming"
            if ttl_val in {None, 0}:
                ttl_val = precision_timeouts.get(fallback_key)
            if precision_state == "precision_ready":
                ttl_numeric: Optional[float] = None
                try:
                    ttl_numeric = float(ttl_val) if ttl_val not in {None, 0} else None
                except (TypeError, ValueError):
                    ttl_numeric = None
                if ttl_numeric is None or ttl_numeric <= 0:
                    ttl_val = 15.0
                else:
                    ttl_val = min(20.0, max(10.0, ttl_numeric))
            ttl_display: Optional[int] = None
            if ttl_val not in {None, 0}:
                try:
                    ttl_display = max(1, int(round(float(ttl_val))))
                except (TypeError, ValueError):
                    ttl_display = None
            limit_desc = "Precision parkolt limit belépő"
            if window_payload:
                limit_desc += f" {window_payload[0]:.5f}–{window_payload[1]:.5f}"
            if ttl_display:
                limit_desc += f" — {ttl_display} perc timeout"
            limit_step: Dict[str, Any] = {
                "step": "precision_limit",
                "description": limit_desc,
                "direction": precision_plan.get("direction") or precision_direction,
                "confidence": precision_plan.get("confidence"),
                "entry_window": window_payload,
            }
            if ttl_display:
                limit_step["timeout_minutes"] = ttl_display
            execution_playbook.append(limit_step)
            limit_note = "Precision limit belépő parkolva"
            if ttl_display:
                limit_note += f" {ttl_display} percig"
            else:
                default_ttl = precision_timeouts.get("ready")
                if default_ttl:
                    limit_note += f" ~{default_ttl} percig"
            if limit_note not in reasons:
                reasons.append(limit_note)
        trigger_levels_raw = precision_plan.get("trigger_levels") or {}
        trigger_levels_payload: Dict[str, Any] = {}
        if isinstance(trigger_levels_raw, dict):
            for key, value in trigger_levels_raw.items():
                try:
                    trigger_levels_payload[key] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    trigger_levels_payload[key] = value
        trigger_state_payload = precision_plan.get("trigger_state") or "standby"
        trigger_step = {
            "step": "precision_trigger",
            "state": trigger_state_payload,
            "description": f"Precision trigger: {trigger_state_payload}",
            "entry_window": window_payload,
            "trigger_levels": trigger_levels_payload or None,
            "ready_ts": precision_plan.get("ready_ts"),
            "score": precision_plan.get("score"),
            "score_threshold": precision_plan.get(
                "score_threshold", PRECISION_SCORE_THRESHOLD_DEFAULT
            ),
            "score_ready": precision_plan.get("score_ready"),
            "trigger_ready": precision_plan.get("trigger_ready"),
            "order_flow_ready": precision_plan.get("order_flow_ready"),
            "order_flow_strength": precision_plan.get("order_flow_strength"),
            "trigger_progress": precision_plan.get("trigger_progress"),
            "trigger_confidence": precision_plan.get("trigger_confidence"),
            "reasons": precision_plan.get("trigger_reasons") or None,
        }
        execution_playbook.append(trigger_step)

    entry_thresholds_meta["precision_score_threshold"] = float(precision_threshold_value)
    entry_thresholds_meta["precision_timeouts"] = {
        "ready": precision_timeouts.get("ready"),
        "arming": precision_timeouts.get("arming"),
    }

    # 9) Session override + mentés: signal.json
    if latency_flags:
        for flag in latency_flags:
            msg = f"Diagnosztika: {flag}"
            if msg not in reasons:
                reasons.append(msg)

    if intervention_summary and intervention_summary.get("irs", 0) >= 40:
        highlight = f"Crypto Watch: IRS {intervention_summary['irs']} ({intervention_summary['band']})"
        if highlight not in reasons:
            reasons.insert(0, highlight)
    if asset == "BTCUSD" and intervention_band:
        if decision == "buy" and intervention_band in {"HIGH", "EXTREME"}:
            block_msg = (
                f"Crypto Watch: IRS {intervention_summary['irs']} ({intervention_band}) → BTCUSD long belépés tiltva"
                if intervention_summary
                else "Crypto Watch: BTCUSD long belépés tiltva"
            )
            if block_msg not in reasons:
                reasons.insert(0, block_msg)
            decision = "no entry"
            entry = sl = tp1 = tp2 = rr = None
            if "intervention_watch" not in missing:
                missing.append("intervention_watch")
        elif decision == "sell" and intervention_band == "EXTREME":
            note_short = "Crypto Watch: új short csak 0.5×ATR5m pullback + RR≥2.5 után"
            if note_short not in reasons:
                reasons.append(note_short)
        if intervention_band == "EXTREME":
            guard_note = "Order guard: market belépés tiltva, limit ≥0.5×ATR5m bufferrel"
            if guard_note not in reasons:
                reasons.append(guard_note)

    if not session_ok_flag:
        status_note = session_meta.get("status_note") or "Session zárva"
        if status_note not in reasons:
            reasons.insert(0, status_note)
        decision = "market closed"
        P = 0
        entry = sl = tp1 = tp2 = rr = None
        mode = "session_closed"
        required_list = ["session"]
        if "session" not in missing:
            missing.append("session")

    missing = list(dict.fromkeys(missing))
    structure_flag = "range"
    if bos5m_short:
        structure_flag = "bos_down"
    elif bos5m_long:
        structure_flag = "bos_up"

    position_note, exit_signal = derive_position_management_note(
        asset,
        session_meta,
        regime_ok,
        effective_bias,
        structure_flag,
        atr1h,
        anchor_bias,
        anchor_timestamp,
        anchor_record,
        P,
        (rel_atr if (rel_atr is not None and not np.isnan(rel_atr)) else None),
        (float(atr5) if atr5 is not None and np.isfinite(float(atr5)) else None),
        price_for_calc,
        invalid_level_buy,
        invalid_level_sell,
        (float(invalid_buffer) if invalid_buffer is not None else None),
        decision,
        sentiment_signal,
    )
    if position_note and position_note not in reasons:
        reasons.append(position_note)

    structure_flip_flag_value = 0.0
    if anchor_bias == "long" and structure_flag == "bos_down":
        structure_flip_flag_value = 1.0
    elif anchor_bias == "short" and structure_flag == "bos_up":
        structure_flip_flag_value = 1.0
    elif anchor_bias is None:
        if effective_bias == "long" and structure_flag == "bos_down":
            structure_flip_flag_value = 1.0
        elif effective_bias == "short" and structure_flag == "bos_up":
            structure_flip_flag_value = 1.0

    precision_score_value = 0.0
    if precision_plan:
        precision_score_value = safe_float(precision_plan.get("score")) or 0.0

    precision_trigger_confidence = 0.0
    precision_trigger_ready_value = 0.0
    precision_trigger_arming_value = 0.0
    precision_trigger_fire_value = 0.0
    if precision_trigger_state:
        state_norm = precision_trigger_state.lower()
        if state_norm in {"ready", "arming", "fire"}:
            precision_trigger_ready_value = 1.0
        if state_norm == "arming":
            precision_trigger_arming_value = 1.0
        if state_norm == "fire":
            precision_trigger_fire_value = 1.0
    if precision_plan:
        precision_trigger_confidence = safe_float(
            precision_plan.get("trigger_confidence")
        ) or 0.0
    precision_order_flow_value = 1.0 if precision_flow_ready else 0.0

    momentum_trail_activation = 0.0
    momentum_trail_lock = 0.0
    momentum_trail_price = 0.0
    if momentum_trailing_plan:
        activation = safe_float(momentum_trailing_plan.get("activation_rr"))
        lock_ratio = safe_float(momentum_trailing_plan.get("lock_ratio"))
        trail_price = safe_float(momentum_trailing_plan.get("trail_price"))
        if activation is not None and np.isfinite(activation):
            momentum_trail_activation = float(activation)
        if lock_ratio is not None and np.isfinite(lock_ratio):
            momentum_trail_lock = float(lock_ratio)
        if trail_price is not None and np.isfinite(trail_price):
            momentum_trail_price = float(trail_price)

    rel_atr_value = 0.0
    if rel_atr is not None:
        try:
            if not np.isnan(rel_atr):
                rel_atr_value = float(rel_atr)
        except TypeError:
            pass

    news_sentiment_value = 0.0
    news_severity_value = 0.0
    if sentiment_signal and ENABLE_SENTIMENT_PROBABILITY:
        news_sentiment_value = sentiment_signal.score or 0.0
        news_severity_value = sentiment_signal.effective_severity or 0.0

    ml_features = {
        "p_score": P,
        "rel_atr": rel_atr_value,
        "ema21_slope": regime_slope_signed,
        "bias_long": 1.0 if effective_bias == "long" else 0.0,
        "bias_short": 1.0 if effective_bias == "short" else 0.0,
        "momentum_vol_ratio": momentum_vol_ratio or 0.0,
        "order_flow_status": order_flow_metrics.get("status"),
        "order_flow_imbalance": order_flow_metrics.get("imbalance"),
        "order_flow_pressure": order_flow_metrics.get("pressure"),
        "order_flow_aggressor": order_flow_metrics.get("aggressor_ratio"),
        "news_sentiment": news_sentiment_value,
        "news_event_severity": news_severity_value,
        "realtime_confidence": realtime_confidence,
        "volatility_ratio": float(overlay_ratio)
        if (overlay_ratio is not None and np.isfinite(float(overlay_ratio)))
        else 0.0,
        "volatility_regime_flag": 1.0
        if overlay_regime in {"implied_elevated", "implied_extreme"}
        else 0.0,
        "precision_score": precision_score_value,
        "precision_trigger_ready": precision_trigger_ready_value,
        "precision_trigger_arming": precision_trigger_arming_value,
        "precision_trigger_fire": precision_trigger_fire_value,
        "precision_trigger_confidence": precision_trigger_confidence,
        "precision_order_flow_ready": precision_order_flow_value,
        "momentum_trail_activation_rr": momentum_trail_activation,
        "momentum_trail_lock_ratio": momentum_trail_lock,
        "momentum_trail_price": momentum_trail_price,
        "structure_flip_flag": structure_flip_flag_value,
    }

    asset_key = asset.upper()
    placeholder_meta = ML_PROBABILITY_PLACEHOLDER_INFO.get(asset_key)
    if (
        ML_PROBABILITY_ACTIVE
        and asset_key in ML_PROBABILITY_PLACEHOLDER_BLOCKLIST
        and placeholder_meta
    ):
        placeholder_payload = {
            "source": "sklearn",
            "status": "placeholder_model",
            "unavailable_reason": "placeholder_model",
        }
        detail = placeholder_meta.get("detail")
        if detail:
            placeholder_payload["detail"] = detail
        model_path = placeholder_meta.get("path")
        if model_path:
            placeholder_payload["model_path"] = model_path
        placeholder_payload.update(
            {
                key: value
                for key, value in placeholder_meta.items()
                if key in {"size_bytes", "size_warning"}
            }
        )
        ml_prediction = ProbabilityPrediction(metadata=placeholder_payload)
    elif ML_PROBABILITY_ACTIVE:
        ml_prediction = predict_signal_probability(asset, ml_features)
    else:
        ml_prediction = ProbabilityPrediction(
            metadata={
                "source": "sklearn",
                "status": "disabled",
                "unavailable_reason": "ml_scoring_disabled",
            }
        )
    ml_probability = ml_prediction.probability
    ml_probability_raw = ml_prediction.raw_probability
    ml_threshold = ml_prediction.threshold
    probability_metadata = ensure_probability_metadata(ml_prediction.metadata)
    probability_metadata = _apply_probability_stack_gap_guard(
        asset,
        probability_metadata,
        now=analysis_now,
        base_dir=Path(PUBLIC_DIR),
    )
    probability_source = probability_metadata.get("source")
    fallback_meta: Optional[Dict[str, Any]] = None
    meta_reason = probability_metadata.get("unavailable_reason")
    raw_fallback = probability_metadata.get("fallback")
    if isinstance(raw_fallback, dict):
        fallback_meta = raw_fallback
    if fallback_meta and probability_source == "fallback":
        fallback_reason = fallback_meta.get("reason") or meta_reason
        reason_map = {
            "model_missing": "ML fallback: modell hiányzik — heurisztikus pontozás aktív",
            "sklearn_missing": "ML fallback: scikit-learn hiányzik — heurisztikus pontozás aktív",
            "model_type_mismatch": "ML fallback: modell típus inkompatibilis",
            "feature_snapshot_missing": "ML fallback: hiányzó feature snapshot",
        }
        if isinstance(fallback_reason, str):
            reason_text = reason_map.get(
                fallback_reason,
                f"ML fallback aktív ({fallback_reason})",
            )
        else:
            reason_text = "ML fallback aktív"
        if reason_text not in reasons:
            reasons.append(reason_text)

    ml_gap_reasons = {"feature_snapshot_missing", "model_missing"}
    if asset.upper() == "BTCUSD" and (
        probability_metadata.get("unavailable_reason") in ml_gap_reasons
        or (fallback_meta and fallback_meta.get("reason") in ml_gap_reasons)
    ):
        gap_reason = "ML modell / feature snapshot hiányzik — belépés tiltva"
        if gap_reason not in reasons:
            reasons.insert(0, gap_reason)
        if "ml_probability" not in missing:
            missing.append("ml_probability")
        decision = "no entry"
        entry = sl = tp1 = tp2 = rr = None
        execution_playbook = []

    combined_probability = P / 100.0
    if ml_probability is not None:
        combined_probability = min(
            1.0, max(0.0, 0.6 * (P / 100.0) + 0.4 * ml_probability)
        )

    ml_confidence_block = False
    if (
        decision in ("buy", "sell")
        and ml_probability is not None
        and ml_threshold is not None
        and ml_probability < ml_threshold
    ):
        ml_confidence_block = True
        if "ml_confidence" not in missing:
            missing.append("ml_confidence")
        reason_ml = (
            f"ML valószínűség {ml_probability:.1%} a küszöb ({ml_threshold:.1%}) alatt"
        )
        if reason_ml not in reasons:
            reasons.append(reason_ml)
        decision = "no entry"
        entry = sl = tp1 = tp2 = rr = None
        execution_playbook = []
        momentum_trailing_plan = None

    missing = list(dict.fromkeys(missing))
    log_entry_gate_decision(asset, last5_closed_ts, missing)

    analysis_timestamp = nowiso()
    probability_percent = int(max(0, min(100, round(combined_probability * 100))))
  
    gate_mode_value = gates_mode_override if gates_mode_override and mode == "core" else mode

    gates_payload: Dict[str, Any] = {
        "mode": gate_mode_value,
        "required": required_list,
        "missing": missing,
    }
    entry_required_snapshot = list(gates_payload.get("required") or [])
    entry_missing_snapshot = list(gates_payload.get("missing") or [])
    if intraday_relaxed_guards:
        gates_payload["relaxed_guards"] = list(intraday_relaxed_guards)
        gates_payload["relax_size_scale"] = intraday_relax_scale
        gates_payload["relax_mode"] = "intraday"
    if session_meta:
        session_window_status = {
            "entry_open": bool(session_meta.get("entry_open")),
            "monitor_window": bool(session_meta.get("within_monitor_window")),
            "status": session_meta.get("status"),
        }
        status_note = session_meta.get("status_note")
        if isinstance(status_note, str) and status_note.strip():
            session_window_status["status_note"] = status_note.strip()
        summary_bits = []
        summary_bits.append("Entry ablak NYITVA" if session_meta.get("entry_open") else "Entry ablak ZÁRVA")
        summary_bits.append("Megfigyelés aktív" if session_meta.get("within_monitor_window") else "Megfigyelési ablakon kívül")
        session_window_status["összefoglaló"] = ", ".join(summary_bits)
        gates_payload["session_window_status"] = session_window_status
    if intraday_bias_gate_meta:
        gates_payload["intraday_bias"] = intraday_bias_gate_meta

    effective_thresholds = {
        "asset": asset,
        "entry_profile": asset_entry_profile,
        "p_score_min": float(p_score_min_local) if p_score_min_local is not None else None,
        "spread_max_atr_pct": float(spread_limit) if spread_limit is not None else None,
        "tp_net_min": float(tp_net_threshold) if tp_net_threshold is not None else None,
        "rr_required": float(rr_required_effective) if rr_required_effective is not None else None,
        "tp_min_profit_pct": float(tp_min_profit_pct) if tp_min_profit_pct is not None else None,
        "min_stoploss_pct": float(min_stoploss_pct) if min_stoploss_pct is not None else None,
        "atr_abs_min_used": float(atr_abs_min) if atr_abs_min is not None else None,
        "atr_low_threshold": float(atr_threshold) if atr_threshold is not None else None,
        "atr_overlay_min": float(atr_overlay_min) if atr_overlay_min is not None else None,
        "bos_lookback": int(bos_lookback) if bos_lookback is not None else None,
        "fib_tolerance": float(fib_tol) if fib_tol is not None else None,
        "position_scale_floor": float(position_scale_floor)
        if position_scale_floor is not None
        else None,
        "staleness_max_age_seconds": int(staleness_max_age) if staleness_max_age is not None else None,
        "session_window": session_window_payload,
    }


    decision_obj = {
        "asset": asset,
        "ok": True,
        "retrieved_at_utc": analysis_timestamp,
        "source": "Twelve Data (lokális JSON)",
        "spot": {
            "price": display_spot,
            "utc": spot_utc,
            "retrieved_at_utc": spot_retrieved,
            "source": spot_source,
            "realtime_override": realtime_used,
            "fallback_used": spot_fallback_used,
            "confidence": realtime_confidence,
            "avg_latency_profile": avg_delay if avg_delay else None,
            "realtime_stats": realtime_stats if realtime_stats else None,
        },
        "signal": decision,
        "probability": probability_percent,
        "probability_raw": int(P),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": (round(rr, 2) if rr else None),
        "leverage": lev,
        "position_size_scale": position_size_scale,
        "range_time_stop": range_time_stop_plan,
        "funding_filter": funding_dir_filter,
        "funding_rate": funding_value,
        "stale_timeframes": dict(stale_timeframes),
        "biases": {
            "raw_4h": raw_bias4h,
            "raw_1h": raw_bias1h,
            "adjusted_4h": bias4h,
            "adjusted_1h": bias1h,
        },
        "gates": gates_payload,
        "gate_skips": list(gate_skips),
        "session_info": session_meta,
        "momentum_diagnostics": momentum_diagnostics,
        "diagnostics": diagnostics_payload(tf_meta, source_files, latency_flags),
        "reasons": (reasons + ([f"missing: {', '.join(missing)}"] if missing else []))
        or ["no signal"],
        "realtime_transport": realtime_transport,
    }
    decision_obj["data_integrity_diagnostics"] = {
        "core_data_ok": bool(core_data_ok),
        "precision_data_ok": bool(precision_data_ok),
        "precision_disabled_due_to_data_gap": bool(precision_disabled_due_to_data_gap),
        "core_required_sources": ["spot", "k5m"],
        "precision_required_sources": ["spot", "k5m", "k1m"],
        "spot": tf_meta.get("spot"),
        "k5m": tf_meta.get("k5m"),
        "k1m": tf_meta.get("k1m"),
    }
    decision_obj["profile_resolution"] = profile_resolution
    decision_obj["effective_thresholds"] = effective_thresholds
    if news_lockout_active:
        decision_obj["news_lockout_active"] = True
        if news_reason:
            decision_obj["news_lockout_reason"] = news_reason
    if probability_source:
        decision_obj["probability_model_source"] = probability_source
    if fallback_meta:
        decision_obj["probability_fallback"] = fallback_meta
    decision_obj["probability_model"] = (
        float(ml_probability) if ml_probability is not None else None
    )
    decision_obj["probability_model_raw"] = (
        float(ml_probability_raw) if ml_probability_raw is not None else None
    )
    decision_obj["probability_calibrated"] = (
        float(ml_probability) if ml_probability is not None else None
    )
    decision_obj["probability_threshold"] = (
        float(ml_threshold) if ml_threshold is not None else None
    )
    decision_obj["probability_stack"] = probability_metadata
    if ml_confidence_block:
        decision_obj["ml_confidence_blocked"] = True
    decision_obj["ema21_slope_1h"] = regime_slope_signed
    decision_obj["ema21_slope_threshold"] = EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT)
    decision_obj["ema21_relation_1h"] = ema21_relation
    decision_obj["last_swing_high_1h"] = float(move_hi) if move_hi is not None else None
    decision_obj["last_swing_low_1h"] = float(move_lo) if move_lo is not None else None
    decision_obj["last_close_1h"] = float(last_close_1h) if np.isfinite(last_close_1h) else None
    decision_obj["bos_5m_dir"] = structure_flag
    decision_obj["atr1h"] = atr1h
    decision_obj["momentum_volume_ratio"] = momentum_vol_ratio
    decision_obj["momentum_liquidity_ok"] = momentum_liquidity_ok
    decision_obj["dynamic_tp_profile"] = dynamic_tp_profile
    decision_obj["order_flow_metrics"] = order_flow_metrics
    decision_obj["intraday_profile"] = intraday_profile
    if asset == "BTCUSD" and btc_level_checks_state:
        entry_thresholds_meta["btc_level_checks"] = btc_level_checks_state
    entry_thresholds_meta.setdefault("atr_threshold_effective", atr_threshold)
    entry_thresholds_meta.setdefault("p_score_min_effective", p_score_min_local)
    decision_obj["entry_thresholds"] = entry_thresholds_meta
    if entry_gate_context_hu:
        decision_obj["entry_gate_context_hu"] = entry_gate_context_hu
    if volatility_overlay:
        decision_obj["volatility_overlay"] = volatility_overlay
    if tick_order_flow:
        decision_obj["order_flow_tick_snapshot"] = tick_order_flow
    if precision_plan:
        decision_obj["precision_plan"] = precision_plan
    if sentiment_signal:
        sentiment_payload = {
            "score": sentiment_signal.score,
            "bias": sentiment_signal.bias,
        }
        if sentiment_signal.headline:
            sentiment_payload["headline"] = sentiment_signal.headline
        if sentiment_signal.expires_at:
            sentiment_payload["expires_at"] = sentiment_signal.expires_at.isoformat()
        sentiment_payload["severity"] = sentiment_signal.effective_severity
        if sentiment_signal.category:
            sentiment_payload["category"] = sentiment_signal.category
        if sentiment_signal.source:
            sentiment_payload["source"] = str(sentiment_signal.source)
        if sentiment_applied_points is not None:
            sentiment_payload["applied_points"] = sentiment_applied_points
        if sentiment_normalized is not None:
            sentiment_payload["normalized_score"] = sentiment_normalized
        decision_obj["news_sentiment"] = sentiment_payload
    if momentum_trailing_plan:
        decision_obj["momentum_trailing_plan"] = momentum_trailing_plan
    if execution_playbook:
        decision_obj["execution_playbook"] = execution_playbook
    decision_obj["invalid_levels"] = {
        "buy": invalid_level_buy,
        "sell": invalid_level_sell,
    }
    decision_obj["invalid_buffer_abs"] = float(invalid_buffer) if invalid_buffer is not None else None
    if position_note:
        decision_obj["position_management"] = position_note
    if exit_signal:
        decision_obj["position_exit_signal"] = exit_signal

    sentiment_exit_summary: Optional[Dict[str, Any]] = None
    if exit_signal and exit_signal.get("category") == "sentiment_risk":
        sentiment_exit_summary = {
            "state": exit_signal.get("state"),
            "severity_label": exit_signal.get("severity"),
            "severity_score": exit_signal.get("sentiment_severity"),
            "score": exit_signal.get("sentiment_score"),
            "bias": exit_signal.get("sentiment_bias"),
            "direction": exit_signal.get("direction"),
            "category": exit_signal.get("sentiment_category")
            or exit_signal.get("category"),
            "triggered_at": exit_signal.get("triggered_at"),
        }
        if sentiment_signal and sentiment_signal.headline:
            sentiment_exit_summary["headline"] = sentiment_signal.headline
        else:
            headlines = exit_signal.get("headlines") or []
            if headlines:
                sentiment_exit_summary["headline"] = headlines[0]
        sentiment_exit_summary = {
            key: value
            for key, value in sentiment_exit_summary.items()
            if value is not None
        }
        decision_obj["sentiment_exit_summary"] = sentiment_exit_summary
        score_repr = sentiment_exit_summary.get("score")
        severity_repr = sentiment_exit_summary.get("severity_score")
        try:
            score_text = f"{float(score_repr):+.2f}" if score_repr is not None else "n/a"
        except (TypeError, ValueError):
            score_text = "n/a"
        try:
            severity_text = (
                f"{float(severity_repr):.2f}" if severity_repr is not None else "n/a"
            )
        except (TypeError, ValueError):
            severity_text = "n/a"
        LOGGER.warning(
            "[sentiment_exit] %s state=%s score=%s severity=%s bias=%s",
            asset,
            sentiment_exit_summary.get("state") or "unknown",
            score_text,
            severity_text,
            sentiment_exit_summary.get("bias") or "n/a",
        )

    action_plan = build_action_plan(
        asset=asset,
        decision=decision,
        session_meta=session_meta,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        leverage=lev,
        probability=probability_percent,
        precision_plan=precision_plan,
        execution_playbook=execution_playbook,
        position_note=position_note,
        exit_signal=exit_signal,
        missing=decision_obj.get("gates", {}).get("missing", []),
        reasons=decision_obj.get("reasons"),
        last_computed_risk=last_computed_risk,
        momentum_trailing_plan=momentum_trailing_plan,
        intraday_profile=intraday_profile,
        btc_profile=(
            btc_profile_name
            if asset == "BTCUSD" and btc_profile_name
            else (_btc_active_profile() if asset == "BTCUSD" else None)
        ),
        entry_thresholds=decision_obj.get("entry_thresholds"),
        entry_thresholds_meta=entry_thresholds_meta,
        entry_gate_context_hu=entry_gate_context_hu,
    )
    entry_blockers_snapshot = (
        list(action_plan.get("blockers") or []) if isinstance(action_plan, dict) else []
    )
    if action_plan:
        decision_obj["action_plan"] = action_plan

    entry_p_score_used: Optional[int] = None
    try:
        entry_p_score_used = int(round(P)) if P is not None else None
    except Exception:
        entry_p_score_used = None
    position_p_score_used = (
        anchor_prev_p if anchor_prev_p is not None else entry_p_score_used
    )
    decision_obj["entry_diagnostics"] = {
        "required": entry_required_snapshot,
        "missing": entry_missing_snapshot,
        "blockers": entry_blockers_snapshot,
        "p_score_used": entry_p_score_used,
    }
    if precision_override_active:
        decision_obj["entry_diagnostics"]["precision_override"] = True
    position_diagnostics = {
        "state": None,
        "severity": None,
        "reasons": [],
        "p_score_used": position_p_score_used,
    }
    if exit_signal:
        position_diagnostics.update(
            {
                "state": exit_signal.get("state") or exit_signal.get("action"),
                "severity": exit_signal.get("severity"),
                "reasons": list(exit_signal.get("reasons") or []),
            }
        )
    decision_obj["position_diagnostics"] = position_diagnostics

    snapshot_metadata = {
        "analysis_timestamp": analysis_timestamp,
        "signal": decision,
        "mode": mode,
        "probability_calibrated": float(ml_probability)
        if ml_probability is not None
        else None,
        "probability_threshold": float(ml_threshold)
        if ml_threshold is not None
        else None,
    }
    if probability_source:
        snapshot_metadata["probability_model_source"] = probability_source
    log_feature_snapshot(asset, ml_features, metadata=snapshot_metadata)
    active_position_meta = {
        "ema21_slope_abs": regime_val,
        "ema21_slope_signed": regime_slope_signed,
        "ema21_slope_threshold": EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT),
        "ema21_relation": ema21_relation,
        "structure_5m": structure_flag,
        "stale_timeframes": dict(stale_timeframes),
        "atr1h": atr1h,
        "atr1h_trail_floor": float(invalid_buffer) if invalid_buffer is not None else None,
        "atr5": float(atr5 or 0.0) if np.isfinite(float(atr5 or 0.0)) else None,
        "atr5_rel": rel_atr if (rel_atr is not None and np.isfinite(rel_atr)) else None,
        "atr5_threshold": atr_threshold,
        "bos1h_long": bool(bos1h_long),
        "bos1h_short": bool(bos1h_short),
        "regime_ok": bool(regime_ok),
        "effective_bias": effective_bias,
        "last_close_1h": float(last_close_1h) if np.isfinite(last_close_1h) else None,
        "last_swing_high_1h": float(move_hi) if move_hi is not None else None,
        "last_swing_low_1h": float(move_lo) if move_lo is not None else None,
        "invalid_level_sell": invalid_level_sell,
        "invalid_level_buy": invalid_level_buy,
        "invalid_buffer_abs": float(invalid_buffer) if invalid_buffer is not None else None,
        "p_score": int(P),
        "probability_calibrated": float(ml_probability)
        if ml_probability is not None
        else None,
        "probability_threshold": float(ml_threshold)
        if ml_threshold is not None
        else None,
        "probability_model_raw": float(ml_probability_raw)
        if ml_probability_raw is not None
        else None,
        "probability_model_source": probability_source,
        "tp_profile": {"tp1_r": current_tp1_mult, "tp2_r": current_tp2_mult},
        "realtime_confidence": realtime_confidence,
        "avg_latency_profile": avg_delay if avg_delay else None,
    }
    if fallback_meta:
        active_position_meta["probability_fallback"] = fallback_meta
    if realtime_stats:
        active_position_meta["realtime_stats"] = realtime_stats
    if anchor_bias:
        active_position_meta["anchor_side"] = anchor_bias
    if anchor_price_state is not None:
        active_position_meta["anchor_price"] = anchor_price_state
    if anchor_timestamp:
        active_position_meta["anchor_timestamp"] = anchor_timestamp
    if anchor_prev_p is not None:
        active_position_meta["previous_p_score"] = anchor_prev_p
    if anchor_drift_state:
        active_position_meta["anchor_drift_state"] = anchor_drift_state
    if anchor_drift_score is not None:
        active_position_meta["anchor_drift_score"] = anchor_drift_score
    if momentum_trailing_plan:
        active_position_meta["momentum_trailing_plan"] = momentum_trailing_plan
    if precision_plan:
        active_position_meta["precision_plan"] = precision_plan
        window_payload: Optional[List[float]] = None
        window_raw = precision_plan.get("entry_window")
        if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
            try:
                window_payload = [float(window_raw[0]), float(window_raw[1])]
            except (TypeError, ValueError):
                window_payload = None
        trigger_meta = {
            "state": precision_plan.get("trigger_state"),
            "ready_ts": precision_plan.get("ready_ts"),
            "entry_window": window_payload,
            "trigger_levels": precision_plan.get("trigger_levels"),
            "score": precision_plan.get("score"),
            "score_threshold": precision_plan.get(
                "score_threshold", PRECISION_SCORE_THRESHOLD_DEFAULT
            ),
            "score_ready": precision_plan.get("score_ready"),
            "trigger_ready": precision_plan.get("trigger_ready"),
            "order_flow_ready": precision_plan.get("order_flow_ready"),
            "order_flow_strength": precision_plan.get("order_flow_strength"),
            "trigger_progress": precision_plan.get("trigger_progress"),
            "trigger_confidence": precision_plan.get("trigger_confidence"),
            "reasons": precision_plan.get("trigger_reasons"),
        }
        active_position_meta["precision_trigger"] = trigger_meta
    if action_plan:
        active_position_meta["action_plan"] = action_plan
    if execution_playbook:
        active_position_meta["execution_playbook"] = execution_playbook
    if exit_signal:
        active_position_meta["exit_signal"] = exit_signal
    decision_obj["active_position_meta"] = active_position_meta
    if intervention_summary:
        decision_obj["intervention_watch"] = intervention_summary

    decision_obj = apply_signal_stability_layer(
        asset,
        decision_obj,
        decision=decision,
        action_plan=action_plan,
        exit_signal=exit_signal,
        gates_missing=decision_obj.get("gates", {}).get("missing", []),
        analysis_timestamp=analysis_timestamp,
        outdir=Path(outdir),
    )

    intent = decision_obj.get("intent")
    if intent in {"manage_position", "hard_exit"}:
        gates_payload["missing"] = []
        if decision_obj.get("gates") is not gates_payload:
            try:
                decision_obj.setdefault("gates", {})["missing"] = []
            except Exception:
                pass
        if action_plan:
            action_plan["blockers"] = []
            action_plan["blockers_raw"] = []

    anchor_metrics_payload = {
        "p_score": P,
        "atr5": float(atr5) if atr5 is not None and np.isfinite(float(atr5)) else None,
        "atr1h": atr1h,
        "rel_atr": rel_atr if (rel_atr is not None and not np.isnan(rel_atr)) else None,
        "analysis_timestamp": decision_obj["retrieved_at_utc"],
        "realtime_confidence": realtime_confidence,
        "realtime_transport": realtime_transport,
        "dynamic_tp_regime": dynamic_tp_profile.get("regime"),
        "order_flow_imbalance": order_flow_metrics.get("imbalance"),
        "order_flow_pressure": order_flow_metrics.get("pressure"),
        "current_price": price_for_calc,
        "spot_price": display_spot,
    }

    if precision_plan:
        anchor_metrics_payload["precision_score"] = precision_plan.get("score")
        anchor_metrics_payload["precision_plan"] = precision_plan
        window_raw = precision_plan.get("entry_window")
        window_payload: Optional[List[float]] = None
        if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
            try:
                window_payload = [float(window_raw[0]), float(window_raw[1])]
            except (TypeError, ValueError):
                window_payload = None
        anchor_metrics_payload["precision_trigger"] = {
            "state": precision_plan.get("trigger_state"),
            "ready_ts": precision_plan.get("ready_ts"),
            "entry_window": window_payload,
            "trigger_levels": precision_plan.get("trigger_levels"),
            "score_threshold": precision_plan.get(
                "score_threshold", PRECISION_SCORE_THRESHOLD_DEFAULT
            ),
            "score_ready": precision_plan.get("score_ready"),
            "trigger_ready": precision_plan.get("trigger_ready"),
            "order_flow_ready": precision_plan.get("order_flow_ready"),
            "order_flow_strength": precision_plan.get("order_flow_strength"),
            "trigger_progress": precision_plan.get("trigger_progress"),
            "trigger_confidence": precision_plan.get("trigger_confidence"),
            "reasons": precision_plan.get("trigger_reasons"),
        }
    if volatility_overlay:
        anchor_metrics_payload["volatility_overlay"] = volatility_overlay
    if tick_order_flow:
        anchor_metrics_payload["order_flow_tick"] = tick_order_flow
    if anchor_drift_state:
        anchor_metrics_payload["anchor_drift_state"] = anchor_drift_state
    if anchor_drift_score is not None:
        anchor_metrics_payload["anchor_drift_score"] = anchor_drift_score
    if sentiment_signal:
        anchor_metrics_payload["news_sentiment"] = sentiment_signal.score
        anchor_metrics_payload["news_sentiment_severity"] = sentiment_signal.effective_severity

    global ANCHOR_STATE_CACHE
    if decision in ("buy", "sell"):
        anchor_price = entry or spot_price
        if anchor_price is None and np.isfinite(last_close_1h):
            anchor_price = float(last_close_1h)
        anchor_payload = dict(anchor_metrics_payload)
        anchor_payload.update(
            {
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "rr_target": rr,
                "initial_risk_abs": last_computed_risk,
                "dynamic_tp_profile": dynamic_tp_profile,
            }
        )
        try:
            ANCHOR_STATE_CACHE = record_anchor(
                asset,
                decision,
                price=anchor_price,
                timestamp=decision_obj["retrieved_at_utc"],
                extras=anchor_payload,
            )
        except Exception:
            # Anchor frissítés hibája ne állítsa meg az elemzést.
            pass
    else:
        try:
            ANCHOR_STATE_CACHE = update_anchor_metrics(asset, anchor_metrics_payload)
        except Exception:
            pass
    _log_gate_summary(asset, decision_obj)
    save_json(os.path.join(outdir, "signal.json"), decision_obj)
    try:
        record_signal_event(asset, decision_obj)
    except Exception:
        # Journaling issues should not break signal generation.
        pass
    return decision_obj
# ------------------------------- főfolyamat ------------------------------------


def _determine_analysis_workers(asset_count: int) -> int:
    if asset_count <= 1:
        return 1

    default_workers = min(asset_count, max(1, os.cpu_count() or 1))
    env_value = os.getenv("ANALYSIS_MAX_WORKERS")
    if env_value:
        try:
            configured = int(env_value)
        except ValueError:
            LOGGER.warning("Ignoring invalid ANALYSIS_MAX_WORKERS value: %r", env_value)
        else:
            if configured > 0:
                default_workers = min(asset_count, configured)
            else:
                LOGGER.warning(
                    "ANALYSIS_MAX_WORKERS must be positive; received %r", env_value
                )
    return max(1, default_workers)


def _analyze_asset_guard(asset: str) -> Tuple[str, Dict[str, Any]]:
    try:
        result = analyze(asset)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Analysis failed for asset %s", asset)
        failure_result: Dict[str, Any] = {"asset": asset, "ok": False, "error": str(exc)}
        try:
            failure_payload = build_analysis_error_signal(asset, exc)
            save_json(os.path.join(PUBLIC_DIR, asset, "signal.json"), failure_payload)
        except Exception:  # pragma: no cover - ensure pipeline resiliency
            LOGGER.exception("Failed to persist analysis error placeholder for %s", asset)
        return asset, failure_result

    # Biztonsági háló: minden jelzés tartalmazzon valószínűségi metadatát.
    if isinstance(result, dict):
        prob_meta = ensure_probability_metadata(result.get("probability_stack"))
        if prob_meta != result.get("probability_stack"):
            LOGGER.warning(
                "probability_stack_missing",
                extra={"asset": asset, "reason": "missing_or_null"},
            )
        result["probability_stack"] = prob_meta
        result.setdefault("probability_model_source", prob_meta.get("source"))

    return asset, result


def _backfill_signal_probability_metadata(
    asset: str, base_dir: Optional[Path] = None
) -> None:
    """Ensure persisted signal files always contain probability metadata.

    A védőháló akkor is pótolja a hiányzó `probability_stack` blokkot, ha
    valamilyen korábbi pipeline futásból maradt meg a jelzés JSON, vagy a
    korábbi elemzés egy része kihagyta a mezőt. A visszatöltés nem módosít
    más kulcsokat, csak beállítja az alapértelmezett forrást és a
    ``probability_model_source`` mezőt, hogy a verifikáció sikerüljön.
    """

    signal_path = Path(base_dir or PUBLIC_DIR) / asset.upper() / "signal.json"
    if not signal_path.exists():
        return

    try:
        payload = load_json(signal_path)
    except Exception as exc:  # pragma: no cover - defensív visszatöltés
        LOGGER.warning("signal_backfill_load_failed", extra={"asset": asset, "err": str(exc)})
        return

    if not isinstance(payload, dict):
        return

    prob_meta = ensure_probability_metadata(payload.get("probability_stack"))
    changed = prob_meta != payload.get("probability_stack")
    if changed:
        LOGGER.warning(
            "signal_probability_stack_missing",
            extra={"asset": asset, "action": "backfilled"},
        )
    payload["probability_stack"] = prob_meta

    if prob_meta.get("source") and not payload.get("probability_model_source"):
        payload["probability_model_source"] = prob_meta.get("source")
        changed = True

    if changed:
        save_json(signal_path, payload)


def _trading_artifact_timestamp(path: Path) -> Optional[datetime]:
    try:
        payload = load_json(str(path))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Nem sikerült beolvasni a trading artefaktot (%s): %s", path, exc)
        payload = None

    if isinstance(payload, dict):
        for key in ("generated_at_utc", "updated_at_utc", "completed_utc"):
            ts = _parse_utc_timestamp(payload.get(key))
            if ts:
                return ts

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _ensure_trading_preconditions(analysis_started_at: datetime) -> Optional[datetime]:
    """Abort the run if the trading artefact is missing or stale."""

    trading_status_path = Path(PUBLIC_DIR) / "pipeline" / "trading_status.json"
    trading_ts: Optional[datetime] = None
    if not trading_status_path.exists():
        public_root = Path(PUBLIC_DIR).resolve()
        default_public_root = Path("public").resolve()
        if public_root != default_public_root:
            trading_status_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(
                str(trading_status_path),
                {"generated_at_utc": analysis_started_at.isoformat()},
            )
            trading_ts = analysis_started_at
            LOGGER.warning(
                "Missing trading artefact (%s) – generated placeholder for non-default PUBLIC_DIR",
                trading_status_path,
            )
        else:
            raise SystemExit(
                "Trading artefakt hiányzik (public/pipeline/trading_status.json) – futtasd le a Trading.py lépést."
            )

    if trading_ts is None:
        trading_ts = _trading_artifact_timestamp(trading_status_path)
    if trading_ts is None:
        raise SystemExit(
            "Trading artefakt időbélyeg hiányzik vagy sérült – futtasd újra a trading lépést."
        )

    now_safe = analysis_started_at
    if trading_ts > now_safe:
        raise SystemExit(
            "Trading időbélyeg későbbi, mint az analysis start – ellenőrizd a pipeline sorrendet."
        )

    age_seconds = (now_safe - trading_ts).total_seconds()
    try:
        lag_budget = float(PIPELINE_MAX_LAG_SECONDS) if PIPELINE_MAX_LAG_SECONDS is not None else None
    except (TypeError, ValueError):
        lag_budget = None

    if lag_budget is not None and age_seconds > lag_budget:
        raise SystemExit(
            "Trading artefakt túl régi (%ds) – futtasd le újra a Trading.py lépést." % int(age_seconds)
        )

    LOGGER.info("Trading artefakt időbélyeg: %s (késés %.1fs)", trading_ts.isoformat(), age_seconds)
    return trading_ts


def main():
    run_id = str(uuid4())
    position_tracker.set_audit_context(source="analysis", run_id=run_id)
    analysis_started_at = datetime.now(timezone.utc)
    pipeline_log_path = None
    if get_pipeline_log_path:
        try:
            pipeline_log_path = get_pipeline_log_path()
        except Exception:
            pipeline_log_path = None
    if pipeline_log_path:
        ensure_json_file_handler(
            LOGGER,
            pipeline_log_path,
            static_fields={"component": "analysis", **(get_run_logging_context() or {})},
        )

    pipeline_payload = None
    analysis_delay_seconds: Optional[float] = None
    lag_threshold = PIPELINE_MAX_LAG_SECONDS
    lag_breached = False
    _ensure_trading_preconditions(analysis_started_at)
    if record_analysis_run:
        try:
            pipeline_payload, analysis_delay_seconds, lag_threshold, lag_breached = record_analysis_run(
                started_at=analysis_started_at,
                max_lag_seconds=PIPELINE_MAX_LAG_SECONDS,
            )
        except Exception as exc:
            LOGGER.warning("Failed to record analysis pipeline metrics: %s", exc)
        else:
            if analysis_delay_seconds is not None:
                LOGGER.info(
                    "Analysis started %.1f seconds after trading run",
                    analysis_delay_seconds,
                )
            if lag_breached and analysis_delay_seconds is not None:
                if lag_threshold is not None:
                    LOGGER.warning(
                        "Analysis start delay %.1f seconds exceeded threshold %ss",
                        analysis_delay_seconds,
                        lag_threshold,
                    )
                else:
                    LOGGER.warning(
                        "Analysis start delay %.1f seconds exceeded threshold",
                        analysis_delay_seconds,
                    )

    heartbeat_path: Path = Path(PUBLIC_DIR) / "system_heartbeat.json"
    heartbeat_ts: Optional[datetime]
    heartbeat_age: Optional[float] = None
    try:
        heartbeat_ts = _load_heartbeat_timestamp(heartbeat_path)
    except Exception as exc:
        LOGGER.warning("Heartbeat betöltése sikertelen: %s", exc)
        heartbeat_ts = None
    if heartbeat_ts:
        heartbeat_age = (analysis_started_at - heartbeat_ts).total_seconds()

    # Fallback: use pipeline monitor timestamps when heartbeat artefact is missing
    # (e.g., downstream analysis triggered from a published trading artifact).
    fallback_ts = None
    fallback_age = None
    staleness_limit = float(lag_threshold or PIPELINE_MAX_LAG_SECONDS or 300)
    if heartbeat_age is None or heartbeat_age > 60:
        candidates: List[datetime] = []
        if isinstance(pipeline_payload, dict):
            candidates.append(parse_utc_timestamp(pipeline_payload.get("updated_utc")))
            trading_meta = pipeline_payload.get("trading") if isinstance(pipeline_payload.get("trading"), dict) else {}
            candidates.append(parse_utc_timestamp(trading_meta.get("completed_utc")))
        fallback_ts = max([c for c in candidates if c is not None], default=None)
        if fallback_ts:
            fallback_age = (analysis_started_at - fallback_ts).total_seconds()

    effective_age = heartbeat_age
    age_source = "heartbeat"
    if heartbeat_age is None or heartbeat_age > 60:
        if fallback_age is not None:
            effective_age = fallback_age
            age_source = "pipeline_monitor"

    if effective_age is None:
        LOGGER.warning(
            "Heartbeat timestamp unavailable; proceeding without stall enforcement (limit %.1fs)",
            staleness_limit,
        )
    elif effective_age > staleness_limit:
        LOGGER.critical("CRITICAL: Data Pipeline Stalled")
        raise SystemExit("Data pipeline heartbeat missing or stale")
    elif age_source != "heartbeat":
        LOGGER.warning(
            "Heartbeat missing or stale; using %s timestamp (age %.1fs, limit %.1fs)",
            age_source,
            effective_age,
            staleness_limit,
        )

    asset_count = len(ASSETS)
    LOGGER.info("Starting analysis run for %d assets", asset_count)
    missing_models = {}
    dependency_issues: Dict[str, str] = {}
    placeholder_models: Dict[str, Dict[str, Any]] = {}
    global ML_PROBABILITY_PLACEHOLDER_BLOCKLIST, ML_PROBABILITY_PLACEHOLDER_INFO
    ML_PROBABILITY_PLACEHOLDER_BLOCKLIST = set()
    ML_PROBABILITY_PLACEHOLDER_INFO = {}
    if ENABLE_ML_PROBABILITY:
        missing_models = missing_model_artifacts(ASSETS)
        dependency_issues = runtime_dependency_issues()
        for asset in ASSETS:
            if asset.upper() in missing_models:
                continue
            info = inspect_model_artifact(asset)
            if info.get("status") == "placeholder":
                placeholder_models[asset] = info
    summary = {
        "ok": True,
        "generated_utc": nowiso(),
        "analysis_started_utc": to_utc_iso(analysis_started_at),
        "assets": {},
        "latency_flags": [],
        "sentiment_alerts": [],
        "troubleshooting": list(REFRESH_TIPS),
    }
    if MISSING_OPTIONAL_DEPENDENCIES:
        summary["degraded_mode"] = True
        summary["degraded_components"] = sorted(MISSING_OPTIONAL_DEPENDENCIES)
        degraded_note = (
            "Guardrail modulok hiányoznak: "
            + ", ".join(sorted(MISSING_OPTIONAL_DEPENDENCIES))
            + " — monitoring/precision/latency hook-ok fallback módot használnak."
        )
        summary["troubleshooting"].append(degraded_note)
        summary["optional_dependency_issues"] = list(OPTIONAL_DEPENDENCY_ISSUES)
    run_context = summary.setdefault("run_context", {})
    run_context.update({k: v for k, v in (get_run_logging_context() or {}).items() if v is not None})
    current_weekday = datetime.now(timezone.utc).weekday()
    run_context["weekday"] = current_weekday
    if current_weekday >= 5:
        run_context["weekend_run"] = True
        weekend_notes = summary.setdefault("notes", [])
        note_text = "Hétvégi snapshot — piaczárás miatt jelzések csak tájékoztató jellegűek"
        if note_text not in weekend_notes:
            weekend_notes.append(note_text)
    revision_info = detect_analysis_revision()
    if revision_info:
        summary["analysis_revision"] = revision_info
        commit = revision_info.get("commit", "unknown")
        dirty_suffix = " (dirty)" if revision_info.get("dirty") else ""
        LOGGER.info("Analysis build revision: %s%s", commit, dirty_suffix)
    if pipeline_payload:
        summary["pipeline_monitoring"] = pipeline_payload
        analysis_meta = pipeline_payload.get("analysis") if isinstance(pipeline_payload, dict) else None
        if isinstance(analysis_meta, dict) and analysis_meta.get("lag_breached"):
            lag_seconds = analysis_meta.get("lag_from_trading_seconds")
            threshold_value = analysis_meta.get("lag_threshold_seconds")
            if lag_seconds is not None:
                lag_text = int(round(float(lag_seconds)))
                threshold_text = (
                    f"{int(threshold_value)}s" if isinstance(threshold_value, (int, float)) else "n/a"
                )
                flag_msg = (
                    f"analysis: trading→analysis késés {lag_text}s (küszöb {threshold_text})"
                )
                if flag_msg not in summary["latency_flags"]:
                    summary["latency_flags"].append(flag_msg)
    global ML_PROBABILITY_ACTIVE
    ml_active = ENABLE_ML_PROBABILITY
    ml_disable_notes: List[str] = []
    if ML_PROBABILITY_MANUAL_OVERRIDE:
        ml_disable_notes.append(ML_PROBABILITY_MANUAL_REASON)
    if dependency_issues:
        summary["ml_runtime_issues"] = dependency_issues
        issue_names = ", ".join(sorted(dependency_issues))
        dep_warning = (
            "ML függőségek hiányoznak: "
            f"{issue_names} – telepítsd a requirements.txt szerinti csomagokat "
            "(pl. pip install -r requirements.txt vagy építsd be a konténerbe)."
        )
        summary["troubleshooting"].append(dep_warning)
        LOGGER.warning("ml runtime dependencies missing: %s", dep_warning)
        ml_active = False
        ml_disable_notes.append("függőségi problémák")
    reminder_required = False
    if missing_models:
        summary["ml_models_missing"] = {
            asset: str(path) for asset, path in missing_models.items()
        }
        missing_list = ", ".join(sorted(missing_models))
        warning = (
            "Hiányzó ML modellek: "
            f"{missing_list} – töltsd fel a public/models/<asset>_gbm.pkl fájlokat "
            "a valószínűségi score aktiválásához; a többi asseten az ML továbbra is fut, "
            "ezeken pedig a fallback heurisztika szolgáltatja a becslést."
        )
        summary["troubleshooting"].append(warning)
        if not SUPPRESS_ML_MODEL_WARNINGS:
            LOGGER.warning("ml artifacts missing: %s", warning)
        ml_disable_notes.append("modell artefakt hiányzik")
    if placeholder_models:
        placeholder_payload = {
            asset.upper(): {
                key: value
                for key, value in info.items()
                if key not in {"asset"} and value is not None
            }
            for asset, info in placeholder_models.items()
        }
        summary["ml_models_placeholder"] = {
            asset: meta for asset, meta in placeholder_payload.items()
        }
        ML_PROBABILITY_PLACEHOLDER_BLOCKLIST = set(placeholder_payload)
        ML_PROBABILITY_PLACEHOLDER_INFO = placeholder_payload
        placeholders = ", ".join(sorted(placeholder_models))
        placeholder_msg = (
            "Placeholder ML modellek: "
            f"{placeholders} – töltsd fel a tényleges GradientBoostingClassifier pickle-t; "
            "ezeken az eszközökön fallback pontozás fut, a többi asset valódi ML-t használ."
        )
        summary["troubleshooting"].append(placeholder_msg)
        if not SUPPRESS_ML_MODEL_WARNINGS:
            LOGGER.warning("ml placeholder artefacts detected: %s", placeholder_msg)
        ml_disable_notes.append("placeholder modell detektálva")
    if record_ml_model_status:
        try:
            _, reminder_required = record_ml_model_status(
                missing=list(sorted(missing_models)),
                placeholders=list(sorted(placeholder_models)),
            )
        except Exception:
            LOGGER.exception("Failed to persist ML model status for monitoring")
            reminder_required = False
    if reminder_required:
        reminder_assets: List[str] = []
        if missing_models:
            reminder_assets.append(
                "hiányzó modellek: " + ", ".join(sorted(missing_models))
            )
        if placeholder_models:
            reminder_assets.append(
                "placeholder modellek: " + ", ".join(sorted(placeholder_models))
            )
        reminder_text = (
            "Heti ML modell státusz emlékeztető – " + "; ".join(reminder_assets)
        )
        summary["troubleshooting"].append(reminder_text)
        if not SUPPRESS_ML_MODEL_WARNINGS:
            LOGGER.warning("ml model reminder: %s", reminder_text)
        else:
            LOGGER.info("ml model reminder (suppressed warnings): %s", reminder_text)
    if not ENABLE_ML_PROBABILITY:
        ml_disable_notes.append("flag letiltva (ENABLE_ML_PROBABILITY=0)")
    ML_PROBABILITY_ACTIVE = ml_active
    if not ML_PROBABILITY_ACTIVE:
        reason_text = "; ".join(ml_disable_notes) if ml_disable_notes else "ismeretlen ok"
        summary["troubleshooting"].append(
            "ML valószínűség számítás ideiglenesen letiltva"
            + (f" ({reason_text})." if reason_text else ".")
        )
    worker_count = _determine_analysis_workers(asset_count)
    if worker_count > 1:
        LOGGER.info("Running asset analysis with up to %d workers", worker_count)
    asset_results: Dict[str, Dict[str, Any]] = {}
    if worker_count <= 1:
        for asset in ASSETS:
            asset_key, result = _analyze_asset_guard(asset)
            asset_results[asset_key] = result
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {pool.submit(_analyze_asset_guard, asset): asset for asset in ASSETS}
            for future in as_completed(future_map):
                asset_name = future_map[future]
                try:
                    asset_key, result = future.result()
                except Exception as exc:  # pragma: no cover - executor safeguard
                    LOGGER.exception("Unhandled analysis error for asset %s", asset_name)
                    asset_results[asset_name] = {
                        "asset": asset_name,
                        "ok": False,
                        "error": str(exc),
                    }
                else:
                    asset_results[asset_key] = result

    for asset in ASSETS:
        res = asset_results.get(asset)
        if res is None:
            continue
        summary["assets"][asset] = res
        diag = res.get("diagnostics", {}) if isinstance(res, dict) else {}
        flags = diag.get("latency_flags") if isinstance(diag, dict) else None
        if flags:
            summary["latency_flags"].extend(flags)
        if isinstance(res, dict):
            sentiment_payload = res.get("sentiment_exit_summary")
            if not sentiment_payload:
                exit_payload = res.get("position_exit_signal")
                if not exit_payload and isinstance(res.get("active_position_meta"), dict):
                    exit_payload = res["active_position_meta"].get("exit_signal")
                if isinstance(exit_payload, dict) and exit_payload.get("category") == "sentiment_risk":
                    sentiment_payload = {
                        key: exit_payload.get(key)
                        for key in (
                            "state",
                            "severity",
                            "sentiment_severity",
                            "sentiment_score",
                            "sentiment_bias",
                            "sentiment_category",
                            "triggered_at",
                        )
                    }
                    headlines = exit_payload.get("headlines")
                    if isinstance(headlines, (list, tuple)) and headlines:
                        headline = headlines[0]
                        if isinstance(headline, str):
                            sentiment_payload["headline"] = headline
                if sentiment_payload:
                    mapped = {
                        "state": sentiment_payload.get("state"),
                        "severity_label": sentiment_payload.get("severity"),
                        "severity_score": sentiment_payload.get("sentiment_severity"),
                        "score": sentiment_payload.get("sentiment_score"),
                        "bias": sentiment_payload.get("sentiment_bias"),
                        "category": sentiment_payload.get("sentiment_category"),
                        "triggered_at": sentiment_payload.get("triggered_at"),
                    }
                    if "headline" in sentiment_payload:
                        mapped["headline"] = sentiment_payload["headline"]
                    sentiment_payload = mapped
            if sentiment_payload:
                alert_payload = {"asset": asset}
                alert_payload.update(
                    {
                        key: sentiment_payload.get(key)
                        for key in (
                            "state",
                            "severity_label",
                            "severity_score",
                            "score",
                            "bias",
                            "category",
                            "triggered_at",
                            "headline",
                        )
                    }
                )
                alert_payload = {
                    key: value for key, value in alert_payload.items() if value is not None
                }
                if alert_payload not in summary["sentiment_alerts"]:
                    summary["sentiment_alerts"].append(alert_payload)

    for asset in ASSETS:
        _backfill_signal_probability_metadata(asset, base_dir=Path(PUBLIC_DIR))

    summary["entry_counts"] = _build_entry_count_summary(asset_results)
    save_json(os.path.join(PUBLIC_DIR, "analysis_summary.json"), summary)

    html = "<!doctype html><meta charset='utf-8'><title>Analysis Summary</title>"
    html += "<h1>Analysis Summary (TD-only)</h1>"
    html += "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>"
    with open(os.path.join(PUBLIC_DIR, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(html)

    try:
        update_signal_health_report(Path(PUBLIC_DIR), summary)
    except Exception:
        pass

    try:
        update_data_latency_report(Path(PUBLIC_DIR), summary)
    except Exception:
        pass

    try:
        update_precision_gate_report()
    except Exception:
        pass

    try:
        build_status_snapshot(summary, Path(PUBLIC_DIR))
    except Exception as exc:
        LOGGER.warning("Failed to persist status snapshot: %s", exc)

    reports_env = os.getenv("REPORTS_DIR")
    if reports_env:
        reports_dir = Path(reports_env)
    else:
        reports_dir = Path(PUBLIC_DIR) / "reports"
    try:
        update_live_validation(Path(PUBLIC_DIR), reports_dir)
    except Exception:
        pass

    analysis_completed_at = datetime.now(timezone.utc)
    if finalize_analysis_run:
        try:
            duration_seconds = max(
                (analysis_completed_at - analysis_started_at).total_seconds(),
                0.0,
            )
            finalize_analysis_run(
                completed_at=analysis_completed_at,
                duration_seconds=duration_seconds,
            )
            LOGGER.info(
                "Analysis run completed in %.1f seconds",
                duration_seconds,
            )
        except Exception as exc:
            LOGGER.warning("Failed to finalize analysis pipeline metrics: %s", exc)

    data_gap_assets = [
        asset
        for asset, payload in (summary.get("assets") or {}).items()
        if isinstance(payload, dict)
        and isinstance(payload.get("probability_stack"), dict)
        and payload["probability_stack"].get("status") == "data_gap"
    ]
    if data_gap_assets:
        message = "Probability stack data gap detected for: " + ", ".join(sorted(data_gap_assets))
        LOGGER.warning(message)

if __name__ == "__main__":
    main()














































































