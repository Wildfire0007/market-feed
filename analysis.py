# -*- coding: utf-8 -*- 
"""
analysis.py — TD-only intraday jelzésképző (lokális JSON-okból).
Forrás: Trading.py által generált fájlok a public/<ASSET>/ alatt.
Kimenet:
  public/<ASSET>/signal.json      — "buy" / "sell" / "no entry" + okok
  public/analysis_summary.json    — összesített státusz
  public/analysis.html            — egyszerű HTML kivonat
"""

import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from zoneinfo import ZoneInfo

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

try:  # Optional monitoring utilities; keep analysis resilient if absent.
    from reports.trade_journal import record_signal_event
except Exception:  # pragma: no cover - optional dependency guard
    def record_signal_event(*_args, **_kwargs):
        return None

try:
    from reports.monitoring import update_signal_health_report, update_data_latency_report
except Exception:  # pragma: no cover - optional dependency guard
    def update_signal_health_report(*_args, **_kwargs):
        return None

    def update_data_latency_report(*_args, **_kwargs):
        return None

try:
    from reports.precision_monitor import update_precision_gate_report
except Exception:  # pragma: no cover - optional dependency guard
    def update_precision_gate_report(*_args, **_kwargs):
        return None

try:
    from reports.backtester import update_live_validation
except Exception:  # pragma: no cover - optional dependency guard
    def update_live_validation(*_args, **_kwargs):
        return None

try:
    from volatility_metrics import load_volatility_overlay
except Exception:  # pragma: no cover - optional helper
    def load_volatility_overlay(asset: str, outdir: Path, k1m: Optional[Any] = None) -> Dict[str, Any]:
        return {}

try:
    from reports.pipeline_monitor import (
        record_analysis_run,
        get_pipeline_log_path,
        DEFAULT_MAX_LAG_SECONDS as PIPELINE_MAX_LAG_SECONDS,
        record_ml_model_status,
    )
except Exception:  # pragma: no cover - optional helper
    record_analysis_run = None
    get_pipeline_log_path = None
    PIPELINE_MAX_LAG_SECONDS = None
    record_ml_model_status = None

import pandas as pd
import numpy as np

# --- Elemzendő eszközök ---
from config.analysis_settings import (
    ACTIVE_INVALID_BUFFER_ABS,
    ASSET_COST_MODEL,
    ATR5_MIN_MULT,
    ATR_ABS_MIN,
    ATR_LOW_TH_ASSET,
    ATR_LOW_TH_DEFAULT,
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
    NEWS_MODE_SETTINGS,
    FX_TP_TARGETS,
    GOLD_HIGH_VOL_WINDOWS,
    GOLD_LOW_VOL_TH,
    INTERVENTION_WATCH_DEFAULT,
    INTRADAY_ATR_RELAX,
    INTRADAY_BIAS_RELAX,
    LEVERAGE,
    MIN_RISK_ABS,
    MOMENTUM_RR_MIN,
    OFI_Z_SETTINGS,
    NVDA_EXTENDED_ATR_REL,
    NVDA_MOMENTUM_ATR_REL,
    P_SCORE_TIME_BONUS,
    SESSION_TIME_RULES,
    SESSION_WEEKDAYS,
    SESSION_WINDOWS_UTC,
    SPREAD_MAX_ATR_PCT,
    SL_BUFFER_RULES,
    SMT_AUTO_CONFIG,
    SMT_PENALTY_VALUE,
    SMT_REQUIRED_BARS,
    SPOT_MAX_AGE_SECONDS,
    TP_MIN_ABS,
    TP_MIN_PCT,
    TP_NET_MIN_ASSET,
    TP_NET_MIN_DEFAULT,
    VWAP_BAND_MULT,
    get_atr_threshold_multiplier,
    get_p_score_min,
)

LOGGER = logging.getLogger(__name__)

# Az asset-specifikus küszöböket a config/analysis_settings.json állomány
# szolgáltatja, így új eszköz felvételekor elegendő azt módosítani.

MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")

PUBLIC_DIR = "public"

MAX_RISK_PCT = 1.8


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
TP1_R   = 2.0
TP2_R   = 3.0
TP1_R_MOMENTUM = 1.5
TP2_R_MOMENTUM = 2.4
MIN_STOPLOSS_PCT = 0.01
# Momentum structure check window (5m candles × lookback bars)
MOMENTUM_BOS_LB = 36
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

OFI_Z_TRIGGER = float(OFI_Z_SETTINGS.get("trigger") or 0.0)
OFI_Z_WEAKENING = float(OFI_Z_SETTINGS.get("weakening") or 0.0)
OFI_Z_LOOKBACK = int(OFI_Z_SETTINGS.get("lookback_bars") or 0)

VWAP_TREND_BAND = float(VWAP_BAND_MULT.get("trend") or 0.8)
VWAP_MEAN_REVERT_BAND = float(VWAP_BAND_MULT.get("mean_revert") or 2.0)

NEWS_LOCKOUT_MINUTES = int(NEWS_MODE_SETTINGS.get("lockout_minutes") or 0)
NEWS_STABILISATION_MINUTES = int(NEWS_MODE_SETTINGS.get("stabilisation_minutes") or 0)
NEWS_SEVERITY_THRESHOLD = float(NEWS_MODE_SETTINGS.get("severity_threshold") or 1.0)
NEWS_CALENDAR_FILES: List[str] = list(NEWS_MODE_SETTINGS.get("calendar_files") or [])
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
ORDER_FLOW_LOOKBACK_MIN = 120
ORDER_FLOW_IMBALANCE_TH = 0.6
ORDER_FLOW_PRESSURE_TH = 0.7
PRECISION_FLOW_IMBALANCE_MARGIN = 1.1
PRECISION_FLOW_PRESSURE_MARGIN = 1.1
PRECISION_SCORE_THRESHOLD = 65.0
PRECISION_TRIGGER_NEAR_MULT = 0.2
REALTIME_JUMP_MULT = 2.0
MICRO_BOS_P_BONUS = 8.0
MOMENTUM_ATR_REL = 0.0006
MOMENTUM_VOLUME_RECENT = 6
MOMENTUM_VOLUME_BASE = 30
MOMENTUM_VOLUME_RATIO_TH = 1.4
MOMENTUM_TRAIL_TRIGGER_R = 1.2
MOMENTUM_TRAIL_LOCK = 0.5
ANCHOR_P_SCORE_DELTA_WARN = 10.0
ANCHOR_ATR_DROP_RATIO = 0.75
INTRADAY_EXHAUSTION_PCT = 0.82
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

def now_utctime_hm() -> Tuple[int,int]:
    t = datetime.now(timezone.utc)
    return t.hour, t.minute

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


def format_utc_minute(minute: int) -> str:
    minute = max(0, min(23 * 60 + 59, minute))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def format_local_range(start_dt: datetime, end_dt: datetime) -> List[str]:
    return [start_dt.strftime("%H:%M"), end_dt.strftime("%H:%M")]


def convert_windows_to_local(
    windows: Optional[List[Tuple[int, int, int, int]]]
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
        start_local = start_dt.astimezone(MARKET_TIMEZONE)
        end_local = end_dt.astimezone(MARKET_TIMEZONE)
        result.append(format_local_range(start_local, end_local))
    return result


def convert_minutes_to_local_range(start: int, end: int) -> List[str]:
    today_utc = datetime.now(timezone.utc).date()
    start_dt = datetime.combine(today_utc, dtime(start // 60, start % 60, tzinfo=timezone.utc))
    end_dt = datetime.combine(today_utc, dtime(end // 60, end % 60, tzinfo=timezone.utc))
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    start_local = start_dt.astimezone(MARKET_TIMEZONE)
    end_local = end_dt.astimezone(MARKET_TIMEZONE)
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

def session_state(asset: str) -> Tuple[bool, Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    minute_of_day = h * 60 + m
    entry_windows, monitor_windows = session_windows_utc(asset)
    monitor_ok = in_any_window_utc(monitor_windows, h, m)
    entry_window_ok = in_any_window_utc(entry_windows, h, m)
    weekday_ok = session_weekday_ok(asset, now)

    special_status: Optional[str] = None
    special_note: Optional[str] = None
    special_reason: Optional[str] = None
    break_active = False
    break_window: Optional[Tuple[int, int]] = None

    rules = SESSION_TIME_RULES.get(asset, {})
    sunday_open = rules.get("sunday_open_minute")
    if sunday_open is not None and now.weekday() == 6 and minute_of_day < sunday_open:
        monitor_ok = False
        entry_window_ok = False
        special_status = "closed_out_of_hours"
        special_reason = "sunday_open_pending"
        special_note = f"Piac zárva (vasárnapi nyitás {format_utc_minute(sunday_open)} UTC után)"

    if special_status is None:
        daily_breaks = rules.get("daily_breaks") or []
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

    friday_close = rules.get("friday_close_minute")
    if friday_close is not None and now.weekday() == 4 and minute_of_day >= friday_close:
        monitor_ok = False
        entry_window_ok = False
        special_status = "closed_out_of_hours"
        special_reason = "friday_close"
        special_note = f"Piac zárva (pénteki zárás {format_utc_minute(friday_close)} UTC)"

    open_now = monitor_ok and weekday_ok
    entry_open = entry_window_ok and weekday_ok and (
        monitor_ok or not monitor_windows
    )

    info: Dict[str, Any] = {
        "open": open_now,
        "entry_open": entry_open,
        "within_window": entry_window_ok,
        "within_entry_window": entry_window_ok,
        "within_monitor_window": monitor_ok,
        "weekday_ok": weekday_ok,
        "now_utc": now.isoformat(),
        "windows_utc": entry_windows,
    }
    if monitor_windows and monitor_windows != entry_windows:
        info["monitor_windows_utc"] = monitor_windows
    info["time_zone"] = "Europe/Berlin"
    entry_local = convert_windows_to_local(entry_windows)
    if entry_local:
        info["windows_local_cet"] = entry_local
    monitor_local = convert_windows_to_local(monitor_windows)
    if monitor_local and monitor_windows != entry_windows:
        info["monitor_windows_local_cet"] = monitor_local
    allowed = SESSION_WEEKDAYS.get(asset)
    if allowed:
        info["allowed_weekdays"] = list(allowed)
    if not weekday_ok:
        status = "closed_weekend"
        status_note = "Piac zárva (hétvége)"
    elif not open_now:
        status = "closed_out_of_hours"
        status_note = "Piac zárva (nyitáson kívül)"
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
    if break_active:
        info["daily_break_active"] = True
    if break_window:
        info["daily_break_window_utc"] = [format_utc_minute(break_window[0]), format_utc_minute(break_window[1])]
        info["daily_break_window_cet"] = convert_minutes_to_local_range(*break_window)
    if special_reason:
        info["special_closure_reason"] = special_reason
    info["status"] = status
    info["status_note"] = status_note
    if not entry_open:
        nxt = next_session_open(asset, now)
        if nxt:
            info["next_open_utc"] = nxt.isoformat()
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
) -> Dict[str, Any]:
    session_meta = session_meta or {}
    execution_playbook = execution_playbook or []
    intraday_profile = intraday_profile or {}
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
        add_note(f"Maximális számla kockázat: {MAX_RISK_PCT:.1f}%.")

    if momentum_trailing_plan:
        activation = momentum_trailing_plan.get("activation_rr")
        lock_ratio = momentum_trailing_plan.get("lock_ratio")
        if activation is not None:
            add_note(
                f"Momentum trail aktiválás: {float(activation):.2f}R, lock {float(lock_ratio or 0) * 100:.0f}%"
            )

    if precision_plan:
        score = precision_plan.get("score")
        threshold = precision_plan.get("score_threshold") or PRECISION_SCORE_THRESHOLD
        confidence = precision_plan.get("confidence")
        trigger_state = precision_plan.get("trigger_state")
        if score is not None:
            add_note(
                f"Precision score: {float(score):.2f} / küszöb {float(threshold):.0f}."
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

    precision_state = str((precision_plan or {}).get("trigger_state") or "")
    precision_direction = (precision_plan or {}).get("direction")
    if decision in {"precision_ready", "precision_arming"} and precision_plan:
        label = "Precision trigger előkészítés"
        if decision == "precision_arming":
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
        update_state("monitor_trigger", "high" if decision == "precision_arming" else "medium")

    if precision_state:
        plan["context"]["precision_state"] = precision_state
    if precision_direction:
        plan["context"]["precision_direction"] = precision_direction

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

    if (decision in {"buy", "sell", "precision_ready", "precision_arming"}) and not entry_open:
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
        if in_any_window_utc(GOLD_HIGH_VOL_WINDOWS, h, m):
            base = ATR_LOW_TH_DEFAULT
        else:
            base = GOLD_LOW_VOL_TH
    else:
        base = ATR_LOW_TH_ASSET.get(asset, ATR_LOW_TH_DEFAULT)
    multiplier = get_atr_threshold_multiplier(asset)
    return base * multiplier

def tp_min_pct_for(asset: str, rel_atr: float, session_flag: bool) -> float:
    base = TP_MIN_PCT.get(asset, TP_MIN_PCT["default"])
    if np.isnan(rel_atr):
        return base
    return base

def tp_net_min_for(asset: str) -> float:
    return TP_NET_MIN_ASSET.get(asset, TP_NET_MIN_DEFAULT)


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


def _normalize_btcusd_sentiment(signal: SentimentSignal) -> float:
    bias = (signal.bias or "").lower()
    direction = 1.0
    if any(flag in bias for flag in ("bear", "risk_off", "usd_bullish")):
        direction = -1.0
    elif any(flag in bias for flag in ("btc_bullish", "risk_on", "bull")):
        direction = 1.0
    return signal.score * direction


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

    atr5_series = atr(df_5m, 14)
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
    relaxed = RELAXED_STALE_FRAMES.get(asset_key)
    if relaxed and frame in relaxed:
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

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        status = ""
    if status:
        metadata["dirty"] = True

    return metadata

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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

def as_df_klines(raw: Any) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open","high","low","close"])
    arr = raw if isinstance(raw, list) else (raw.get("values") or [])
    rows: List[Dict[str, Any]] = []
    for x in arr:
        try:
            if "datetime" in x:
                dt = pd.to_datetime(x["datetime"], utc=True)
                o = float(x["open"]); h = float(x["high"]); l = float(x["low"]); c = float(x["close"])
                v = float(x.get("volume", 0.0) or 0.0)
            elif "t" in x:
                dt = pd.to_datetime(x["t"], utc=True)
                o = float(x["o"]); h = float(x["h"]); l = float(x["l"]); c = float(x["c"])
                v = float(x.get("v", 0.0) or 0.0)
            else:
                continue
            rows.append({"time": dt, "open": o, "high": h, "low": l, "close": c, "volume": v})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows).sort_values("time").set_index("time")
    cols = ["open", "high", "low", "close", "volume"]
    return df[cols]


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
    return r.fillna(method="bfill").fillna(50.0)

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
    return adx_series.fillna(method="bfill").dropna()


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
    buckets = ATR_PERCENTILE_BUCKETS
    minute_now = _min_of_day(now.hour, now.minute)
    bucket_name = _bucket_for_minute(minute_now, buckets)
    details["bucket"] = bucket_name
    percentile_target = None
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
    cumsum_volume = volume.fillna(method="ffill").fillna(method="bfill").cumsum()
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


def evaluate_news_lockout(asset: str, now: datetime) -> Tuple[bool, Optional[str]]:
    if NEWS_LOCKOUT_MINUTES <= 0 and NEWS_STABILISATION_MINUTES <= 0:
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
        if severity_val < NEWS_SEVERITY_THRESHOLD:
            continue
        delta_minutes = (now - event_time).total_seconds() / 60.0
        if abs(delta_minutes) <= NEWS_LOCKOUT_MINUTES:
            lockout = True
            reason = event.get("title") or event.get("name") or "High impact news"
            break
        if 0 < delta_minutes < NEWS_STABILISATION_MINUTES:
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
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "imbalance": None,
        "pressure": None,
        "delta_volume": None,
        "aggressor_ratio": None,
        "imbalance_z": None,
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
        return metrics

    if k1m.empty or "volume" not in k1m.columns or len(k1m) < ORDER_FLOW_LOOKBACK_MIN:
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

def broke_structure(df: pd.DataFrame, direction: str, lookback: int = MOMENTUM_BOS_LB) -> bool:
    """Egyszerű szerkezeti törés: utolsó high/low áttöri az előző N bar csúcsát/alját."""
    if df.empty or len(df) < lookback + 2:
        return False
    ref = df.iloc[-(lookback+1):-1]
    last = df.iloc[-1]
    if direction == "long":
        return last["high"] > ref["high"].max()
    if direction == "short":
        return last["low"] < ref["low"].min()
    return False

def retest_level(df: pd.DataFrame, direction: str, lookback: int = MOMENTUM_BOS_LB) -> bool:
    if df.empty or len(df) < 2:
        return False
    if len(df) < lookback + 1:
        ref = df.iloc[:-1]
    else:
        ref = df.iloc[-(lookback+1):-1]
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

def structure_break_with_retest(df: pd.DataFrame, direction: str, lookback: int = MOMENTUM_BOS_LB) -> bool:
    if direction not in ("long", "short"):
        return False
    if not broke_structure(df, direction, lookback):
        return False
    return retest_level(df, direction, lookback)

def micro_bos_with_retest(k1m: pd.DataFrame, k5m: pd.DataFrame, direction: str) -> bool:
    if direction not in ("long", "short"):
        return False
    if k1m.empty or len(k1m) < 10:
        return False
    if not detect_bos(k1m, direction):
        return False
    return retest_level(k5m, direction, MOMENTUM_BOS_LB)


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
    order_flow_metrics: Dict[str, Optional[float]],
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
        "score_threshold": PRECISION_SCORE_THRESHOLD,
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
    flow_conditions: List[bool] = []
    flow_strength = 0.0
    flow_reasons: List[str] = []
    if imb is not None:
        if direction == "buy" and imb > ORDER_FLOW_IMBALANCE_TH:
            score += 12.0
            flow_strength += min(1.0, float(imb) / ORDER_FLOW_IMBALANCE_TH)
            factors.append(f"order flow imbalance +{imb:.2f}")
            flow_reasons.append(f"imbalance {imb:.2f}")
            flow_conditions.append(
                imb >= ORDER_FLOW_IMBALANCE_TH * PRECISION_FLOW_IMBALANCE_MARGIN
            )
        elif direction == "sell" and imb < -ORDER_FLOW_IMBALANCE_TH:
            score += 12.0
            flow_strength += min(1.0, abs(float(imb)) / ORDER_FLOW_IMBALANCE_TH)
            factors.append(f"order flow imbalance {imb:.2f}")
            flow_reasons.append(f"imbalance {imb:.2f}")
            flow_conditions.append(
                imb <= -ORDER_FLOW_IMBALANCE_TH * PRECISION_FLOW_IMBALANCE_MARGIN
            )
        else:
            flow_conditions.append(False)

    pressure = order_flow_metrics.get("pressure")
    if pressure is not None:
        if direction == "buy" and pressure > ORDER_FLOW_PRESSURE_TH:
            score += 10.0
            flow_strength += min(1.0, float(pressure) / ORDER_FLOW_PRESSURE_TH)
            factors.append(f"order flow pressure +{pressure:.2f}")
            flow_reasons.append(f"pressure {pressure:.2f}")
            flow_conditions.append(
                pressure >= ORDER_FLOW_PRESSURE_TH * PRECISION_FLOW_PRESSURE_MARGIN
            )
        elif direction == "sell" and pressure < -ORDER_FLOW_PRESSURE_TH:
            score += 10.0
            flow_strength += min(1.0, abs(float(pressure)) / ORDER_FLOW_PRESSURE_TH)
            factors.append(f"order flow pressure {pressure:.2f}")
            flow_reasons.append(f"pressure {pressure:.2f}")
            flow_conditions.append(
                pressure <= -ORDER_FLOW_PRESSURE_TH * PRECISION_FLOW_PRESSURE_MARGIN
            )
        else:
            flow_conditions.append(False)

    delta_volume = order_flow_metrics.get("delta_volume")
    if delta_volume is not None:
        try:
            dv = float(delta_volume)
            if direction == "buy" and dv > 0:
                flow_strength += min(0.5, abs(dv))
                flow_reasons.append(f"delta +{dv:.1f}")
                flow_conditions.append(True)
            elif direction == "sell" and dv < 0:
                flow_strength += min(0.5, abs(dv))
                flow_reasons.append(f"delta {dv:.1f}")
                flow_conditions.append(True)
        except (TypeError, ValueError):
            pass

    flow_ready = bool(flow_conditions) and all(flow_conditions)
    if flow_conditions:
        plan["order_flow_strength"] = round(
            min(2.0, flow_strength / max(len(flow_conditions), 1)), 2
        )
    plan["order_flow_ready"] = flow_ready
    if flow_reasons:
        plan["trigger_reasons"].extend(flow_reasons)

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
                candidate.floor("S").isoformat().replace("+00:00", "Z")
            )
    except Exception:
        ready_ts = None
    plan["ready_ts"] = ready_ts

    try:
        plan_score = float(plan.get("score") or 0.0)
    except (TypeError, ValueError):
        plan_score = 0.0
    plan["score_ready"] = plan_score >= PRECISION_SCORE_THRESHOLD

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
        tp1_core = TP1_R * 1.15
        tp2_core = TP2_R * 1.2
        rr_core = rr_core * 1.05
        tp1_mom = TP1_R_MOMENTUM * 1.1
        tp2_mom = TP2_R_MOMENTUM * 1.15
        rr_mom = rr_mom * 1.05

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
    price_ref = profile["price_reference"]
    if price_ref is None:
        price_ref = last_close
    profile["price_reference"] = price_ref

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
            "elapsed_minutes": elapsed_minutes,
            "notes": notes,
        }
    )

    return profile

# ------------------------------ elemzés egy eszközre ---------------------------

def analyze(asset: str) -> Dict[str, Any]:
    outdir = os.path.join(PUBLIC_DIR, asset)
    os.makedirs(outdir, exist_ok=True)

    # 1) Bemenetek
    latency_profile = load_latency_profile(outdir)
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
    spot_utc = "-"
    spot_retrieved = "-"
    display_spot: Optional[float] = None
    realtime_used = False
    realtime_reason: Optional[str] = None
    realtime_confidence: float = 1.0
    realtime_transport = str(spot_realtime.get("transport") or "http").lower() if isinstance(spot_realtime, dict) else "http"
    entry_thresholds_meta: Dict[str, Any] = {"profile": ENTRY_THRESHOLD_PROFILE_NAME}
    if spot:
        spot_price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
        spot_utc = spot.get("utc") or spot.get("timestamp") or "-"
        spot_retrieved = spot.get("retrieved_at_utc") or spot.get("retrieved") or "-"

    rt_price = None
    rt_utc = None
    if spot_realtime:
        rt_price = spot_realtime.get("price") if spot_realtime.get("price") is not None else spot_realtime.get("price_usd")
        rt_utc = spot_realtime.get("utc") or spot_realtime.get("timestamp") or spot_realtime.get("retrieved_at_utc")

    rt_ts = parse_utc_timestamp(rt_utc)
    if rt_price is not None and (not spot_price or (rt_ts and parse_utc_timestamp(spot_utc) and rt_ts > parse_utc_timestamp(spot_utc)) or (rt_ts and not parse_utc_timestamp(spot_utc))):
        spot_price = rt_price
        display_spot = safe_float(rt_price)
        if rt_ts:
            spot_utc = to_utc_iso(rt_ts)
        if spot_realtime.get("retrieved_at_utc"):
            spot_retrieved = spot_realtime.get("retrieved_at_utc")
        realtime_used = True
        realtime_reason = "Realtime spot feed override"

    now = datetime.now(timezone.utc)
    session_ok_flag, session_meta = session_state(asset)
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
    spot_max_age = SPOT_MAX_AGE_SECONDS.get(asset, SPOT_MAX_AGE_SECONDS["default"])
    relaxed_spot_reason: Optional[str] = None
    if spot_ts:
        delta = now - spot_ts
        if delta.total_seconds() < 0:
            spot_latency_sec = 0
        else:
            spot_latency_sec = int(delta.total_seconds())
        if spot_latency_sec is not None and spot_latency_sec > spot_max_age:
            age_min = spot_latency_sec // 60
            limit_min = spot_max_age // 60
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

    spread_abs: Optional[float] = None
    if isinstance(spot_realtime, dict):
        bid_val = safe_float(spot_realtime.get("bid") or spot_realtime.get("best_bid"))
        ask_val = safe_float(spot_realtime.get("ask") or spot_realtime.get("best_ask"))
        if bid_val is not None and ask_val is not None and ask_val >= bid_val:
            spread_abs = float(ask_val - bid_val)
    if spread_abs is None and rt_price is not None and spot_price is not None:
        try:
            spread_abs = abs(float(rt_price) - float(spot_price))
        except (TypeError, ValueError):
            spread_abs = None

    analysis_now = now

    intervention_summary: Optional[Dict[str, Any]] = None
    intervention_band: Optional[str] = None
    sentiment_signal = None
    sentiment_applied_points: Optional[float] = None
    sentiment_normalized: Optional[float] = None
    if asset == "BTCUSD":
        intervention_config = load_intervention_config(outdir)
        news_flag = load_intervention_news_flag(outdir, intervention_config)
        sentiment_signal = load_sentiment(asset, Path(outdir))
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
    if critical_reasons:
        reasons_payload = ["Critical data latency — belépés tiltva"] + critical_reasons
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

    # 2/b Rezsim (EMA21 meredekség 1h)
    regime_ok, regime_val, regime_slope_signed = ema_slope_ok(
        k1h_closed,
        EMA_SLOPE_PERIOD,
        EMA_SLOPE_LOOKBACK,
        EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT),
    )
    slope_threshold = EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT)
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
    atr_series_5 = atr(k5m_closed)
    atr5 = atr_series_5.iloc[-1]
    rel_atr = float(atr5 / price_for_calc) if (atr5 and price_for_calc) else float("nan")
    adx_value = latest_adx(k5m_closed)
    if adx_value is not None and not np.isfinite(adx_value):
        adx_value = None
    entry_thresholds_meta["adx_value"] = adx_value
    adx_regime = "unknown"
    if adx_value is not None:
        if ADX_TREND_MIN and adx_value >= ADX_TREND_MIN:
            adx_regime = "trend"
        elif adx_value < ADX_RANGE_MAX:
            adx_regime = "range"
        else:
            adx_regime = "balanced"
    entry_thresholds_meta["adx_regime_initial"] = adx_regime
    atr_profile_multiplier = get_atr_threshold_multiplier(asset)
    atr_threshold = atr_low_threshold(asset)
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
    spread_limit = SPREAD_MAX_ATR_PCT.get(asset, SPREAD_MAX_ATR_PCT.get("default"))
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
    atr_abs_min = ATR_ABS_MIN.get(asset)
    atr_abs_ok = True
    if atr_abs_min is not None:
        try:
            atr_abs_ok = float(atr5) >= atr_abs_min
        except Exception:
            atr_abs_ok = False
    atr_ok = not (np.isnan(rel_atr) or rel_atr < atr_threshold or not atr_abs_ok)
    if not spread_gate_ok:
        atr_ok = False
    if not atr_overlay_gate:
        atr_ok = False
    if stale_timeframes.get("k5m"):
        atr_ok = False

    entry_thresholds_meta["atr_threshold_effective"] = atr_threshold
    entry_thresholds_meta["spread_gate_ok"] = spread_gate_ok

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
    fib_ok = fib_zone_ok(
        move_hi, move_lo, price_for_calc,
        low=0.618, high=0.886,
        tol_abs=atr1h_tol * 0.75,   # SZÉLESÍTVE: ±0.75×ATR(1h)
        tol_frac=0.02
    )

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

    struct_retest_long  = structure_break_with_retest(k5m_closed, "long", MOMENTUM_BOS_LB)
    struct_retest_short = structure_break_with_retest(k5m_closed, "short", MOMENTUM_BOS_LB)
    if stale_timeframes.get("k5m"):
        struct_retest_long = struct_retest_short = False

    micro_bos_long = micro_bos_with_retest(k1m_closed, k5m_closed, "long")
    micro_bos_short = micro_bos_with_retest(k1m_closed, k5m_closed, "short")
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

    nvda_cross_long = nvda_cross_short = False
    if asset == "NVDA":
        nvda_cross_long = ema_cross_long
        nvda_cross_short = ema_cross_short

    effective_bias = trend_bias
    bias_override_used = False
    bias_override_reason: Optional[str] = None
    intraday_bias_gate_meta: Optional[Dict[str, Any]] = None
    bias_gate_notes: List[str] = []
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
                    return (
                        momentum_vol_ratio is not None
                        and momentum_vol_ratio >= MOMENTUM_VOLUME_RATIO_TH
                    )
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

    recent_break_long = broke_structure(k5m_closed, "long", MOMENTUM_BOS_LB)
    recent_break_short = broke_structure(k5m_closed, "short", MOMENTUM_BOS_LB)
    if stale_timeframes.get("k5m"):
        recent_break_long = recent_break_short = False

    ofi_zscore = order_flow_metrics.get("imbalance_z") if isinstance(order_flow_metrics, dict) else None
    try:
        if ofi_zscore is not None and not np.isfinite(ofi_zscore):
            ofi_zscore = None
    except Exception:
        ofi_zscore = None

    structure_components: Dict[str, bool] = {"bos": False, "liquidity": False, "ofi": False}
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
    structure_components["liquidity"] = bool(
        liquidity_ok
        or liquidity_ok_base
        or vwap_confluence.get("trend_pullback")
        or vwap_confluence.get("mean_revert")
    )
    if ofi_zscore is not None and OFI_Z_TRIGGER > 0:
        if effective_bias == "long":
            structure_components["ofi"] = ofi_zscore >= OFI_Z_TRIGGER
        elif effective_bias == "short":
            structure_components["ofi"] = ofi_zscore <= -OFI_Z_TRIGGER
    if structure_components.get("ofi") and ofi_zscore is not None:
        structure_notes.append(f"OFI z-score {ofi_zscore:.2f} támogatja az irányt")
    structure_gate = sum(1 for flag in structure_components.values() if flag) >= 2
    entry_thresholds_meta["structure_components"] = structure_components

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

    for note in structure_notes:
        if note not in reasons:
            reasons.append(note)

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

    P = max(0.0, min(100.0, P))

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
    if asset == "NVDA":
        h, m = now_utctime_hm()
        minute = h * 60 + m
        cash_start = _min_of_day(13, 30)
        cash_end = _min_of_day(20, 0)
        in_cash_session = cash_start <= minute <= cash_end
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
    liquidity_ok = liquidity_ok_base or bool(vwap_confluence.get("trend_pullback")) or bool(vwap_confluence.get("mean_revert"))
    structure_notes: List[str] = []
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
        if high_atr_push and directional_confirmation:
            liquidity_ok = True
            liquidity_relaxed = True

    p_score_min_base = get_p_score_min(asset)
    p_score_min_local = p_score_min_base
    entry_thresholds_meta["p_score_min_base"] = p_score_min_base
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

    core_rr_min = CORE_RR_MIN.get(asset, CORE_RR_MIN["default"])
    momentum_rr_min = MOMENTUM_RR_MIN.get(asset, MOMENTUM_RR_MIN["default"])
    position_size_scale = 1.0
    funding_dir_filter: Optional[str] = None
    funding_reason: Optional[str] = None
    range_time_stop_plan: Optional[Dict[str, Any]] = None
    if adx_regime == "trend":
        core_rr_min = max(core_rr_min, ADX_TREND_CORE_RR)
        momentum_rr_min = max(momentum_rr_min, ADX_TREND_MOM_RR)
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

    entry_thresholds_meta["position_size_scale"] = position_size_scale
    if funding_dir_filter:
        entry_thresholds_meta["funding_filter"] = funding_dir_filter
    if funding_value is not None:
        entry_thresholds_meta["funding_rate"] = funding_value

    core_tp1_mult = dynamic_tp_profile["core"]["tp1"]
    core_tp2_mult = dynamic_tp_profile["core"]["tp2"]
    core_rr_min = max(core_rr_min, dynamic_tp_profile["core"]["rr"])
    mom_tp1_mult = dynamic_tp_profile["momentum"]["tp1"]
    mom_tp2_mult = dynamic_tp_profile["momentum"]["tp2"]
    momentum_rr_min = max(momentum_rr_min, dynamic_tp_profile["momentum"]["rr"])

    range_guard_label = "intraday_range_guard"
    structure_label = "structure(2of3)"

    core_required = [
        "session",
        "regime",
        "bias",
        structure_label,
        range_guard_label,
        "atr",
        f"rr_math>={core_rr_min:.1f}",
        "tp_min_profit",
        "min_stoploss",
        tp_net_label,
    ]

    conds_core = {
        "session": bool(session_ok_flag),
        "regime": bool(regime_ok),
        "bias": effective_bias in ("long", "short"),
        structure_label: bool(structure_gate),
        "atr": bool(atr_ok),
        range_guard_label: range_guard_ok,
    }
    if funding_dir_filter:
        core_required.append("funding_alignment")
        if effective_bias in ("long", "short"):
            conds_core["funding_alignment"] = effective_bias == funding_dir_filter
        else:
            conds_core["funding_alignment"] = False
    base_core_ok = all(conds_core.values())
    can_enter_core = (P >= p_score_min_local) and base_core_ok
    missing_core = [k for k, v in conds_core.items() if not v]
    if P < p_score_min_local:
        if float(p_score_min_local).is_integer():
            p_score_label = str(int(round(p_score_min_local)))
        else:
            p_score_label = f"{p_score_min_local:.1f}"
        missing_core.append(f"P_score>={p_score_label}")
    if liquidity_relaxed:
        reasons.append("Likviditási kapu lazítva erős momentum miatt")

    # --- Momentum feltételek (override) — kriptókra (zárt 5m-ből) ---
    momentum_used = False
    mom_dir: Optional[str] = None
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
    if asset in ENABLE_MOMENTUM_ASSETS:
        direction = effective_bias if effective_bias in {"long", "short"} else None
        if not session_ok_flag:
            missing_mom.append("session")
        if not regime_ok:
            missing_mom.append("regime")
        if direction is None:
            missing_mom.append("bias")
        if funding_dir_filter and direction in {"long", "short"}:
            if direction != funding_dir_filter:
                missing_mom.append("funding_alignment")
        if not range_guard_ok:
            missing_mom.append(range_guard_label)

        if momentum_vol_ratio is None or momentum_vol_ratio < MOMENTUM_VOLUME_RATIO_TH:
            momentum_liquidity_ok = False
            missing_mom.append("liquidity")

        of_imb = order_flow_metrics.get("imbalance")
        if of_imb is not None and abs(of_imb) < ORDER_FLOW_IMBALANCE_TH:
            momentum_liquidity_ok = False
            missing_mom.append("order_flow")
        of_pressure = order_flow_metrics.get("pressure")
        if of_pressure is not None and direction is not None:
            if direction == "long" and of_pressure < ORDER_FLOW_PRESSURE_TH:
                momentum_liquidity_ok = False
                missing_mom.append("order_flow_pressure")
            elif direction == "short" and of_pressure > -ORDER_FLOW_PRESSURE_TH:
                momentum_liquidity_ok = False
                missing_mom.append("order_flow_pressure")
        ofi_confirm = True
        if ofi_zscore is not None and OFI_Z_TRIGGER > 0 and direction in {"long", "short"}:
            if direction == "long":
                ofi_confirm = ofi_zscore >= OFI_Z_TRIGGER
            else:
                ofi_confirm = ofi_zscore <= -OFI_Z_TRIGGER
        if not ofi_confirm:
            missing_mom.append("ofi")

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
                and mom_atr_ok
                and cross_flag
                and momentum_liquidity_ok
                and ofi_confirm
            ):
                mom_dir = "buy" if direction == "long" else "sell"
                mom_trigger_desc = "EMA9×21 momentum cross"
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
    precision_plan: Optional[Dict[str, Any]] = None
    precision_ready_for_entry = False
    precision_flow_ready = False
    precision_trigger_ready = False
    precision_trigger_state: Optional[str] = None
    precision_gate_label = f"precision_score>={int(PRECISION_SCORE_THRESHOLD)}"
    precision_flow_gate_label = "precision_flow_alignment"
    precision_trigger_gate_label = "precision_trigger_sync"

    def compute_levels(decision_side: str, rr_required: float, tp1_mult: float = TP1_R, tp2_mult: float = TP2_R):
        nonlocal entry, sl, tp1, tp2, rr, missing, min_stoploss_ok, tp1_net_pct_value, last_computed_risk, current_tp1_mult, current_tp2_mult
        current_tp1_mult = tp1_mult
        current_tp2_mult = tp2_mult
        atr5_val  = float(atr5 or 0.0)

        buf_rule = SL_BUFFER_RULES.get(asset, SL_BUFFER_RULES["default"])
        buf = max(buf_rule.get("atr_mult", 0.2) * atr5_val, buf_rule.get("abs_min", 0.5))
      
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
        min_stoploss_ok_local = risk >= entry * MIN_STOPLOSS_PCT - 1e-9
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

        rel_atr_local = float(rel_atr) if not np.isnan(rel_atr) else float("nan")
        high_vol = (not np.isnan(rel_atr_local)) and (rel_atr_local >= ATR_VOL_HIGH_REL)
        cost_mult = COST_MULT_HIGH_VOL if high_vol else COST_MULT_DEFAULT
        tp_min_pct = tp_min_pct_for(asset, rel_atr_local, session_ok_flag)
        overnight_days = estimate_overnight_days(asset, analysis_now)
        cost_round_pct, overnight_pct = compute_cost_components(asset, entry, overnight_days)
        total_cost_pct = cost_mult * cost_round_pct + overnight_pct
        net_pct = gross_pct - total_cost_pct
        tp1_net_pct_value = net_pct

        min_profit_abs = max(
            TP_MIN_ABS.get(asset, TP_MIN_ABS["default"]),
            tp_min_pct * entry,
            (cost_mult * cost_round_pct + overnight_pct) * entry,
            ATR5_MIN_MULT * atr5_val,
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

    if can_enter_core:
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
            elif tp1_net_pct_value is not None:
                msg_net = f"TP1 nettó profit ≈ {tp1_net_pct_value*100:.2f}%"
                if msg_net not in reasons:
                    reasons.append(msg_net)
    else:
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
                ):
                    allowed_slip = 0.2 * last_computed_risk
                    if decision == "buy":
                        slip = max(0.0, price_for_calc - last5_close)
                    else:
                        slip = max(0.0, last5_close - price_for_calc)
                    slip_info = {
                        "slip": float(slip),
                        "allowed": float(allowed_slip),
                    }
                    if slip > allowed_slip + 1e-9:
                        no_chase_violation = True
                if slip_info:
                    slip_info["violated"] = no_chase_violation
                    entry_thresholds_meta["momentum_no_chase"] = slip_info
                if no_chase_violation:
                    reasons.append(
                        "Momentum no-chase szabály: aktuális ár kedvezőtlenebb mint 0.2R"
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
        elif asset in ENABLE_MOMENTUM_ASSETS and missing_mom:
            mode = "momentum"
            required_list = list(mom_required)
            missing = list(dict.fromkeys(missing_mom))  # uniq

    precision_direction: Optional[str] = None
    execution_playbook: List[Dict[str, Any]] = []
    if decision in ("buy", "sell"):
        precision_direction = decision
    elif effective_bias == "long":
        precision_direction = "buy"
    elif effective_bias == "short":
        precision_direction = "sell"

    if precision_direction:
        atr5_value = float(atr5) if atr5 is not None and np.isfinite(float(atr5)) else None
        precision_plan = compute_precision_entry(
            asset,
            precision_direction,
            k1m_closed,
            k5m_closed,
            price_for_calc,
            atr5_value,
            order_flow_metrics,
        )

    if precision_plan:
        precision_plan.setdefault("score_threshold", PRECISION_SCORE_THRESHOLD)
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
        precision_ready_for_entry = precision_score_val >= PRECISION_SCORE_THRESHOLD
        precision_plan["score_ready"] = bool(precision_plan.get("score_ready") or precision_ready_for_entry)
        precision_trigger_state = str(precision_plan.get("trigger_state") or "standby")
        precision_plan["trigger_state"] = precision_trigger_state
        precision_flow_ready = bool(precision_plan.get("order_flow_ready"))
        precision_trigger_ready = bool(
            precision_plan.get("trigger_ready")
            or precision_trigger_state in {"arming", "fire"}
        )
        precision_plan["trigger_ready"] = precision_trigger_ready
        if precision_plan.get("trigger_levels") is None:
            precision_plan["trigger_levels"] = {}
        if precision_gate_label not in required_list:
            required_list.append(precision_gate_label)
        if not precision_ready_for_entry and precision_gate_label not in missing:
            missing.append(precision_gate_label)
        if precision_flow_gate_label not in required_list:
            required_list.append(precision_flow_gate_label)
        if not precision_flow_ready and precision_flow_gate_label not in missing:
            missing.append(precision_flow_gate_label)
        if precision_trigger_gate_label not in required_list:
            required_list.append(precision_trigger_gate_label)
        if not precision_trigger_ready and precision_trigger_gate_label not in missing:
            missing.append(precision_trigger_gate_label)
        if precision_flow_ready:
            missing = [item for item in missing if item != precision_flow_gate_label]
        if precision_trigger_ready:
            missing = [item for item in missing if item != precision_trigger_gate_label]

    if precision_plan:
        if not precision_ready_for_entry:
            if decision in ("buy", "sell"):
                score_note = (
                    f"Precision score {precision_score_val:.2f} < {PRECISION_SCORE_THRESHOLD:.0f}"
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
                if not precision_flow_ready:
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
        if decision in ("buy", "sell") and range_time_stop_plan:
            execution_playbook.append(
                {
                    "step": "time_stop",
                    "description": (
                        f"Time-stop {range_time_stop_plan['timeout']} percig, ha nem érjük el {range_time_stop_plan['breakeven_trigger']:.2f}R-t"
                    ),
                    "timeout_minutes": range_time_stop_plan["timeout"],
                    "breakeven_trigger_r": range_time_stop_plan["breakeven_trigger"],
                }
            )

    if precision_plan:
        window_payload: Optional[List[float]] = None
        window = precision_plan.get("entry_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            try:
                window_payload = [float(window[0]), float(window[1])]
            except (TypeError, ValueError):
                window_payload = None
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
            "score_threshold": precision_plan.get("score_threshold", PRECISION_SCORE_THRESHOLD),
            "score_ready": precision_plan.get("score_ready"),
            "trigger_ready": precision_plan.get("trigger_ready"),
            "order_flow_ready": precision_plan.get("order_flow_ready"),
            "order_flow_strength": precision_plan.get("order_flow_strength"),
            "trigger_progress": precision_plan.get("trigger_progress"),
            "trigger_confidence": precision_plan.get("trigger_confidence"),
            "reasons": precision_plan.get("trigger_reasons") or None,
        }
        execution_playbook.append(trigger_step)

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
        "order_flow_imbalance": order_flow_metrics.get("imbalance") or 0.0,
        "order_flow_pressure": order_flow_metrics.get("pressure") or 0.0,
        "order_flow_aggressor": order_flow_metrics.get("aggressor_ratio") or 0.0,
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
    probability_source = None
    fallback_meta: Optional[Dict[str, Any]] = None
    if ml_prediction.metadata:
        probability_source = ml_prediction.metadata.get("source")
        meta_reason = ml_prediction.metadata.get("unavailable_reason")
        raw_fallback = ml_prediction.metadata.get("fallback")
        if isinstance(raw_fallback, dict):
            fallback_meta = raw_fallback
        if fallback_meta and probability_source == "fallback":
            fallback_reason = fallback_meta.get("reason") or meta_reason
            reason_map = {
                "model_missing": "ML fallback: modell hiányzik — heurisztikus pontozás aktív",
                "sklearn_missing": "ML fallback: scikit-learn hiányzik — heurisztikus pontozás aktív",
                "model_type_mismatch": "ML fallback: modell típus inkompatibilis",
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

    analysis_timestamp = nowiso()
    probability_percent = int(max(0, min(100, round(combined_probability * 100))))

    gates_payload: Dict[str, Any] = {
        "mode": mode,
        "required": required_list,
        "missing": missing,
    }
    if intraday_bias_gate_meta:
        gates_payload["intraday_bias"] = intraday_bias_gate_meta

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
        "session_info": session_meta,
        "diagnostics": diagnostics_payload(tf_meta, source_files, latency_flags),
        "reasons": (reasons + ([f"missing: {', '.join(missing)}"] if missing else []))
        or ["no signal"],
        "realtime_transport": realtime_transport,
    }
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
    if ml_prediction.metadata:
        decision_obj["probability_stack"] = ml_prediction.metadata
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
    entry_thresholds_meta.setdefault("atr_threshold_effective", atr_threshold)
    entry_thresholds_meta.setdefault("p_score_min_effective", p_score_min_local)
    decision_obj["entry_thresholds"] = entry_thresholds_meta
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
    )
    if action_plan:
        decision_obj["action_plan"] = action_plan

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
            "score_threshold": precision_plan.get("score_threshold", PRECISION_SCORE_THRESHOLD),
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
            "score_threshold": precision_plan.get("score_threshold", PRECISION_SCORE_THRESHOLD),
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
    save_json(os.path.join(outdir, "signal.json"), decision_obj)
    try:
        record_signal_event(asset, decision_obj)
    except Exception:
        # Journaling issues should not break signal generation.
        pass
    return decision_obj
# ------------------------------- főfolyamat ------------------------------------

def main():
    analysis_started_at = datetime.now(timezone.utc)
    pipeline_log_path = None
    if get_pipeline_log_path:
        try:
            pipeline_log_path = get_pipeline_log_path()
        except Exception:
            pipeline_log_path = None
    if pipeline_log_path:
        pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(handler, "_pipeline_log", False) for handler in LOGGER.handlers):
            handler = logging.FileHandler(pipeline_log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
            handler._pipeline_log = True  # type: ignore[attr-defined]
            LOGGER.addHandler(handler)
        if LOGGER.level > logging.INFO:
            LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False

    pipeline_payload = None
    analysis_delay_seconds: Optional[float] = None
    lag_threshold = PIPELINE_MAX_LAG_SECONDS
    lag_breached = False
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

    LOGGER.info("Starting analysis run for %d assets", len(ASSETS))
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
    for asset in ASSETS:
        try:
            res = analyze(asset)
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
        except Exception as exc:
            LOGGER.exception("Analysis failed for asset %s", asset)
            summary["assets"][asset] = {"asset": asset, "ok": False, "error": str(exc)}
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

    reports_env = os.getenv("REPORTS_DIR")
    if reports_env:
        reports_dir = Path(reports_env)
    else:
        reports_dir = Path(PUBLIC_DIR) / "reports"
    try:
        update_live_validation(Path(PUBLIC_DIR), reports_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()






































