# -*- coding: utf-8 -*- 
"""
analysis.py — TD-only intraday jelzésképző (lokális JSON-okból).
Forrás: Trading.py által generált fájlok a public/<ASSET>/ alatt.
Kimenet:
  public/<ASSET>/signal.json      — "buy" / "sell" / "no entry" + okok
  public/analysis_summary.json    — összesített státusz
  public/analysis.html            — egyszerű HTML kivonat
"""

import os, json
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from active_anchor import load_anchor_state, record_anchor

import pandas as pd
import numpy as np

# --- Elemzendő eszközök ---
ASSETS = ["EURUSD", "GOLD_CFD", "USDJPY", "USOIL", "NVDA", "SRTY"]

MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")

PUBLIC_DIR = "public"

LEVERAGE = {
    "EURUSD": 5.0,
    "USDJPY": 5.0,
    "GOLD_CFD": 2.0,
    "USOIL": 2.0,
    "NVDA": 2.0,
    "SRTY": 2.0,
}

MAX_RISK_PCT = 1.8
FIB_TOL = 0.02

# --- ATR küszöbök ---
ATR_LOW_TH_DEFAULT = 0.0007   # 0.07%
ATR_LOW_TH_ASSET = {
    "EURUSD": 0.0005,  # 0.05%
    "USDJPY": 0.0005,
    "NVDA": 0.0012,
    "SRTY": 0.0015,
}
GOLD_HIGH_VOL_WINDOWS = [(6, 30, 21, 30)]  # európai nyitástól US zárásig lazább
GOLD_LOW_VOL_TH = 0.0006
SMT_PENALTY_VALUE = 7
SMT_REQUIRED_BARS = 2

# --- Kereskedési/egz. küszöbök (RR/TP) ---
MIN_R_CORE      = 2.0
MIN_R_MOMENTUM  = 1.6
TP1_R   = 2.0
TP2_R   = 3.0
MIN_STOPLOSS_PCT = 0.01
TP_NET_MIN_DEFAULT = 0.01
TP_NET_MIN_ASSET = {
    "EURUSD": 0.0035,
    "USDJPY": 0.0030,
    "GOLD_CFD": 0.0025,
    "USOIL": 0.0030,
    "NVDA": 0.0120,
    "SRTY": 0.0150,
}

# --- Per-asset TP minimumok, költségek és SL pufferek ---
TP_MIN_PCT = {        # min. TP1 távolság %-ban (entry-hez képest)
    "default": 0.0020,  # 0.20%
    "GOLD_CFD": 0.0025, # 0.25%
    "USOIL":    0.0030, # 0.30%
    "EURUSD":   0.0035, # 0.35%
    "USDJPY":   0.0030, # 0.30%
    "NVDA":     0.0120, # 1.20%
    "SRTY":     0.0150, # 1.50%
}
TP_MIN_ABS = {        # min. TP1 távolság abszolútban (tick/árjegyzés miatt)
    "default": 0.50,
    "GOLD_CFD": 3.0,
    "USOIL":    0.70,
    "EURUSD":   0.0015,
    "USDJPY":   0.20,
    "NVDA":     4.0,
    "SRTY":     0.35,
}
SL_BUFFER_RULES = {
    "default": {"atr_mult": 0.2, "abs_min": 0.5},
    "GOLD_CFD": {"atr_mult": 0.2, "abs_min": 2.0},
    "USOIL":    {"atr_mult": 0.25, "abs_min": 0.15},
    "EURUSD":   {"atr_mult": 0.2, "abs_min": 0.0008},
    "USDJPY":   {"atr_mult": 0.2, "abs_min": 0.10},
    "NVDA":     {"atr_mult": 0.10, "abs_min": 2.0},
    "SRTY":     {"atr_mult": 0.12, "abs_min": 0.25},
}
MIN_RISK_ABS = {
    "default": 0.5,
    "GOLD_CFD": 2.0,
    "USOIL": 0.15,
    "EURUSD": 0.0008,
    "USDJPY": 0.10,
    "NVDA": 2.0,
    "SRTY": 0.25,
}

ACTIVE_INVALID_BUFFER_ABS = {
    "GOLD_CFD": 2.0,
    "USOIL": 0.15,
}

ASSET_COST_MODEL = {
    "GOLD_CFD": {"type": "pct", "round_trip_pct": 0.0005, "overnight_pct": 0.00035},
    "USOIL":   {"type": "pct", "round_trip_pct": 0.0008, "overnight_pct": 0.00050},
    "EURUSD":  {"type": "pip", "round_trip_pips": 1.0, "overnight_pips": 0.6},
    "USDJPY":  {"type": "pip", "round_trip_pips": 1.0, "overnight_pips": 0.7},
    "NVDA":    {"type": "pct", "round_trip_pct": 0.0008, "overnight_pct": 0.00040},
    "SRTY":    {"type": "pct", "round_trip_pct": 0.0010, "overnight_pct": 0.00045},
}
DEFAULT_COST_MODEL = {"type": "pct", "round_trip_pct": 0.0010, "overnight_pct": 0.00030}
COST_MULT_DEFAULT = 1.5
COST_MULT_HIGH_VOL = 1.3
ATR5_MIN_MULT  = 0.5     # min. profit >= 0.5× ATR(5m)
ATR_VOL_HIGH_REL = 0.002  # 0.20% relatív ATR felett lazítjuk a költség-multit

EMA_SLOPE_TH_DEFAULT = 0.0008
EMA_SLOPE_TH_ASSET = {
    "EURUSD": 0.0006,
    "USDJPY": 0.0006,
    "GOLD_CFD": 0.0008,
    "USOIL": 0.0008,
    "NVDA": 0.0010,
    "SRTY": 0.0010,
}

# --- Asset-specifikus ATR és RR küszöbök ---
ATR_ABS_MIN = {
    "EURUSD": 0.00035,
    "USDJPY": 0.20,
    "GOLD_CFD": 1.8,
    "USOIL": 0.25,
    "NVDA": 4.0,
    "SRTY": 0.35,
}

CORE_RR_MIN = {
    "default": MIN_R_CORE,
    "NVDA": 2.2,
    "SRTY": 2.5,
}

MOMENTUM_RR_MIN = {
    "default": MIN_R_MOMENTUM,
    "NVDA": 1.6,
    "SRTY": 1.8,
}

FX_TP_TARGETS = {
    "EURUSD": 0.0035,
    "USDJPY": 0.0030,
}

NVDA_EXTENDED_ATR_REL = 0.0015
NVDA_MOMENTUM_ATR_REL = 0.0010
SRTY_MOMENTUM_ATR_REL = 0.0010

# --- Momentum override ---
ENABLE_MOMENTUM_ASSETS: set = {"NVDA", "SRTY"}
MOMENTUM_BARS    = 7             # 5m EMA9–EMA21 legfeljebb 7 baron belüli jel
MOMENTUM_ATR_REL = 0.0010        # alap momentum ATR küszöb (asset-specifikus felülírás)
MOMENTUM_BOS_LB  = 15            # szerkezeti töréshez nézett ablak (bar)

ANCHOR_STATE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def current_anchor_state() -> Dict[str, Dict[str, Any]]:
    global ANCHOR_STATE_CACHE
    if ANCHOR_STATE_CACHE is None:
        ANCHOR_STATE_CACHE = load_anchor_state()
    return ANCHOR_STATE_CACHE

P_SCORE_MIN = 60
MICRO_BOS_P_BONUS = 8

# --- Rezsim és session beállítások ---
EMA_SLOPE_PERIOD   = 21          # 1h EMA21
EMA_SLOPE_LOOKBACK = 3           # hány baron mérjük a változást
EMA_SLOPE_TH       = EMA_SLOPE_TH_DEFAULT      # alap küszöb (ha nincs asset-specifikus)

# UTC idősávok: mind az entry (új belépő) mind a 24/5 megfigyelés számára.
# "entry": kereskedési (új pozíció) ablakok
# "monitor": tágabb, eToro-kompatibilis felügyeleti ablakok (24/5, napi szünettel)
SESSION_WINDOWS_UTC: Dict[str, Dict[str, Optional[List[Tuple[int, int, int, int]]]]] = {
    "EURUSD": {
        "entry": [
            (0, 0, 23, 59),
        ],
        "monitor": [
            (0, 0, 23, 59),
        ],
    },
    "USDJPY": {
        "entry": [
            (0, 0, 23, 59),
        ],
        "monitor": [
            (0, 0, 23, 59),
        ],
    },
    # Arany/olaj: szűk entry, de 24/5 kitettség felügyelet (napi 1h karbantartási szünet).
    "GOLD_CFD": {
        "entry": [
            (7, 0, 17, 30),
            (8, 0, 18, 30),
        ],
        "monitor": [
            (0, 0, 23, 59),
        ],
    },
    "USOIL": {
        "entry": [
            (7, 0, 18, 0),
            (8, 0, 19, 0),
        ],
        "monitor": [
            (0, 0, 23, 59),
        ],
    },
    "NVDA": {
        "entry": [
            (13, 30, 20, 0),
        ],
        "monitor": [
            (12, 0, 21, 30),
        ],
    },
    "SRTY": {
        "entry": [
            (13, 30, 20, 0),
        ],
        "monitor": [
            (12, 0, 21, 30),
        ],
    },
}

# Spot-adat elavulás küszöbök (másodperc)
SPOT_MAX_AGE_SECONDS: Dict[str, int] = {
    "default": 20 * 60,      # 20 perc
    "GOLD_CFD": 45 * 60,     # 45 perc — lazább, mert cash session
    "USOIL":    45 * 60,     # 45 perc — CME/NYMEX CFD feed esti szünettel
    "NVDA":    15 * 60,
    "SRTY":    15 * 60,
}

INTERVENTION_CONFIG_FILENAME = "intervention_watch_config.json"
INTERVENTION_NEWS_FILENAME = "intervention_watch_news_flag.json"
INTERVENTION_STATE_FILENAME = "intervention_watch_state.json"
INTERVENTION_SUMMARY_FILENAME = "analysis_summary.json"
INTERVENTION_P_SCORE_ADD = 15

INTERVENTION_WATCH_DEFAULT: Dict[str, Any] = {
    "USDJPY": {
        "intervention_watch": {
            "big_figures": [150.0, 152.0, 155.0, 160.0, 165.0],
            "session_primary_utc": ["00:00", "09:00"],
            "session_secondary_utc": ["18:00", "21:00"],
            "speed_thresholds_pips_30m": [40, 60, 80],
            "spike_multiplier_atr5": 1.2,
            "wick_ratio_trigger": 0.55,
            "vr_threshold": 1.4,
            "atr_spike_ratio": 1.6,
            "irs_bands": {
                "LOW": [0, 39],
                "ELEVATED": [40, 59],
                "HIGH": [60, 79],
                "IMMINENT": [80, 100],
            },
            "actions": {
                "HIGH": {
                    "block_new_longs": True,
                    "reduce_long_size_pct": 50,
                    "p_score_long_add": 15,
                    "tighten_sl": {"atr5_mult": 0.3, "min_pct": 0.10},
                },
                "IMMINENT": {
                    "force_partial_close_long_pct": 50,
                    "trailing_sl": {"atr5_mult": 0.5, "min_pct": 0.20},
                    "allow_new_shorts_after_pullback": {
                        "pullback_atr5_mult": 0.5,
                        "rr_min": 2.5,
                    },
                    "leverage_cap_jpy_cross": 2,
                },
            },
        }
    }
}

# Diagnosztikai tippek — a dashboard / Worker frissítéséhez.
REFRESH_TIPS = [
    "Az analysis.py mindig a legutóbbi ZÁRT gyertyával számol (5m: max. ~5 perc késés).",
    "CI/CD-ben kösd össze a Trading és Analysis futást: az analysis job csak a trading után induljon (needs: trading).",
    "A kliens kéréséhez adj cache-busting query paramot (pl. ?v=<timestamp>) és no-store cache-control fejlécet.",
    "Cloudflare Worker stale policy: 5m feedre állítsd 120s-re, hogy hamar átjöjjön az új jel.",
    "A dashboard stabilizáló (2 azonos jel + 10 perc cooldown) lassíthatja a kártya frissítését — lazítsd, ha realtime kell.",
]

# Heti naptári korlátozások (Python weekday: hétfő=0 ... vasárnap=6). None = mindig.
SESSION_WEEKDAYS: Dict[str, Optional[List[int]]] = {
    "EURUSD": [0, 1, 2, 3, 4, 6],  # hétfő–péntek + vasárnap esti nyitás
    "USDJPY": [0, 1, 2, 3, 4, 6],
    "GOLD_CFD": [0, 1, 2, 3, 4, 6],     # vasárnap esti nyitás – szombat zárva
    "USOIL": [0, 1, 2, 3, 4, 6],        # vasárnap esti nyitás – szombat zárva
    "NVDA": [0, 1, 2, 3, 4],
    "SRTY": [0, 1, 2, 3, 4],
}


def _min_of_day(hour: int, minute: int = 0) -> int:
    return hour * 60 + minute


SESSION_TIME_RULES: Dict[str, Dict[str, Any]] = {
    "EURUSD": {
        "sunday_open_minute": _min_of_day(21, 5),    # 23:05 CEST → 21:05 UTC
        "friday_close_minute": _min_of_day(21, 55),   # 23:55 CEST → 21:55 UTC
        "daily_breaks": [(_min_of_day(22, 0), _min_of_day(22, 5))],
    },
    "USDJPY": {
        "sunday_open_minute": _min_of_day(21, 5),
        "friday_close_minute": _min_of_day(21, 55),
        "daily_breaks": [(_min_of_day(22, 0), _min_of_day(22, 5))],
    },
    "GOLD_CFD": {
        "sunday_open_minute": _min_of_day(22, 0),     # 00:00 CEST → 22:00 UTC
        "friday_close_minute": _min_of_day(20, 30),
        "daily_breaks": [(_min_of_day(21, 0), _min_of_day(22, 0))],
    },
    "USOIL": {
        "sunday_open_minute": _min_of_day(22, 0),
        "friday_close_minute": _min_of_day(20, 30),
        "daily_breaks": [(_min_of_day(21, 0), _min_of_day(22, 0))],
    },
    "NVDA": {
        "sunday_open_minute": _min_of_day(22, 0),
        "friday_close_minute": _min_of_day(20, 30),
        "daily_breaks": [(_min_of_day(21, 0), _min_of_day(21, 15))],
    },
    "SRTY": {
        "sunday_open_minute": _min_of_day(22, 0),
        "friday_close_minute": _min_of_day(20, 30),
        "daily_breaks": [(_min_of_day(21, 0), _min_of_day(21, 15))],
    },
}

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

def in_any_window_utc(windows: Optional[List[Tuple[int,int,int,int]]], h: int, m: int) -> bool:
    if not windows:
        return True
    minutes = h*60 + m
    for sh, sm, eh, em in windows:
        s = sh*60 + sm
        e = eh*60 + em
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
    if asset in {"EURUSD", "USDJPY"}:
        return f"1h ATR ≈ {atr_value:.4f}"
    if atr_value >= 100:
        return f"1h ATR ≈ {atr_value:.0f}"
    if atr_value >= 10:
        return f"1h ATR ≈ {atr_value:.1f}"
    return f"1h ATR ≈ {atr_value:.2f}"


def derive_position_management_note(
    asset: str,
    session_meta: Dict[str, Any],
    regime_ok: bool,
    effective_bias: str,
    structure_flag: str,
    atr1h: Optional[float],
    anchor_bias: Optional[str],
    anchor_timestamp: Optional[str],
) -> Optional[str]:
    if not session_meta.get("open"):
        return None
    if session_meta.get("status") in {"maintenance", "closed_out_of_hours"}:
        return None

    monitor_only = not session_meta.get("entry_open")
    anchor_active = anchor_bias in {"long", "short"}
    if not monitor_only and not anchor_active:
        return None

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

    if anchor_active and bias in {"long", "short"} and anchor_bias != bias:
        return (
            prefix
            + f"aktív {anchor_bias} pozíció a jelenlegi bias ({bias}) ellen → defenzív menedzsment, részleges zárás vagy szoros SL"
            + hint_suffix
        )

    direction = anchor_bias or bias
    if direction == "long":
        if regime_ok and structure_flag == "bos_up":
            return prefix + "long trend aktív → pozíció tartható, SL igazítás az 1h swing alatt" + hint_suffix
        if regime_ok:
            return prefix + "long bias, de friss BOS nincs → részleges realizálás vagy szorosabb SL" + hint_suffix
        return prefix + "long kitettség gyenge trendben → méretcsökkentés vagy zárás mérlegelendő" + hint_suffix
    if direction == "short":
        if regime_ok and structure_flag == "bos_down":
            return prefix + "short trend aktív → pozíció tartható, SL az 1h csúcsa felett" + hint_suffix
        if regime_ok:
            return prefix + "short bias, de szerkezeti megerősítés hiányzik → részleges realizálás / SL szűkítés" + hint_suffix
        return prefix + "short kitettség gyenge trendben → méretcsökkentés vagy zárás mérlegelendő" + hint_suffix
    return prefix + "nincs egyértelmű bias → defenzív menedzsment, részleges zárás vagy szoros SL" + hint_suffix

def atr_low_threshold(asset: str) -> float:
    h, m = now_utctime_hm()
    if asset == "GOLD_CFD":
        if in_any_window_utc(GOLD_HIGH_VOL_WINDOWS, h, m):
            return ATR_LOW_TH_DEFAULT
        return GOLD_LOW_VOL_TH
    return ATR_LOW_TH_ASSET.get(asset, ATR_LOW_TH_DEFAULT)

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
    if asset in {"EURUSD", "USDJPY"}:
        return 3 if wd == 2 else 1   # FX: szerdai tripla díj
    if asset in {"GOLD_CFD", "USOIL", "NVDA", "SRTY"}:
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
        return "IMMINENT"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "ELEVATED"
    return "LOW"


def compute_irs_usdjpy(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    now_utc: datetime,
    config: Dict[str, Any],
    news_flag: int = 0,
) -> Tuple[int, str, Dict[str, Any]]:
    cfg = config.get("intervention_watch", config)
    metrics: Dict[str, Any] = {
        "price": None,
        "nearest_big_figure": None,
        "dist_pips": None,
        "roc_30m_pips": None,
        "d5_pips": None,
        "atr5_pips": None,
        "atr5_median_pips": None,
        "rv_ratio": None,
        "session_points": 0,
        "comms_points": 0,
        "components": {},
    }

    if df_1m.empty or df_5m.empty:
        return 0, "LOW", metrics

    price = safe_float(df_1m["close"].iloc[-1])
    if price is None:
        return 0, "LOW", metrics

    big_figures = cfg.get("big_figures") or []
    if big_figures:
        nearest = min(big_figures, key=lambda x: abs(price - float(x)))
    else:
        nearest = round(price)
    nearest = float(nearest)
    dist_pips = abs(price - nearest) * 100.0
    lps = max(0.0, min(30.0, 30.0 - dist_pips / 5.0))

    roc_30 = 0.0
    speed_score = 0.0
    speed_thresholds = cfg.get("speed_thresholds_pips_30m", [40, 60, 80])
    if len(df_1m) >= 31:
        price_30 = safe_float(df_1m["close"].iloc[-30])
        if price_30 is not None:
            roc_30 = abs(price - price_30) * 100.0
            if roc_30 >= float(speed_thresholds[2] if len(speed_thresholds) > 2 else 80):
                speed_score += 18
            elif roc_30 >= float(speed_thresholds[1] if len(speed_thresholds) > 1 else 60):
                speed_score += 12
            elif roc_30 >= float(speed_thresholds[0] if speed_thresholds else 40):
                speed_score += 7

    d5 = 0.0
    atr5_val = None
    atr5_series = atr(df_5m, 14)
    if len(df_5m) >= 2:
        close_curr = safe_float(df_5m["close"].iloc[-1])
        close_prev = safe_float(df_5m["close"].iloc[-2])
        if close_curr is not None and close_prev is not None:
            d5 = abs(close_curr - close_prev) * 100.0
    if not atr5_series.empty:
        atr5_raw = safe_float(atr5_series.iloc[-1])
        if atr5_raw is not None:
            atr5_val = atr5_raw
    atr5_pips = (atr5_val or 0.0) * 100.0
    atr5_med = None
    if not atr5_series.empty:
        tail = atr5_series.tail(48).to_numpy(dtype=float)
        if tail.size:
            atr5_med = float(np.nanmedian(tail))
    atr5_med_pips = (atr5_med or 0.0) * 100.0

    spike_mult = float(cfg.get("spike_multiplier_atr5", 1.2) or 1.2)
    delta5_spike = False
    if d5 >= 25.0 and atr5_pips > 0 and d5 >= spike_mult * atr5_pips:
        speed_score += 12
        delta5_spike = True
    speed_score = min(30.0, speed_score)

    vol_spike = 0.0
    atr_spike_ratio = float(cfg.get("atr_spike_ratio", 1.6) or 1.6)
    atr_spike = False
    if atr5_med_pips > 0 and atr5_pips > 0:
        if (atr5_pips / atr5_med_pips) >= atr_spike_ratio:
            vol_spike += 12
            atr_spike = True

    rv_ratio = None
    vr_threshold = float(cfg.get("vr_threshold", 1.4) or 1.4)
    vr_spike = False
    if len(df_1m) >= 60 and len(df_5m) >= 12:
        rv1 = safe_float(np.var(np.diff(df_1m["close"].iloc[-60:].to_numpy(dtype=float))))
        rv5 = safe_float(np.var(np.diff(df_5m["close"].iloc[-12:].to_numpy(dtype=float))))
        if rv1 is not None and rv5 is not None and rv5 > 0:
            rv_ratio = rv1 / rv5
            if rv_ratio >= vr_threshold:
                vol_spike += 8
                vr_spike = True

    wick_pressure = False
    wick_threshold = float(cfg.get("wick_ratio_trigger", 0.55) or 0.55)
    if len(df_5m) >= 3:
        last_bars = df_5m.iloc[-3:]
        wick_hits = 0
        for _, row in last_bars.iterrows():
            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            open_ = safe_float(row.get("open"))
            close = safe_float(row.get("close"))
            if None in (high, low, open_, close):
                continue
            total = high - low
            if total <= 0:
                continue
            if close >= open_:
                upper = high - max(open_, close)
                if upper / total >= wick_threshold:
                    wick_hits += 1
        if wick_hits >= 2:
            vol_spike += 4
            wick_pressure = True

    vol_spike = min(20.0, vol_spike)

    session_points = 0.0
    primary = cfg.get("session_primary_utc")
    secondary = cfg.get("session_secondary_utc")
    if isinstance(primary, (list, tuple)) and len(primary) == 2:
        if in_utc_range(now_utc, str(primary[0]), str(primary[1])):
            session_points += 8
    if isinstance(secondary, (list, tuple)) and len(secondary) == 2:
        if in_utc_range(now_utc, str(secondary[0]), str(secondary[1])):
            session_points += 3
    session_points = min(10.0, session_points)

    comms = max(0, min(10, int(news_flag or 0)))

    total_score = lps + speed_score + vol_spike + session_points + comms
    irs = int(round(max(0.0, min(100.0, total_score))))
    band = intervention_band(irs, cfg.get("irs_bands", {}))

    metrics.update(
        {
            "price": price,
            "nearest_big_figure": nearest,
            "dist_pips": dist_pips,
            "roc_30m_pips": roc_30,
            "d5_pips": d5,
            "atr5_pips": atr5_pips,
            "atr5_median_pips": atr5_med_pips,
            "rv_ratio": rv_ratio,
            "session_points": session_points,
            "comms_points": comms,
            "components": {
                "level_pressure": lps,
                "speed": speed_score,
                "vol_spike": vol_spike,
                "session": session_points,
                "comms": comms,
                "delta5_spike": delta5_spike,
                "atr_spike": atr_spike,
                "vr_spike": vr_spike,
                "wick_pressure": wick_pressure,
            },
        }
    )

    return irs, band, metrics


def load_intervention_config(outdir: str) -> Dict[str, Any]:
    base_cfg = deepcopy(INTERVENTION_WATCH_DEFAULT.get("USDJPY", {}))
    path = os.path.join(outdir, INTERVENTION_CONFIG_FILENAME)
    override = load_json(path)
    if isinstance(override, dict):
        candidate = override
        if "USDJPY" in candidate and isinstance(candidate["USDJPY"], dict):
            candidate = candidate["USDJPY"]
        if "intervention_watch" in candidate and isinstance(candidate["intervention_watch"], dict):
            candidate = candidate["intervention_watch"]
        if isinstance(candidate, dict):
            base_cfg = deep_update(deepcopy(base_cfg), candidate)
    return base_cfg


def load_intervention_news_flag(outdir: str, config: Dict[str, Any]) -> int:
    cfg = config.get("intervention_watch", config)
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
    cfg = config.get("intervention_watch", config)
    dist = safe_float(metrics.get("dist_pips"))
    nearest = safe_float(metrics.get("nearest_big_figure"))
    if dist is not None and nearest is not None and dist <= 30:
        reasons.append(f"Near {nearest:.2f} (dist {dist:.0f} pip)")

    roc = safe_float(metrics.get("roc_30m_pips"))
    speed_thresholds = cfg.get("speed_thresholds_pips_30m", [40, 60, 80])
    if roc is not None and speed_thresholds:
        if roc >= float(speed_thresholds[-1]):
            reasons.append(f"+Speed 30m={roc:.0f} pip")
        elif roc >= float(speed_thresholds[0]):
            reasons.append(f"Speed 30m={roc:.0f} pip")

    if metrics.get("components", {}).get("delta5_spike"):
        d5 = safe_float(metrics.get("d5_pips"))
        atr5 = safe_float(metrics.get("atr5_pips"))
        ratio = (d5 / atr5) if d5 and atr5 else None
        if ratio:
            reasons.append(f"5m Δ={d5:.0f} pip ({ratio:.1f}×ATR5)")
        elif d5:
            reasons.append(f"5m Δ={d5:.0f} pip spike")

    if metrics.get("components", {}).get("atr_spike"):
        reasons.append("ATR5 spike vs 48-bar median")

    if metrics.get("components", {}).get("vr_spike"):
        vr = safe_float(metrics.get("rv_ratio"))
        if vr:
            reasons.append(f"Variance-ratio={vr:.2f} (>={cfg.get('vr_threshold', 1.4)})")
        else:
            reasons.append("Variance-ratio spike")

    if metrics.get("components", {}).get("wick_pressure"):
        reasons.append("Upper wick pressure (parabola risk)")

    session_pts = safe_float(metrics.get("session_points"))
    if session_pts and session_pts >= 6:
        reasons.append("Tokyo session focus")
    elif session_pts and session_pts > 0:
        reasons.append("US afternoon window")

    comms_pts = metrics.get("comms_points")
    if comms_pts:
        reasons.append(f"Comms/news flag +{int(comms_pts)}")

    return reasons


def build_intervention_policy(band: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    policy: List[str] = [
        "spread_guard_top3bars>=1.4×normal → pause entries 5-10m",
        "discord_alert_cooldown:10m",
    ]
    cfg = config.get("intervention_watch", config)
    actions = cfg.get("actions", {})

    if band in {"HIGH", "IMMINENT"}:
        high_actions = actions.get("HIGH", {})
        if high_actions.get("block_new_longs"):
            policy.append("block_new_longs")
        reduce_pct = high_actions.get("reduce_long_size_pct")
        if reduce_pct:
            policy.append(f"reduce_long_size_pct:{int(reduce_pct)}")
        p_add = high_actions.get("p_score_long_add")
        if p_add:
            policy.append(f"p_score_long_add:+{int(p_add)}")
        if high_actions.get("tighten_sl"):
            ts = high_actions["tighten_sl"]
            policy.append(
                "tighten_sl>=max({:.1f}×ATR5m,{:.2f}%)".format(
                    float(ts.get("atr5_mult", 0.3)), float(ts.get("min_pct", 0.10))
                )
            )
        policy.append("jpy_cross_longs_blocked")
        policy.append("jpy_cross_shorts_slippage_buffer")

    if band == "IMMINENT":
        immin_actions = actions.get("IMMINENT", {})
        force_pct = immin_actions.get("force_partial_close_long_pct")
        if force_pct:
            policy.append(f"force_partial_close_long_pct:{int(force_pct)}")
        trailing = immin_actions.get("trailing_sl")
        if trailing:
            policy.append(
                "trailing_sl>=max({:.1f}×ATR5m,{:.2f}%)".format(
                    float(trailing.get("atr5_mult", 0.5)), float(trailing.get("min_pct", 0.20))
                )
            )
        pb = immin_actions.get("allow_new_shorts_after_pullback", {})
        if pb:
            pullback_mult = float(pb.get("pullback_atr5_mult", 0.5) or 0.5)
            rr_min = float(pb.get("rr_min", 2.5) or 2.5)
            policy.append(
                "new_shorts_after_pullback≥{:.1f}×ATR5m(rr≥{:.1f})".format(
                    pullback_mult,
                    rr_min,
                )
            )
        lev_cap = immin_actions.get("leverage_cap_jpy_cross")
        if lev_cap:
            policy.append(f"jpy_cross_leverage_cap:{float(lev_cap):.0f}x")
        policy.append("tp_targets_jpy_cross:1.8R/2.8R")
        policy.append("market_entries_blocked_use_limit_with≥0.5×ATR5m_buffer")

    return policy


def save_intervention_asset_summary(outdir: str, risk_summary: Dict[str, Any]) -> None:
    payload = {
        "asset": "USDJPY",
        "generated_utc": nowiso(),
        "boj_mof_risk": risk_summary,
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


def build_data_gap_signal(asset: str,
                          spot_price: Any,
                          spot_utc: str,
                          spot_retrieved: str,
                          leverage: float,
                          reasons: List[str],
                          display_spot: Optional[float],
                          diagnostics: Dict[str, Any]) -> Dict[str, Any]:
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
        "signal": "no entry",
        "probability": 0,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "rr": None,
        "leverage": leverage,
        "gates": {
            "mode": "data_gap",
            "required": ["data_integrity"],
            "missing": ["data_integrity"],
        },
        "session_info": {
            "open": None,
            "within_window": None,
            "weekday_ok": None,
            "status": "unavailable",
            "status_note": "Hiányzó adat",
        },
        "diagnostics": diagnostics,
        "reasons": reasons,
    }

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

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
        return pd.DataFrame(columns=["open","high","low","close"])
    df = pd.DataFrame(rows).sort_values("time").set_index("time")
    return df[["open","high","low","close"]]

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

def smt_penalty(asset: str) -> Tuple[int, Optional[str]]:
    path = os.path.join(PUBLIC_DIR, asset, "smt.json")
    data = load_json(path)
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

# ------------------------------ elemzés egy eszközre ---------------------------

def analyze(asset: str) -> Dict[str, Any]:
    outdir = os.path.join(PUBLIC_DIR, asset)
    os.makedirs(outdir, exist_ok=True)

    # 1) Bemenetek
    spot = load_json(os.path.join(outdir, "spot.json")) or {}
    k1m_raw = load_json(os.path.join(outdir, "klines_1m.json"))
    k5m_raw = load_json(os.path.join(outdir, "klines_5m.json"))
    k1h_raw = load_json(os.path.join(outdir, "klines_1h.json"))
    k4h_raw = load_json(os.path.join(outdir, "klines_4h.json"))

    k1m, k5m, k1h, k4h = as_df_klines(k1m_raw), as_df_klines(k5m_raw), as_df_klines(k1h_raw), as_df_klines(k4h_raw)

    spot_price = None
    spot_utc = "-"
    spot_retrieved = "-"
    if spot:
        spot_price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
        spot_utc = spot.get("utc") or spot.get("timestamp") or "-"
        spot_retrieved = spot.get("retrieved_at_utc") or spot.get("retrieved") or "-"

    now = datetime.now(timezone.utc)

    spot_ts_primary = parse_utc_timestamp(spot_utc)
    spot_ts_fallback = parse_utc_timestamp(spot_retrieved)
    spot_ts = spot_ts_primary or spot_ts_fallback
    spot_latency_sec: Optional[int] = None
    spot_stale_reason: Optional[str] = None
    spot_max_age = SPOT_MAX_AGE_SECONDS.get(asset, SPOT_MAX_AGE_SECONDS["default"])
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

    display_spot = safe_float(spot_price)
    k1m_closed = k1m.iloc[:-1] if len(k1m) > 1 else k1m.copy()
    k5m_closed = k5m.iloc[:-1] if len(k5m) > 1 else k5m.copy()
    k1h_closed = k1h.iloc[:-1] if len(k1h) > 1 else k1h.copy()
    k4h_closed = k4h.iloc[:-1] if len(k4h) > 1 else k4h.copy()

    last5_close: Optional[float] = None
    last5_closed_ts = df_last_timestamp(k5m_closed)
    if not k5m_closed.empty:
        try:
            last5_close = float(k5m_closed["close"].iloc[-1])
            if not np.isfinite(last5_close):
                last5_close = None
        except Exception:
            last5_close = None

    spot_issue_initial: Optional[str] = None
    if spot_price is None:
        spot_issue_initial = "Spot price missing"
    elif spot_stale_reason:
        spot_issue_initial = spot_stale_reason

    spot_source = "quote"
    spot_fallback_used = False
    spot_latency_notes: List[str] = []
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
            spot_latency_sec = 0 if delta.total_seconds() < 0 else int(delta.total_seconds())
        else:
            spot_latency_sec = None
        spot_stale_reason = None

    if spot_fallback_used:
        if spot_issue_initial:
            spot_latency_notes.append(
                f"spot: {spot_issue_initial.lower()} — 5m zárt gyertya árát használjuk"
            )
        else:
            spot_latency_notes.append("spot: 5m zárt gyertya árát használjuk")

    analysis_now = datetime.now(timezone.utc)

    intervention_summary: Optional[Dict[str, Any]] = None
    intervention_band: Optional[str] = None
    if asset == "USDJPY":
        intervention_config = load_intervention_config(outdir)
        news_flag = load_intervention_news_flag(outdir, intervention_config)
        irs_value, irs_band, irs_metrics = compute_irs_usdjpy(
            k1m_closed,
            k5m_closed,
            analysis_now,
            intervention_config,
            news_flag=news_flag,
        )
        since_utc = update_intervention_state(outdir, irs_band, analysis_now)
        iw_reasons = build_intervention_reasons(irs_metrics, intervention_config)
        policy = build_intervention_policy(irs_band, irs_metrics, intervention_config)
        score_breakdown = {
            "level_pressure": irs_metrics.get("components", {}).get("level_pressure", 0.0),
            "speed": irs_metrics.get("components", {}).get("speed", 0.0),
            "vol_spike": irs_metrics.get("components", {}).get("vol_spike", 0.0),
            "session": irs_metrics.get("components", {}).get("session", 0.0),
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
        intervention_band = irs_band
        save_intervention_asset_summary(outdir, intervention_summary)

    expected_delays = {"k1m": 60, "k5m": 300, "k1h": 3600, "k4h": 4*3600}
    tf_meta: Dict[str, Dict[str, Any]] = {}
    latency_flags: List[str] = []
    latency_flags.extend(spot_latency_notes)
    tf_inputs = [
        ("k1m", k1m, k1m_closed, os.path.join(outdir, "klines_1m.json")),
        ("k5m", k5m, k5m_closed, os.path.join(outdir, "klines_5m.json")),
        ("k1h", k1h, k1h_closed, os.path.join(outdir, "klines_1h.json")),
        ("k4h", k4h, k4h_closed, os.path.join(outdir, "klines_4h.json")),
    ]
    for key, df_full, df_closed, path in tf_inputs:
        last_raw = df_last_timestamp(df_full)
        last_closed = df_last_timestamp(df_closed)
        latency_sec = None
        if last_closed:
            latency_sec = int((now - last_closed).total_seconds())
            expected = expected_delays.get(key, 0)
            if expected and latency_sec > expected + 240:
                latency_flags.append(f"{key}: utolsó zárt gyertya {latency_sec//60} perc késésben van")
        tf_meta[key] = {
            "last_raw_utc": to_utc_iso(last_raw) if last_raw else None,
            "last_closed_utc": to_utc_iso(last_closed) if last_closed else None,
            "latency_seconds": latency_sec,
            "expected_max_delay_seconds": expected_delays.get(key),
            "source_mtime_utc": file_mtime(path),
        }

    tf_meta["spot"] = {
        "last_raw_utc": to_utc_iso(spot_ts) if spot_ts else None,
        "latency_seconds": spot_latency_sec,
        "expected_max_delay_seconds": spot_max_age,
        "source_mtime_utc": file_mtime(os.path.join(outdir, "spot.json")),
        "source": spot_source,
        "fallback_used": spot_fallback_used,
    }
    if spot_issue_initial:
        tf_meta["spot"]["original_issue"] = spot_issue_initial
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
        )
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    price_for_calc = last5_close

    if display_spot is None and price_for_calc is not None and np.isfinite(price_for_calc):
        display_spot = price_for_calc

    # 2) Bias 4H→1H (zárt 1h/4h)
    bias4h = bias_from_emas(k4h_closed)
    bias1h = bias_from_emas(k1h_closed)
    trend_bias = (
        "long"
        if (bias4h == "long" and bias1h != "short")
        else (
            "short"
            if (bias4h == "short" and bias1h != "long")
            else "neutral"
        )
    )

    if asset in {"EURUSD", "USDJPY", "NVDA", "SRTY"}:
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
    if asset in {"EURUSD", "USDJPY", "NVDA", "SRTY"} and desired_bias in {"long", "short"}:
        if desired_bias == "long":
            slope_sign_ok = regime_slope_signed >= slope_threshold
        else:
            slope_sign_ok = regime_slope_signed <= -slope_threshold
        regime_ok = regime_ok and slope_sign_ok
    # 3) HTF sweep (zárt 1h/4h)
    sw1h = detect_sweep(k1h_closed, 24); sw4h = detect_sweep(k4h_closed, 24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]

    # 4) 5M BOS a trend irányába (zárt 5m)
    bos5m_long = detect_bos(k5m_closed, "long")
    bos5m_short = detect_bos(k5m_closed, "short")

    bos1h_long = detect_bos(k1h_closed, "long")
    bos1h_short = detect_bos(k1h_closed, "short")

    # 5) ATR szűrő (relatív) — a stabil árhoz viszonyítjuk (zárt 5m)
    atr5 = atr(k5m_closed).iloc[-1]
    rel_atr = float(atr5 / price_for_calc) if (atr5 and price_for_calc) else float("nan")
    atr_threshold = atr_low_threshold(asset)
    atr_abs_min = ATR_ABS_MIN.get(asset)
    atr_abs_ok = True
    if atr_abs_min is not None:
        try:
            atr_abs_ok = float(atr5) >= atr_abs_min
        except Exception:
            atr_abs_ok = False
    atr_ok = not (np.isnan(rel_atr) or rel_atr < atr_threshold or not atr_abs_ok)

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

    struct_retest_long  = structure_break_with_retest(k5m_closed, "long", MOMENTUM_BOS_LB)
    struct_retest_short = structure_break_with_retest(k5m_closed, "short", MOMENTUM_BOS_LB)

    micro_bos_long = micro_bos_with_retest(k1m_closed, k5m_closed, "long")
    micro_bos_short = micro_bos_with_retest(k1m_closed, k5m_closed, "short")

    effective_bias = trend_bias
    bias_override_used = False
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

    if asset == "SRTY" and effective_bias != "short":
        effective_bias = "neutral"
    if asset == "NVDA" and effective_bias == "neutral" and bias1h in {"long", "short"} and bias4h == bias1h and regime_ok:
        effective_bias = bias1h

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

    structure_ok_long = bool(bos5m_long or struct_retest_long)
    structure_ok_short = bool(bos5m_short or struct_retest_short)
    nvda_cross_long = nvda_cross_short = False
    if asset == "NVDA":
        ema9_5m = ema(k5m_closed["close"], 9)
        ema21_5m = ema(k5m_closed["close"], 21)
        nvda_cross_long = ema_cross_recent(ema9_5m, ema21_5m, bars=7, direction="long")
        nvda_cross_short = ema_cross_recent(ema9_5m, ema21_5m, bars=7, direction="short")
    structure_gate = False
    if effective_bias == "long":
        structure_gate = structure_ok_long
        if asset == "NVDA":
            structure_gate = structure_gate or nvda_cross_long
    elif effective_bias == "short":
        structure_gate = structure_ok_short
        if asset == "NVDA":
            structure_gate = structure_gate or nvda_cross_short
        if asset == "SRTY":
            structure_gate = bool(bos5m_short)
    else:
        structure_gate = False

    # 7) P-score (egyszerű súlyozás)
    P, reasons = 20, []
    if effective_bias != "neutral":
        P += 20
        if bias_override_used:
            reasons.append(f"Bias override: 1h trend {effective_bias} + momentum támogatás")
        else:
            reasons.append(f"Bias(4H→1H)={effective_bias}")
    if regime_ok:
        P += 8
        reasons.append("Regime ok (EMA21 slope)")
    if swept:
        P += 15
        reasons.append("HTF sweep ok")
    struct_retest_active = ((effective_bias == "long" and struct_retest_long) or
                            (effective_bias == "short" and struct_retest_short))
    if bos5m:
        P += 18
        reasons.append("5M BOS trendirányba")
    elif struct_retest_active:
        P += 12
        reasons.append("5m szerkezeti törés + retest a trend irányába")
    elif asset == "NVDA" and (
        (effective_bias == "long" and nvda_cross_long)
        or (effective_bias == "short" and nvda_cross_short)
    ):
        P += 15
        reasons.append("5m EMA9×21 momentum kereszt megerősítés")
    elif micro_bos_active:
        if atr_ok and not np.isnan(rel_atr) and rel_atr >= max(atr_threshold, MOMENTUM_ATR_REL):
            P += MICRO_BOS_P_BONUS
            reasons.append("Micro BOS megerősítés (1m szerkezet + magas ATR)")
        else:
            reasons.append("1m BOS + 5m retest — várjuk a 5m megerősítést")
    if fib_ok:
        P += 20
        reasons.append("Fib zóna konfluencia (0.618–0.886)")
    elif ema21_dist_ok:
        P += 12
        reasons.append("Ár 1h EMA21 zónában (ATR tolerancia)")
    if atr_ok:
        P += 9
        reasons.append("ATR rendben")

    smt_pen, smt_reason = smt_penalty(asset)
    if smt_pen and smt_reason:
        P -= smt_pen
        reasons.append(f"SMT büntetés −{smt_pen}% ({smt_reason})")
    P = max(0, min(100, P))

    # --- Kapuk (liquidity = Fib zóna VAGY sweep) + session + regime ---
    session_ok_flag, session_meta = session_state(asset)

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

    anchor_state = current_anchor_state()
    anchor_record = anchor_state.get(asset.upper()) if isinstance(anchor_state, dict) else None
    anchor_bias = None
    anchor_timestamp = None
    anchor_price_state: Optional[float] = None
    if isinstance(anchor_record, dict):
        side_raw = (anchor_record.get("side") or "").lower()
        if side_raw == "buy":
            anchor_bias = "long"
        elif side_raw == "sell":
            anchor_bias = "short"
        anchor_timestamp = anchor_record.get("timestamp")
        anchor_price_state = safe_float(anchor_record.get("price"))
    if asset in {"EURUSD", "USDJPY"}:
        liquidity_ok_base = bool(fib_ok)
    elif asset == "SRTY":
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
    candidate_dir = effective_bias if effective_bias in ("long", "short") else (bias1h if bias1h in ("long", "short") else None)
    strong_momentum = False
    if candidate_dir == "long":
        strong_momentum = bool(bos5m_long or struct_retest_long or micro_bos_long)
        if asset == "NVDA":
            strong_momentum = strong_momentum or nvda_cross_long
    elif candidate_dir == "short":
        strong_momentum = bool(bos5m_short or struct_retest_short or micro_bos_short)
        if asset == "NVDA":
            strong_momentum = strong_momentum or nvda_cross_short
    liquidity_relaxed = False
    liquidity_ok = liquidity_ok_base
    if asset != "SRTY":
        if not liquidity_ok_base and strong_momentum:
            high_atr_push = bool(
                atr_ok and not np.isnan(rel_atr)
                and rel_atr >= max(atr_threshold * 1.3, MOMENTUM_ATR_REL)
            )
            if high_atr_push or P >= 65:
                liquidity_ok = True
                liquidity_relaxed = True

    p_score_min_local = P_SCORE_MIN
    if asset == "USDJPY" and intervention_band in {"HIGH", "IMMINENT"} and effective_bias == "long":
        p_score_min_local += INTERVENTION_P_SCORE_ADD
        note = f"Intervention Watch: USDJPY long P-score küszöb +{INTERVENTION_P_SCORE_ADD} (IRS {intervention_summary['irs']} {intervention_band})" if intervention_summary else None
        if note and note not in reasons:
            reasons.append(note)

    tp_net_threshold = tp_net_min_for(asset)
    tp_net_pct_display = f"{tp_net_threshold * 100:.2f}".rstrip("0").rstrip(".")
    tp_net_label = f"tp1_net>=+{tp_net_pct_display}%"

    core_rr_min = CORE_RR_MIN.get(asset, CORE_RR_MIN["default"])
    momentum_rr_min = MOMENTUM_RR_MIN.get(asset, MOMENTUM_RR_MIN["default"])

    liquidity_label = "liquidity(fib|sweep|ema21|retest)"
    if asset in {"EURUSD", "USDJPY", "SRTY"}:
        liquidity_label = "liquidity(fib zone)"
    elif asset == "NVDA":
        liquidity_label = "liquidity(fib|retest)"

    core_required = [
        "session",
        "regime",
        "bias",
        "bos5m",
        liquidity_label,
        "atr",
        f"rr_math>={core_rr_min:.1f}",
        "tp_min_profit",
        "min_stoploss",
        tp_net_label,
    ]

    conds_core = {
        "session": bool(session_ok_flag),
        "regime":  bool(regime_ok),
        "bias":    effective_bias in ("long","short"),
        "bos5m":   bool(structure_gate),
        "liquidity": liquidity_ok,
        "atr":     bool(atr_ok),
    }
    base_core_ok = all(v for k, v in conds_core.items() if k != "bos5m")
    bos_gate_ok = conds_core["bos5m"] or micro_bos_active
    can_enter_core = (P >= p_score_min_local) and base_core_ok and bos_gate_ok
    missing_core = [k for k, v in conds_core.items() if not v]
    if P < p_score_min_local:
        missing_core.append(f"P_score>={p_score_min_local}")
    if micro_bos_active and not conds_core["bos5m"]:
        if "bos5m" not in missing_core:
            missing_core.append("bos5m")
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
        "bos5m",
        "atr",
        f"rr_math>={momentum_rr_min:.1f}",
        "tp_min_profit",
        "min_stoploss",
        tp_net_label,
    ]
    missing_mom: List[str] = []
    mom_trigger_desc: Optional[str] = None

    if asset in ENABLE_MOMENTUM_ASSETS:
        direction = effective_bias if effective_bias in {"long", "short"} else None
        if not session_ok_flag:
            missing_mom.append("session")
        if not regime_ok:
            missing_mom.append("regime")
        if direction is None:
            missing_mom.append("bias")

        if asset == "NVDA" and direction is not None:
            mom_atr_ok = not np.isnan(rel_atr) and rel_atr >= NVDA_MOMENTUM_ATR_REL and atr_abs_ok
            cross_flag = nvda_cross_long if direction == "long" else nvda_cross_short
            if session_ok_flag and regime_ok and mom_atr_ok and cross_flag:
                mom_dir = "buy" if direction == "long" else "sell"
                mom_trigger_desc = "EMA9×21 momentum cross"
                missing_mom = []
            else:
                if not mom_atr_ok:
                    missing_mom.append("atr")
                if not cross_flag:
                    missing_mom.append("momentum_trigger")
        elif asset == "SRTY" and direction == "short":
            h, m = now_utctime_hm()
            minute = h * 60 + m
            window_ok = _min_of_day(13, 30) <= minute <= _min_of_day(15, 0)
            mom_atr_ok = not np.isnan(rel_atr) and rel_atr >= SRTY_MOMENTUM_ATR_REL and atr_abs_ok
            bos_ok = bool(bos5m_short)
            if session_ok_flag and regime_ok and mom_atr_ok and bos_ok and window_ok:
                mom_dir = "sell"
                mom_trigger_desc = "Momentum window 13:30–15:00 UTC"
                missing_mom = []
            else:
                if not window_ok:
                    missing_mom.append("momentum_trigger")
                if not mom_atr_ok:
                    missing_mom.append("atr")
                if not bos_ok:
                    missing_mom.append("bos5m")
        else:
            if asset == "SRTY" and direction != "short":
                missing_mom.append("bias")

    # 8) Döntés + szintek (RR/TP matek) — core vagy momentum
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)
    mode = "core"
    missing = list(missing_core)
    required_list: List[str] = list(core_required)
    min_stoploss_ok = True
    tp1_net_pct_value: Optional[float] = None

    def compute_levels(decision_side: str, rr_required: float):
        nonlocal entry, sl, tp1, tp2, rr, missing, min_stoploss_ok, tp1_net_pct_value
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
            tp1 = entry + TP1_R * risk
            tp2 = entry + TP2_R * risk
            tp1_dist = tp1 - entry
            ok_math = (sl < entry < tp1 <= tp2)
        else:
            base_sl = hi5 if hi5 is not None else (entry + atr5_val)
            sl = base_sl + buf
            risk = sl - entry
            if risk < 0:
                sl = entry + buf
                risk = sl - entry
            tp1 = entry - TP1_R * risk
            tp2 = entry - TP2_R * risk
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
            tp1 = entry + TP1_R * risk
            tp2 = entry + TP2_R * risk
            rr  = (tp2 - entry) / risk
            tp1_dist = tp1 - entry
            ok_math = ok_math and (sl < entry < tp1 <= tp2)
            gross_pct = tp1_dist / entry
        else:
            tp1 = entry - TP1_R * risk
            tp2 = entry - TP2_R * risk
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
            if not compute_levels(decision, core_rr_min):
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
            if not compute_levels(decision, momentum_rr_min):
                decision = "no entry"
            else:
                reason_msg = "Momentum override"
                if mom_trigger_desc:
                    reason_msg += f" ({mom_trigger_desc})"
                reasons.append(reason_msg)
                reasons.append("Momentum: rész-realizálás javasolt 2.5R-n")
                P = max(P, 75)
                if tp1_net_pct_value is not None:
                    msg_net = f"TP1 nettó profit ≈ {tp1_net_pct_value*100:.2f}%"
                    if msg_net not in reasons:
                        reasons.append(msg_net)
        elif asset in ENABLE_MOMENTUM_ASSETS and missing_mom:
            mode = "momentum"
            required_list = list(mom_required)
            missing = list(dict.fromkeys(missing_mom))  # uniq

    # 9) Session override + mentés: signal.json
    if latency_flags:
        for flag in latency_flags:
            msg = f"Diagnosztika: {flag}"
            if msg not in reasons:
                reasons.append(msg)

    if intervention_summary and intervention_summary.get("irs", 0) >= 40:
        highlight = f"BoJ/MoF Intervention Watch: IRS {intervention_summary['irs']} ({intervention_summary['band']})"
        if highlight not in reasons:
            reasons.insert(0, highlight)
    if asset == "USDJPY" and intervention_band:
        if decision == "buy" and intervention_band in {"HIGH", "IMMINENT"}:
            block_msg = (
                f"Intervention Watch: IRS {intervention_summary['irs']} ({intervention_band}) → USDJPY long belépés tiltva"
                if intervention_summary else "Intervention Watch: USDJPY long belépés tiltva"
            )
            if block_msg not in reasons:
                reasons.insert(0, block_msg)
            decision = "no entry"
            entry = sl = tp1 = tp2 = rr = None
            if "intervention_watch" not in missing:
                missing.append("intervention_watch")
        elif decision == "sell" and intervention_band == "IMMINENT":
            note_short = "Intervention Watch: új short csak 0.5×ATR5m pullback + RR≥2.5 után"
            if note_short not in reasons:
                reasons.append(note_short)
        if intervention_band == "IMMINENT":
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

    position_note = derive_position_management_note(
        asset,
        session_meta,
        regime_ok,
        effective_bias,
        structure_flag,
        atr1h,
        anchor_bias,
        anchor_timestamp,
    )
    if position_note and position_note not in reasons:
        reasons.append(position_note)

    decision_obj = {
        "asset": asset,
        "ok": True,
        "retrieved_at_utc": nowiso(),
        "source": "Twelve Data (lokális JSON)",
        "spot": {"price": display_spot, "utc": spot_utc, "retrieved_at_utc": spot_retrieved},
        "signal": decision,
        "probability": int(P),
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": (round(rr,2) if rr else None),
        "leverage": lev,
        "gates": {
            "mode": mode,
            "required": required_list,
            "missing": missing,
        },
        "session_info": session_meta,
        "diagnostics": diagnostics_payload(tf_meta, source_files, latency_flags),
        "reasons": (reasons + ([f"missing: {', '.join(missing)}"] if missing else [])) or ["no signal"],
    }
    decision_obj["ema21_slope_1h"] = regime_slope_signed
    decision_obj["ema21_slope_threshold"] = EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT)
    decision_obj["ema21_relation_1h"] = ema21_relation
    decision_obj["last_swing_high_1h"] = float(move_hi) if move_hi is not None else None
    decision_obj["last_swing_low_1h"] = float(move_lo) if move_lo is not None else None
    decision_obj["last_close_1h"] = float(last_close_1h) if np.isfinite(last_close_1h) else None
    decision_obj["bos_5m_dir"] = structure_flag
    decision_obj["atr1h"] = atr1h
    decision_obj["invalid_levels"] = {
        "buy": invalid_level_buy,
        "sell": invalid_level_sell,
    }
    decision_obj["invalid_buffer_abs"] = float(invalid_buffer) if invalid_buffer is not None else None
    if position_note:
        decision_obj["position_management"] = position_note
    active_position_meta = {
        "ema21_slope_abs": regime_val,
        "ema21_slope_signed": regime_slope_signed,
        "ema21_slope_threshold": EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_DEFAULT),
        "ema21_relation": ema21_relation,
        "structure_5m": structure_flag,
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
    }
    if anchor_bias:
        active_position_meta["anchor_side"] = anchor_bias
    if anchor_price_state is not None:
        active_position_meta["anchor_price"] = anchor_price_state
    if anchor_timestamp:
        active_position_meta["anchor_timestamp"] = anchor_timestamp
    decision_obj["active_position_meta"] = active_position_meta
    if intervention_summary:
        decision_obj["intervention_watch"] = intervention_summary

    if decision in ("buy", "sell"):
        anchor_price = entry or spot_price
        if anchor_price is None and np.isfinite(last_close_1h):
            anchor_price = float(last_close_1h)
        try:
            global ANCHOR_STATE_CACHE
            ANCHOR_STATE_CACHE = record_anchor(
                asset,
                decision,
                price=anchor_price,
                timestamp=decision_obj["retrieved_at_utc"],
            )
        except Exception:
            # Anchor frissítés hibája ne állítsa meg az elemzést.
            pass
    save_json(os.path.join(outdir, "signal.json"), decision_obj)
    return decision_obj
# ------------------------------- főfolyamat ------------------------------------

def main():
    summary = {
        "ok": True,
        "generated_utc": nowiso(),
        "assets": {},
        "latency_flags": [],
        "troubleshooting": list(REFRESH_TIPS),
    }
    for asset in ASSETS:
        try:
            res = analyze(asset)
            summary["assets"][asset] = res
            diag = res.get("diagnostics", {}) if isinstance(res, dict) else {}
            flags = diag.get("latency_flags") if isinstance(diag, dict) else None
            if flags:
                summary["latency_flags"].extend(flags)
        except Exception as e:
            summary["assets"][asset] = {"asset": asset, "ok": False, "error": str(e)}
    save_json(os.path.join(PUBLIC_DIR, "analysis_summary.json"), summary)

    html = "<!doctype html><meta charset='utf-8'><title>Analysis Summary</title>"
    html += "<h1>Analysis Summary (TD-only)</h1>"
    html += "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>"
    with open(os.path.join(PUBLIC_DIR, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()



