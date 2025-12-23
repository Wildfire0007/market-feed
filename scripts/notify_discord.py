#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Esemény-alapú Discord riasztó + óránkénti összefoglaló (per-eszköz panelek)

Stílus:
- Külön embed minden eszköznek, saját emojival.
- Félkövér eszköznév a címben. A leírás elején 🟢/🔴 ikon.
- BUY/SELL = zöld sáv, NO ENTRY = piros sáv, stabilizálás alatt = sárga sáv.
- RR/TP/SL/Entry számok backtick-ben.

Küldés:
- STABIL (>= STABILITY_RUNS) BUY/SELL ➜ "normal"
- Ellenirányú stabil jel flip ➜ "flip"
- Korábban küldött BUY/SELL stabilan NO ENTRY ➜ "invalidate"
- ÓRÁNKÉNTI HEARTBEAT (minden órában), akkor is, ha nincs riasztás.
  Ha adott órában már ment event (normal/flip/invalidate), külön heartbeat nem megy ki.
  --force / --manual kapcsolóval (vagy DISCORD_FORCE_NOTIFY=1) kézi futtatáskor is kimegy az összefoglaló.
  Kézi futtatáskor elfogadjuk a "manual"/"force" kulcsszavakat is flag nélkül.

ENV:
- DISCORD_WEBHOOK_URL
- DISCORD_WEBHOOK_URL_LIVE (opcionális: #🚨-live-signals)
- DISCORD_WEBHOOK_URL_MANAGEMENT (opcionális: #💼-management)
- DISCORD_WEBHOOK_URL_MARKET_SCAN (opcionális: #📊-market-scan)
- DISCORD_COOLDOWN_MIN (perc, default 5)
- DISCORD_FORCE_NOTIFY=1 ➜ cooldown figyelmen kívül hagyása + összefoglaló kényszerítése
- DISCORD_FORCE_HEARTBEAT=1 ➜ csak az összefoglalót kényszerítjük (cooldown marad)
"""

import os, json, sys, logging, requests, time
from uuid import uuid4
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
import numpy as np
from typing import Iterable, Optional, Set, Tuple, Dict, Any, List
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo  # Py3.9+

# --- Ensure repository root on sys.path when executed as a script ---
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_anchor import load_anchor_state, touch_anchor
from config.analysis_settings import ASSETS as CONFIG_ASSETS
from config import analysis_settings as settings
from logging_utils import ensure_json_stream_handler
from reports.pipeline_monitor import (
    compute_run_timing_deltas,
    load_pipeline_payload,
)
import position_tracker

LOGGER = logging.getLogger("market_feed.notify")
ensure_json_stream_handler(LOGGER, static_fields={"component": "notify"})

PUBLIC_DIR = "public"
ASSETS: List[str] = list(CONFIG_ASSETS)
STATE_ARCHIVE_PATH = f"{PUBLIC_DIR}/_notify_state.archive.json"
ACTIVE_ASSET_SET: Set[str] = {asset.upper() for asset in ASSETS}
DEFAULT_ASSET_STATE: Dict[str, Any] = {
    "last": "no entry",
    "count": 0,
    "last_sent": None,
    "last_sent_decision": None,
    "last_sent_mode": None,
    "last_sent_known": False,
    "cooldown_until": None,
}

ENTRY_GATE_STATS_PATH = _REPO_ROOT / PUBLIC_DIR / "debug" / "entry_gate_stats.json"
PIPELINE_MONITOR_PATH = _REPO_ROOT / PUBLIC_DIR / "monitoring" / "pipeline_timing.json"

# ---- Active position helper config ----
TDSTATUS_PATH = f"{PUBLIC_DIR}/tdstatus.json"
EIA_OVERRIDES_PATH = f"{PUBLIC_DIR}/USOIL/eia_schedule_overrides.json"

EMA21_SLOPE_MIN = 0.0008  # 0.08%
ATR_TRAIL_MIN_ABS = 0.15

ACTIVE_POSITION_STATE_PATH = f"{PUBLIC_DIR}/_active_position_state.json"

# Baseline watcher overrides; extended with every configured asset at runtime.
_ACTIVE_WATCHER_OVERRIDES = {
    "USOIL": {
        "invalid_buffer_abs": 0.15,
        "atr_rel_min_5m": 0.0007,
        "event": {
            "name": "EIA WPSR",
            "schedule": "Szerda 10:30 ET",
            "pre_window_min": 60,
        },
    },
    "GOLD_CFD": {
        "invalid_buffer_abs": 2.0,
        "atr_rel_min_5m": 0.0007,
        "event": None,
    },
}


def _build_active_watcher_config() -> Dict[str, Any]:
    assets_cfg = {asset.upper(): {} for asset in ASSETS}
    assets_cfg.update(_ACTIVE_WATCHER_OVERRIDES)
    return {
        "common": {
            "ema21_slope_min_abs": 0.0008,
            "atr_trail_k": (0.6, 1.0),
            "bos_tf": "5m",
            "rollover_warn_gmt": "21:00",
            "rollover_window_min": 20,
            "send_on_state_change_only": True,
        },
        "assets": assets_cfg,
    }
  
EXIT_DEDUP_MINUTES_BY_ASSET = {
    "EURUSD": 30,
}


def _dedup_minutes_for_exit(asset: str) -> int:
    return int(EXIT_DEDUP_MINUTES_BY_ASSET.get(asset, 0))


@lru_cache(maxsize=1)
def _signal_stability_config() -> Dict[str, Any]:
    cfg = settings.load_config().get("signal_stability") or {}
    return cfg if isinstance(cfg, dict) else {}


def _resolve_asset_value(config_map: Any, asset: str, default: int) -> int:
    if isinstance(config_map, dict):
        try:
            return int(config_map.get(asset, config_map.get("default", default)))
        except (TypeError, ValueError):
            return int(default)
    return int(default)


# ---- Mobil-optimalizált kártyák segédfüggvényei ----
HB_TZ = timezone.utc  # Alapértelmezés, ha nincs pytz
try:
    HB_TZ = ZoneInfo("Europe/Budapest")
except Exception:
    pass

ASSET_EMOJIS = {
    "EURUSD": "💶",
    "USDJPY": "💴",
    "GBPUSD": "💷",
    "BTCUSD": "🚀",
    "ETHUSD": "💎",
    "GOLD_CFD": "🥇",
    "XAUUSD": "🥇",
    "XAGUSD": "🥈",
    "USOIL": "🛢️",
    "WTI": "🛢️",
    "NVDA": "🤖",
    "TSLA": "🚗",
    "SPX500": "📈",
    "NAS100": "💻",
}

COLORS = {
    "LONG": 0x2ECC71,
    "SHORT": 0xE74C3C,
    "WAIT": 0xF1C40F,
    "NO": 0x95A5A6,
    "FLIP": 0xE67E22,
    "A": 0x2ECC71,
    "B": 0xF39C12,
    "C": 0xBDC3C7,
}


def _get_emoji(asset: str) -> str:
    return ASSET_EMOJIS.get((asset or "").upper(), "📉")


def _translate_market_closed_reason(reason: Optional[str]) -> str:
    """Magyar nyelvű, felhasználóbarát indok a piac zártságára."""

    if not reason:
        return "Ismeretlen ok"

    reason_key = str(reason).strip().lower()
    translations = {
        "weekend": "Hétvége",
        "outside_hours": "Kereskedési időn kívül",
        "holiday": "Ünnepnap",
        "market_holiday": "Tőzsdei szünnap",
        "status_profile_forced": "Profil által lezárva",
        "maintenance": "Karbantartás",
        "no_data": "Nincs friss adat",
        "frozen": "Jegelve",
    }

    if reason_key in translations:
        return translations[reason_key]

    humanized = reason_key.replace("_", " ").strip()
    return humanized.capitalize() if humanized else "Ismeretlen ok"


 



def _tracked_levels_from_manual_positions(
    asset: str, manual_positions: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    entry = None
    if isinstance(manual_positions, dict):
        candidate = manual_positions.get(asset)
        entry = candidate if isinstance(candidate, dict) else None
    if not isinstance(entry, dict):
        return {}
    return {
        key: entry.get(key)
        for key in ("entry", "sl", "tp2", "opened_at_utc", "side")
        if entry.get(key) is not None
    }


def _format_manual_position_line(
    asset: str, position_state: Dict[str, Any], tracked_levels: Dict[str, Any]
) -> Optional[str]:
    if not position_state.get("has_position"):
        return None

    side = (position_state.get("side") or tracked_levels.get("side") or "").lower()
    side_txt = "long" if side in {"buy", "long"} else "short" if side in {"sell", "short"} else "open"

    entry = tracked_levels.get("entry") or position_state.get("entry")
    sl = tracked_levels.get("sl") or position_state.get("sl")
    tp2 = tracked_levels.get("tp2") or position_state.get("tp2")
    opened_at = tracked_levels.get("opened_at_utc") or position_state.get("opened_at_utc")

    parts: List[str] = []
    if opened_at:
        parts.append(f"opened_at: {opened_at}")
    if entry is not None:
        parts.append(f"Entry {format_price(entry, asset)}")
    if sl is not None:
        parts.append(f"SL {format_price(sl, asset)}")
    if tp2 is not None:
        parts.append(f"TP2 {format_price(tp2, asset)}")

    suffix = " — " + " • ".join(parts) if parts else ""
    return f"Pozíciómenedzsment: aktív {side_txt} pozíció{suffix}"
  
def draw_progress_bar(value: float, length: int = 10) -> str:
    """ASCII sáv: [■■■■■■■□□□]"""

    try:
        pct = max(0.0, min(1.0, float(value) / 100.0))
        filled = int(round(length * pct))
        inner = "■" * filled + "□" * (length - filled)
        return f"[{inner}]"
    except Exception:
        return "[" + ("□" * length) + "]"


def format_price(val: Any, asset: str) -> str:
    """Eszköz-specifikus árformázás"""

    if val is None:
        return "n/a"
    try:
        fval = float(val)
        upper_asset = (asset or "").upper()
        if any(x in upper_asset for x in ("JPY", "NVDA", "USOIL", "GOLD")):
            return f"{fval:,.2f}"
        if "BTC" in upper_asset:
            return f"{fval:,.0f}"
        return f"{fval:,.4f}"
    except Exception:
        return str(val)


def translate_reasons(missing_list: List[str]) -> str:
    """Technikai kulcsok magyarra fordítása"""

    map_dict = {
        "atr": "Alacsony volatilitás (ATR)",
        "atr_gate": "ATR küszöb alatt / felett",
        "spread": "Túl magas spread",
        "spread_gate": "Spread kapu blokkol",
        "bias": "Trend (Bias) semleges/ellentétes",
        "regime": "Piaci rezsim hiba",
        "choppy": "Oldalazás (Choppy)",
        "session": "Piac zárva",
        "liquidity": "Likviditás hiány (Fib/Sweep)",
        "p_score": "Alacsony P-score",
        "structure": "Struktúra hiba",
        "intervention_watch": "Beavatkozási figyelés",
        "no_chase": "Túl késő (No Chase)",
        "structure(2of3)": "Struktúra kapu (2/3 komponens) nem teljesült",
      
        # Momentum + order-flow
        "momentum_trigger": "Momentum trigger hiányzik",
        "ofi": "Order-flow (OFI) megerősítés hiányzik",

        # Precision / trigger kapuk
        "triggers": "Core belépő jel hiányzik",
        "precision_flow_alignment": "Precision: order-flow nincs összhangban",
        "precision_trigger_sync": "Precision: trigger szinkronra vár",

        # RR / TP / SL / range
        "intraday_range_guard": "Napi tartomány-védelem (range guard) blokkol",
        "min_stoploss": "Stop-loss feltétel nem teljesül",

        # Meta
        "p_score": "Alacsony P-score",
        "intervention_watch": "Beavatkozási figyelés aktív",
        "choppy": "Oldalazó (choppy) piaci szakasz",
        "no_chase": "Ne üldözd az árat (No Chase szabály)",
    }

    clean_reasons: List[str] = []
    seen = set()

    for missing in missing_list:
        key = str(missing or "").strip()
        if not key:
            continue

        # RR-hez kapcsolódó kulcsok (pl. rr_math>=1.6)
        if "rr_" in key:
            txt = "Gyenge RR arány / RR kapu nem teljesül"

        # TP / profit-kapuk (pl. tp_min_profit, tp1_net>=+75%)
        elif key.startswith("tp"):
            txt = "Kicsi profit potenciál / TP kapu nem teljesül"

        # Precision kapuk (pl. precision_score>=52, precision_flow_alignment, precision_trigger_sync)
        elif key.startswith("precision_"):
            if "flow_alignment" in key:
                txt = "Precision: order-flow nincs összhangban"
            elif "trigger_sync" in key:
                txt = "Precision: trigger szinkronra vár"
            elif "score" in key:
                txt = "Precision: P-score küszöb nem teljesül"
            else:
                txt = "Precision kapu feltételei nem teljesülnek"

        else:
            # Ha van konkrét magyar fordítás, azt használjuk, különben a nyers kulcsot
            txt = map_dict.get(key, key)

        if txt and txt not in seen:
            clean_reasons.append(txt)
            seen.add(txt)

    return ", ".join(clean_reasons) if clean_reasons else "—"


def extract_regime(signal_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Fő piaci rezsim (TREND / CHOPPY / BALANCED …) és az ADX érték kiolvasása.

    Elsődlegesen az entry_thresholds.adx_regime mezőt használjuk,
    ha ez nincs, akkor a dynamic_score_engine.regime_penalty.label mezőt.
    """
    if not isinstance(signal_data, dict):
        return None, None

    et = signal_data.get("entry_thresholds") or {}

    # ADX érték
    adx_raw = et.get("adx_value")
    try:
        adx_value = float(adx_raw) if adx_raw is not None else None
    except (TypeError, ValueError):
        adx_value = None

    # Rezsim string
    adx_regime = str(
        et.get("adx_regime") or et.get("adx_regime_initial") or ""
    ).strip().lower()

    label: Optional[str] = None
    if adx_regime:
        if "trend" in adx_regime:
            label = "TREND"
        elif "choppy" in adx_regime or "range" in adx_regime:
            label = "CHOPPY"
        elif "balanced" in adx_regime:
            label = "BALANCED"
        else:
            label = adx_regime.upper()
    else:
        dse = et.get("dynamic_score_engine") or {}
        reg_pen = dse.get("regime_penalty") or {}
        raw_label = reg_pen.get("label") or ""
        if raw_label:
            label = str(raw_label).strip().upper()

    return label, adx_value

def classify_setup(p_score: float, gates: Dict[str, Any], decision: str) -> Dict[str, Any]:
    """Felhasználói szabály alapú A/B/C setup besorolás."""

    missing = gates.get("missing", []) if isinstance(gates, dict) else []
    mode = (gates or {}).get("mode", "core") if isinstance(gates, dict) else "core"

    is_active_signal = (decision or "").upper() in {"BUY", "SELL"}

    if p_score >= 80 and is_active_signal:
        return {
            "grade": "A",
            "title": "A Setup (Prémium)",
            "action": "Teljes méret, agresszív menedzsment.",
            "color": COLORS["A"],
        }

    soft_blockers = ["atr", "bias", "regime", "choppy"]
    is_soft_blocked = bool(missing) and all(m in soft_blockers for m in missing)

    if p_score >= 30:
        if is_active_signal:
            return {
                "grade": "B",
                "title": "B Setup (Standard)",
                "action": "Fél pozícióméret, szigorúbb Stop Loss.",
                "color": COLORS["B"],
            }
        if is_soft_blocked:
            return {
                "grade": "B",
                "title": "B Setup (Sikertelen)",
                "action": "Fél pozíció (ha manuálisan belépsz).",
                "color": COLORS["B"],
            }

    if p_score >= 25:
        return {
            "grade": "C",
            "title": "C Setup (Spekulatív)",
            "action": "Negyed méret, vagy csak megerősítésre.",
            "color": COLORS["C"],
        }

    return {
        "grade": "-",
        "title": "Nincs Setup",
        "action": "Kivárás.",
        "color": COLORS["NO"],
    }


def resolve_setup_grade_for_signal(signal_data: Dict[str, Any], decision: str) -> Optional[str]:
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
    setup_meta = classify_setup(p_score, gates_for_setup if isinstance(gates_for_setup, dict) else {}, decision)
    grade = setup_meta.get("grade") if isinstance(setup_meta, dict) else None
    if isinstance(grade, str) and grade.strip() in {"A", "B", "C"}:
        return grade.strip()
    return None


def extract_trade_levels(signal_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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

    return entry, sl, tp2


def build_mobile_embed_for_asset(
    asset: str,
    state: Dict[str, Any],
    signal_data: Dict[str, Any],
    decision: str,
    mode: str,
    is_stable: bool,
    is_flip: bool,
    is_invalidate: bool,
    *,
    kind: str = "normal",
    manual_positions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mobil-optimalizált kereskedési kártya."""

    session = (signal_data or {}).get("session_info", {})
    if not session.get("open", True):
        raw_reason = session.get("market_closed_reason") or "Hétvége"
        reason = _translate_market_closed_reason(raw_reason)
        next_open = session.get("next_open_utc", "Ismeretlen")
        return {
            "title": f"{_get_emoji(asset)} {asset}",
            "description": f"🔴 **PIAC ZÁRVA**\nOk: {reason}\nNyitás: {next_open}",
            "color": 0x2C3E50,
        }

    p_score = signal_data.get("probability_raw", 0) if isinstance(signal_data, dict) else 0
    spot = (signal_data.get("spot") or {}).get("price") if isinstance(signal_data, dict) else None
    ts_raw = signal_data.get("retrieved_at_utc") if isinstance(signal_data, dict) else None
    entry_diag_raw = signal_data.get("entry_diagnostics") if isinstance(signal_data, dict) else {}
    entry_diag = entry_diag_raw if isinstance(entry_diag_raw, dict) else {}

    position_diag_raw = (
        signal_data.get("position_diagnostics") if isinstance(signal_data, dict) else {}
    )
    position_diag = position_diag_raw if isinstance(position_diag_raw, dict) else {}
    tracked_levels = (
        signal_data.get("tracked_levels") if isinstance(signal_data, dict) else {}
    )
    tracked_levels = tracked_levels if isinstance(tracked_levels, dict) else {}
  
    try:
        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        local_time = dt.astimezone(HB_TZ).strftime("%H:%M")
    except Exception:
        local_time = "--:--"
    
    gates_for_setup = (signal_data or {}).get("gates", {})
    if isinstance(gates_for_setup, dict) and isinstance(entry_diag, dict):
        gates_for_setup = {**gates_for_setup, "missing": entry_diag.get("missing", [])}
    setup_info = classify_setup(float(p_score or 0), gates_for_setup, decision)

    # --- Piaci rezsim (CHOPPY / TREND) kiolvasása és magyar szöveg készítése ---
    regime_label, adx_value = extract_regime(signal_data)
    regime_line = None
    if regime_label or adx_value is not None:
        regime_icons = {
            "TREND": "📈",
            "CHOPPY": "🌊",
            "BALANCED": "⚖️",
        }
        icon = regime_icons.get(regime_label or "", "📊")

        extra_hu = ""
        if regime_label == "TREND":
            extra_hu = " (irányított trend)"
        elif regime_label == "CHOPPY":
            extra_hu = " (oldalazó/range piac)"
        elif regime_label == "BALANCED":
            extra_hu = " (átmeneti/balanced)"

        if regime_label and adx_value is not None:
            regime_line = (
                f"{icon} Piaci rezsim: **{regime_label}**{extra_hu} • ADX≈`{adx_value:.1f}`"
            )
        elif regime_label:
            regime_line = f"{icon} Piaci rezsim: **{regime_label}**{extra_hu}"
        elif adx_value is not None:
            regime_line = f"{icon} ADX≈`{adx_value:.1f}`"


    

    status_text = "NINCS BELÉPŐ"
    color = COLORS["NO"]
    status_icon = "⚪"

    decision_upper = (decision or "").upper()
    notify_meta = signal_data.get("notify") if isinstance(signal_data, dict) else {}
    notify_reason = notify_meta.get("reason") if isinstance(notify_meta, dict) else None
    position_state = signal_data.get("position_state") if isinstance(signal_data, dict) else {}
    intent = (signal_data or {}).get("intent")
    has_tracked_position = bool(position_state.get("has_position"))
    cooldown_active = bool(position_state.get("cooldown_active"))
    cooldown_until = position_state.get("cooldown_until_utc") if isinstance(position_state, dict) else None
    reason_override: Optional[str] = None
    entry_block_reason: Optional[str] = None
    if notify_reason == "position_already_open":
        side_label = None
        side = (position_state or {}).get("side") if isinstance(position_state, dict) else None
        if isinstance(side, str):
            side_label = "LONG" if side.lower() == "buy" else "SHORT" if side.lower() == "sell" else None
        reason_override = f"NO NEW ENTRY — position open ({side_label or 'OPEN'})"
    elif notify_reason == "cooldown_active":
        cd_until = None
        if isinstance(notify_meta, dict):
            cd_until = notify_meta.get("cooldown_until_utc")
        if not cd_until and isinstance(position_state, dict):
            cd_until = position_state.get("cooldown_until_utc")
        reason_override = (
            f"COOLDOWN — no entry until {cd_until}" if cd_until else "COOLDOWN — no entry"
        )
    elif notify_reason == "no_open_position_tracked" and intent in {"hard_exit", "manage_position"}:
        reason_override = "NO MGMT — no open position tracked"
    if decision_upper == "BUY":
        status_text = "LONG"
        color = COLORS["LONG"]
        status_icon = "🟢"
    elif decision_upper == "SELL":
        status_text = "SHORT"
        color = COLORS["SHORT"]
        status_icon = "🔴"

    if is_flip:
        status_text = "FORDULAT (FLIP)"
        color = COLORS["FLIP"]
        status_icon = "🟠"
    if not is_stable and decision_upper in {"BUY", "SELL"}:
        status_text = "VÁRAKOZÁS (Stabilizálás...)"
        color = COLORS["WAIT"]
        status_icon = "🟡"

    if reason_override:
        entry_block_reason = reason_override
        decision_upper = "NO ENTRY"
      
    entry_status_text = status_text
    entry_status_icon = status_icon
    entry_color = color

    primary_header = None
    if intent == "hard_exit":
        primary_header = "⛔ HARD EXIT — tracked pozíció zárása (assumed)"
        status_text = "HARD EXIT"
        status_icon = "⛔"
        color = COLORS.get("SHORT", COLORS["NO"])
    elif cooldown_active:
        primary_header = "⏳ COOLDOWN"
        status_text = "COOLDOWN"
        status_icon = "⏳"
        color = COLORS.get("WAIT", COLORS["NO"])
    elif has_tracked_position or intent == "manage_position":
        primary_header = "🧭 AKTÍV POZÍCIÓ / MENEDZSMENT"
        status_text = "AKTÍV POZÍCIÓ"
        status_icon = "🧭"
        side_label = None
        side = (position_state or {}).get("side") if isinstance(position_state, dict) else None
        if isinstance(side, str):
            side_lower = side.lower()
            side_label = "LONG" if side_lower == "buy" else "SHORT" if side_lower == "sell" else None
        if side_label == "LONG":
            color = COLORS.get("LONG", color)
        elif side_label == "SHORT":
            color = COLORS.get("SHORT", color)
        else:
            color = COLORS.get("WAIT", color)
    elif intent == "entry" and decision_upper in {"BUY", "SELL"}:
        primary_header = f"🚀 ENTRY ({decision_upper})"
    elif reason_override:
        primary_header = reason_override

    if intent == "hard_exit":
        entry_status_text = "HARD EXIT"
        entry_status_icon = "⛔"
    elif intent == "manage_position":
        entry_status_text = "MENEDZSMENT"
        entry_status_icon = "🧭"
          
    mode_hu = "Bázis" if "core" in str(mode).lower() else "Lendület"

    title = f"{_get_emoji(asset)} {asset}"  # csak eszköz azonosító a push értesítés vágásának elkerülésére

    event_suffix = ""
    if kind == "flip":
        event_suffix = " • 🔁 Flip"
    elif kind == "invalidate":
        event_suffix = " • ❌ Invalidate"
    elif kind == "heartbeat":
        event_suffix = " • ℹ️ Állapot"

    p_bar = draw_progress_bar(p_score)
    line_score = f"📊 `{p_bar}` {int(p_score)}%"
    line_price = f"💵 {format_price(spot, asset)} • 🕒 {local_time}"

    grade_icon = "🟢" if setup_info["grade"] == "A" else "🟡" if setup_info["grade"] == "B" else "⚪"
    line_setup = f"🎯 {grade_icon} **{setup_info['title']}** — {setup_info['action']}"

    # Pozíciómenedzsment
    position_note = None
    if isinstance(signal_data, dict):
        raw_note = signal_data.get("position_management")
        if not raw_note:
            pm_reasons = signal_data.get("reasons")
            if isinstance(pm_reasons, list):
                for reason in pm_reasons:
                    if isinstance(reason, str) and reason.strip().lower().startswith("pozíciómenedzsment"):
                        raw_note = reason
                        break
        if isinstance(raw_note, str):
            raw_note = raw_note.strip()
        position_note = raw_note
        if not position_note and position_state.get("has_position"):
            fallback_levels = tracked_levels or _tracked_levels_from_manual_positions(
                asset, manual_positions
            )
            position_note = _format_manual_position_line(
                asset, position_state, fallback_levels
            )

    entry = sl = tp1 = tp2 = rr = None
    if isinstance(signal_data, dict):
        entry = signal_data.get("entry")
        sl = signal_data.get("sl")
        tp1 = signal_data.get("tp1")
        tp2 = signal_data.get("tp2")
        rr = signal_data.get("rr")
  
    # --- Mobil + pszicho struktúra (7–8 sor) ---
    tracked_entry = tracked_levels.get("entry") or entry
    tracked_sl = tracked_levels.get("sl") or sl
    tracked_tp1 = tracked_levels.get("tp1") or tp1
    tracked_tp2 = tracked_levels.get("tp2") or tp2
    opened_at = tracked_levels.get("opened_at_utc") or position_state.get("opened_at_utc")

    position_levels = tracked_levels or _tracked_levels_from_manual_positions(
        asset, manual_positions
    )
    position_entry = position_levels.get("entry") or position_state.get("entry")
    position_sl = position_levels.get("sl") or position_state.get("sl")
    position_tp1 = position_levels.get("tp1") or position_state.get("tp1")
    position_tp2 = position_levels.get("tp2") or position_state.get("tp2")

    position_levels_line = None
    position_level_parts: List[str] = []
    if position_entry is not None:
        position_level_parts.append(f"entry `{format_price(position_entry, asset)}`")
    if position_sl is not None:
        position_level_parts.append(f"SL `{format_price(position_sl, asset)}`")
    if position_tp1 is not None:
        position_level_parts.append(f"TP1 `{format_price(position_tp1, asset)}`")
    if position_tp2 is not None:
        position_level_parts.append(f"TP2 `{format_price(position_tp2, asset)}`")
    if position_level_parts:
        position_levels_line = "🧭 Position levels: " + " • ".join(position_level_parts)
      
    lines: List[str] = []
    entry_lines: List[str] = []
   
    grade_emoji = "🟢" if setup_info["grade"] == "A" else "🟡" if setup_info["grade"] == "B" else "⚪"
    setup_direction = resolve_setup_direction(signal_data, decision_upper)
    direction_suffix = ""
    if setup_info["grade"] in {"A", "B", "C"} and setup_direction:
        direction_suffix = f" ({setup_direction.upper()})"

    if primary_header:
        lines.append(primary_header)

    # Fő blokk (pozíció/cooldown/entry)
    if intent == "hard_exit":
        hard_exit_reasons = translate_reasons(position_diag.get("reasons") or []) if isinstance(position_diag, dict) else None
        lines.append(f"{status_icon} HARD EXIT — tracked pozíció zárása (assumed)")
        if hard_exit_reasons:
            lines.append(f"Ok: {hard_exit_reasons}")
        if position_levels_line:
            lines.append(position_levels_line)
        if cooldown_until:
            lines.append(f"Cooldown indul: {cooldown_until}")
        lines.append(line_price)
        lines.append(line_score)
    elif cooldown_active:
        lines.append(f"{status_icon} COOLDOWN — új belépők tiltva")
        if cooldown_until:
            lines.append(f"Lejár: {cooldown_until}")
        lines.append(line_price)
        if regime_line:
            lines.append(regime_line)
        lines.append(line_score)
    elif has_tracked_position:
        side_label = side_label if "side_label" in locals() else None
        if not side_label:
            side_label = "OPEN"
        lines.append(f"{status_icon} {status_text} — {side_label}")
        level_parts = [
            f"SL {format_price(tracked_sl, asset) if tracked_sl is not None else '-'}",
            f"TP1 {format_price(tracked_tp1, asset) if tracked_tp1 is not None else '-'}",
            f"TP2 {format_price(tracked_tp2, asset) if tracked_tp2 is not None else '-'}",
        ]
        lines.append(
            f"Nyitva: {opened_at or '-'} • " + " • ".join(level_parts)
        )
        if intent in {"hard_exit", "manage_position"} and position_levels_line:
            lines.append(position_levels_line)
        if position_note:
            lines.append(f"🧭 {position_note}")
        lines.append(line_price)
        if regime_line:
            lines.append(regime_line)
        lines.append(line_score)
    else:
        lines.append(
            f"{entry_status_icon} {decision_upper or 'NINCS'} • {setup_info['grade']} setup{direction_suffix} • {setup_info['action']}"
        )
        lines.append(line_price)
        if regime_line:
            lines.append(regime_line)
        if (
            intent == "entry"
            and decision_upper in {"BUY", "SELL"}
            and all(v is not None for v in (entry, sl, tp1, tp2))
        ):
            rr_txt = f"RR≈`{rr}`" if rr is not None else ""
            lines.append(
                f"🎯 Belépő `{format_price(entry, asset)}` • SL `{format_price(sl, asset)}` • TP1 `{format_price(tp1, asset)}` • TP2 `{format_price(tp2, asset)}` • {rr_txt}".strip(
                    " • "
                )
            )
        if position_note:
            lines.append(f"🧭 {position_note}")
        if intent in {"hard_exit", "manage_position"}:
            pos_state = position_diag.get("state") if isinstance(position_diag, dict) else None
            pos_severity = position_diag.get("severity") if isinstance(position_diag, dict) else None
            diag_parts: List[str] = []
            if pos_state:
                diag_parts.append(str(pos_state))
            if pos_severity:
                diag_parts.append(f"súlyosság: {pos_severity}")
            if diag_parts:
                lines.append(f"🧭 Pozíció diagnosztika: {' • '.join(diag_parts)}")
            pos_reasons = position_diag.get("reasons") if isinstance(position_diag, dict) else []
            if pos_reasons:
                reasons_hu = translate_reasons([str(reason) for reason in pos_reasons])
                lines.append(f"➤ Okok: {reasons_hu}")

        if isinstance(entry_diag, dict):
            entry_missing = entry_diag.get("missing") or []
        else:
            entry_missing = []
        gates_missing = entry_missing if intent == "entry" else []
        if gates_missing and entry_status_text != "NINCS BELÉPŐ":
            reasons = translate_reasons(gates_missing)
            lines.append(f"🧠 Figyelem: {reasons}")
        lines.append(line_score)
        if entry_status_text == "NINCS BELÉPŐ" and intent == "entry":
            if gates_missing:
                reasons_hu = translate_reasons(gates_missing)
                lines.append(f"⛔ Blokkolók: {reasons_hu}")

    # Új belépők blokk pozíció vagy cooldown mellett
    if has_tracked_position or cooldown_active:
        if isinstance(entry_diag, dict):
            entry_missing = entry_diag.get("missing") or []
        else:
            entry_missing = []
        gates_missing = entry_missing if intent == "entry" else []

        entry_lines.append("➕ ÚJ BELÉPŐK")
        if cooldown_active:
            cd_line = f"⏳ Cooldown aktív — új belépők tiltva"
            if cooldown_until:
                cd_line = f"{cd_line} (eddig: {cooldown_until})"
            entry_lines.append(cd_line)
        elif has_tracked_position:
            entry_lines.append("⛔ Új belépők tiltva: aktív tracked pozíció, kivárás új trade-re.")
        if entry_block_reason and entry_block_reason not in entry_lines:
            entry_lines.append(entry_block_reason)
        entry_lines.append(
            f"{entry_status_icon} {entry_status_text} • {setup_info['grade']} setup{direction_suffix} • {setup_info['action']}"
        )
        entry_lines.append(line_price)
        if regime_line:
            entry_lines.append(regime_line)
        if (
            intent == "entry"
            and decision_upper in {"BUY", "SELL"}
            and all(v is not None for v in (entry, sl, tp1, tp2))
        ):
            rr_txt = f"RR≈`{rr}`" if rr is not None else ""
            entry_lines.append(
                f"🎯 Belépő `{format_price(entry, asset)}` • SL `{format_price(sl, asset)}` • TP1 `{format_price(tp1, asset)}` • TP2 `{format_price(tp2, asset)}` • {rr_txt}".strip(
                    " • "
                )
            )
        if gates_missing and entry_status_text != "NINCS BELÉPŐ":
            reasons = translate_reasons(gates_missing)
            entry_lines.append(f"🧠 Figyelem: {reasons}")
        entry_lines.append(line_score)
        if entry_status_text == "NINCS BELÉPŐ" and intent == "entry" and gates_missing:
            reasons_hu = translate_reasons(gates_missing)
            entry_lines.append(f"⛔ Blokkolók: {reasons_hu}")

    if entry_lines:
        if lines:
            lines.append("")
        lines.extend(entry_lines)
   
    description = "\n".join(lines)

    final_color = color
    if not (has_tracked_position or cooldown_active or intent == "hard_exit"):
        final_color = entry_color if entry_status_text != "NINCS BELÉPŐ" else setup_info["color"]

    return {
        "title": title,
        "description": description,
        "color": final_color,
    }
  
# ---- Debounce / stabilitás / cooldown ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2
HEARTBEAT_STALE_MIN = 55  # ennyi perc után küldünk összefoglalót akkor is, ha az óra nem váltott
LAST_SENT_RETENTION_DAYS = 120  # ennyi nap után töröljük/archiváljuk a last_sent mezőt
LAST_SENT_FUTURE_GRACE_MIN = 15  # jövőbe mutató timestamp esetén ennyi percet engedünk meg
def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"WARN: {name}='{raw}' nem értelmezhető egész számként, {default}-ot használunk.",
            file=sys.stderr,
        )
        return default

NETWORK_RETRIES = int_env("DISCORD_NETWORK_RETRIES", 3)
NETWORK_BACKOFF_BASE = max(1.0, float(os.getenv("DISCORD_NETWORK_BACKOFF_BASE", "2")))
NETWORK_BACKOFF_CAP = max(NETWORK_BACKOFF_BASE, float(os.getenv("DISCORD_NETWORK_BACKOFF_CAP", "20")))
NETWORK_COOLDOWN_MIN = max(1, int_env("DISCORD_NETWORK_COOLDOWN_MIN", 5))
_WEBHOOK_COOLDOWN_UNTIL: Dict[str, float] = {}



def load_webhooks() -> Dict[str, str]:
    base = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    def pick(env_var: str) -> str:
        val = os.getenv(env_var, "").strip()
        return val or base

    return {
        "_base": base,
        "live": pick("DISCORD_WEBHOOK_URL_LIVE"),
        "management": pick("DISCORD_WEBHOOK_URL_MANAGEMENT"),
        "market_scan": pick("DISCORD_WEBHOOK_URL_MARKET_SCAN"),
    }


def classify_signal_channel(decision: str, kind: str, is_stable: bool) -> str:
    """Routing szabályok a fő jel-kártyákhoz.

    - Zöld BUY/SELL (stabil, normál küldés) ➜ #🚨-live-signals
    - Minden más jel- és státuszkártya ➜ #📊-market-scan
    """

    decision = (decision or "").lower()
    if decision in {"buy", "sell"} and is_stable and kind == "normal":
        return "live"
    return "market_scan"


def post_batches(hook: str, content: str, embeds: List[Dict[str, Any]]) -> None:
    """Küldjünk 10-es csomagokban az embedeket egy webhookra."""

    if not hook:
        return
    now = time.time()
    cooldown_until = _WEBHOOK_COOLDOWN_UNTIL.get(hook)
    if cooldown_until and now < cooldown_until:
        LOGGER.warning(
            "notify_webhook_cooldown_active",
            extra={"hook": hook[:32], "retry_at_epoch": cooldown_until},
        )
        return

    def _sleep_with_cap(delay: float) -> None:
        time.sleep(min(delay, NETWORK_BACKOFF_CAP))

    def _retry_delay(attempt: int, response: Optional[requests.Response]) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
            if retry_after:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    pass
        return min(NETWORK_BACKOFF_CAP, NETWORK_BACKOFF_BASE * (2 ** (attempt - 1)))
      
    batches = [embeds[i : i + 10] for i in range(0, len(embeds), 10)]
    for batch in batches:
        last_error: Optional[Exception] = None
        for attempt in range(1, NETWORK_RETRIES + 1):
            try:
                r = requests.post(hook, json={"content": content, "embeds": batch}, timeout=20)
                r.raise_for_status()
                break
            except requests.HTTPError as exc:  # pragma: no cover - exercised via RequestException
                last_error = exc
                status = exc.response.status_code if exc.response else None
                delay = _retry_delay(attempt, exc.response)
                LOGGER.warning(
                    "notify_webhook_http_error",
                    extra={"status": status, "attempt": attempt, "delay": delay},
                )
                if status == 429:
                    _WEBHOOK_COOLDOWN_UNTIL[hook] = time.time() + NETWORK_COOLDOWN_MIN * 60
                if attempt == NETWORK_RETRIES:
                    raise
                _sleep_with_cap(delay)
            except requests.RequestException as exc:
                last_error = exc
                delay = _retry_delay(attempt, None)
                LOGGER.warning(
                    "notify_webhook_network_error",
                    extra={"attempt": attempt, "delay": delay, "error": str(exc)},
                )
                if attempt == NETWORK_RETRIES:
                    raise
                _sleep_with_cap(delay)
        else:
            if last_error:
                raise last_error
      
def build_entry_gate_summary_embed() -> Optional[Dict[str, Any]]:
    """Return a compact embed with top entry gate elutasítási okok."""

    try:
        if not ENTRY_GATE_STATS_PATH.exists():
            return None
        payload = json.loads(ENTRY_GATE_STATS_PATH.read_text(encoding="utf-8"))
        reason_counts: Dict[str, int] = {}
        asset_lines: List[Tuple[int, str]] = []

        for asset, entries in sorted(payload.items()):
            if not isinstance(entries, list):
                continue
              
            asset_reason_counts: Dict[str, int] = {}
            reject_count = 0
            for item in entries:
                if not isinstance(item, dict):
                    continue
                reasons = item.get("missing") or item.get("precision_hiany") or []
                if reasons:
                    reject_count += 1
                for reason in reasons:
                    txt = str(reason)
                    reason_counts[txt] = reason_counts.get(txt, 0) + 1
                    asset_reason_counts[txt] = asset_reason_counts.get(txt, 0) + 1

            if reject_count:
                top_reasons = sorted(asset_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
                reason_summary = ", ".join(f"{reason} {count}x" for reason, count in top_reasons)
                asset_lines.append((reject_count, f"• {asset}: {reject_count}x blokkolva ({reason_summary})"))
              
        if not reason_counts:
            return None
          
        top = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        lines = [f"• {reason}: {count}x" for reason, count in top]
      
        asset_lines.sort(key=lambda item: (-item[0], item[1]))
        asset_field: Optional[Dict[str, Any]] = None
        if asset_lines:
            asset_field = {
                "name": "Érintett instrumentumok",
                "value": "\n".join(line for _, line in asset_lines[:6]),
                "inline": False,
            }

        embed: Dict[str, Any] = {
            "title": "Entry gate toplista (24h)",
            "description": "\n".join(lines),
            "color": 0xC0392B,
        }
        if asset_field:
            embed["fields"] = [asset_field]
        return embed
    except Exception:
        LOGGER.debug("entry_gate_summary_embed_failed", exc_info=True)
        return None


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    minutes = value / 60.0
    if minutes >= 1:
        return f"{minutes:.1f}p"
    return f"{value:.0f} mp"


def build_pipeline_diag_embed(
    payload: Optional[Dict[str, Any]] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Összefoglaló a pipeline időbélyeg-diffjeiről és artefakt-hashekről."""

    try:
        data = payload or load_pipeline_payload(PIPELINE_MONITOR_PATH)
    except Exception:
        LOGGER.debug("pipeline_diag_load_failed", exc_info=True)
        return None

    if not data:
        return None

    deltas = compute_run_timing_deltas(data, now=now)
    run_section = data.get("run") if isinstance(data, dict) else {}
    artifacts = data.get("artifacts") if isinstance(data, dict) else {}
    hashes = artifacts.get("hashes") if isinstance(artifacts, dict) else {}

    lines = []
    run_id = run_section.get("run_id") if isinstance(run_section, dict) else None
    if run_id:
        lines.append(f"run_id: `{run_id}`")
    if deltas.get("trading_duration_seconds") is not None:
        lines.append(
            f"Trading időtartam: {_format_seconds(deltas['trading_duration_seconds'])}"
        )
    if deltas.get("trading_to_analysis_gap_seconds") is not None:
        lines.append(
            "Trading→analysis késés: "
            f"{_format_seconds(deltas['trading_to_analysis_gap_seconds'])}"
        )
    if deltas.get("analysis_duration_seconds") is not None:
        lines.append(
            f"Analysis futásidő: {_format_seconds(deltas['analysis_duration_seconds'])}"
        )
    if deltas.get("analysis_age_seconds") is not None:
        lines.append(
            f"Utolsó analysis kora: {_format_seconds(deltas['analysis_age_seconds'])}"
        )
    if deltas.get("run_capture_offset_seconds") is not None:
        lines.append(
            "Run start-capture eltérés: "
            f"{_format_seconds(deltas['run_capture_offset_seconds'])}"
        )

    if not lines and not hashes:
        return None

    fields: List[Dict[str, Any]] = []
    if hashes:
        ordered = list(hashes.items())[:5]
        hash_lines = []
        for path_str, meta in ordered:
            name = Path(path_str).name
            if not meta:
                hash_lines.append(f"{name}: hiányzik")
                continue
            digest = meta.get("sha256") or ""
            size = meta.get("size")
            short = digest[:8] if digest else "n/a"
            if size is None:
                hash_lines.append(f"{name}: {short}")
            else:
                hash_lines.append(f"{name}: {short} ({size} B)")
        fields.append({"name": "Artefakt-hash", "value": "\n".join(hash_lines), "inline": False})

    return {
        "title": "Pipeline diagnosztika",
        "description": "\n".join(lines),
        "color": 0x4F6BED,
        "fields": fields,
    }
  

COOLDOWN_MIN   = int_env("DISCORD_COOLDOWN_MIN", 5)  # perc; 0 = off
MOMENTUM_COOLDOWN_MIN = int_env("DISCORD_COOLDOWN_MOMENTUM_MIN", 5)
FLIP_COOLDOWN_MINUTES_BY_ASSET = {
    "EURUSD": 30,
}


def _flip_cooldown_minutes(asset: str) -> int:
    return int(FLIP_COOLDOWN_MINUTES_BY_ASSET.get(asset, 0))


def env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def normalize_cli_flags(argv: Iterable[str]) -> Set[str]:
    """Egyszerű (argparse nélküli) flag normalizálás kézi futtatáshoz."""
    norm: Set[str] = set()
    for arg in argv:
        if not arg:
            continue
        if arg.startswith("--"):
            norm.add(arg[2:].lower())
        elif arg.startswith("-"):
            # jelenleg csak -f/-F érdekes, de legyen rugalmas
            if arg.lower() in {"-f", "-force"}:
                norm.add("force")
            else:
                norm.add(arg[1:].lower())
        else:
            norm.add(arg.lower())
    return norm

def flag_any(flags: Set[str], *candidates: str) -> bool:
    """Rugalmas flag-azonosítás (force/manual/heartbeat variációk)."""

    if not flags or not candidates:
        return False

    normalized_flags: Set[str] = set()
    for flag in flags:
        f = flag.lower()
        normalized_flags.add(f)
        normalized_flags.add(f.replace("-", ""))
        normalized_flags.add(f.replace("_", ""))

    normalized_candidates: Set[str] = set()
    for cand in candidates:
        c = (cand or "").lower()
        if not c:
            continue
        normalized_candidates.add(c)
        normalized_candidates.add(c.replace("-", ""))
        normalized_candidates.add(c.replace("_", ""))

    if normalized_candidates.intersection(normalized_flags):
        return True

    for flag in normalized_flags:
        for cand in normalized_candidates:
            if len(cand) >= 3 and cand in flag:
                return True

    return False

# ---- Időzóna a fejlécben / órakulcshoz ----
try:
    HB_TZ = ZoneInfo("Europe/Budapest")
except Exception as exc:  # pragma: no cover - környezeti hiányosságokra
    print(
        "WARN: Europe/Budapest időzóna nem elérhető, UTC-re esünk vissza.",
        f"({exc})",
        file=sys.stderr,
    )
    HB_TZ = timezone.utc

try:
    NY_TZ = ZoneInfo("America/New_York")
except Exception as exc:  # pragma: no cover - fallback
    print(
        "WARN: America/New_York időzóna nem elérhető, UTC-re esünk vissza.",
        f"({exc})",
        file=sys.stderr,
    )
    NY_TZ = timezone.utc

# ---- Megjelenés / emoji / színek ----
EMOJI = {
    "EURUSD": "💶",
    "BTCUSD": "🚀",
    "GOLD_CFD": "💰",
    "XAGUSD": "🥈",
    "USOIL": "🛢️",
    "NVDA": "🤖",    
}
COLOR = {
    "LONG":   0x2ecc71,  # zöld (csak tényleges Buy/Sell döntésnél)
    "SELL":  0x2ecc71,  # zöld (csak tényleges Buy/Sell döntésnél)
    "NO":    0xe74c3c,  # piros (invalidate)
    "WAIT":  0xf7dc6f,  # citromsárga (várakozás/stabilizálás)
    "FLIP":  0xe67e22,  # narancssárga (flip)
    "INFO":  0x95a5a6,  # semleges
}

# ---------------- util ----------------

def bud_now():
    return datetime.now(HB_TZ)

def bud_hh_key(dt=None) -> str:
    dt = dt or bud_now()
    return dt.strftime("%Y%m%d%H")

def bud_time_str(dt=None) -> str:
    dt = dt or bud_now()
    return dt.strftime("%Y-%m-%d %H:%M ") + (dt.tzname() or "CET")

def draw_progress_bar(value, min_val=0, max_val=100, length=10):
    """
    ASCII progress bar generálása a P-score vizualizálásához.
    Pl: [■■■■■■■□□□] 70%
    """
    try:
        val = float(value)
        pct = (val - min_val) / (max_val - min_val)
        pct = max(0.0, min(1.0, pct))
        filled = int(round(length * pct))
        # ■ karakter a teli, □ az üres részre
        bar = "■" * filled + "□" * (length - filled)
        return bar
    except:
        return "□" * length

def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utcnow_epoch():
    return int(datetime.now(timezone.utc).timestamp())

def iso_to_epoch(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
    except Exception:
        return 0

def parse_utc(value):
    if value is None or value == "-":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                return None
    return None


def to_utc_iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def update_asset_send_state(
    st: Dict[str, Any],
    *,
    decision: str,
    now: datetime,
    cooldown_minutes: int = 0,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Frissíti az eszköz értesítési állapotát küldéskor.

    A ``last_sent``, ``last_sent_decision`` és ``cooldown_until`` mezőket
    UTC-ben állítja be, determinisztikus, másodpercre kerekített időbélyegekkel.

    Rollback értelmezés: ha a friss állapotírás problémát okoz, a visszaállítás
    azt jelenti, hogy a küldési ágakban nem hívjuk meg ezt a függvényt, így a
    ``_notify_state.json`` érintetlen marad. Erre azért van szükség, mert
    hibás timestamp vagy döntés mentése torzíthatja a cooldown-logikát és
    a következő értesítések sorrendjét; írás letiltásával a korábbi stabil
    állapot konzerválható a hiba kivizsgálásáig.
    """

    now_iso = to_utc_iso(now)
    st = dict(st) if st is not None else _default_asset_state()
    st["last_sent"] = now_iso
    st["last_sent_decision"] = decision
    st["last_sent_mode"] = mode
    st["last_sent_known"] = True

    if cooldown_minutes and cooldown_minutes > 0:
        st["cooldown_until"] = to_utc_iso(now + timedelta(minutes=cooldown_minutes))
    else:
        st["cooldown_until"] = None

    return st


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def normalize_asset_key(asset: str) -> str:
    if not asset:
        return ""
    return str(asset).upper().strip()

def extract_tdstatus_meta(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None

    base = candidate
    for key in ("position", "data", "details", "info"):
        nested = candidate.get(key)
        if isinstance(nested, dict) and {"has_open_position", "open", "is_open", "side", "position_side"}.intersection(nested.keys()):
            base = nested
            break

    has_open = bool(base.get("has_open_position") or base.get("open") or base.get("is_open"))
    side = (
        base.get("side")
        or base.get("position_side")
        or base.get("direction")
        or candidate.get("side")
        or ""
    )
    avg_entry = safe_float(
        base.get("avg_entry")
        or base.get("average_entry")
        or candidate.get("avg_entry")
        or candidate.get("average_entry")
    )
    size = safe_float(
        base.get("size")
        or base.get("position_size")
        or candidate.get("size")
        or candidate.get("position_size")
    )

    if not (has_open or side or avg_entry is not None or size is not None):
        return None

    return {
        "has_open_position": has_open,
        "side": str(side).lower(),
        "avg_entry": avg_entry,
        "size": size,
    }


def load_tdstatus() -> Dict[str, Dict[str, Any]]:
    data = load(TDSTATUS_PATH)
    if not data:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    def handle_candidate(meta: Any, default_asset: Optional[str] = None):
        info = extract_tdstatus_meta(meta) if isinstance(meta, dict) else None
        if not info:
            return
        asset = normalize_asset_key(
            (meta.get("asset") if isinstance(meta, dict) else None)
            or (meta.get("symbol") if isinstance(meta, dict) else None)
            or default_asset
        )
        if not asset:
            return
        results[asset] = info

    if isinstance(data, dict):
        handle_candidate(data)
        for key in ("positions", "assets", "open_positions", "data", "tdstatus", "symbols"):
            block = data.get(key)
            if isinstance(block, dict):
                for asset_key, meta in block.items():
                    handle_candidate(meta, asset_key)
            elif isinstance(block, list):
                for item in block:
                    handle_candidate(item)
    elif isinstance(data, list):
        for item in data:
            handle_candidate(item)

    return results


def tdstatus_for_asset(tdstatus: Dict[str, Dict[str, Any]], asset: str) -> Dict[str, Any]:
    if not tdstatus or not asset:
        return {}
    asset_key = normalize_asset_key(asset)
    return tdstatus.get(asset_key, {})


def parse_utc_list(values: Iterable) -> list:
    parsed = []
    for item in values or []:
        if isinstance(item, dict):
            raw = item.get("utc") or item.get("datetime") or item.get("ts")
        else:
            raw = item
        dt = parse_utc(raw)
        if dt is not None:
            parsed.append(dt)
    return parsed


def load_eia_overrides() -> list:
    data = load(EIA_OVERRIDES_PATH)
    if data is None:
        return []
    if isinstance(data, dict):
        if "events" in data:
            return parse_utc_list(data.get("events"))
        if "overrides" in data:
            return parse_utc_list(data.get("overrides"))
        return parse_utc_list(data.values())
    if isinstance(data, list):
        return parse_utc_list(data)
    return []


def next_eia_release(now: Optional[datetime] = None) -> Optional[datetime]:
    now = now or datetime.now(timezone.utc)
    overrides = [dt for dt in load_eia_overrides() if isinstance(dt, datetime) and dt.tzinfo]
    overrides = [dt for dt in overrides if dt >= now - timedelta(days=1)]
    if overrides:
        return min(overrides)

    ny_now = now.astimezone(NY_TZ)
    weekday = ny_now.weekday()
    release_time = datetime(
        ny_now.year,
        ny_now.month,
        ny_now.day,
        10,
        30,
        tzinfo=NY_TZ,
    )

    if weekday > 2 or (weekday == 2 and ny_now >= release_time):
        days_ahead = (9 - weekday) % 7  # next Wednesday
        if days_ahead == 0:
            days_ahead = 7
        release_date = (ny_now + timedelta(days=days_ahead)).date()
    else:
        days_ahead = (2 - weekday)
        release_date = (ny_now + timedelta(days=days_ahead)).date()

    event_ny = datetime.combine(
        release_date,
        datetime.min.time(),
        tzinfo=NY_TZ,
    ).replace(hour=10, minute=30)
    return event_ny.astimezone(timezone.utc)


def format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    sign = "in"
    if total_seconds < 0:
        sign = "+"
        total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        base = f"{hours}h {minutes}m"
    else:
        base = f"{minutes}m"
    if sign == "in":
        return f"in {base}"
    return f"+{base}"


def eia_countdown(now: Optional[datetime] = None) -> Tuple[Optional[str], Optional[float]]:
    now = now or datetime.now(timezone.utc)
    event = next_eia_release(now)
    if not event:
        return None, None
    delta = event - now
    text = format_timedelta(delta)
    return text, delta.total_seconds() / 60.0

def _normalise_asset_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = str(value).strip().upper()
    return name or None


def _default_asset_state() -> Dict[str, Any]:
    return dict(DEFAULT_ASSET_STATE)


def _load_prior_open_commits(audit_path: Path) -> Set[str]:
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return set()

    assets: Set[str] = set()
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("event") == "OPEN_COMMIT":
            asset = payload.get("asset")
            if asset:
                assets.add(str(asset))
    return assets


def _apply_manual_position_transitions(
    *,
    asset: str,
    intent: str,
    decision: str,
    setup_grade: str,
    notify_meta: Optional[Dict[str, Any]],
    signal_payload: Dict[str, Any],
    manual_tracking_enabled: bool,
    can_write_positions: bool,
    manual_state: Dict[str, Any],
    manual_positions: Dict[str, Any],
    tracking_cfg: Dict[str, Any],
    now_dt: datetime,
    now_iso: str,
    send_kind: str,
    display_stable: bool,
    missing_list: Iterable[str],
    cooldown_map: Dict[str, Any],
    cooldown_default: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, bool]:
    positions_changed = False
    entry_opened = False

    if (
        manual_tracking_enabled
        and can_write_positions
        and intent == "hard_exit"
        and manual_state.get("has_position")
    ):
        manual_positions = position_tracker.close_position(
            asset,
            reason="hard_exit",
            closed_at_utc=now_iso,
            cooldown_minutes=_resolve_asset_value(cooldown_map, asset, cooldown_default),
            positions=manual_positions,
        )
        manual_state = position_tracker.compute_state(
            asset, tracking_cfg, manual_positions, now_dt
        )
        LOGGER.debug(
            "CLOSE state transition %s reason=%s cooldown_until=%s",
            asset,
            "hard_exit",
            manual_state.get("cooldown_until_utc"),
        )
        positions_changed = True

    if (
        manual_tracking_enabled
        and can_write_positions
        and intent == "entry"
        and manual_state.get("is_flat")
        and bool((notify_meta or {}).get("should_notify", True))
        and setup_grade in {"A", "B"}
        and decision in ("buy", "sell")
    ):
        entry_level, sl_level, tp2_level = extract_trade_levels(signal_payload)
        position_tracker.log_audit_event(
            "entry open attempt",
            event="OPEN_ATTEMPT",
            asset=asset,
            intent=intent,
            decision=decision,
            entry_side=decision,
            setup_grade=setup_grade,
            stable=bool(display_stable),
            gates_missing=missing_list,
            notify_should_notify=bool((notify_meta or {}).get("should_notify", True)),
            notify_reason=(notify_meta or {}).get("reason"),
            manual_tracking_enabled=manual_tracking_enabled,
            manual_has_position=manual_state.get("has_position"),
            manual_cooldown_active=manual_state.get("cooldown_active"),
            entry_level=entry_level,
            sl=sl_level,
            tp2=tp2_level,
            send_kind=send_kind,
        )
        manual_positions = position_tracker.open_position(
            asset,
            side="long" if decision == "buy" else "short",
            entry=entry_level,
            sl=sl_level,
            tp2=tp2_level,
            opened_at_utc=now_iso,
            positions=manual_positions,
        )
        manual_state = position_tracker.compute_state(
            asset, tracking_cfg, manual_positions, now_dt
        )
        LOGGER.debug(
            "OPEN state transition %s %s entry=%s sl=%s tp2=%s opened_at=%s",
            asset,
            decision,
            entry_level,
            sl_level,
            tp2_level,
            now_iso,
        )
        positions_changed = True
        entry_opened = True
    elif (
        manual_tracking_enabled
        and not can_write_positions
        and intent == "entry"
        and manual_state.get("is_flat")
        and bool((notify_meta or {}).get("should_notify", True))
        and setup_grade in {"A", "B"}
        and decision in ("buy", "sell")
    ):
        entry_level, sl_level, tp2_level = extract_trade_levels(signal_payload)
        position_tracker.log_audit_event(
            "entry suppressed: notify is read-only",
            event="ENTRY_SUPPRESSED",
            asset=asset,
            intent=intent,
            decision=decision,
            entry_side=decision,
            setup_grade=setup_grade,
            stable=bool(display_stable),
            gates_missing=missing_list,
            notify_should_notify=bool((notify_meta or {}).get("should_notify", True)),
            notify_reason=(notify_meta or {}).get("reason"),
            manual_tracking_enabled=manual_tracking_enabled,
            manual_has_position=manual_state.get("has_position"),
            manual_cooldown_active=manual_state.get("cooldown_active"),
            entry_level=entry_level,
            sl=sl_level,
            tp2=tp2_level,
            send_kind=send_kind,
            suppression_reason="writer_is_analysis",
        )

    return manual_positions, manual_state, positions_changed, entry_opened
  
  
def _archive_last_sent_entries(entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return

    archive_path = Path(STATE_ARCHIVE_PATH)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with archive_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except Exception:
        existing = []

    if not isinstance(existing, list):
        existing = []

    timestamp = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        payload = dict(entry)
        payload["archived_utc"] = timestamp
        existing.append(payload)

    with archive_path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2)


def _sanitize_last_sent(
    asset: str,
    state: Dict[str, Any],
    archived: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> None:
    raw_value = state.get("last_sent")
    state["last_sent_known"] = bool(state.get("last_sent_known"))

    if raw_value in (None, ""):
        state["last_sent"] = None
        if raw_value:
            state["last_sent_known"] = True
        return

    parsed = None
    parsed_reason = None

    if isinstance(raw_value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
        except Exception:
            parsed = None
            parsed_reason = "invalid-type"
    elif isinstance(raw_value, str):
        parsed = parse_utc(raw_value)
        if parsed is None:
            parsed_reason = "invalid-format"
    else:
        parsed_reason = "invalid-type"

    if parsed is not None and parsed_reason is None:
        now = now or datetime.now(timezone.utc)
        future_threshold = now + timedelta(minutes=LAST_SENT_FUTURE_GRACE_MIN)
        stale_threshold = now - timedelta(days=LAST_SENT_RETENTION_DAYS)

        if parsed > future_threshold:
            parsed_reason = "future"
        elif parsed < stale_threshold:
            parsed_reason = "stale"
        else:
            state["last_sent"] = to_utc_iso(parsed)
            state["last_sent_known"] = True
            return

    # Ha idáig eljutottunk, a timestamp-et töröljük, de megjegyezzük, hogy volt érték
    state["last_sent"] = None
    if raw_value not in (None, ""):
        state["last_sent_known"] = True

    archived.append(
        {
            "asset": asset,
            "last_sent_raw": raw_value,
            "reason": parsed_reason or "unknown",
            "last_sent_decision": state.get("last_sent_decision"),
        }
    )


def _archive_removed_state(removed: Dict[str, Any]) -> None:
    if not removed:
        return
    archive_path = Path(STATE_ARCHIVE_PATH)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except Exception:
        existing = []
    if not isinstance(existing, list):
        existing = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for asset, payload in removed.items():
        entry = {
            "asset": str(asset),
            "archived_utc": timestamp,
            "state": payload,
        }
        existing.append(entry)
    with archive_path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2)


def _ensure_state_structure(state: Any, *, persist_archive: bool = False) -> Dict[str, Any]:
    state_dict = state if isinstance(state, dict) else {}
    meta = state_dict.get("_meta") if isinstance(state_dict, dict) else None
    cleaned_meta: Dict[str, Any] = meta if isinstance(meta, dict) else {}
    cleaned_meta.setdefault("run_id", None)
    cleaned_meta.setdefault("last_analysis_utc", None)
    cleaned: Dict[str, Any] = {"_meta": cleaned_meta}
    recognised: Dict[str, Dict[str, Any]] = {}
    removed: Dict[str, Any] = {}
    archived_last_sent: List[Dict[str, Any]] = []

    items = state_dict.items() if isinstance(state_dict, dict) else []
    for key, value in items:
        if key == "_meta":
            continue
        normalised = _normalise_asset_name(key)
        if normalised and normalised in ACTIVE_ASSET_SET:
            payload = dict(value) if isinstance(value, dict) else {}
            merged = _default_asset_state()
            merged.update(payload)
            _sanitize_last_sent(normalised, merged, archived_last_sent)
            recognised[normalised] = merged
        else:
            removed[str(key)] = value

    for asset in ASSETS:
        normalised_asset = _normalise_asset_name(asset)
        if not normalised_asset:
            continue
        cleaned[normalised_asset] = recognised.get(normalised_asset, _default_asset_state())

    if persist_archive and removed:
        _archive_removed_state(removed)
    if persist_archive and archived_last_sent:
        _archive_last_sent_entries(archived_last_sent)

    return cleaned
  

def build_default_state(
    *,
    now: Optional[datetime] = None,
    reason: str = "scheduled_reset",
) -> Dict[str, Any]:
    """Return a baseline notification state with reset metadata."""

    base = _ensure_state_structure({}, persist_archive=False)
    meta = base.setdefault("_meta", {})
    reset_ts = to_utc_iso((now or datetime.now(timezone.utc)))
    meta["last_reset_utc"] = reset_ts
    meta["last_reset_reason"] = reason

    for asset in list(base.keys()):
        if asset == "_meta":
            continue
        base[asset] = _default_asset_state()

    return base
  
def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return _ensure_state_structure(data, persist_archive=True)

def save_state(st):
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    normalised = _ensure_state_structure(st, persist_archive=False)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(normalised, f, ensure_ascii=False, indent=2)

def mark_heartbeat(meta: Dict[str, Any], bud_key: str, now_iso: str) -> None:
    if meta is None:
        return
    meta["last_heartbeat_key"] = bud_key
    meta["last_heartbeat_utc"] = now_iso


def load_active_position_state() -> Dict[str, Any]:
    try:
        with open(ACTIVE_POSITION_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_active_position_state(state: Dict[str, Any]) -> None:
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(ACTIVE_POSITION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fmt_num(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def spot_from_sig_or_file(asset: str, sig: dict):
    spot = (sig or {}).get("spot") or {}
    price = spot.get("price") or spot.get("price_usd")
    utc = spot.get("utc") or spot.get("timestamp")
    if price is None:
        js = load(f"{PUBLIC_DIR}/{asset}/spot.json") or {}
        price = js.get("price") or js.get("price_usd")
        utc = utc or js.get("utc") or js.get("timestamp")
    spot_ts = parse_utc(utc)

    kline_data = load(f"{PUBLIC_DIR}/{asset}/klines_5m.json") or {}
    rows = []
    if isinstance(kline_data, dict):
        rows = kline_data.get("values") or []
    elif isinstance(kline_data, list):
        rows = kline_data

    fallback_price = None
    fallback_ts = None
    for row in rows:
        ts_raw = row.get("datetime") or row.get("t")
        close_raw = row.get("close") or row.get("c")
        if ts_raw is None or close_raw in (None, ""):
            continue
        ts = parse_utc(ts_raw)
        if ts is None:
            continue
        try:
            close_val = float(close_raw)
        except (TypeError, ValueError):
            continue
        if fallback_ts is None or ts > fallback_ts:
            fallback_ts = ts
            fallback_price = close_val

    use_fallback = False
    if fallback_price is not None:
        if price is None or spot_ts is None:
            use_fallback = True
        elif fallback_ts and spot_ts and fallback_ts > spot_ts:
            use_fallback = True

    if use_fallback:
        price = fallback_price
        utc = to_utc_iso(fallback_ts) or utc
      
    return price, utc

def missing_from_sig(sig: dict):
    gates = (sig or {}).get("gates") or {}
    miss = gates.get("missing") or []
    if not miss:
        return ""
    pretty = {
        "session": "Session",
        "regime": "Regime",
        "bias": "Bias",
        "bos5m": "BOS (5m)",
        "liquidity": "Liquidity",
        "liquidity(fib_zone|sweep)": "Liquidity",
        "liquidity(fib|sweep|ema21|retest)": "Liquidity",
        "atr": "ATR",
        "tp_min_profit": "TP min. profit",
        "tp1_net>=+1.0%": "TP1 nettó ≥ +1.0%",
        "min_stoploss": "Minimum stoploss",
        "RR≥1.5": "RR≥1.5",
        "rr_math>=2.0": "RR≥2.0",
        # momentum
        "momentum(ema9x21)": "Momentum (EMA9×21)",
        "bos5m|struct_break": "BOS/Structure",
        "precision warning": "Precision figyelmeztetés",
        "precision_warning": "Precision figyelmeztetés",
    }
    out = []
    for k in miss:
        key = "RR≥2.0" if k.startswith("rr_math") else k
        out.append(pretty.get(k, key))
    uniq = list(dict.fromkeys(out))
    return ", ".join(uniq)

def gates_mode(sig: dict) -> str:
    return ((sig or {}).get("gates") or {}).get("mode") or "-"

def decision_of(sig: dict) -> str:
    closed, _ = market_closed_info(sig)
    if closed:
        return "no entry"
    d = (sig or {}).get("signal", "no entry")
    d = (d or "").lower()
    if d not in ("buy","sell"):
        return "no entry"
    return d
  
def note_implies_open(note: str) -> bool:
    n = (note or "").strip().lower()
    if not n:
        return False
    hints = {
        "market open",
        "market opened",
        "market opening",
        "piac nyitva",
        "nyitva",
        "open",
        "opened",
    }
    return any(h in n for h in hints)


def format_closed_note(base_note: str = "", reason: Optional[str] = None) -> str:
    note = (base_note or "").strip()
    reason_text = (reason or "").strip()
  
    if note_implies_open(note):
        note = ""

    combined = f"{note} {reason_text}".lower()
    data_issue = any(
        keyword in combined for keyword in {"adat", "data", "latency", "delay", "stale", "cache"}
    )
  
    if not note:
        note = "Hiányzó adat" if data_issue else "Piac zárva"

    lower = note.lower()
    if not data_issue:
        if "market" not in lower:
            if "piac" not in lower:
                note = f"{note} • Market closed"
            else:
                note = f"{note} (market closed)"

    if reason_text:
        lower = note.lower()
        if reason_text.lower() not in lower:
            note = f"{note} – {reason_text}" if note else reason_text

    return note


def market_closed_info(sig: dict) -> Tuple[bool, str]:
    sig = sig or {}
    session = sig.get("session_info") or {}
    open_flag = session.get("open")
    status = str(session.get("status") or "").lower()
    note = session.get("status_note") or ""

    closed_statuses = {
        "closed",
        "closed_out_of_hours",
        "halted",
        "halted_limit",
        "maintenance",
        "holiday",
    }

    if isinstance(open_flag, bool) and not open_flag:
        return True, format_closed_note(note)

    if status:
        if status in closed_statuses or status.startswith("closed") or status.startswith("halt"):
            return True, format_closed_note(note)

    diagnostics = sig.get("diagnostics") or {}
    tf_spot = (diagnostics.get("timeframes") or {}).get("spot") or {}
    latency = safe_float(tf_spot.get("latency_seconds"))
    expected = safe_float(tf_spot.get("expected_max_delay_seconds"))
    if latency is not None:
        base_limit = 1800.0  # 30 perc — ha nincs explicit limit
        limit = expected if expected and expected > 0 else base_limit
        limit = max(limit, base_limit)
        if latency > limit:
            latency_minutes = max(1, int(latency // 60))
            reason = f"adat késik ≈{latency_minutes} perc"
            return True, format_closed_note(note, reason)

    spot = sig.get("spot") or {}
    spot_ts = parse_utc(spot.get("utc") or spot.get("timestamp"))
    if spot_ts is not None:
        age = datetime.now(timezone.utc) - spot_ts
        limit_seconds = max((expected or 0), 1800)
        if age > timedelta(seconds=limit_seconds):
            age_minutes = int(age.total_seconds() // 60)
            if age_minutes <= 0:
                age_minutes = 1
            reason = f"adat {age_minutes} perc óta nem frissült"
            return True, format_closed_note(note, reason)

    return False, ""

# ------------- embed-renderek -------------

def card_color(dec: str, is_stable: bool, kind: str, setup_grade: Optional[str] = None) -> int:
    if kind == "flip":
        return COLOR["WAIT"]
    if kind == "invalidate":
        return COLOR["NO"]
    if dec in ("BUY", "SELL"):
        return COLOR["BUY"] if is_stable else COLOR["NO"]
    return COLOR["NO"]


def colorize_setup_text(text: str, setup_grade: Optional[str]) -> str:
    """ABC setup jelölés egységes betűtípussal, színkódolt emojival."""

    emoji_prefix = {
        "A": "🟢",  # zöld
        "B": "🟡",  # citromsárga
        "C": "⚪️",  # szürke/semleges
    }
    prefix = emoji_prefix.get(setup_grade)
    return f"{prefix} {text}" if prefix else text


def resolve_setup_direction(sig: dict, decision: str) -> Optional[str]:
    """Próbálja meg kideríteni a long/short irányt a jelből vagy a precíziós tervből."""

    if decision in {"BUY", "SELL"}:
        return decision

    direction_sources = []
    precision_plan = (sig or {}).get("precision_plan") or {}
    direction_sources.append(precision_plan.get("direction"))
    direction_sources.append((precision_plan.get("context") or {}).get("direction"))

    for cand in direction_sources:
        if isinstance(cand, str) and cand.strip():
            upper = cand.strip().upper()
            if upper in {"BUY", "SELL", "LONG", "SHORT"}:
                return "BUY" if upper in {"BUY", "LONG"} else "SELL"
    return None
  
def slope_status_icon(slope: Optional[float], threshold: float, side: str) -> str:
    if slope is None:
        return "⚠️"
    side = (side or "").lower()
    if side == "buy":
        if slope >= threshold:
            return "✅"
        if slope > 0:
            return "⚠️"
        return "❌"
    # default sell interpretation
    if slope <= -threshold:
        return "✅"
    if slope < 0:
        return "⚠️"
    return "❌"


def structure_label(flag: Optional[str]) -> str:
    mapping = {
        "bos_down": "BOS↓",
        "bos_up": "BOS↑",
        "range": "Range",
    }
    return mapping.get((flag or "").lower(), flag or "-")


def format_percentage(value: Optional[float]) -> str:
    if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def format_signed_percentage(value: Optional[float]) -> str:
    if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:+.2f}%"


def format_tminus(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds >= 0:
        prefix = "T−"
    else:
        prefix = "T+"
        total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return f"{prefix}{' '.join(parts)}"


HU_WEEKDAYS = ["Hét", "Ked", "Sze", "Csü", "Pén", "Szo", "Vas"]


def weekday_short_hu(dt: datetime) -> str:
    try:
        return HU_WEEKDAYS[dt.weekday() % 7]
    except Exception:
        return dt.strftime("%a")


def format_hu_countdown(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    prefix = "T-" if total_seconds >= 0 else "T+"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if hours:
        base = f"{hours}ó {minutes}p"
    else:
        base = f"{minutes}p"
    if not base.strip():
        base = "0p"
    return f"{prefix}{base}"


def _now_utc_iso(dt: Optional[datetime] = None) -> str:
    return to_utc_iso(dt or datetime.now(timezone.utc))


class ActivePositionWatcher:
    ASSET_ORDER = ("USOIL", "GOLD_CFD")

    def __init__(
        self,
        config: Dict[str, Any],
        tdstatus: Dict[str, Dict[str, Any]],
        signals: Dict[str, Dict[str, Any]],
        now: Optional[datetime] = None,
        manual_positions: Optional[Dict[str, Any]] = None,
        manual_tracking_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}
        self.config_common = self.config.get("common", {})
        self.config_assets = self.config.get("assets", {})
        self.tdstatus = tdstatus or {}
        self.signals = signals or {}
        self.now = now or datetime.now(timezone.utc)
        self.manual_positions = manual_positions if isinstance(manual_positions, dict) else {}
        self.manual_tracking_cfg = manual_tracking_cfg or (
            _signal_stability_config().get("manual_position_tracking") or {}
        )
        self.anchor_state = load_anchor_state()
        self.state_cache = load_active_position_state()
        self.updated_state = False
        self.embeds: List[Dict[str, Any]] = []
        self.latest_cards: Dict[str, Dict[str, Any]] = {}
        self.changed_assets: Set[str] = set()
        dynamic_assets: List[str] = list(self.ASSET_ORDER)
        for asset in sorted(self.manual_positions.keys()):
            if asset not in dynamic_assets:
                dynamic_assets.append(asset)
        for asset in sorted((self.config_assets or {}).keys()):
            if asset not in dynamic_assets:
                dynamic_assets.append(asset)
        self.asset_order = tuple(dynamic_assets)

    # -------------------- helpers --------------------
    def run(self, allowed_assets: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        self.latest_cards = {}
        self.changed_assets = set()
        self.embeds = []
        allowed_keys = None
        if allowed_assets is not None:
            allowed_keys = {normalize_asset_key(x) for x in allowed_assets if x}
        for asset in self.asset_order:
            embed = self._evaluate_asset(asset)
            if embed:
                if allowed_keys is not None and normalize_asset_key(asset) not in allowed_keys:
                    LOGGER.info("watcher suppressed: %s no manual position", asset)
                    continue
                self.embeds.append(embed)
        if self.updated_state:
            save_active_position_state(self.state_cache)
        return self.embeds

    def _asset_config(self, asset: str) -> Dict[str, Any]:
        cfg = dict(self.config_common)
        asset_cfg = (self.config_assets or {}).get(asset, {})
        if isinstance(asset_cfg, dict):
            cfg.update(asset_cfg)
        return cfg

    def _tdstatus(self, asset: str) -> Dict[str, Any]:
        return tdstatus_for_asset(self.tdstatus, asset)

    def _signal(self, asset: str) -> Dict[str, Any]:
        return self.signals.get(asset, {}) or {}

    def _title_for_asset(self, asset: str, side: str) -> str:
        base = "GOLD" if asset == "GOLD_CFD" else asset
        side_txt = (side or "").upper() or "-"
        return f"{base} • Active Position ({side_txt})"

    def _resolve_anchor(self, asset: str, status: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        asset_key = asset.upper()
        record = self.anchor_state.get(asset_key, {})
        side = (record.get("side") or "").lower()
        avg_entry = safe_float(status.get("avg_entry"))
        if not side:
            status_side = (status.get("side") or "").lower()
            if status_side:
                self.anchor_state = touch_anchor(asset, status_side, price=avg_entry)
                record = self.anchor_state.get(asset_key, {})
                side = (record.get("side") or "").lower()
        else:
            status_side = (status.get("side") or "").lower()
            if status_side and status_side != side:
                # igazítsuk az anchor-t az aktuális pozíció irányához
                self.anchor_state = touch_anchor(asset, status_side, price=avg_entry)
                record = self.anchor_state.get(asset_key, {})
                side = (record.get("side") or "").lower()
        price = safe_float(record.get("price"))
        ts = record.get("timestamp")
        if price is None:
            price = avg_entry
        return side, price, ts

    def _event_info(self, asset: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_cfg = cfg.get("event") if isinstance(cfg, dict) else None
        if asset != "USOIL" or not event_cfg:
            return None
        event_dt = next_eia_release(self.now)
        if not event_dt:
            return None
        delta = event_dt - self.now
        pre_window = int(event_cfg.get("pre_window_min") or 0)
        minutes = delta.total_seconds() / 60.0
        event_mode = pre_window > 0 and 0 <= minutes <= pre_window
        et_dt = event_dt.astimezone(NY_TZ)
        bud_dt = event_dt.astimezone(HB_TZ)
        countdown = format_tminus(delta)
        name = event_cfg.get("name", "EIA")
        et_label = et_dt.strftime("%a %H:%M")
        bud_label = bud_dt.strftime("%H:%M")
        field_value = f"{name}: {et_label} ET (Bp {bud_label}) • {countdown}"
        return {
            "name": name,
            "datetime": event_dt,
            "display": field_value,
            "event_mode": event_mode,
            "minutes": minutes,
            "countdown": countdown,
        }

    def _rollover_active(self, cfg: Dict[str, Any]) -> bool:
        warn = (cfg.get("rollover_warn_gmt") or self.config_common.get("rollover_warn_gmt") or "").strip()
        window = int(cfg.get("rollover_window_min") or self.config_common.get("rollover_window_min") or 0)
        if not warn or window <= 0:
            return False
        try:
            hour_str, minute_str = warn.split(":", 1)
            target_hour = int(hour_str)
            target_minute = int(minute_str)
        except ValueError:
            return False
        base = self.now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        candidates = [base, base + timedelta(days=1), base - timedelta(days=1)]
        for cand in candidates:
            diff = abs((self.now - cand).total_seconds()) / 60.0
            if diff <= window:
                return True
        return False

    def _state_cache_entry(self, asset: str) -> Dict[str, Any]:
        entry = self.state_cache.get(asset)
        if isinstance(entry, dict):
            return entry
        return {}

    # -------------------- core evaluation --------------------
    def _evaluate_asset(self, asset: str) -> Optional[Dict[str, Any]]:
        cfg = self._asset_config(asset)
        status = self._tdstatus(asset)
        manual_state: Dict[str, Any] = {}
        try:
            manual_state = position_tracker.compute_state(
                asset, self.manual_tracking_cfg, self.manual_positions, self.now
            )
        except Exception:
            manual_state = {}
          
        has_open = bool(status.get("has_open_position")) and bool((status.get("side") or "").strip())
        if manual_state.get("has_position"):
            has_open = True
        if not has_open and manual_state.get("cooldown_active"):
            self.latest_cards.pop(asset, None)
            return None
        side_override = manual_state.get("side") if isinstance(manual_state, dict) else None
        if side_override:
            status = dict(status or {})
            status["side"] = side_override
        prev_entry = self._state_cache_entry(asset)

        if not has_open:
            self.latest_cards.pop(asset, None)
            if prev_entry.get("state") != "FLAT":
                self.state_cache[asset] = {
                    "state": "FLAT",
                    "key": "FLAT",
                    "updated": _now_utc_iso(self.now),
                }
                self.updated_state = True
            print(f"watcher: no-open-position ({asset})")
            return None

        signal = self._signal(asset)
        meta = (signal.get("active_position_meta") or {}) if isinstance(signal, dict) else {}
        
        anchor_side, anchor_price, _ = self._resolve_anchor(asset, status)
        if anchor_side not in {"buy", "sell"}:
            return None

        slope = safe_float(meta.get("ema21_slope_signed") or signal.get("ema21_slope_1h"))
        slope_th = safe_float(meta.get("ema21_slope_threshold") or signal.get("ema21_slope_threshold"))
        if slope_th is None:
            slope_th = safe_float(cfg.get("ema21_slope_min_abs")) or EMA21_SLOPE_MIN

        structure_raw = (meta.get("structure_5m") or signal.get("bos_5m_dir") or "").lower()
        if structure_raw in {"bos↑", "bosup", "bos-up"}:
            structure_flag = "bos_up"
        elif structure_raw in {"bos↓", "bosdown", "bos-down"}:
            structure_flag = "bos_down"
        else:
            structure_flag = structure_raw
        structure_display = structure_label(meta.get("structure_5m") or signal.get("bos_5m_dir"))

        atr1h = safe_float(meta.get("atr1h") or signal.get("atr1h"))
        invalid_buffer_abs = safe_float(
            meta.get("invalid_buffer_abs")
            or signal.get("invalid_buffer_abs")
            or cfg.get("invalid_buffer_abs")
        )

        invalid_levels = signal.get("invalid_levels") if isinstance(signal, dict) else {}
        invalid_level_sell = safe_float(meta.get("invalid_level_sell") or (invalid_levels or {}).get("sell"))
        invalid_level_buy = safe_float(meta.get("invalid_level_buy") or (invalid_levels or {}).get("buy"))

        last_close_1h = safe_float(meta.get("last_close_1h") or signal.get("last_close_1h"))

        event_info = self._event_info(asset, cfg)
        event_mode = bool(event_info and event_info.get("event_mode"))

        tp1_raw = meta.get("tp1_reached")
        if tp1_raw is None and isinstance(signal, dict):
            tp1_raw = signal.get("tp1_reached")
        if isinstance(tp1_raw, str):
            tp1_reached = tp1_raw.strip().lower() in {"1", "true", "yes", "hit", "ok"}
        elif isinstance(tp1_raw, (int, float)):
            tp1_reached = bool(tp1_raw)
        else:
            tp1_reached = bool(tp1_raw) if isinstance(tp1_raw, bool) else None

        size = safe_float(status.get("size"))
        avg_entry = safe_float(status.get("avg_entry"))
        anchor_price_display = anchor_price or avg_entry

        if anchor_side == "sell":
            invalid_level = invalid_level_sell
            invalid_hit = (
                invalid_level is not None
                and last_close_1h is not None
                and last_close_1h > invalid_level
            )
            regime_flip = slope is not None and slope_th is not None and slope >= slope_th
            structure_opposite = structure_flag == "bos_up"
            exit_arrow = "↑"
        else:
            invalid_level = invalid_level_buy
            invalid_hit = (
                invalid_level is not None
                and last_close_1h is not None
                and last_close_1h < invalid_level
            )
            regime_flip = slope is not None and slope_th is not None and slope <= -slope_th
            structure_opposite = structure_flag == "bos_down"
            exit_arrow = "↓"

        exit_condition = invalid_hit and structure_opposite
        reduce_condition = regime_flip and structure_opposite

        if exit_condition:
            state = "EXIT"
        elif event_mode:
            state = "EVENT"
        elif reduce_condition:
            state = "REDUCE"
        else:
            state = "HOLD"

        action_map = {
            "HOLD": "✅ HOLD",
            "REDUCE": "⚠️ REDUCE 30–50%",
            "EVENT": "⚠️ REDUCE 30–50% (event window)",
            "EXIT": "⛔ EXIT now",
        }
        action_field = {"name": "Action", "value": action_map.get(state, "✅ HOLD"), "inline": False}

        anchor_parts: List[str] = []
        if anchor_price_display is not None:
            anchor_parts.append(f"@ {fmt_num(anchor_price_display, digits=2)}")
        if size is not None:
            anchor_parts.append(f"size {fmt_num(size, digits=2)}")
        anchor_value = f"{anchor_side.upper()}" + (" " + " • ".join(anchor_parts) if anchor_parts else "")

        if invalid_level is not None:
            invalid_text = f"{fmt_num(invalid_level, digits=2)} (1h close{exit_arrow} ⇒ EXIT)"
        else:
            invalid_text = "n/a"

        slope_icon = slope_status_icon(slope, slope_th or EMA21_SLOPE_MIN, anchor_side)
        threshold_text = format_percentage(abs(slope_th or EMA21_SLOPE_MIN))
        regime_text = f"{format_signed_percentage(slope)} • {slope_icon} (küszöb: {threshold_text} abs.)"

        atr_floor_candidates: List[float] = []
        if atr1h is not None:
            atr_floor_candidates.append(0.5 * atr1h)
            if asset == "USOIL":
                atr_floor_candidates.append(ATR_TRAIL_MIN_ABS)
        if invalid_buffer_abs is not None:
            atr_floor_candidates.append(invalid_buffer_abs)
        trail_floor = max(atr_floor_candidates) if atr_floor_candidates else None        
        if atr1h is not None:
            atr_text_base = fmt_num(atr1h, digits=2)
        else:
            atr_text_base = "n/a"
        if atr1h is not None and trail_floor is not None and atr1h not in (0, 0.0):
            trail_text = f"${fmt_num(trail_floor, digits=2)}"
            k_val = trail_floor / atr1h if atr1h else None
            if k_val is not None and np.isfinite(k_val):
                k_text = f"{k_val:.2f}".rstrip("0").rstrip(".")
            else:
                k_text = "n/a"
        else:
            trail_text = "n/a"
            k_text = "n/a"
        atr_field_text = f"{atr_text_base} / {trail_text}"
        if k_text != "n/a":
            atr_field_text += f" (K={k_text})"

        fields = [
            action_field,
            {"name": "Anchor", "value": anchor_value or "-", "inline": True},
            {"name": "Invalid (1H)", "value": invalid_text, "inline": True},
            {"name": "Regime (1h EMA21 slope)", "value": regime_text, "inline": True},
            {"name": "5m structure", "value": structure_display or "-", "inline": True},
            {"name": "ATR(1h) / Trail", "value": atr_field_text, "inline": True},
        ]

        if event_info:            
            fields.append(
                {
                    "name": "Next Event",
                    "value": event_info.get("display") or "-",
                    "inline": False,
                }
            )

        desc_lines: List[str] = []
        if tp1_reached is True:
            desc_lines.append("TP1 reached → BE + cost, ATR trailing active.")
        elif tp1_reached is False:
            desc_lines.append("TP1 pending → manage core size cautiously.")
        if state == "EXIT" and invalid_level is not None and last_close_1h is not None:
            desc_lines.append(
                f"Trigger: 1h close {fmt_num(last_close_1h, digits=2)} vs invalid {fmt_num(invalid_level, digits=2)} + 5m BOS flip."
            )
        elif state in {"REDUCE", "EVENT"} and structure_opposite:
            reason = "Regime + 5m BOS opposite" if state == "REDUCE" else "Event window active"
            if event_mode and event_info:
                reason = f"EIA window {event_info.get('countdown')}"
            desc_lines.append(f"Trigger: {reason}.")
        elif state == "HOLD":
            desc_lines.append("Anchor bias intact – defensive management only.")
          
        if state == "EXIT" and asset == "EURUSD":
            dedup_line = "EXIT jel deduplikálva: ugyanarra a pozícióra 30 percig nem küldünk ismétlést."
            if dedup_line not in desc_lines:
                desc_lines.append(dedup_line)

        color_map = {
            "HOLD": 0x2ecc71,
            "REDUCE": 0xf1c40f,
            "EVENT": 0xf1c40f,
            "EXIT": 0xe74c3c,
        }

        embed: Dict[str, Any] = {
            "title": self._title_for_asset(asset, anchor_side),
            "color": color_map.get(state, 0x2ecc71),
            "fields": fields,
            "footer": {"text": "Active-position menedzsment; nem új belépő."},
        }
        if desc_lines:
            embed["description"] = "\n".join(desc_lines)

        state_key = f"{state}|{anchor_side}|event={1 if event_mode else 0}"
        state_record = {
            "state": state,
            "key": state_key,
            "anchor": anchor_side,
            "updated": _now_utc_iso(self.now),
        }
        self.latest_cards[asset] = deepcopy(embed)
        exit_dedup_min = _dedup_minutes_for_exit(asset)
        if state == "EXIT" and exit_dedup_min > 0:
            last_exit_sent_iso = prev_entry.get("last_exit_sent_utc")
            if last_exit_sent_iso:
                age_sec = iso_to_epoch(_now_utc_iso(self.now)) - iso_to_epoch(last_exit_sent_iso)
                if age_sec >= 0 and age_sec < exit_dedup_min * 60:
                    state_record["last_exit_sent_utc"] = last_exit_sent_iso
                    self.state_cache[asset] = state_record
                    self.updated_state = True
                    return None

            state_record["last_exit_sent_utc"] = _now_utc_iso(self.now)
            self.updated_state = True
        elif state != "EXIT" and "last_exit_sent_utc" in prev_entry and "last_exit_sent_utc" not in state_record:
            state_record["last_exit_sent_utc"] = prev_entry["last_exit_sent_utc"]
  
        if prev_entry.get("key") != state_key:
            self.state_cache[asset] = state_record
            self.updated_state = True
            self.changed_assets.add(asset)
            return embed

        state_record["updated"] = prev_entry.get("updated", state_record["updated"])
        self.state_cache[asset] = state_record
        return None
    
    def snapshot_embeds(self, exclude: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        exclude_keys = {normalize_asset_key(x) for x in (exclude or []) if x}
        embeds: List[Dict[str, Any]] = []
        for asset in self.ASSET_ORDER:
            card = self.latest_cards.get(asset)
            if not card:
                continue
            if exclude_keys and normalize_asset_key(asset) in exclude_keys:
                continue
            embeds.append(deepcopy(card))
        return embeds

def build_embed_for_asset(asset: str, sig: dict, is_stable: bool, kind: str = "normal", prev_decision: str = None, tdstatus: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    kind: "normal" | "invalidate" | "flip" | "heartbeat"
    """
    emoji = EMOJI.get(asset, "📊")
    closed, closed_note = market_closed_info(sig)
    dec_effective = decision_of(sig).upper()
    dec = dec_effective if dec_effective in ("BUY", "SELL") else "NO ENTRY"

    session_info = (sig or {}).get("session_info") or {}
    entry_open = session_info.get("entry_open")
    monitor_open = session_info.get("open")

    p_raw = int(sig.get("probability", 0) or 0)
    p = 0 if closed else p_raw
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")
    mode = gates_mode(sig)
    mode_pretty = {
        "analysis_error": "analysis hiba",
        "data_gap": "adat hiány",
        "unavailable": "adat nem elérhető",
    }
    display_mode = mode_pretty.get(mode, mode)
    missing_list = ((sig.get("gates") or {}).get("missing") or [])
    core_bos_pending = (mode == "core") and ("bos5m" in missing_list)

    entry_thresholds = sig.get("entry_thresholds") if isinstance(sig, dict) else {}
    entry_diagnostics = sig.get("entry_diagnostics") if isinstance(sig, dict) else {}
    dynamic_lines: List[str] = []
    setup_classification: Optional[str] = None
    setup_classification_line: Optional[str] = None
    setup_grade: Optional[str] = None
    setup_score: Optional[float] = None
    setup_issues: List[str] = []
    setup_direction = resolve_setup_direction(sig, dec)
    if isinstance(entry_diagnostics, dict) and entry_diagnostics.get("precision_override"):
        dynamic_lines.append("Precision override (reduced size)")
    if isinstance(entry_thresholds, dict):
        atr_soft_meta = entry_thresholds.get("atr_soft_gate") or {}
        atr_soft_used = bool(entry_thresholds.get("atr_soft_gate_used"))
        atr_soft_mode = str(atr_soft_meta.get("mode") or "").lower()
        if atr_soft_used or atr_soft_mode == "soft_pass":
            atr_penalty = safe_float(atr_soft_meta.get("penalty")) or 0.0
            atr_missing = safe_float(atr_soft_meta.get("diff_ratio"))
            tolerance = safe_float(atr_soft_meta.get("tolerance_pct"))
            miss_txt = f"hiány {format_percentage(atr_missing)}" if atr_missing is not None else "toleranciás belépés"
            tol_txt = f" (tolerancia {format_percentage(tolerance)})" if tolerance is not None else ""
            pen_txt = f" −{atr_penalty:.1f}P" if atr_penalty else ""
            dynamic_lines.append(f"ATR Soft Gate: {miss_txt}{tol_txt}{pen_txt}")
            setup_issues.append("ATR hiány/lazítás")

        latency_meta = entry_thresholds.get("latency_relaxation") or {}
        latency_mode = str(latency_meta.get("mode") or "").lower()
        if latency_mode == "penalized":
            age_min = None
            try:
                age_min = int((latency_meta.get("age_seconds") or 0) // 60)
            except Exception:
                age_min = None
            profile = latency_meta.get("profile")
            latency_penalty = safe_float(latency_meta.get("penalty")) or 0.0
            age_txt = f"≈{age_min} perc" if age_min is not None else "késleltetés"
            profile_txt = f"{profile} profil" if profile else "relaxált guard"
            pen_txt = f" −{latency_penalty:.1f}P" if latency_penalty else ""
            dynamic_lines.append(f"Lazított latency guard ({profile_txt}) — {age_txt}{pen_txt}")
            setup_issues.append("késleltetett adat")

        score_meta = entry_thresholds.get("dynamic_score_engine") or {}
        setup_score = safe_float(score_meta.get("final_score"))
        if setup_score is None:
            setup_score = safe_float(score_meta.get("base_score"))
        regime_meta = score_meta.get("regime_penalty") if isinstance(score_meta, dict) else None
        vol_meta = score_meta.get("volatility_bonus") if isinstance(score_meta, dict) else None
        if isinstance(regime_meta, dict) and regime_meta.get("points"):
            points = safe_float(regime_meta.get("points")) or 0.0
            label = (regime_meta.get("label") or "").upper()
            sign = "−" if points < 0 else "+"
            dynamic_lines.append(f"Regime {label}: {sign}{abs(points):.1f}P")
            if points < 0:
                setup_issues.append(f"regime {label}")
        if isinstance(vol_meta, dict) and vol_meta.get("points"):
            points = safe_float(vol_meta.get("points")) or 0.0
            z_val = safe_float(vol_meta.get("volatility_z"))
            z_txt = f" z={z_val:.2f}" if z_val is not None else ""
            dynamic_lines.append(f"Volatilitás bónusz +{points:.1f}P{z_txt}")
          
    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"
    
    setup_score = setup_score if setup_score is not None else safe_float(p_raw)
    if setup_score is not None:
        if setup_score >= 60 and not setup_issues:
            setup_grade = "A"
            setup_classification = "A Setup (Prémium) — Teljes pozícióméret, agresszív menedzsment."
        elif setup_score >= 30:
            setup_grade = "B"
            issue_txt = ", ".join(setup_issues) if setup_issues else "legalább egy feltétel gyenge vagy hiányzik"
            setup_classification = (
                "B Setup (Standard) — Fél pozícióméret, szigorúbb Stop Loss. "
                f"Gyenge/hiányzó: {issue_txt}."
            )
        elif setup_score >= 25:
            setup_grade = "C"
            setup_classification = (
                "C Setup (Speculatív) — Negyed méret vagy manuális megerősítés. "
                "Csak erős triggerrel (sweep/hír/divergencia) vállald."
            )
        else:
            setup_grade = "X"
            setup_classification = (
                "❌ Setup túl gyenge — P-score <25. Csak figyelés, belépő nem ajánlott."
            )

    if setup_classification:
        direction_txt = (setup_direction or "n/a").upper()
        setup_with_direction = f"{setup_classification} — Irány: {direction_txt}"
        setup_classification_line = colorize_setup_text(setup_with_direction, setup_grade)

    # státusz
    status_emoji = "🔴"
    if dec in ("BUY", "SELL"):
        if core_bos_pending or not is_stable:
            status_emoji = "🟡"
        else:
            status_emoji = "🟢"

    base_label = dec.title() if dec else ""
    if base_label in ("Buy", "Sell") and setup_grade:
        status_label = f"{base_label} (Grade {setup_grade})"
    elif setup_grade:
        status_label = f"{setup_grade} Setup"
    else:
        status_label = base_label or dec
    status_bold  = f"{status_emoji} **{status_label}**"

    lines = [
        f"{status_bold} • P={p}% • mód: `{display_mode}`",
        f"Spot: `{spot_s}` • UTC: `{utc_s}`",
    ] 
    if setup_classification_line:
        lines.append(setup_classification_line)

    no_entry_reason = None
    missing_note = None
    if mode == "analysis_error":
        reason_detail = None
        reasons = sig.get("reasons") if isinstance(sig, dict) else None
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str) and reason.strip():
                    reason_detail = reason.strip()
                    break
        diagnostics = sig.get("diagnostics") if isinstance(sig, dict) else None
        diag_msg = None
        if isinstance(diagnostics, dict):
            diag_err = diagnostics.get("error")
            if isinstance(diag_err, dict):
                diag_msg = diag_err.get("message")
        detail = reason_detail or diag_msg
        note_suffix = f" — {detail}" if detail else ""
        no_entry_reason = f"⚠️ Analysis hiba{note_suffix}."
    missing_names = []
    for gate in missing_list:
        gate_s = str(gate or "").strip()
        if not gate_s:
            continue
        gate_s = gate_s.replace("_", " ")
        gate_s = gate_s.replace("bos", "BOS")
        missing_names.append(gate_s)

    if missing_names:
        missing_txt = ", ".join(missing_names)
        missing_note = f"⏳ Nincs belépési trigger — hiányzik: {missing_txt}."
      
          
    if dynamic_lines:
        lines.append("⚙️ Dinamikus: " + " | ".join(dynamic_lines))

    if no_entry_reason:
        lines.append(no_entry_reason)

    if missing_note and missing_note not in lines:
        lines.append(missing_note)

    if closed:
        lines.append(f"🔒 {closed_note or 'Piac zárva (market closed)'}")
    elif monitor_open and entry_open is False:
        lines.append("🌙 Entry ablak zárva — csak pozíció menedzsment, új belépő tiltva.")

    position_note = None
    if isinstance(sig, dict):
        raw_note = sig.get("position_management")
        if not raw_note:
            pm_reasons = sig.get("reasons")
            if isinstance(pm_reasons, list):
                for reason in pm_reasons:
                    if isinstance(reason, str) and reason.strip().lower().startswith("pozíciómenedzsment"):
                        raw_note = reason
                        break
        if isinstance(raw_note, str):
            raw_note = raw_note.strip()
        position_note = raw_note

    if position_note:
        if not any(line.strip() == position_note for line in lines):
            lines.append(f"🧭 {position_note}")
    # RR/TP/SL sor (ha minden adat megvan)
    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2, rr)):
        lines.append(f"@ `{fmt_num(entry)}` • SL `{fmt_num(sl)}` • TP1 `{fmt_num(t1)}` • TP2 `{fmt_num(t2)}` • RR≈`{rr}`")
    # Stabilizálás információ
    if dec in ("BUY", "SELL") and kind in ("normal", "heartbeat"):
        if core_bos_pending:
            lines.append("⏳ Állapot: *stabilizálás alatt (5m BOS megerősítésre várunk)*")
        elif not is_stable:
            lines.append("⏳ Állapot: *stabilizálás alatt*")

    # Hiányzó feltételek — ha vannak, mindig mutatjuk
    miss = missing_from_sig(sig)
    if miss and not (no_entry_reason and "hiányzik" in no_entry_reason.lower()) and not missing_note:
        lines.append(f"Hiányzó: *{miss}*")

    # cím + szín
    title = f"{emoji} **{asset}**"
    if kind == "invalidate":
        title += " • ❌ Invalidate"
    elif kind == "flip":
        arrow = "→"
        title += f" • 🔁 Flip ({(prev_decision or '').upper()} {arrow} {dec})"
    elif kind == "heartbeat":
        title += " • ℹ️ Állapot"

    color = card_color(dec, is_stable, kind, setup_grade)

    return {
        "title": title,
        "description": "\n".join(lines),
        "color": color,
    }

# ---------------- főlogika ----------------

def main():
    audit_run_id = str(uuid4())
    position_tracker.set_audit_context(source="notify", run_id=audit_run_id)
    run_meta = {
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "workflow": os.getenv("GITHUB_WORKFLOW"),
    }

    def log_event(event: str, **fields: Any) -> None:
        payload = {
            key: value
            for key, value in run_meta.items()
            if value is not None and str(value).strip()
        }
        for key, value in fields.items():
            if isinstance(value, set):
                payload[key] = sorted(value)
            elif value is not None:
                payload[key] = value
        payload["event"] = event
        LOGGER.info(event, extra=payload)

    webhooks = load_webhooks()
    if not any(val for key, val in webhooks.items() if key != "_base"):
        log_event("notify_skipped", reason="missing_webhook")
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    argv = sys.argv[1:]
    flags = normalize_cli_flags(argv)
  
    force_env = env_flag("DISCORD_FORCE_NOTIFY")
    force_heartbeat_env = env_flag("DISCORD_FORCE_HEARTBEAT")
  
    manual_flag = flag_any(flags, "manual", "manual-run", "manualmode", "m", "man")
    force_flag = flag_any(flags, "force", "force-notify", "notify-force", "f")
    heartbeat_flag = flag_any(flags, "force-heartbeat", "heartbeat", "hb", "summary", "all")
    skip_cooldown_flag = flag_any(flags, "skip-cooldown", "no-cooldown", "nocooldown", "skipcooldown")

    # Ezek a jelzők (manual/force + DISCORD_FORCE_NOTIFY) jelentik a valódi kézi kényszerítést.
    manual_context = manual_flag or force_flag or force_env

    # Ha TTY-ból futtatjuk kézzel és nincs külön flag, tekintsük manuális kényszerítésnek.
    if not flags and (sys.stdin.isatty() or sys.stdout.isatty()):
        manual_flag = True
        manual_context = True

    force_send = manual_context or skip_cooldown_flag
    force_heartbeat = manual_context or heartbeat_flag or force_heartbeat_env

    log_event(
        "notify_run_start",
        manual_flag=bool(manual_flag),
        force_flag=bool(force_flag),
        heartbeat_flag=bool(heartbeat_flag),
        manual_context=bool(manual_context),
        force_send=bool(force_send),
        force_heartbeat=bool(force_heartbeat),
        skip_cooldown=bool(skip_cooldown_flag),
        force_env=bool(force_env),
        force_heartbeat_env=bool(force_heartbeat_env),
        asset_count=len(ASSETS),
        flag_args=sorted(flags),
    )

    tdstatus = load_tdstatus()
    state = load_state()
    meta  = state.get("_meta", {})
    signal_stability_cfg = _signal_stability_config()
    tracking_cfg = (signal_stability_cfg.get("manual_position_tracking") or {})
    manual_writer = str(tracking_cfg.get("writer") or "notify").lower()
    can_write_positions = manual_writer == "notify"
    manual_tracking_enabled = bool(tracking_cfg.get("enabled"))
    positions_path = tracking_cfg.get("positions_file") or "public/_manual_positions.json"
    treat_missing_positions = bool(tracking_cfg.get("treat_missing_file_as_flat", False))
    manual_positions = position_tracker.load_positions(positions_path, treat_missing_positions)
    cooldown_map = tracking_cfg.get("post_exit_cooldown_minutes") or {}
    cooldown_default = 20
    open_commits_this_run: Set[str] = set()
    manual_states: Dict[str, Any] = {}
  
    analysis_summary = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
    run_id = run_meta.get("run_id") or os.getenv("GITHUB_RUN_ID")
    last_analysis_utc = analysis_summary.get("generated_utc") or meta.get("last_analysis_utc")

    if run_id:
        meta["run_id"] = str(run_id)
    if last_analysis_utc:
        meta["last_analysis_utc"] = last_analysis_utc

    log_event(
        "notify_state_meta",
        run_id=meta.get("run_id"),
        last_analysis_utc=meta.get("last_analysis_utc"),
        assets=len(ASSETS),
    )
    last_heartbeat_prev = meta.get("last_heartbeat_key")
    last_heartbeat_iso = meta.get("last_heartbeat_utc")
    asset_embeds = {}
    asset_channels: Dict[str, str] = {}
    now_dt  = datetime.now(timezone.utc)
    now_iso = to_utc_iso(now_dt)
    now_ep  = int(now_dt.timestamp())
    bud_dt  = bud_now()
    bud_key = bud_hh_key(bud_dt)

    per_asset_sigs = {}
    per_asset_is_stable = {}
    watcher_embeds: List[Dict[str, Any]] = []
    auto_close_embeds: List[Dict[str, Any]] = []
    manual_open_assets: Set[str] = set()
    asset_send_records: Dict[str, Dict[str, Any]] = {}

    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            sig = (analysis_summary.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0} 
        per_asset_sigs[asset] = sig

        manual_state = position_tracker.compute_state(
            asset, tracking_cfg, manual_positions, now_dt
        )
        if isinstance(sig, dict):
            sig.setdefault("position_state", manual_state)

        entry_level: Optional[float] = None
        sl_level: Optional[float] = None
        tp2_level: Optional[float] = None
        try:
            entry_level, sl_level, tp2_level = extract_trade_levels(sig)
        except Exception:
            entry_level, sl_level, tp2_level = None, None, None

        spot_price, _ = spot_from_sig_or_file(asset, sig)
        if manual_state.get("has_position"):
            changed, reason, manual_positions = position_tracker.check_close_by_levels(
                asset,
                manual_positions,
                spot_price,
                now_dt,
                _resolve_asset_value(cooldown_map, asset, cooldown_default),
            )
            if changed:
                manual_state = position_tracker.compute_state(
                    asset, tracking_cfg, manual_positions, now_dt
                )
                reason_label = "SL hit" if reason == "sl_hit" else "TP2 hit" if reason == "tp2_hit" else str(reason)
                auto_close_embeds.append(
                    {
                        "title": f"{_get_emoji(asset)} {asset} — POSITION CLOSED (AUTO)",
                        "description": f"Reason: {reason_label}\nSpot: {format_price(spot_price, asset)}\nTime: {to_utc_iso(now_dt)}",
                        "color": COLORS.get("NO", 0x95A5A6),
                    }
                )
                LOGGER.debug(
                    "CLOSE state transition %s reason=%s cooldown_until=%s",
                    asset,
                    reason,
                    manual_state.get("cooldown_until_utc"),
                )
                position_tracker.save_positions_atomic(positions_path, manual_positions)

        # --- stabilitás számítása ---
        mode_current = gates_mode(sig)
        eff = decision_of(sig)  # 'buy' | 'sell' | 'no entry'
        setup_grade = resolve_setup_grade_for_signal(sig, eff)

        st = state.get(asset, {
            "last": None, "count": 0,
            "last_sent": None,
            "last_sent_decision": None,
            "last_sent_mode": None,
            "last_sent_known": False,
            "cooldown_until": None,
        })

        if eff == st.get("last"):
            st["count"] = int(st.get("count", 0)) + 1
        else:
            st["last"]  = eff
            st["count"] = 1

        missing_list = ((sig.get("gates") or {}).get("missing") or [])
        core_bos_pending = (mode_current == "core") and ("bos5m" in missing_list)

        is_stable = st["count"] >= STABILITY_RUNS
        display_stable = is_stable and not core_bos_pending
        per_asset_is_stable[asset] = display_stable
        is_actionable_now = (eff in ("buy","sell")) and is_stable and not core_bos_pending

        notify_meta = sig.get("notify") if isinstance(sig, dict) else {}
        intent = sig.get("intent") if isinstance(sig, dict) else None
        if manual_tracking_enabled and intent in {"manage_position", "hard_exit"}:
            if not manual_state.get("has_position"):
                notify_meta = dict(notify_meta or {})
                notify_meta["should_notify"] = False
                notify_meta["reason"] = "no_open_position_tracked"
                sig["notify"] = notify_meta
        if manual_tracking_enabled and intent == "entry":
            if manual_state.get("cooldown_active"):
                notify_meta = dict(notify_meta or {})
                notify_meta.setdefault("should_notify", False)
                notify_meta.setdefault("reason", "cooldown_active")
                notify_meta.setdefault("cooldown_until_utc", manual_state.get("cooldown_until_utc"))
                sig["notify"] = notify_meta
            elif manual_state.get("has_position"):
                notify_meta = dict(notify_meta or {})
                notify_meta.setdefault("should_notify", False)
                notify_meta.setdefault("reason", "position_already_open")
                sig["notify"] = notify_meta
        should_notify = True
        if isinstance(notify_meta, dict):
            should_notify = bool(notify_meta.get("should_notify", True))
        if not should_notify:
            if intent == "entry":
                position_tracker.log_audit_event(
                    "entry suppressed before notify dispatch",
                    event="ENTRY_SUPPRESSED",
                    asset=asset,
                    intent=intent,
                    decision=eff,
                    entry_side=eff if eff in {"buy", "sell"} else None,
                    setup_grade=setup_grade,
                    stable=bool(display_stable),
                    gates_missing=missing_list,
                    notify_should_notify=False,
                    notify_reason=(notify_meta or {}).get("reason"),
                    manual_tracking_enabled=manual_tracking_enabled,
                    manual_has_position=manual_state.get("has_position"),
                    manual_cooldown_active=manual_state.get("cooldown_active"),
                    cooldown_until_utc=manual_state.get("cooldown_until_utc"),
                    suppression_reason=(notify_meta or {}).get("reason") or "notify_blocked",
                    send_kind=None,
                )
            state[asset] = st
            per_asset_sigs[asset] = sig
            per_asset_is_stable[asset] = display_stable
            continue

        cooldown_until_iso = st.get("cooldown_until")
        cooldown_active = False
        if COOLDOWN_MIN > 0 and cooldown_until_iso:
            cooldown_active = now_ep < iso_to_epoch(cooldown_until_iso)
        if force_send:
            cooldown_active = False

        prev_sent_decision = st.get("last_sent_decision")
        flip_cd_min = _flip_cooldown_minutes(asset)
        last_sent_iso = st.get("last_sent")
        flip_cd_active = False
        age_sec: Optional[int] = None
        if (
            flip_cd_min > 0
            and last_sent_iso
            and prev_sent_decision in ("buy", "sell")
            and eff in ("buy", "sell")
            and eff != prev_sent_decision
        ):
            age_sec = now_ep - iso_to_epoch(last_sent_iso)
            if age_sec >= 0 and age_sec < flip_cd_min * 60:
                flip_cd_active = True

        # --- küldési döntés ---
        send_kind = None  # None | "normal" | "invalidate" | "flip"

        if is_actionable_now:
            if prev_sent_decision in ("buy","sell"):
                if eff != prev_sent_decision:
                    if flip_cd_active:
                        print(f"notify: flip suppressed by cooldown ({asset}) age_sec={age_sec} cd_min={flip_cd_min}")
                        send_kind = None
                    else:
                        send_kind = "flip"
                else:
                    if not cooldown_active:
                        send_kind = "normal"
            else:
                if not cooldown_active:
                    send_kind = "normal"
        else:
            if prev_sent_decision in ("buy","sell") and eff == "no entry" and is_stable:
                send_kind = "invalidate"

        if send_kind is None and intent in {"hard_exit", "manage_position"}:
            send_kind = "normal"
            display_stable = True

        if send_kind == "invalidate" and manual_state.get("has_position"):
            send_kind = None

        if send_kind is None and intent == "entry":
            suppression_reason = (notify_meta or {}).get("reason")
            if suppression_reason is None:
                if not is_actionable_now:
                    suppression_reason = "not_actionable"
                elif cooldown_active:
                    suppression_reason = "cooldown_active"
                elif flip_cd_active:
                    suppression_reason = "flip_cooldown_active"
                else:
                    suppression_reason = "send_kind_none"
            position_tracker.log_audit_event(
                "entry suppressed by dispatcher",
                event="ENTRY_SUPPRESSED",
                asset=asset,
                intent=intent,
                decision=eff,
                entry_side=eff if eff in {"buy", "sell"} else None,
                setup_grade=setup_grade,
                stable=bool(display_stable),
                gates_missing=missing_list,
                notify_should_notify=should_notify,
                notify_reason=(notify_meta or {}).get("reason"),
                manual_tracking_enabled=manual_tracking_enabled,
                manual_has_position=manual_state.get("has_position"),
                manual_cooldown_active=manual_state.get("cooldown_active"),
                cooldown_until_utc=manual_state.get("cooldown_until_utc"),
                suppression_reason=suppression_reason,
                send_kind=None,
            )
          
        # --- embed + állapot frissítés ---
        if send_kind:
            embed = build_mobile_embed_for_asset(
                asset,
                state,
                sig,
                eff,
                mode_current,
                display_stable,
                is_flip=send_kind == "flip",
                is_invalidate=send_kind == "invalidate",
                kind=send_kind,
                manual_positions=manual_positions,
            )
            asset_embeds[asset] = embed
            channel = classify_signal_channel(eff, send_kind, display_stable)
            if intent in {"hard_exit", "manage_position"}:
                channel = "management"
            asset_channels[asset] = channel
            asset_send_records[asset] = {
                "asset": asset,
                "channel": channel,
                "send_kind": send_kind,
                "intent": intent,
                "decision": eff,
                "setup_grade": setup_grade,
                "stable": bool(display_stable),
                "entry_level": entry_level,
                "sl": sl_level,
                "tp2": tp2_level,
                "manual_tracking_enabled": manual_tracking_enabled,
                "manual_has_position": manual_state.get("has_position"),
                "manual_cooldown_active": manual_state.get("cooldown_active"),
                "cooldown_until_utc": manual_state.get("cooldown_until_utc"),
                "notify_should_notify": bool((notify_meta or {}).get("should_notify", True)),
                "notify_reason": (notify_meta or {}).get("reason"),
            }
            if send_kind in ("normal","flip"):
                cooldown_minutes = COOLDOWN_MIN
                if COOLDOWN_MIN > 0 and mode_current == "momentum":
                    cooldown_minutes = MOMENTUM_COOLDOWN_MIN
                st = update_asset_send_state(
                    st,
                    decision=eff,
                    now=datetime.fromtimestamp(now_ep, tz=timezone.utc),
                    cooldown_minutes=cooldown_minutes,
                    mode=mode_current,
                )
                mark_heartbeat(meta, bud_key, now_iso)                
            elif send_kind == "invalidate":
                st = update_asset_send_state(
                    st,
                    decision="no entry",
                    now=datetime.fromtimestamp(now_ep, tz=timezone.utc),
                    cooldown_minutes=0,
                    mode=None,
                )
                mark_heartbeat(meta, bud_key, now_iso)

        manual_positions, manual_state, positions_changed, entry_opened = _apply_manual_position_transitions(
            asset=asset,
            intent=intent,
            decision=eff,
            setup_grade=setup_grade,
            notify_meta=notify_meta,
            signal_payload=sig,
            manual_tracking_enabled=manual_tracking_enabled,
            can_write_positions=can_write_positions,
            manual_state=manual_state,
            manual_positions=manual_positions,
            tracking_cfg=tracking_cfg,
            now_dt=now_dt,
            now_iso=now_iso,
            send_kind=send_kind,
            display_stable=display_stable,
            missing_list=missing_list,
            cooldown_map=cooldown_map,
            cooldown_default=cooldown_default,
        )

        if positions_changed:
            position_tracker.save_positions_atomic(positions_path, manual_positions)
            sig["position_state"] = manual_state

        if positions_changed and entry_opened:
            entry_level, sl_level, tp2_level = extract_trade_levels(sig)
            position_tracker.log_audit_event(
                "entry open committed",
                event="OPEN_COMMIT",
                asset=asset,
                intent=intent,
                decision=eff,
                entry_side=eff,
                setup_grade=setup_grade,
                entry=entry_level,
                sl=sl_level,
                tp2=tp2_level,
                positions_file=positions_path,
                send_kind=send_kind,
            )
            open_commits_this_run.add(asset)

        manual_states[asset] = manual_state

        state[asset] = st

        if manual_tracking_enabled and manual_state.get("has_position"):
            manual_open_assets.add(asset)

    audit_path = position_tracker.resolve_repo_path("public/_manual_positions_audit.jsonl")
    prior_open_commits = _load_prior_open_commits(audit_path)
    for asset, mstate in manual_states.items():
        if (
            mstate.get("has_position")
            and asset not in open_commits_this_run
            and asset not in prior_open_commits
        ):
            position_tracker.log_audit_event(
                "manual position missing OPEN_COMMIT",
                event="INCONSISTENT_STATE",
                asset=asset,
                opened_at_utc=mstate.get("opened_at_utc"),
                positions_file=positions_path,
            )
          
    watcher = ActivePositionWatcher(
        _build_active_watcher_config(),
        tdstatus=tdstatus,
        signals=per_asset_sigs,
        now=datetime.now(timezone.utc),
        manual_positions=manual_positions,
        manual_tracking_cfg=tracking_cfg,
    )
    watcher_embeds = watcher.run(
        allowed_assets=manual_open_assets if manual_tracking_enabled else None
    )

    # --- Heartbeat: MINDEN órában, ha az órában még nem ment ki event ---
    heartbeat_due = last_heartbeat_prev != bud_key
    if not heartbeat_due:
        last_hb_dt = parse_utc(last_heartbeat_iso)
        if last_hb_dt is None:
            heartbeat_due = True
        else:
            delta = now_dt - last_hb_dt
            if delta < timedelta(0) or delta >= timedelta(minutes=HEARTBEAT_STALE_MIN):
                heartbeat_due = True
    want_heartbeat = force_heartbeat or heartbeat_due
    heartbeat_added = False
    heartbeat_snapshots: List[Dict[str, Any]] = []
    if want_heartbeat:
        for asset in ASSETS:
            sig = per_asset_sigs.get(asset) or {"asset": asset, "signal": "no entry", "probability": 0}
            is_stable = per_asset_is_stable.get(asset, True)        
            if asset not in asset_embeds:
                asset_embeds[asset] = build_mobile_embed_for_asset(
                    asset,
                    state,
                    sig,
                    decision_of(sig),
                    gates_mode(sig),
                    is_stable=is_stable,
                    is_flip=False,
                    is_invalidate=False,
                    kind="heartbeat",
                    manual_positions=manual_positions,                  
                )
                asset_channels.setdefault(asset, "market_scan")
                heartbeat_added = True

        heartbeat_snapshots = watcher.snapshot_embeds(exclude=watcher.changed_assets)
        if heartbeat_snapshots:
            heartbeat_added = True

        if heartbeat_added:
            mark_heartbeat(meta, bud_key, now_iso)

    state["_meta"] = meta
    save_state(state)

    live_embeds = [asset_embeds[a] for a in ASSETS if asset_channels.get(a) == "live"]
    management_embeds = list(watcher_embeds)
    management_embeds.extend(auto_close_embeds)
    market_scan_embeds = [
        asset_embeds[a]
        for a in ASSETS
        if a in asset_embeds and asset_channels.get(a, "market_scan") != "live"
    ]
    market_scan_embeds.extend(heartbeat_snapshots)
    gate_embed = build_entry_gate_summary_embed()
    if gate_embed:
        market_scan_embeds.append(gate_embed)

    pipeline_embed = build_pipeline_diag_embed(now=now_dt)
    if pipeline_embed:
        market_scan_embeds.append(pipeline_embed)

    if not (live_embeds or management_embeds or market_scan_embeds):
        print("Discord notify: nothing to send.")
        return

    bud_str = bud_time_str(bud_dt)
    title  = f"📣 eToro-Riasztás • Budapest: {bud_str}"
    headers = {
        "live": "Aktív BUY/SELL jelek (#🚨-live-signals)",
        "management": "Pozíció menedzsment / zárás (#💼-management)",
        "market_scan": "Piaci státusz, várakozás (#📊-market-scan)",
    }

    channel_payloads = {
        "live": (webhooks.get("live"), live_embeds),
        "management": (webhooks.get("management"), management_embeds),
        "market_scan": (webhooks.get("market_scan"), market_scan_embeds),
    }

    dispatched = False
    for channel, (hook, embeds) in channel_payloads.items():
        if not embeds:
            continue
        content = f"**{title}**\n{headers[channel]}"
        try:
            post_batches(hook, content, embeds)
            for asset, record in asset_send_records.items():
                if record.get("channel") != channel:
                    continue
                position_tracker.log_audit_event(
                    "Discord dispatch completed",
                    event="DISCORD_SEND",
                    **record,
                )
            dispatched = True
        except Exception as e:
            print(f"Discord notify FAILED ({channel}):", e)

    if dispatched:
        print("Discord notify OK.")
    
if __name__ == "__main__":
    main()
