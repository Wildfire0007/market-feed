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
- DISCORD_COOLDOWN_MIN (perc, default 10)
- DISCORD_FORCE_NOTIFY=1 ➜ cooldown figyelmen kívül hagyása + összefoglaló kényszerítése
- DISCORD_FORCE_HEARTBEAT=1 ➜ csak az összefoglalót kényszerítjük (cooldown marad)
"""

import os, json, sys, requests
from pathlib import Path
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

PUBLIC_DIR = "public"
ASSETS = ["EURUSD", "NSDQ100", "GOLD_CFD", "USDJPY", "USOIL"]

# ---- Active position helper config ----
TDSTATUS_PATH = f"{PUBLIC_DIR}/tdstatus.json"
EIA_OVERRIDES_PATH = f"{PUBLIC_DIR}/USOIL/eia_schedule_overrides.json"

EMA21_SLOPE_MIN = 0.0008  # 0.08%
ATR_TRAIL_MULT_DEFAULT = 0.5
ATR_TRAIL_MIN_ABS = 0.15
ATR_REL_MIN = 0.0007
EIA_RISK_WINDOW_MIN = 60

ACTIVE_POSITION_STATE_PATH = f"{PUBLIC_DIR}/_active_position_state.json"
ACTIVE_WATCHER_CONFIG = {
    "common": {
        "ema21_slope_min_abs": 0.0008,
        "atr_trail_k": (0.6, 1.0),
        "bos_tf": "5m",
        "rollover_warn_gmt": "21:00",
        "rollover_window_min": 20,
        "send_on_state_change_only": True,
    },
    "assets": {
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
    },
}

# ---- Debounce / stabilitás / cooldown ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2
HEARTBEAT_STALE_MIN = 55  # ennyi perc után küldünk összefoglalót akkor is, ha az óra nem váltott
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


COOLDOWN_MIN   = int_env("DISCORD_COOLDOWN_MIN", 10)  # perc; 0 = off
MOMENTUM_COOLDOWN_MIN = int_env("DISCORD_COOLDOWN_MOMENTUM_MIN", 8)

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
    "NSDQ100": "📈",
    "GOLD_CFD": "💰",
    "USDJPY": "💴",
    "USOIL": "🛢️",
}
COLOR = {
    "BUY":   0x2ecc71,  # zöld
    "SELL":  0x2ecc71,  # zöld
    "NO":    0xe74c3c,  # piros
    "WAIT":  0xf1c40f,  # sárga
    "FLIP":  0x3498db,  # kék
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

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

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
    if note_implies_open(note):
        note = ""

    if not note:
        note = "Piac zárva"

    lower = note.lower()
    if "market" not in lower:
        if "piac" not in lower:
            note = f"{note} • Market closed"
        else:
            note = f"{note} (market closed)"

    if reason:
        note = f"{note} – {reason}"

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

def card_color(dec: str, is_stable: bool, kind: str) -> int:
    if kind == "flip":
        return COLOR["FLIP"]
    if kind == "invalidate":
        return COLOR["NO"]
    if dec in ("BUY","SELL"):
        return COLOR["BUY"] if is_stable else COLOR["WAIT"]
    return COLOR["NO"]

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
    ) -> None:
        self.config = config or {}
        self.config_common = self.config.get("common", {})
        self.config_assets = self.config.get("assets", {})
        self.tdstatus = tdstatus or {}
        self.signals = signals or {}
        self.now = now or datetime.now(timezone.utc)
        self.anchor_state = load_anchor_state()
        self.state_cache = load_active_position_state()
        self.updated_state = False
        self.embeds: List[Dict[str, Any]] = []

    # -------------------- helpers --------------------
    def run(self) -> List[Dict[str, Any]]:
        for asset in self.ASSET_ORDER:
            embed = self._evaluate_asset(asset)
            if embed:
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
        return f"{base} • Aktív pozíció ({side_txt})"

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
        bud_dt = event_dt.astimezone(HB_TZ)
        countdown = format_hu_countdown(delta)
        pre_window = int(event_cfg.get("pre_window_min") or 0)
        minutes = delta.total_seconds() / 60.0
        event_mode = pre_window > 0 and 0 <= minutes <= pre_window
        return {
            "name": event_cfg.get("name", "EIA"),
            "datetime": event_dt,
            "budapest": f"{weekday_short_hu(bud_dt)} {bud_dt.strftime('%H:%M')} (Bp) – {countdown}",
            "delta": delta,
            "event_mode": event_mode,
            "minutes": minutes,
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
        has_open = bool(status.get("has_open_position")) and bool((status.get("side") or "").strip())
        prev_entry = self._state_cache_entry(asset)

        if not has_open:
            if prev_entry.get("state") != "FLAT":
                self.state_cache[asset] = {
                    "state": "FLAT",
                    "key": "FLAT",
                    "updated": _now_utc_iso(self.now),
                }
                self.updated_state = True
            return None

        signal = self._signal(asset)
        meta = (signal.get("active_position_meta") or {}) if isinstance(signal, dict) else {}
        decision_now = decision_of(signal)

        anchor_side, anchor_price, anchor_ts = self._resolve_anchor(asset, status)
        if anchor_side not in {"buy", "sell"}:
            return None

        slope = safe_float(meta.get("ema21_slope_signed") or signal.get("ema21_slope_1h"))
        slope_th = safe_float(meta.get("ema21_slope_threshold") or signal.get("ema21_slope_threshold"))
        if slope_th is None:
            slope_th = safe_float(cfg.get("ema21_slope_min_abs")) or EMA21_SLOPE_MIN

        structure_flag = (meta.get("structure_5m") or signal.get("bos_5m_dir") or "").lower()
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

        bos1h_long = bool(meta.get("bos1h_long"))
        bos1h_short = bool(meta.get("bos1h_short"))

        event_info = self._event_info(asset, cfg)
        rollover_active = self._rollover_active(cfg)

        status_side = (status.get("side") or "").lower()
        size = safe_float(status.get("size"))
        avg_entry = safe_float(status.get("avg_entry"))
        anchor_price_display = anchor_price or avg_entry

        exit_invalid = False
        if anchor_side == "sell":
            if invalid_level_sell is not None and last_close_1h is not None:
                exit_invalid = last_close_1h > invalid_level_sell
        else:
            if invalid_level_buy is not None and last_close_1h is not None:
                exit_invalid = last_close_1h < invalid_level_buy

        exit_bos = (anchor_side == "sell" and bos1h_long) or (anchor_side == "buy" and bos1h_short)
        exit_flip = (anchor_side == "sell" and decision_now == "buy") or (
            anchor_side == "buy" and decision_now == "sell"
        )
        exit_triggered = exit_invalid or exit_bos or exit_flip

        slope_flip = False
        if anchor_side == "sell":
            slope_flip = slope is not None and slope_th is not None and slope >= slope_th
            bos_opposite = structure_flag == "bos_up"
        else:
            slope_flip = slope is not None and slope_th is not None and slope <= -(slope_th)
            bos_opposite = structure_flag == "bos_down"

        reduce_trigger = slope_flip and bos_opposite
        event_mode = bool(event_info and event_info.get("event_mode"))

        state = "HOLD"
        exit_reasons: List[str] = []
        reduce_reasons: List[str] = []

        if exit_triggered:
            state = "EXIT"
            if exit_invalid:
                exit_reasons.append("invalid szint sérült")
            if exit_bos:
                exit_reasons.append("1h BOS fordult")
            if exit_flip:
                exit_reasons.append("rendszer FLIP ellentétes")
        elif event_mode:
            state = "EVENT"
        elif reduce_trigger:
            state = "REDUCE"
            if slope_flip:
                reduce_reasons.append("EMA21 slope ellened")
            if bos_opposite:
                reduce_reasons.append("5m BOS ellenirányú")
        elif rollover_active:
            state = "ROLLOVER"

        action_text = "✅ Akció: TART"
        if state == "REDUCE":
            detail = f" – {', '.join(reduce_reasons)}" if reduce_reasons else ""
            action_text = f"⚠ Akció: CSÖKKENT 30–50%{detail}"
        elif state == "EVENT":
            action_text = "⚠ Akció: CSÖKKENT 30–50% (EIA-ablak)"
        elif state == "EXIT":
            detail = f" – {', '.join(exit_reasons)}" if exit_reasons else ""
            action_text = f"⛔ Akció: TELJES EXIT{detail}"
        elif state == "ROLLOVER":
            action_text = "ℹ️ Akció: TART (Rollover – spread tágulhat)"

        action_field = {"name": "Akció", "value": action_text, "inline": False}

        anchor_parts = []
        if anchor_price_display is not None:
            anchor_parts.append(f"@ {fmt_num(anchor_price_display, digits=2)}")
        if size is not None:
            anchor_parts.append(f"méret {fmt_num(size, digits=2)}")
        anchor_value = f"{anchor_side.upper()}" + (" " + " • ".join(anchor_parts) if anchor_parts else "")

        invalid_level = invalid_level_sell if anchor_side == "sell" else invalid_level_buy
        arrow = "↑" if anchor_side == "sell" else "↓"
        invalid_text = (
            f"{fmt_num(invalid_level, digits=2)} (zárás {arrow} ⇒ EXIT)"
            if invalid_level is not None
            else "n/a"
        )

        slope_icon = slope_status_icon(slope, slope_th or EMA21_SLOPE_MIN, anchor_side)
        regime_text = f"EMA21 meredekség {format_percentage(slope)} ({slope_icon})"

        atr_floor_candidates = []
        if atr1h is not None:
            atr_floor_candidates.append(0.5 * atr1h)
        if invalid_buffer_abs is not None:
            atr_floor_candidates.append(invalid_buffer_abs)
        trail_floor = max(atr_floor_candidates) if atr_floor_candidates else None
        atr_range = cfg.get("atr_trail_k") or (0.6, 1.0)
        if atr1h is not None:
            base = f"{fmt_num(atr1h, digits=2)}"
            if trail_floor is not None:
                if invalid_buffer_abs is not None:
                    trail_desc = f"max(0.5×ATR, ${fmt_num(invalid_buffer_abs, digits=2)}) = ${fmt_num(trail_floor, digits=2)}"
                else:
                    trail_desc = f"≥ ${fmt_num(trail_floor, digits=2)}"
            else:
                trail_desc = "n/a"
            atr_text = f"{base} → Trail: {trail_desc}"
        else:
            atr_text = "n/a"
        atr_text += f" (aj.: {atr_range[0]:.1f}–{atr_range[1]:.1f}×ATR)"

        fields = [
            action_field,
            {"name": "Anchor", "value": anchor_value, "inline": True},
            {"name": "Invalid (1H)", "value": invalid_text, "inline": True},
            {"name": "Regime", "value": regime_text, "inline": True},
            {"name": "5m struktúra", "value": structure_display, "inline": True},
            {"name": "ATR(1h) / Trail", "value": atr_text, "inline": True},
        ]

        if event_info:
            suffix = " – Event mód" if event_mode else ""
            fields.append(
                {
                    "name": event_info.get("name", "Következő adat"),
                    "value": f"{event_info['budapest']}{suffix}",
                    "inline": False,
                }
            )

        desc_lines = [
            "TP1 után: 25–50% rész-zárás, maradék BE + költség, trail: ATR×0.6–1.0.",
        ]
        if event_mode:
            desc_lines.append("EIA -60…0 perc: momentum belépők tiltva, méret csökkentendő.")
        if rollover_active:
            desc_lines.append("Rollover – spread tágulhat.")

        embed = {
            "title": self._title_for_asset(asset, anchor_side),
            "color": (
                COLOR["NO"]
                if state == "EXIT"
                else COLOR["WAIT"]
                if state in {"REDUCE", "EVENT"}
                else COLOR["INFO"]
                if state == "ROLLOVER"
                else COLOR["BUY"]
            ),
            "fields": fields,
            "footer": {
                "text": "Ajánlás aktív pozíció menedzselésére; nem új belépő. Üzenet csak állapot-változáskor küldve (rate-limit safe).",
            },
        }
        if desc_lines:
            embed["description"] = "\n".join(desc_lines)

        state_key = f"{state}|{anchor_side}"
        if prev_entry.get("key") != state_key:
            self.state_cache[asset] = {
                "state": state,
                "key": state_key,
                "anchor": anchor_side,
                "updated": _now_utc_iso(self.now),
            }
            self.updated_state = True
            return embed

        # Állapot változatlan – frissítsük az időbélyeget, de ne küldjünk új üzenetet.
        self.state_cache[asset] = {
            "state": prev_entry.get("state", state),
            "key": prev_entry.get("key", state_key),
            "anchor": anchor_side,
            "updated": prev_entry.get("updated", _now_utc_iso(self.now)),
        }
        return None

def build_active_position_helper(
    asset: str,
    sig: dict,
    dec: str,
    kind: str,
    tdstatus: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[list]:
    if asset != "USOIL":
        return None
    status = tdstatus_for_asset(tdstatus or {}, asset)
    if not status or not status.get("has_open_position"):
        return None
    side = (status.get("side") or "").lower()
    if side not in {"sell", "buy"}:
        return None
    if dec in ("BUY", "SELL") and kind not in {"invalidate"}:
        return None

    meta = (sig or {}).get("active_position_meta") or {}
    slope_signed = safe_float(meta.get("ema21_slope_signed"))
    threshold = safe_float(meta.get("ema21_slope_threshold")) or EMA21_SLOPE_MIN
    slope_text = format_percentage(slope_signed)
    th_text = format_percentage(threshold)
    icon = slope_status_icon(slope_signed, threshold, side)

    ema_relation = (meta.get("ema21_relation") or "-").capitalize()
    structure = structure_label(meta.get("structure_5m"))

    atr1h = safe_float(meta.get("atr1h"))
    trail = None
    if atr1h is not None:
        trail = max(ATR_TRAIL_MIN_ABS, atr1h * ATR_TRAIL_MULT_DEFAULT)

    countdown_text, minutes_to_event = eia_countdown()

    avg_entry = safe_float(status.get("avg_entry"))
    size = safe_float(status.get("size"))
    pos_line = None
    if avg_entry is not None or size is not None:
        parts = []
        if avg_entry is not None:
            parts.append(f"@ {fmt_num(avg_entry, digits=2)}")
        if size is not None:
            parts.append(f"méret {fmt_num(size, digits=2)}")
        if parts:
            pos_line = f"Pozíció: {side.upper()} " + ", ".join(parts)

    helper_lines = [f"**USOIL • Active-Position Policy ({side.upper()} open)**"]
    helper_lines.append(f"Regime: {icon} (EMA21 slope: {slope_text} vs {th_text})")
    helper_lines.append(f"1h vs EMA21: {ema_relation}")
    helper_lines.append(f"Structure(5m): {structure}")
    if atr1h is not None:
        helper_lines.append(
            f"ATR(1h): {fmt_num(atr1h, digits=2)} → Trail ≈ ${fmt_num(trail, digits=2)} (≥ ${ATR_TRAIL_MIN_ABS:.2f})"
        )
    if countdown_text:
        helper_lines.append(f"Next data: EIA {countdown_text}")
    if pos_line:
        helper_lines.append(pos_line)

    actions = []
    ema_relation_lower = ema_relation.lower()
    structure_flag = (meta.get("structure_5m") or "").lower()

    if side == "sell":
        if slope_signed is not None and slope_signed <= -threshold and ema_relation_lower == "below":
            actions.append("Hold (defenzív): Tartsd, trail: ATR×0.5–1.0, TP1-nél BE+cost.")
        if ema_relation_lower == "above" and structure_flag == "bos_up":
            actions.append("Reduce 30–50%: 1h zár EMA21 fölött + 5m HH/HL (BOS↑).")
        bos_key = "bos1h_long"
        flip_target = "buy"
        exit_text = "Full Exit: 1h BOS↑ vagy rendszer FLIP BUY."
        final_note = (
            "A fenti pontok pozíció-menedzsment javaslatok, nem új belépő. Új short csak Regime=OK + 5m BOS↓ + ATR_rel≥0.07% esetén."
        )
    else:
        if slope_signed is not None and slope_signed >= threshold and ema_relation_lower == "above":
            actions.append("Hold (defenzív): Tartsd, trail: ATR×0.5–1.0, TP1-nél BE+cost.")
        if ema_relation_lower == "below" and structure_flag == "bos_down":
            actions.append("Reduce 30–50%: 1h zár EMA21 alatt + 5m LL/LH (BOS↓).")
        bos_key = "bos1h_short"
        flip_target = "sell"
        exit_text = "Full Exit: 1h BOS↓ vagy rendszer FLIP SELL."
        final_note = (
            "A fenti pontok pozíció-menedzsment javaslatok, nem új belépő. Új long csak Regime=OK + 5m BOS↑ + ATR_rel≥0.07% esetén."
        )

    bos_trigger = bool(meta.get(bos_key))
    reasons = (sig or {}).get("reasons") or []
    flip_hit = any(
        isinstance(r, str) and "flip" in r.lower() and flip_target in r.lower()
        for r in reasons
    )
    if bos_trigger or flip_hit:
        actions.append(exit_text)

    if minutes_to_event is not None and 0 <= minutes_to_event <= EIA_RISK_WINDOW_MIN:
        actions.append("Event mode: EIA −60–0 perc: reduce 30–50%, momentum off.")

    if actions:
        helper_lines.append("Akció-javaslatok:")
        helper_lines.extend(f"- {line}" for line in actions)

    helper_lines.append(final_note)

    return helper_lines


def build_embed_for_asset(asset: str, sig: dict, is_stable: bool, kind: str = "normal", prev_decision: str = None, tdstatus: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    kind: "normal" | "invalidate" | "flip" | "heartbeat"
    """
    emoji = EMOJI.get(asset, "📊")
    closed, closed_note = market_closed_info(sig)
    dec_effective = decision_of(sig).upper()
    dec = dec_effective if dec_effective in ("BUY", "SELL") else "NO ENTRY"

    p_raw = int(sig.get("probability", 0) or 0)
    p = 0 if closed else p_raw
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")
    mode = gates_mode(sig)
    missing_list = ((sig.get("gates") or {}).get("missing") or [])
    core_bos_pending = (mode == "core") and ("bos5m" in missing_list)

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    # státusz
    if core_bos_pending and dec in ("BUY","SELL"):
        status_emoji = "🟡"
    else:
        status_emoji = "🟢" if dec in ("BUY","SELL") else "🔴"
    status_bold  = f"{status_emoji} **{dec}**"

    lines = [
        f"{status_bold} • P={p}% • mód: `{mode}`",
        f"Spot: `{spot_s}` • UTC: `{utc_s}`",
    ]

    if closed:
        lines.append(f"🔒 {closed_note or 'Piac zárva (market closed)'}")

    # RR/TP/SL sor (ha minden adat megvan)
    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2, rr)):
        lines.append(f"@ `{fmt_num(entry)}` • SL `{fmt_num(sl)}` • TP1 `{fmt_num(t1)}` • TP2 `{fmt_num(t2)}` • RR≈`{rr}`")
    # Stabilizálás információ
    if dec in ("BUY","SELL") and kind in ("normal","heartbeat"):
        if core_bos_pending:
            lines.append("⏳ Állapot: *stabilizálás alatt (5m BOS megerősítésre várunk)*")
        elif not is_stable:
            lines.append("⏳ Állapot: *stabilizálás alatt*")

    # Hiányzó feltételek — ha vannak, mindig mutatjuk
    miss = missing_from_sig(sig)
    if miss:
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

    color = card_color(dec, is_stable, kind)

    helper_lines = build_active_position_helper(asset, sig, dec, kind, tdstatus)
    if helper_lines:
        lines.append("")
        lines.extend(helper_lines)

    return {
        "title": title,
        "description": "\n".join(lines),
        "color": color,
    }

# ---------------- főlogika ----------------

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
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

    tdstatus = load_tdstatus()
    state = load_state()
    meta  = state.get("_meta", {})
    last_heartbeat_prev = meta.get("last_heartbeat_key")
    last_heartbeat_iso = meta.get("last_heartbeat_utc")
    asset_embeds = {}
    actionable_any = False
    now_dt  = datetime.now(timezone.utc)
    now_iso = to_utc_iso(now_dt)
    now_ep  = int(now_dt.timestamp())
    bud_dt  = bud_now()
    bud_key = bud_hh_key(bud_dt)

    per_asset_sigs = {}
    per_asset_is_stable = {}
    watcher_embeds: List[Dict[str, Any]] = []

    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}
        per_asset_sigs[asset] = sig

        # --- stabilitás számítása ---
        mode_current = gates_mode(sig)
        eff = decision_of(sig)  # 'buy' | 'sell' | 'no entry'

        st = state.get(asset, {
            "last": None, "count": 0,
            "last_sent": None,
            "last_sent_decision": None,
            "last_sent_mode": None,
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
        actionable_any = actionable_any or is_actionable_now

        cooldown_until_iso = st.get("cooldown_until")
        cooldown_active = False
        if COOLDOWN_MIN > 0 and cooldown_until_iso:
            cooldown_active = now_ep < iso_to_epoch(cooldown_until_iso)
        if force_send:
            cooldown_active = False

        prev_sent_decision = st.get("last_sent_decision")

        # --- küldési döntés ---
        send_kind = None  # None | "normal" | "invalidate" | "flip"

        if is_actionable_now:
            if prev_sent_decision in ("buy","sell"):
                if eff != prev_sent_decision:
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

        # --- embed + állapot frissítés ---
        if send_kind:
            asset_embeds[asset] = build_embed_for_asset(
                asset,
                sig,
                display_stable,
                kind=send_kind,
                prev_decision=prev_sent_decision,
                tdstatus=tdstatus,
            )
            if send_kind in ("normal","flip"):
                cooldown_minutes = COOLDOWN_MIN
                if COOLDOWN_MIN > 0 and mode_current == "momentum":
                    cooldown_minutes = MOMENTUM_COOLDOWN_MIN
                if cooldown_minutes > 0:
                    st["cooldown_until"] = datetime.fromtimestamp(
                        now_ep + cooldown_minutes*60, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    st["cooldown_until"] = None
                st["last_sent"] = now_iso
                st["last_sent_decision"] = eff
                st["last_sent_mode"] = mode_current
                mark_heartbeat(meta, bud_key, now_iso)
            elif send_kind == "invalidate":
                st["last_sent"] = now_iso
                st["last_sent_decision"] = "no entry"
                st["last_sent_mode"] = None
                mark_heartbeat(meta, bud_key, now_iso)

        state[asset] = st

    watcher = ActivePositionWatcher(
        ACTIVE_WATCHER_CONFIG,
        tdstatus=tdstatus,
        signals=per_asset_sigs,
        now=datetime.now(timezone.utc),
    )
    watcher_embeds = watcher.run()

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
    if want_heartbeat:
        for asset in ASSETS:
            sig = per_asset_sigs.get(asset) or {"asset": asset, "signal": "no entry", "probability": 0}
            is_stable = per_asset_is_stable.get(asset, True)
            if asset not in asset_embeds:
                asset_embeds[asset] = build_embed_for_asset(
                    asset,
                    sig,
                    is_stable=is_stable,
                    kind="heartbeat",
                    tdstatus=tdstatus,
                )
                heartbeat_added = True

        if heartbeat_added:
            mark_heartbeat(meta, bud_key, now_iso)

    state["_meta"] = meta
    save_state(state)

    ordered_embeds = [asset_embeds[a] for a in ASSETS if a in asset_embeds]
    ordered_embeds.extend(watcher_embeds)

    if not ordered_embeds:
        print("Discord notify: nothing to send.")
        return

    # Fejléc: Budapest-idővel
    bud_str = bud_time_str(bud_dt)
    title  = f"📣 eToro-Riasztás • Budapest: {bud_str}"
    header = "Aktív jelzés(ek):" if actionable_any else "Összefoglaló / változás:"
    content = f"**{title}**\n{header}"

    try:
        r = requests.post(hook, json={"content": content, "embeds": ordered_embeds[:10]}, timeout=20)
        r.raise_for_status()
        print("Discord notify OK.")
    except Exception as e:
        print("Discord notify FAILED:", e)

if __name__ == "__main__":
    main()
