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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- Elemzendő eszközök (GER40 -> USOIL) ---
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD", "BNB", "USOIL"]

PUBLIC_DIR = "public"

LEVERAGE = {
    "SOL": 3.0,
    "NSDQ100": 3.0,
    "GOLD_CFD": 2.0,
    "BNB": 3.0,
    "USOIL": 2.0,   # WTI olaj
}

MAX_RISK_PCT = 1.8
FIB_TOL = 0.02

# --- ATR küszöbök ---
ATR_LOW_TH_DEFAULT = 0.0007   # 0.07%
ATR_LOW_TH_ASSET = {
    "SOL": 0.0005,  # lazított küszöb: 0.05%
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

# --- Per-asset TP minimumok és round-trip költség + ATR alapú elvárás ---
# Ezek határozzák meg a TP1 minimális távolságát az entrytől.
TP_MIN_PCT = {        # min. TP1 távolság %-ban (entry-hez képest)
    "default": 0.0030,  # 0.30%
    "GOLD_CFD": 0.0015, # 0.15%  (arany)
    "USOIL":    0.0020, # 0.20%
    "NSDQ100":  0.0012, # 0.12% alap; cash+magas vol esetén enged 0.10%-ig
    "SOL":      0.0030, # 0.30%
    "BNB":      0.0040, # 0.40%
}
TP_MIN_ABS = {        # min. TP1 távolság abszolútban (tick/árjegyzés miatt)
    "default": 0.50,
    "GOLD_CFD": 3.0,   # arany ~3 pont
    "USOIL":    0.08,  # olaj ~0.08
    "NSDQ100":  0.80,
    "SOL":      0.50,
    "BNB":      0.50,
}
COST_ROUND_PCT_ASSET = {  # várható round-trip költség (spread+jutalék+slip) %
    "default": 0.0015,  # 0.15%
    "GOLD_CFD": 0.0008, # 0.08%
    "USOIL":    0.0010, # 0.10%
    "NSDQ100":  0.0007, # 0.07%
    "SOL":      0.0020, # 0.20%
    "BNB":      0.0020, # 0.20%
}
COST_MULT_DEFAULT = 1.5
COST_MULT_HIGH_VOL = 1.3
ATR5_MIN_MULT  = 0.5     # min. profit >= 0.5× ATR(5m)
ATR_VOL_HIGH_REL = 0.002  # 0.20% relatív ATR felett lazítjuk a költség-multit

# --- Momentum override csak kriptókra (SOL, BNB) ---
ENABLE_MOMENTUM_ASSETS = {"SOL", "BNB"}
MOMENTUM_BARS    = 5             # 5m EMA9–EMA21 legalább 5 bar
MOMENTUM_ATR_REL = 0.0008        # >= 0.08% 5m relatív ATR
MOMENTUM_BOS_LB  = 15            # szerkezeti töréshez nézett ablak (bar)

P_SCORE_MIN = 50
MICRO_BOS_P_BONUS = 8

# --- Rezsim és session beállítások ---
EMA_SLOPE_PERIOD   = 21          # 1h EMA21
EMA_SLOPE_LOOKBACK = 3           # hány baron mérjük a változást
EMA_SLOPE_TH       = 0.0007      # ~0.10% relatív elmozdulás (abs) a lookback alatt

# UTC idősávok: [(start_h, start_m, end_h, end_m), ...]; None = mindig
SESSIONS_UTC: Dict[str, Optional[List[Tuple[int,int,int,int]]]] = {
    "SOL": None,
    "BNB": None,
    "NSDQ100": [(0,0, 23,59)],    # teljes nap (0-24)
    "GOLD_CFD": None,             # gyakorlatilag egész nap
    "USOIL": None,                # WTI szinte 23h/nap — nem korlátozzuk
}

# -------------------------- segédek -----------------------------------

def nowiso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def now_utctime_hm() -> Tuple[int,int]:
    t = datetime.now(timezone.utc)
    return t.hour, t.minute

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

def session_ok(asset: str) -> bool:
    return in_any_window_utc(SESSIONS_UTC.get(asset), *now_utctime_hm())

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
    if asset == "NSDQ100" and session_flag and rel_atr >= ATR_VOL_HIGH_REL:
        return min(base, 0.0010)
    return base

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

def ema_slope_ok(df_1h: pd.DataFrame,
                 period: int = EMA_SLOPE_PERIOD,
                 lookback: int = EMA_SLOPE_LOOKBACK,
                 th: float = EMA_SLOPE_TH) -> Tuple[bool, float]:
    """EMA21 relatív meredekség 1h-n: abs(ema_now - ema_prev)/price_now >= th"""
    if df_1h.empty or len(df_1h) < period + lookback + 1:
        return False, 0.0
    c = df_1h["close"]
    e = ema(c, period)
    ema_now = float(e.iloc[-1])
    ema_prev = float(e.iloc[-1 - lookback])
    price_now = float(c.iloc[-1])
    rel = abs(ema_now - ema_prev) / max(1e-9, price_now)
    return (rel >= th), rel

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

    if (spot_price is None) or k5m.empty or k1h.empty or k4h.empty:
        msg = {
            "asset": asset,
            "ok": False,
            "retrieved_at_utc": nowiso(),
            "source": "Twelve Data (lokális JSON)",
            "spot": {"price": spot_price, "utc": spot_utc, "retrieved_at_utc": spot_retrieved},
            "signal": "no entry",
            "probability": 0,
            "entry": None, "sl": None, "tp1": None, "tp2": None, "rr": None,
            "leverage": LEVERAGE.get(asset, 2.0),
            "reasons": ["Insufficient data (spot/k5m/k1h/k4h)"],
        }
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    # --- ZÁRT gyertyák + kijelzett vs. számításhoz használt ár ---
    display_spot = float(spot_price)                                # kijelzés
    k1m_closed = k1m.iloc[:-1] if len(k1m) > 1 else k1m.copy()
    k5m_closed = k5m.iloc[:-1] if len(k5m) > 1 else k5m.copy()
    k1h_closed = k1h.iloc[:-1] if len(k1h) > 1 else k1h.copy()
    k4h_closed = k4h.iloc[:-1] if len(k4h) > 1 else k4h.copy()

    last5_close    = float(k5m_closed["close"].iloc[-1])            # stabil, lezárt 5m
    price_for_calc = last5_close

    # 2) Bias 4H→1H (zárt 1h/4h)
    bias4h = bias_from_emas(k4h_closed)
    bias1h = bias_from_emas(k1h_closed)
    trend_bias = "long" if (bias4h=="long" and bias1h!="short") else ("short" if (bias4h=="short" and bias1h!="long") else "neutral")

    # 2/b Rezsim (EMA21 meredekség 1h)
    regime_ok, regime_val = ema_slope_ok(k1h_closed, EMA_SLOPE_PERIOD, EMA_SLOPE_LOOKBACK, EMA_SLOPE_TH)

    # 3) HTF sweep (zárt 1h/4h)
    sw1h = detect_sweep(k1h_closed, 24); sw4h = detect_sweep(k4h_closed, 24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]

    # 4) 5M BOS a trend irányába (zárt 5m)
    bos5m_long = detect_bos(k5m_closed, "long")
    bos5m_short = detect_bos(k5m_closed, "short")

    # 5) ATR szűrő (relatív) — a stabil árhoz viszonyítjuk (zárt 5m)
    atr5 = atr(k5m_closed).iloc[-1]
    rel_atr = float(atr5 / price_for_calc) if (atr5 and price_for_calc) else float("nan")
    atr_threshold = atr_low_threshold(asset)
    atr_ok = not (np.isnan(rel_atr) or rel_atr < atr_threshold)

    # 6) Fib zóna (0.618–0.886) 1H swingekre, ATR(1h) alapú tűréssel — zárt 1h
    k1h_sw = find_swings(k1h_closed, lb=2)
    move_hi, move_lo = last_swing_levels(k1h_sw)
    atr1h = float(atr(k1h_closed).iloc[-1]) if not k1h_closed.empty else 0.0
    fib_ok = fib_zone_ok(
        move_hi, move_lo, price_for_calc,
        low=0.618, high=0.886,
        tol_abs=atr1h * 0.75,   # SZÉLESÍTVE: ±0.75×ATR(1h)
        tol_frac=0.02
    )

    # 6/b) Kiegészítő likviditás kontextus (1h EMA21 közelség + szerkezeti retest)
    ema21_1h = float(ema(k1h_closed["close"], 21).iloc[-1]) if not k1h_closed.empty else float("nan")
    ema21_dist_ok = (
        np.isfinite(ema21_1h)
        and not np.isnan(atr1h)
        and abs(price_for_calc - ema21_1h) <= max(atr1h, 0.0008 * price_for_calc)
    )

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
    session_ok_flag = session_ok(asset)
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
    elif candidate_dir == "short":
        strong_momentum = bool(bos5m_short or struct_retest_short or micro_bos_short)
    liquidity_relaxed = False
    liquidity_ok = liquidity_ok_base
    if not liquidity_ok_base and strong_momentum:
        high_atr_push = bool(
            atr_ok and not np.isnan(rel_atr)
            and rel_atr >= max(atr_threshold * 1.3, MOMENTUM_ATR_REL)
        )
        if high_atr_push or P >= 65:
            liquidity_ok = True
            liquidity_relaxed = True

    conds_core = {
        "session": bool(session_ok_flag),
        "regime":  bool(regime_ok),
        "bias":    effective_bias in ("long","short"),
        "bos5m":   bool(bos5m or (effective_bias == "long" and struct_retest_long) or (effective_bias == "short" and struct_retest_short)),
        "liquidity": liquidity_ok,
        "atr":     bool(atr_ok),
    }
    base_core_ok = all(v for k, v in conds_core.items() if k != "bos5m")
    bos_gate_ok = conds_core["bos5m"] or micro_bos_active
    can_enter_core = (P >= P_SCORE_MIN) and base_core_ok and bos_gate_ok
    missing_core = [k for k, v in conds_core.items() if not v]
    if micro_bos_active and not conds_core["bos5m"]:
        if "bos5m" not in missing_core:
            missing_core.append("bos5m")
    if liquidity_relaxed:
        reasons.append("Likviditási kapu lazítva erős momentum miatt")

    # --- Momentum feltételek (override) — kriptókra (zárt 5m-ből) ---
    momentum_used = False
    mom_dir: Optional[str] = None
    mom_micro_long = False
    mom_micro_short = False
    mom_required = ["session", "momentum(ema9x21)", "bos5m|struct_break", "atr", f"rr_math>={MIN_R_MOMENTUM:.1f}", "tp_min_profit"]
    missing_mom: List[str] = []

    if asset in ENABLE_MOMENTUM_ASSETS:
        e9_5 = ema(k5m_closed["close"], 9)
        e21_5 = ema(k5m_closed["close"], 21)
        bear = (e9_5 < e21_5).tail(MOMENTUM_BARS).all()
        bull = (e9_5 > e21_5).tail(MOMENTUM_BARS).all()
        bos_struct_short = struct_retest_short
        bos_struct_long  = struct_retest_long
        mom_atr_ok = not np.isnan(rel_atr) and (rel_atr >= MOMENTUM_ATR_REL)
        mom_micro_short = bool(micro_bos_short and mom_atr_ok)
        mom_micro_long  = bool(micro_bos_long and mom_atr_ok)
        bos_any_short = bool(bos5m_short or bos_struct_short or mom_micro_short)
        bos_any_long  = bool(bos5m_long or bos_struct_long or mom_micro_long)

        if session_ok_flag:
            if bear and bos_any_short and mom_atr_ok:
                mom_dir = "sell"
            elif bull and bos_any_long and mom_atr_ok:
                mom_dir = "buy"

        if mom_dir is None:
            if not session_ok_flag: missing_mom.append("session")
            if not (bear or bull):  missing_mom.append("momentum(ema9x21)")
            if bear and not bos_any_short:
                missing_mom.append("bos5m|struct_break")
            if bull and not bos_any_long:
                missing_mom.append("bos5m|struct_break")
            if not mom_atr_ok: missing_mom.append("atr")

    # 8) Döntés + szintek (RR/TP matek) — core vagy momentum
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)
    mode = "core"
    missing = list(missing_core)

    def compute_levels(decision_side: str, rr_required: float):
        nonlocal entry, sl, tp1, tp2, rr, missing
        atr5_val  = float(atr5 or 0.0)
        atr1h_val = float(atr1h or 0.0)

        # vol/költség alapú minimum kockázat és puffer
        risk_min = max(0.6 * atr5_val, 0.0035 * price_for_calc, 0.50)
        atr1h_cap = 0.12 * atr1h_val
        atr1h_component = min(max(0.0, atr1h_cap), 0.0008 * price_for_calc)
        buf      = max(0.2 * atr5_val, atr1h_component)

        k5_sw = find_swings(k5m_closed, lb=2)
        hi5, lo5 = last_swing_levels(k5_sw)

        entry = price_for_calc
        if decision_side == "buy":
            sl = (lo5 if lo5 is not None else (entry - atr5_val)) - buf
            if (entry - sl) < risk_min: sl = entry - risk_min
            risk = max(1e-6, entry - sl)
            tp1 = entry + TP1_R * risk
            tp2 = entry + TP2_R * risk
            rr  = (tp2 - entry) / risk
            tp1_dist = tp1 - entry
            ok_math = (sl < entry < tp1 <= tp2)
        else:
            sl = (hi5 if hi5 is not None else (entry + atr5_val)) + buf
            if (sl - entry) < risk_min: sl = entry + risk_min
            risk = max(1e-6, sl - entry)
            tp1 = entry - TP1_R * risk
            tp2 = entry - TP2_R * risk
            rr  = (entry - tp2) / risk
            tp1_dist = entry - tp1
            ok_math = (tp2 <= tp1 < entry < sl)

        # --- ÚJ: per-asset TP minimum + költségbuffer + ATR(5m) minimum
        cost_pct = COST_ROUND_PCT_ASSET.get(asset, COST_ROUND_PCT_ASSET["default"])
        rel_atr_local = float(rel_atr) if not np.isnan(rel_atr) else float("nan")
        high_vol = (not np.isnan(rel_atr_local)) and (rel_atr_local >= ATR_VOL_HIGH_REL)
        cost_mult = COST_MULT_HIGH_VOL if high_vol else COST_MULT_DEFAULT
        tp_min_pct = tp_min_pct_for(asset, rel_atr_local, session_ok_flag)

        min_profit_abs = max(
            TP_MIN_ABS.get(asset, TP_MIN_ABS["default"]),
            tp_min_pct * entry,
            cost_mult * cost_pct * entry,
            ATR5_MIN_MULT * atr5_val
        )

        if (not ok_math) or (rr is None) or (rr < rr_required) or (tp1_dist < min_profit_abs):
            if rr is None or rr < rr_required: missing.append(f"rr_math>={rr_required:.1f}")
            if tp1_dist < min_profit_abs: missing.append("tp_min_profit")
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
        if decision in ("buy", "sell") and not compute_levels(decision, MIN_R_CORE):
            decision = "no entry"
    else:
        if mom_dir is not None:
            mode = "momentum"
            missing = []
            momentum_used = True
            decision = mom_dir
            if not compute_levels(decision, MIN_R_MOMENTUM):
                decision = "no entry"
            else:
                reasons.append("Momentum override (5m EMA + ATR + BOS)")
                reasons.append("Momentum: rész-realizálás javasolt 2.5R-n")
                if (mom_dir == "buy" and mom_micro_long) or (mom_dir == "sell" and mom_micro_short):
                    reasons.append("Momentum: micro BOS elfogadva (1m szerkezet)")
                P = max(P, 75)
        elif asset in ENABLE_MOMENTUM_ASSETS and missing_mom:
            mode = "momentum"
            missing = list(dict.fromkeys(missing_mom))  # uniq

    # 9) Mentés: signal.json
    missing = list(dict.fromkeys(missing))
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
            "required": (
                ["session", "regime", "bias", "bos5m", "liquidity(fib|sweep|ema21|retest)", "atr", f"rr_math>={MIN_R_CORE:.1f}", "tp_min_profit"]
                if mode == "core" else
                ["session", "momentum(ema9x21)", "bos5m|struct_break", "atr", f"rr_math>={MIN_R_MOMENTUM:.1f}", "tp_min_profit"]
            ),
            "missing": missing,
        },
        "reasons": (reasons + ([f"missing: {', '.join(missing)}"] if missing else [])) or ["no signal"],
    }
    save_json(os.path.join(outdir, "signal.json"), decision_obj)
    return decision_obj

# ------------------------------- főfolyamat ------------------------------------

def main():
    summary = {"ok": True, "generated_utc": nowiso(), "assets": {}}
    for asset in ASSETS:
        try:
            res = analyze(asset)
            summary["assets"][asset] = res
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

