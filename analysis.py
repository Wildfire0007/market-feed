# -*- coding: utf-8 -*-
"""
analysis.py — TD-only intraday jelzésképző (lokális JSON-okból).
Forrás: Trading.py által generált fájlok a public/<ASSET>/ alatt.
Kimenet:
  public/<ASSET>/signal.json      — "buy" / "sell" / "no entry" + okok
  public/analysis_summary.json    — összesített státusz
  public/analysis.html            — egyszerű HTML kivonat

FRISSÍTÉSEK (2025-10-09):
- ATR_LOW_TH asset-függő (SOL/BNB 0.08%, mások 0.07%).
- EMA_SLOPE_TH 0.12% -> 0.10% (alap), GOLD_CFD/USOIL: 0.08%.
- Momentum: 7 bar, ATR_rel 0.10%; NSDQ100-ra csak 13:30–14:00 UTC közt.
- Core BOS kapu: BOS **vagy** 15-baros struktúratörés is elfogadott.
- Fib tolerancia: ±2.5% és 1.0×ATR(1h).
- P-score súlyok változatlanok, de BOS pont a BOS|struct esetén is jár.
- TP1-minimum: adaptív költségpuffer magas volnál (1.7×), különben 2.0×.
- NSDQ100 cash órákban TP_MIN_PCT 0.10% (min).

Robusztussági fixek:
- ATR(1h) és ATR(5m) NaN-védelem (fib tolerancia és SL/TP pufferek számításánál).
- ÚJ: “félkész” 5m bar szűrése — ha az utolsó 5m gyertya kora < UNFINISHED_5M_AGE_SEC, zártnak tekintjük a megelőzőt (levágjuk az utolsót).
"""

import os, json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- Elemzendő eszközök ---
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD", "BNB", "USOIL"]

PUBLIC_DIR = "public"

LEVERAGE = {
    "SOL": 3.0,
    "NSDQ100": 3.0,
    "GOLD_CFD": 2.0,
    "BNB": 3.0,
    "USOIL": 2.0,
}

MAX_RISK_PCT = 1.8

# --- “félkész” 5m bar szűrés (mp) ---
UNFINISHED_5M_AGE_SEC = int(os.getenv("UNFINISHED_5M_AGE_SEC", "240"))

# --- Fib toleranciák (ÚJ, enyhítve) ---
FIB_TOL_FRAC = 0.025              # ±2.5%
FIB_TOL_ABS_ATR1H_MULT = 1.0      # ±1.0 × ATR(1h)

# --- ATR küszöbök (REL) — asset-függő ---
ATR_LOW_TH_ASSET = {
    "default": 0.0007,  # 0.07%
    "SOL":     0.0008,  # 0.08%
    "BNB":     0.0008,  # 0.08%
    "NSDQ100": 0.0007,  # 0.07%
    "GOLD_CFD":0.0007,  # 0.07%
    "USOIL":   0.0007,  # 0.07%
}
def atr_threshold(asset: str) -> float:
    return ATR_LOW_TH_ASSET.get(asset, ATR_LOW_TH_ASSET["default"])

# --- Kereskedési/egz. küszöbök (RR/TP) ---
MIN_R   = 2.0
TP1_R   = 2.0
TP2_R   = 3.0

# --- TP1 minimumok és költségpuffer ---
TP_MIN_PCT = {
    "default": 0.0030,  # 0.30%
    "GOLD_CFD": 0.0015, # 0.15%
    "USOIL":    0.0020, # 0.20%
    "NSDQ100":  0.0020, # 0.20% (cash órákban 0.10%)
    "SOL":      0.0040, # 0.40%
    "BNB":      0.0040, # 0.40%
}
TP_MIN_ABS = {
    "default": 0.50,
    "GOLD_CFD": 3.0,
    "USOIL":    0.08,
    "NSDQ100":  0.80,
    "SOL":      0.80,
    "BNB":      0.50,
}
COST_ROUND_PCT_ASSET = {
    "default": 0.0015,  # 0.15%
    "GOLD_CFD": 0.0008, # 0.08%
    "USOIL":    0.0010, # 0.10%
    "NSDQ100":  0.0007, # 0.07%
    "SOL":      0.0020, # 0.20%
    "BNB":      0.0020, # 0.20%
}
COST_MULT_BASE      = 2.0
COST_MULT_HIGH_VOL  = 1.7
HIGH_VOL_TH         = 0.0020   # 0.20%  (ATR5/entry)
ATR5_MIN_MULT       = 0.5      # min. profit >= 0.5× ATR(5m)

# --- Momentum override eszközök + idősáv NSDQ100-ra (ÚJ) ---
ENABLE_MOMENTUM_ASSETS = {"SOL", "BNB", "NSDQ100"}
MOMENTUM_BARS    = 7
MOMENTUM_ATR_REL = 0.0010
MOMENTUM_BOS_LB  = 15

def momentum_time_ok(asset: str, now_utc: Optional[datetime] = None) -> bool:
    """Kriptók: mindig. NSDQ100: csak 13:30–14:00 UTC között."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if asset in {"SOL", "BNB"}:
        return True
    if asset == "NSDQ100":
        hm = now_utc.strftime("%H:%M")
        return "13:30" <= hm < "14:00"
    return False

# --- Rezsim és session ---
EMA_SLOPE_PERIOD   = 21
EMA_SLOPE_LOOKBACK = 3
# Eszköz-specifikus EMA-slope küszöb (ÚJ)
EMA_SLOPE_TH_ASSET = {
    "default": 0.0010,   # 0.10%
    "GOLD_CFD": 0.0008,  # 0.08%
    "USOIL":    0.0008,  # 0.08%
}
def ema_slope_threshold(asset: str) -> float:
    return EMA_SLOPE_TH_ASSET.get(asset, EMA_SLOPE_TH_ASSET["default"])

# UTC idősávok
SESSIONS_UTC: Dict[str, Optional[List[Tuple[int,int,int,int]]]] = {
    "SOL": None,
    "BNB": None,
    "NSDQ100": [(13,30, 20,0)],
    "GOLD_CFD": None,
    "USOIL": None,
}

# -------------------------- segédek -----------------------------------
def nowiso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def now_utctime_hm() -> Tuple[int,int]:
    t = datetime.now(timezone.utc); return t.hour, t.minute

def in_any_window_utc(windows: Optional[List[Tuple[int,int,int,int]]], h: int, m: int) -> bool:
    if not windows: return True
    minutes = h*60 + m
    for sh, sm, eh, em in windows:
        s = sh*60 + sm; e = eh*60 + em
        if s <= minutes <= e: return True
    return False

def session_ok(asset: str) -> bool:
    return in_any_window_utc(SESSIONS_UTC.get(asset), *now_utctime_hm())

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
    if direction not in ("long", "short"): return False
    sw = find_swings(df, lb=2)
    hi, lo = last_swing_levels(sw.iloc[:-1])
    if direction == "long" and hi is not None:
        return sw["high"].iloc[-1] > hi
    if direction == "short" and lo is not None:
        return sw["low"].iloc[-1] < lo
    return False

def broke_structure(df: pd.DataFrame, direction: str, lookback: int = MOMENTUM_BOS_LB) -> bool:
    """Egyszerű struktúratörés: az utolsó gyertya áttöri az előző N bar csúcsát/alját."""
    if df.empty or len(df) < lookback + 2: return False
    ref = df.iloc[-(lookback+1):-1]
    last = df.iloc[-1]
    if direction == "long":
        return last["high"] > ref["high"].max()
    if direction == "short":
        return last["low"] < ref["low"].min()
    return False

def fib_zone_ok(move_hi, move_lo, price_now,
                low=0.618, high=0.886,
                tol_abs=0.0, tol_frac=0.02) -> bool:
    if move_hi is None or move_lo is None or move_hi == move_lo: return False
    length = move_hi - move_lo
    if length == 0: return False
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
                 th: float = 0.0010) -> Tuple[bool, float]:
    """EMA21 relatív meredekség 1h-n: abs(ema_now - ema_prev)/price_now >= th"""
    if df_1h.empty or len(df_1h) < period + lookback + 1: return False, 0.0
    c = df_1h["close"]; e = ema(c, period)
    ema_now = float(e.iloc[-1]); ema_prev = float(e.iloc[-1 - lookback]); price_now = float(c.iloc[-1])
    rel = abs(ema_now - ema_prev) / max(1e-9, price_now)
    return (rel >= th), rel

# --- ÚJ: záratlan 5m bar szűrés a DataFrame-eken -----------------------
def _last_index_age_seconds(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    last_dt = df.index[-1].to_pydatetime()
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - last_dt).total_seconds()

def drop_unfinished_last_5m(df: pd.DataFrame, min_age_sec: int = UNFINISHED_5M_AGE_SEC) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    age = _last_index_age_seconds(df)
    if age is not None and age < min_age_sec and len(df) >= 2:
        return df.iloc[:-1].copy()
    return df.copy()

# ------------------------------ elemzés egy eszközre ---------------------------
def analyze(asset: str) -> Dict[str, Any]:
    outdir = os.path.join(PUBLIC_DIR, asset)
    os.makedirs(outdir, exist_ok=True)

    # 1) Bemenetek
    spot = load_json(os.path.join(outdir, "spot.json")) or {}
    k5m_raw = load_json(os.path.join(outdir, "klines_5m.json"))
    k1h_raw = load_json(os.path.join(outdir, "klines_1h.json"))
    k4h_raw = load_json(os.path.join(outdir, "klines_4h.json"))

    k5m, k1h, k4h = as_df_klines(k5m_raw), as_df_klines(k1h_raw), as_df_klines(k4h_raw)

    spot_price = None; spot_utc = "-"
    if spot:
        spot_price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
        spot_utc = spot.get("utc") or spot.get("timestamp") or "-"

    if (spot_price is None) or k5m.empty or k1h.empty or k4h.empty:
        msg = {
            "asset": asset, "ok": False, "retrieved_at_utc": nowiso(),
            "source": "Twelve Data (lokális JSON)",
            "spot": {"price": spot_price, "utc": spot_utc},
            "signal": "no entry", "probability": 0,
            "entry": None, "sl": None, "tp1": None, "tp2": None, "rr": None,
            "leverage": LEVERAGE.get(asset, 2.0),
            "reasons": ["Insufficient data (spot/k5m/k1h/k4h)"],
        }
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    # --- ZÁRT gyertyák + számítási ár ---
    # 5m: csak akkor vágjuk le az utolsót, ha túl friss (félkész)
    k5m_closed = drop_unfinished_last_5m(k5m, UNFINISHED_5M_AGE_SEC)
    # 1h/4h: konzervatívan mindig zárt gyertyákkal dolgozunk
    k1h_closed = k1h.iloc[:-1].copy() if len(k1h) > 1 else k1h.copy()
    k4h_closed = k4h.iloc[:-1].copy() if len(k4h) > 1 else k4h.copy()

    if k5m_closed.empty or k1h_closed.empty or k4h_closed.empty:
        msg = {
            "asset": asset, "ok": False, "retrieved_at_utc": nowiso(),
            "source": "Twelve Data (lokális JSON)",
            "spot": {"price": float(spot_price), "utc": spot_utc},
            "signal": "no entry", "probability": 0,
            "reasons": ["Insufficient closed bars after unfinished-5m filter"],
        }
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    display_spot = float(spot_price)
    last5_close    = float(k5m_closed["close"].iloc[-1])
    price_for_calc = last5_close

    # 2) Bias 4H→1H
    bias4h = bias_from_emas(k4h_closed); bias1h = bias_from_emas(k1h_closed)
    trend_bias = "long" if (bias4h=="long" and bias1h!="short") else ("short" if (bias4h=="short" and bias1h!="long") else "neutral")

    # 2/b Rezsim (EMA21 meredekség 1h) — eszközfüggő küszöb (ÚJ)
    regime_ok, regime_val = ema_slope_ok(k1h_closed, EMA_SLOPE_PERIOD, EMA_SLOPE_LOOKBACK, ema_slope_threshold(asset))

    # 3) HTF sweep
    sw1h = detect_sweep(k1h_closed, 24); sw4h = detect_sweep(k4h_closed, 24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]

    # 4) Core BOS kapu: BOS **vagy** 15-baros struktúratörés (ÚJ)
    bos5m_long  = detect_bos(k5m_closed, "long")
    bos5m_short = detect_bos(k5m_closed, "short")
    bos_struct_long  = broke_structure(k5m_closed, "long",  MOMENTUM_BOS_LB)
    bos_struct_short = broke_structure(k5m_closed, "short", MOMENTUM_BOS_LB)
    if trend_bias == "long":
        bos_core = bool(bos5m_long or bos_struct_long)
    elif trend_bias == "short":
        bos_core = bool(bos5m_short or bos_struct_short)
    else:
        bos_core = False

    # 5) ATR szűrő (relatív) — zárt 5m (NaN-védett)
    atr5_series = atr(k5m_closed)
    atr5 = float(atr5_series.iloc[-1]) if len(atr5_series) else float("nan")
    rel_atr = (atr5 / price_for_calc) if (np.isfinite(atr5) and price_for_calc) else float("nan")
    atr_ok = not (np.isnan(rel_atr) or rel_atr < atr_threshold(asset))

    # 6) Fib zóna (0.618–0.886) 1H swingekre, ENYHÍTETT toleranciával (ÚJ, NaN-védett ATR1h)
    k1h_sw = find_swings(k1h_closed, lb=2)
    move_hi, move_lo = last_swing_levels(k1h_sw)
    atr1h_series = atr(k1h_closed)
    atr1h = float(atr1h_series.iloc[-1]) if len(atr1h_series) else 0.0
    if not np.isfinite(atr1h):
        atr1h = 0.0

    fib_ok = fib_zone_ok(
        move_hi, move_lo, price_for_calc,
        low=0.618, high=0.886,
        tol_abs=atr1h * FIB_TOL_ABS_ATR1H_MULT,
        tol_frac=FIB_TOL_FRAC
    )

    # 7) P-score
    P, reasons = 20, []
    if trend_bias != "neutral": P += 20; reasons.append(f"Bias(4H→1H)={trend_bias}")
    if regime_ok:               P += 8;  reasons.append("Regime ok (EMA21 slope)")
    if swept:                   P += 15; reasons.append("HTF sweep ok")
    if bos_core:
        P += 18
        if (trend_bias=="long" and not bos5m_long) or (trend_bias=="short" and not bos5m_short):
            reasons.append("5M struktúratörés (15-bar) a BOS helyett (core)")
        else:
            reasons.append("5M BOS trendirányba")
    if fib_ok:                  P += 20; reasons.append("Fib zóna konfluencia (0.618–0.886)")
    if atr_ok:                  P += 9;  reasons.append("ATR rendben")
    P = max(0, min(100, P))

    # --- Kapuk
    liquidity_ok = bool(fib_ok or swept)
    session_ok_flag = session_ok(asset)
    conds_core = {
        "session": bool(session_ok_flag),
        "regime":  bool(regime_ok),
        "bias":    trend_bias in ("long","short"),
        "bos5m|struct_break":   bool(bos_core),
        "liquidity(fib_zone|sweep)": liquidity_ok,
        "atr":     bool(atr_ok),
    }
    can_enter_core = (P >= 60) and all(conds_core.values())
    missing_core = [k for k, v in conds_core.items() if not v]

    # --- Momentum feltételek (override)
    momentum_used = False
    mom_dir: Optional[str] = None
    missing_mom: List[str] = []

    if (asset in ENABLE_MOMENTUM_ASSETS) and momentum_time_ok(asset):
        e9_5 = ema(k5m_closed["close"], 9); e21_5 = ema(k5m_closed["close"], 21)
        bear = (e9_5 < e21_5).tail(MOMENTUM_BARS).all()
        bull = (e9_5 > e21_5).tail(MOMENTUM_BARS).all()
        bos_any_short = bool(detect_bos(k5m_closed, "short") or broke_structure(k5m_closed, "short", MOMENTUM_BOS_LB))
        bos_any_long  = bool(detect_bos(k5m_closed, "long")  or broke_structure(k5m_closed, "long",  MOMENTUM_BOS_LB))
        mom_atr_ok = not np.isnan(rel_atr) and (rel_atr >= MOMENTUM_ATR_REL)

        if session_ok_flag:
            if bear and bos_any_short and mom_atr_ok: mom_dir = "sell"
            elif bull and bos_any_long and mom_atr_ok: mom_dir = "buy"

        if mom_dir is None:
            if not session_ok_flag: missing_mom.append("session")
            if not (bear or bull):  missing_mom.append("momentum(ema9x21)")
            if not (bos_any_short if bear else bos_any_long): missing_mom.append("bos5m|struct_break")
            if not mom_atr_ok: missing_mom.append("atr")
    elif asset in ENABLE_MOMENTUM_ASSETS and asset == "NSDQ100":
        missing_mom.append("momentum_window(13:30–14:00Z)")

    # 8) Döntés + szintek
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)
    mode = "core"
    missing = list(missing_core)

    def nsdq_cash_session_now() -> bool:
        return asset == "NSDQ100" and session_ok("NSDQ100")

    def compute_levels(decision_side: str):
        nonlocal entry, sl, tp1, tp2, rr, missing
        # NaN-biztos ATR értékek
        atr5_val  = float(atr5)  if np.isfinite(atr5)  else 0.0
        atr1h_val = float(atr1h) if np.isfinite(atr1h) else 0.0

        risk_min = max(0.6 * atr5_val, 0.0035 * price_for_calc, 0.50)
        buf      = max(0.3 * atr5_val, 0.1 * atr1h_val)

        k5_sw = find_swings(k5m_closed, lb=2)
        hi5, lo5 = last_swing_levels(k5_sw)

        entry = price_for_calc
        if decision_side == "buy":
            sl = (lo5 if lo5 is not None else (entry - atr5_val)) - buf
            if (entry - sl) < risk_min: sl = entry - risk_min
            risk = max(1e-6, entry - sl)
            tp1 = entry + TP1_R * risk; tp2 = entry + TP2_R * risk
            rr  = (tp2 - entry) / risk
            tp1_dist = tp1 - entry
            ok_math = (sl < entry < tp1 <= tp2)
        else:
            sl = (hi5 if hi5 is not None else (entry + atr5_val)) + buf
            if (sl - entry) < risk_min: sl = entry + risk_min
            risk = max(1e-6, sl - entry)
            tp1 = entry - TP1_R * risk; tp2 = entry - TP2_R * risk
            rr  = (entry - tp2) / risk
            tp1_dist = entry - tp1
            ok_math = (tp2 <= tp1 < entry < sl)

        rel_atr_local = float(atr5_val / entry) if entry else 0.0
        cost_mult = COST_MULT_HIGH_VOL if rel_atr_local >= HIGH_VOL_TH else COST_MULT_BASE

        tp_min_pct_asset = TP_MIN_PCT.get(asset, TP_MIN_PCT["default"])
        if asset == "NSDQ100" and nsdq_cash_session_now():
            tp_min_pct_asset = min(tp_min_pct_asset, 0.0010)  # 0.10%

        min_profit_abs = max(
            TP_MIN_ABS.get(asset, TP_MIN_ABS["default"]),
            tp_min_pct_asset * entry,
            cost_mult * COST_ROUND_PCT_ASSET.get(asset, COST_ROUND_PCT_ASSET["default"]) * entry,
            ATR5_MIN_MULT * atr5_val
        )

        if (not ok_math) or (rr is None) or (rr < MIN_R) or (tp1_dist < min_profit_abs):
            if rr is None or rr < MIN_R: missing.append(f"rr_math>={MIN_R}")
            if tp1_dist < min_profit_abs: missing.append("tp_min_profit")
            return False
        return True

    if can_enter_core:
        decision = "buy" if trend_bias=="long" else "sell"
        mode = "core"
        if not compute_levels(decision): decision = "no entry"
    else:
        if mom_dir is not None:
            mode = "momentum"; missing = []; momentum_used = True
            decision = mom_dir
            if not compute_levels(decision):
                decision = "no entry"
            else:
                reasons.append("Momentum override (5m EMA + ATR + BOS/struct)")
                P = max(P, 75)
        elif (asset in ENABLE_MOMENTUM_ASSETS) and missing_mom:
            mode = "momentum"; missing = list(dict.fromkeys(missing_mom))

    # 9) Mentés
    decision_obj = {
        "asset": asset, "ok": True, "retrieved_at_utc": nowiso(),
        "source": "Twelve Data (lokális JSON)",
        "spot": {"price": display_spot, "utc": spot_utc},
        "signal": decision, "probability": int(P),
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": (round(rr,2) if rr else None),
        "leverage": lev,
        "gates": {
            "mode": mode,
            "required": (
                ["session", "regime", "bias", "bos5m|struct_break", "liquidity(fib_zone|sweep)", "atr", f"rr_math>={MIN_R}", "tp_min_profit"]
                if mode == "core" else
                ["session", "momentum(ema9x21)", "bos5m|struct_break", "atr", f"rr_math>={MIN_R}", "tp_min_profit"]
            ),
            "missing": missing,
        },
        "reasons": (reasons + ([f"missing: {', '.join(missing)}"] if missing else [])) or ["no signal"],
        "notes": {
            "unfinished_5m_age_sec": UNFINISHED_5M_AGE_SEC
        }
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
