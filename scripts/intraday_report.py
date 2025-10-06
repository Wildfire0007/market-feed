#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intraday_report.py — TD-only verzió
-----------------------------------
Kizárólag a Trading.py által generált lokális JSON-okra épít:
  public/<ASSET>/spot.json
  public/<ASSET>/klines_5m.json
  public/<ASSET>/klines_1h.json
  public/<ASSET>/klines_4h.json
  public/<ASSET>/k1d.json (opcionális)

Kimenet:
  reports/analysis_report.md
  reports/summary.csv

Stratégiai szabályok (rövidített):
  - Top-down bias: 4H → 1H (EMA200/50/21/9 + piaci struktúra)
  - Sweep feltétel: 1H/4H utolsó gyertya/impulzus „kiszúrás” (H/L fölé-alá wick) és visszazárás
  - 5M BOS megerősítés: lokális swing áttörése a trend irányába
  - 79% Fibonacci retrace konfluencia (±2% tolerancia)
  - ATR-szűrő: túl alacsony volatilitás = no trade; extrém magas = méretcsökkentés (info jelleggel)
  - RR ≥ 1.5; P ≥ 60%
  - Leverage limit: kripto/index ≤3×, nemesfém ≤2×

Megjegyzés: Egyszerűsített, determinisztikus szabályrendszer — célja a stabil riport generálás.
"""

import os, json, math, csv, pathlib, statistics
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# --- Konstansok és beállítások ------------------------------------------------

ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
PUBLIC = os.getenv("PUBLIC_DIR", "public")
REPORT_DIR = os.getenv("REPORT_DIR", "reports")

LEVERAGE = {
    "SOL": 3.0,
    "NSDQ100": 3.0,
    "GOLD_CFD": 2.0,
}

MAX_RISK_PCT = 1.8  # csak kijelzésre használjuk itt
FIB_TOL = 0.02      # 79% ±2% sáv
ATR_LOW_TH = 0.0008 # túl alacsony vol. (relatív) → no-trade
ATR_HIGH_TH = 0.02  # extrém vol. jelzés (méretcsökkentés javasolt)

# --- Kisegítő I/O --------------------------------------------------------------

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# --- TA függvények -------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bbands(series: pd.Series, period: int = 20, stds: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + stds * sd
    lower = ma - stds * sd
    return lower, ma, upper

def as_df_klines(raw: Any) -> pd.DataFrame:
    """Elfogadja:
       - list[{'t','o','h','l','c','v'}]
       - {'values': [...]} (TD)
       - {'raw': {'values': [...]} } (Trading.py mentése)
       és a 'datetime/open/high/low/close/volume' kulcsokat is.
    """
    if not raw:
        return pd.DataFrame()

    # 1) emeljük ki az 'arr' listát
    arr = None
    if isinstance(raw, list):
        arr = raw
    elif isinstance(raw, dict):
        if "values" in raw and isinstance(raw["values"], list):
            arr = raw["values"]
        elif "raw" in raw and isinstance(raw["raw"], dict) and "values" in raw["raw"]:
            arr = raw["raw"]["values"]
        elif "data" in raw and isinstance(raw["data"], list):
            arr = raw["data"]
    if not arr:
        return pd.DataFrame()

    # 2) normalizálás -> egységes oszlopok
    rows = []
    for x in arr:
        try:
            t = x.get("t") or x.get("datetime")
            o = x.get("o") or x.get("open")
            h = x.get("h") or x.get("high")
            l = x.get("l") or x.get("low")
            c = x.get("c") or x.get("close")
            v = x.get("v") or x.get("volume") or 0.0
            rows.append({
                "time":  pd.to_datetime(t, utc=True),
                "open":  float(o), "high": float(h),
                "low":   float(l), "close": float(c),
                "volume": float(v),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time").set_index("time")

# --- Struktúra / swing segédek -------------------------------------------------

def find_swings(df: pd.DataFrame, lb: int = 3) -> pd.DataFrame:
    """Egyszerű swing-high/low jelölés (pivot LB/RT)."""
    if df.empty:
        return df
    hi = df["high"]
    lo = df["low"]
    swing_hi = (hi.shift(lb) == hi.rolling(lb*2+1, center=True).max())
    swing_lo = (lo.shift(lb) == lo.rolling(lb*2+1, center=True).min())
    out = df.copy()
    out["swing_hi"] = swing_hi.fillna(False)
    out["swing_lo"] = swing_lo.fillna(False)
    return out

def last_swing_levels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Utolsó valid swing high/low értékek."""
    if df.empty or ("swing_hi" not in df.columns):
        return None, None
    hi_level = df[df["swing_hi"]].tail(1)["high"].values
    lo_level = df[df["swing_lo"]].tail(1)["low"].values
    hi_val = float(hi_level[0]) if len(hi_level) else None
    lo_val = float(lo_level[0]) if len(lo_level) else None
    return hi_val, lo_val

def detect_sweep(df: pd.DataFrame, lookback: int = 24) -> Dict[str, Any]:
    """Egyszerű sweep: utolsó gyertya wick-je áttöri az előző N high/low-t, de a zárás visszatér."""
    out = {"sweep_high": False, "sweep_low": False}
    if len(df) < lookback + 2:
        return out
    ref = df.iloc[-(lookback+1):-1]
    last = df.iloc[-1]
    prev_max = ref["high"].max()
    prev_min = ref["low"].min()
    # High sweep
    if last["high"] > prev_max and last["close"] < prev_max:
        out["sweep_high"] = True
    # Low sweep
    if last["low"] < prev_min and last["close"] > prev_min:
        out["sweep_low"] = True
    return out

def detect_bos(df: pd.DataFrame, direction: str, lb: int = 10) -> bool:
    """Egyszerű BOS: trendirányban áttöri az utolsó swinget az utolsó 'lb' gyertyában."""
    if df.empty:
        return False
    df_sw = find_swings(df, lb=2)
    hi, lo = last_swing_levels(df_sw.iloc[:-1])
    if direction == "long" and hi is not None:
        return df_sw["high"].iloc[-1] > hi
    if direction == "short" and lo is not None:
        return df_sw["low"].iloc[-1] < lo
    return False

def fib79_ok(move_hi: float, move_lo: float, price_now: float, tol: float = FIB_TOL) -> bool:
    if move_hi is None or move_lo is None:
        return False
    if move_hi == move_lo:
        return False
    # Long eset: impulzus low -> high, retrace = 79% lefelé
    if price_now <= move_hi and price_now >= move_lo:
        length = move_hi - move_lo
        level = move_lo + 0.79 * length
        return abs(price_now - level) / max(1e-9, length) <= tol
    # Short eset: impulzus high -> low, retrace = 79% felfelé
    if price_now >= move_lo and price_now <= move_hi:
        length = move_hi - move_lo
        level = move_hi - 0.79 * length
        return abs(price_now - level) / max(1e-9, length) <= tol
    return False

# --- Bias meghatározás ---------------------------------------------------------

def bias_from_emas(df: pd.DataFrame) -> str:
    """Egyszerű EMA-k alapján: long/short/neutral"""
    if df.empty:
        return "neutral"
    c = df["close"]
    e9 = ema(c, 9).iloc[-1]
    e21 = ema(c, 21).iloc[-1]
    e50 = ema(c, 50).iloc[-1]
    e200 = ema(c, 200).iloc[-1]
    last = c.iloc[-1]
    if last > e200 and e50 > e200 and e9 > e21:
        return "long"
    if last < e200 and e50 < e200 and e9 < e21:
        return "short"
    return "neutral"

# --- Jelzésképzés --------------------------------------------------------------

@dataclass
class Signal:
    asset: str
    decision: str            # "LONG"/"SHORT"/"no entry"
    probability: int
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    rr: Optional[float] = None
    leverage: Optional[float] = None
    reason: str = ""
    spot_price: Optional[float] = None
    spot_utc: Optional[str] = None

def build_signal(asset: str, spot: Dict[str, Any], k5m: pd.DataFrame, k1h: pd.DataFrame, k4h: pd.DataFrame) -> Signal:
    # Alapadatok
    spot_price = float(spot.get("price") or spot.get("price_usd") or np.nan)
    spot_utc = spot.get("utc", "")

    # Bias
    bias4h = bias_from_emas(k4h)
    bias1h = bias_from_emas(k1h)
    trend_bias = "long" if (bias4h == "long" and bias1h != "short") else ("short" if (bias4h == "short" and bias1h != "long") else "neutral")

    # Sweep a HTF-eken
    sw1h = detect_sweep(k1h, lookback=24)
    sw4h = detect_sweep(k4h, lookback=24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]

    # 5M BOS a bias irányába
    bos5m = detect_bos(k5m, "long" if trend_bias=="long" else "short")

    # ATR a 5M-n
    atr5 = atr(k5m).iloc[-1] if not k5m.empty else np.nan
    rel_atr = float(atr5 / spot_price) if (atr5 and spot_price) else np.nan
    atr_ok = not (np.isnan(rel_atr) or rel_atr < ATR_LOW_TH)

    # 79% Fib konfluencia: egyszerűen az utolsó 1H swing hi/lo (ha van)
    k1h_sw = find_swings(k1h, lb=2)
    move_hi, move_lo = last_swing_levels(k1h_sw)
    fib_ok = fib79_ok(move_hi, move_lo, spot_price)

    # P-score
    P = 20
    reasons = []
    if trend_bias != "neutral":
        P += 20; reasons.append(f"Bias(4H→1H)={trend_bias}")
    if swept:
        P += 15; reasons.append("HTF sweep ok")
    if bos5m:
        P += 15; reasons.append("5M BOS trendirányba")
    if fib_ok:
        P += 20; reasons.append("79% Fib konfluencia")
    if atr_ok:
        P += 10; reasons.append("ATR rendben")
    P = max(0, min(100, P))

    # Döntés & szintek (egyszerűsített szintlogika)
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)

    if P >= 60 and trend_bias in ("long","short") and bos5m and fib_ok and atr_ok:
        decision = "LONG" if trend_bias=="long" else "SHORT"
        # SL: swing + 0.2*ATR
        k5_sw = find_swings(k5m, lb=2)
        hi5, lo5 = last_swing_levels(k5_sw)
        atrbuf = float(atr5 or 0.0) * 0.2
        if decision == "LONG":
            entry = spot_price
            sl = (lo5 if lo5 is not None else (spot_price - atr5)) - atrbuf
            risk = max(1e-6, entry - sl)
            tp1 = entry + 1.0 * risk
            tp2 = entry + 2.5 * risk
            rr = (tp2 - entry) / risk
        else:
            entry = spot_price
            sl = (hi5 if hi5 is not None else (spot_price + atr5)) + atrbuf
            risk = max(1e-6, sl - entry)
            tp1 = entry - 1.0 * risk
            tp2 = entry - 2.5 * risk
            rr = (entry - tp2) / risk

        # Matematikai ellenőrzés & RR
        ok_math = (
            (decision=="LONG"  and (sl < entry < tp1 <= tp2)) or
            (decision=="SHORT" and (tp2 <= tp1 < entry < sl))
        )
        if not ok_math or rr < 1.5:
            decision = "no entry"

    reason = "; ".join(reasons) if reasons else "nincs elég konfluencia"
    return Signal(asset, decision, int(P), entry, sl, tp1, tp2, (round(rr,2) if rr else None), lev, reason, spot_price, spot_utc)

# --- Jelentés generálás --------------------------------------------------------

def pct_move(a: float, b: float) -> float:
    return abs(b - a) / max(1e-9, a)

def pl_at_100(entry: float, target: float, lev: float) -> float:
    return 100.0 * lev * pct_move(entry, target)

def format_signal(sig: Signal) -> str:
    def fmtf(x, digits=4):
        try:
            return ("{:." + str(digits) + "f}").format(float(x))
        except Exception:
            return "—"

    price_s = fmtf(sig.spot_price)
    utc_s = sig.spot_utc or "-"
    hdr = (
        "### {asset}\n\n"
        "Spot (USD): **{price}** • UTC: `{utc}`\n"
        "Valószínűség: **P = {p}%**\n"
        "Forrás: Twelve Data (lokális JSON)\n\n"
    ).format(asset=sig.asset, price=price_s, utc=utc_s, p=sig.probability)

    if sig.decision == "no entry":
        return hdr + "**Állapot:** no entry — {r}\n\n".format(r=sig.reason)

    e_s  = fmtf(sig.entry)
    sl_s = fmtf(sig.sl)
    t1_s = fmtf(sig.tp1)
    t2_s = fmtf(sig.tp2)
    rr_s = str(sig.rr) if sig.rr is not None else "—"

    line = (
        "[{dec} @ {e}; SL: {sl}; TP1: {t1}; TP2: {t2}; "
        "Ajánlott tőkeáttétel: {lev:.1f}×; R:R≈{rr}]\n"
    ).format(dec=sig.decision, e=e_s, sl=sl_s, t1=t1_s, t2=t2_s, lev=(sig.leverage or 0.0), rr=rr_s)

    pl1 = pl_at_100(sig.entry, sig.tp1, sig.leverage) if (sig.entry is not None and sig.tp1 is not None) else None
    pl2 = pl_at_100(sig.entry, sig.tp2, sig.leverage) if (sig.entry is not None and sig.tp2 is not None) else None
    pls = ""
    if (pl1 is not None and pl2 is not None):
        pls = "P/L@$100 → TP1: {p1} • TP2: {p2}\n".format(p1=fmtf(pl1,2), p2=fmtf(pl2,2))

    return hdr + line + pls + "Indoklás: {r}\n\n".format(r=sig.reason)

def write_markdown(signals: List[Signal]):
    ensure_dir(REPORT_DIR)
    path = os.path.join(REPORT_DIR, "analysis_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Intraday riport (Twelve Data-only)\n\n")
        f.write(f"Generálva (UTC): `{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}`\n\n")
        for sig in signals:
            f.write(format_signal(sig))
        # Rövid ellenőrző lista
        f.write("#### Elemzés & döntés checklist\n")
        f.write("- 4H→1H trend bias\n- 1H/4H sweep + retesztszitu\n- 5M BOS megerősítés\n- 79% Fib konfluencia\n- RR ≥ 1.5R, kockázat ≤ 1.8%\n")

def write_summary_csv(signals: List[Signal]):
    ensure_dir(REPORT_DIR)
    path = os.path.join(REPORT_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Asset","Decision","P%","Entry","SL","TP1","TP2","RR","Leverage","Spot","UTC"])
        for s in signals:
            w.writerow([
                s.asset, s.decision, s.probability,
                f"{s.entry:.6f}" if s.entry else "",
                f"{s.sl:.6f}" if s.sl else "",
                f"{s.tp1:.6f}" if s.tp1 else "",
                f"{s.tp2:.6f}" if s.tp2 else "",
                s.rr if s.rr else "",
                s.leverage or "",
                f"{s.spot_price:.6f}" if s.spot_price else "",
                s.spot_utc or ""
            ])

# --- Főfolyamat ----------------------------------------------------------------

def load_asset_frames(asset: str) -> Tuple[Dict[str,Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.join(PUBLIC, asset)
    spot = load_json(os.path.join(base, "spot.json")) or {}
    k5m = as_df_klines(load_json(os.path.join(base, "klines_5m.json")))
    k1h = as_df_klines(load_json(os.path.join(base, "klines_1h.json")))
    k4h = as_df_klines(load_json(os.path.join(base, "klines_4h.json")))
    return spot, k5m, k1h, k4h

def main():
    signals : List[Signal] = []
    for asset in ASSETS:
        try:
            spot, k5m, k1h, k4h = load_asset_frames(asset)
            if not spot or k5m.empty or k1h.empty or k4h.empty:
                signals.append(Signal(asset, "no entry", 0, reason="Insufficient data (spot/k5m/k1h/k4h)"))
                continue
            sig = build_signal(asset, spot, k5m, k1h, k4h)
            signals.append(sig)
        except Exception as e:
            signals.append(Signal(asset, "no entry", 0, reason=f"Error: {e}"))
    write_markdown(signals)
    write_summary_csv(signals)
    print(f"Kész: {REPORT_DIR}/analysis_report.md, summary.csv")

if __name__ == "__main__":
    main()
