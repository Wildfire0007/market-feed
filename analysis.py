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

# --- Elemzendő eszközök ---
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]

PUBLIC_DIR = "public"

LEVERAGE = {"SOL": 3.0, "NSDQ100": 3.0, "GOLD_CFD": 2.0}
MAX_RISK_PCT = 1.8
FIB_TOL = 0.02        # 79% ±2%
ATR_LOW_TH = 0.0008   # túl alacsony rel. vol → no-trade

# -------------------------- segédek -----------------------------------

def nowiso() -> str:
    # ISO-8601 Z-suffix (UTC)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
    """
    Twelve Data time_series formátum beolvasása:
    - preferált: values[ {datetime, open, high, low, close, volume?} ]
    - kompat.:   values[ {t, o, h, l, c, v?} ]
    A kimenet indexe UTC datetime, oszlopok: open, high, low, close (float).
    """
    if not raw:
        return pd.DataFrame(columns=["open","high","low","close"])

    arr = raw if isinstance(raw, list) else (raw.get("values") or [])
    rows: List[Dict[str, Any]] = []
    for x in arr:
        try:
            # TD standard kulcsok
            if "datetime" in x:
                dt = pd.to_datetime(x["datetime"], utc=True)
                o = float(x["open"]); h = float(x["high"]); l = float(x["low"]); c = float(x["close"])
                v = float(x.get("volume", 0.0) or 0.0)
            # rövid kulcsok (kompatibilitás)
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

# --- TA: EMA/RSI/ATR, swingek, sweep, BOS, 79% fib ----------------------------

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
    if direction not in ("long", "short"):  # neutral → nincs BOS vizsgálat
        return False
    sw = find_swings(df, lb=2)
    hi, lo = last_swing_levels(sw.iloc[:-1])
    if direction == "long" and hi is not None:
        return sw["high"].iloc[-1] > hi
    if direction == "short" and lo is not None:
        return sw["low"].iloc[-1] < lo
    return False

def fib79_ok(move_hi: float, move_lo: float, price_now: float, tol: float = FIB_TOL) -> bool:
    if move_hi is None or move_lo is None or move_hi == move_lo: return False
    length = move_hi - move_lo
    if length == 0: return False
    level_long  = move_lo + 0.79 * length         # long retrace
    level_short = move_hi - 0.79 * length         # short retrace
    return (abs(price_now - level_long)/abs(length) <= tol) or (abs(price_now - level_short)/abs(length) <= tol)

def bias_from_emas(df: pd.DataFrame) -> str:
    if df.empty: return "neutral"
    c = df["close"]
    e9, e21, e50, e200 = ema(c,9).iloc[-1], ema(c,21).iloc[-1], ema(c,50).iloc[-1], ema(c,200).iloc[-1]
    last = c.iloc[-1]
    if last > e200 and e50 > e200 and e9 > e21:  return "long"
    if last < e200 and e50 < e200 and e9 < e21:  return "short"
    return "neutral"

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

    spot_price = None
    spot_utc = "-"
    if spot:
        spot_price = spot.get("price")
        if spot_price is None:
            spot_price = spot.get("price_usd")
        spot_utc = spot.get("utc") or spot.get("timestamp") or "-"

    if (spot_price is None) or k5m.empty or k1h.empty or k4h.empty:
        msg = {
            "asset": asset,
            "ok": False,
            "retrieved_at_utc": nowiso(),
            "source": "Twelve Data (lokális JSON)",
            "spot": {"price": spot_price, "utc": spot_utc},
            "signal": "no entry",
            "probability": 0,
            "entry": None, "sl": None, "tp1": None, "tp2": None, "rr": None,
            "leverage": LEVERAGE.get(asset, 2.0),
            "reasons": ["Insufficient data (spot/k5m/k1h/k4h)"],
        }
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    spot_price = float(spot_price)

    # 2) Bias 4H→1H
    bias4h = bias_from_emas(k4h)
    bias1h = bias_from_emas(k1h)
    trend_bias = "long" if (bias4h=="long" and bias1h!="short") else ("short" if (bias4h=="short" and bias1h!="long") else "neutral")

    # 3) HTF sweep
    sw1h = detect_sweep(k1h, 24); sw4h = detect_sweep(k4h, 24)
    swept = sw1h["sweep_high"] or sw1h["sweep_low"] or sw4h["sweep_high"] or sw4h["sweep_low"]

    # 4) 5M BOS a trend irányába (csak ha nem neutral)
    bos5m = detect_bos(k5m, "long" if trend_bias=="long" else ("short" if trend_bias=="short" else "neutral"))

    # 5) ATR szűrő (relatív)
    atr5 = atr(k5m).iloc[-1]
    rel_atr = float(atr5 / spot_price) if (atr5 and spot_price) else float("nan")
    atr_ok = not (np.isnan(rel_atr) or rel_atr < ATR_LOW_TH)

    # 6) 79% Fib (1H swingek)
    k1h_sw = find_swings(k1h, lb=2)
    move_hi, move_lo = last_swing_levels(k1h_sw)
    fib_ok = fib79_ok(move_hi, move_lo, spot_price)

    # 7) P-score (egyszerű súlyozás)
    P, reasons = 20, []
    if trend_bias != "neutral": P += 20; reasons.append(f"Bias(4H→1H)={trend_bias}")
    if swept:                   P += 15; reasons.append("HTF sweep ok")
    if bos5m:                   P += 15; reasons.append("5M BOS trendirányba")
    if fib_ok:                  P += 20; reasons.append("79% Fib konfluencia")
    if atr_ok:                  P += 10; reasons.append("ATR rendben")
    P = max(0, min(100, P))

    # --- Kapuk összegyűjtése
    conds = {
        "bias": trend_bias in ("long","short"),
        "bos5m": bool(bos5m),
        "fib79": bool(fib_ok),
        "atr": bool(atr_ok),
    }
    can_enter = (P >= 60) and all(conds.values())
    missing = [k for k, v in conds.items() if not v]

    # 8) Döntés + szintek
    decision = "no entry"
    entry = sl = tp1 = tp2 = rr = None
    lev = LEVERAGE.get(asset, 2.0)

    if can_enter:
        decision = "buy" if trend_bias=="long" else "sell"
        # SL: utolsó 5M swing +/- 0.2 ATR puffer
        k5_sw = find_swings(k5m, lb=2)
        hi5, lo5 = last_swing_levels(k5_sw)
        atrbuf = float(atr5 or 0.0) * 0.2
        if decision == "buy":
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

        ok_math = ((decision=="buy"  and (sl < entry < tp1 <= tp2)) or
                   (decision=="sell" and (tp2 <= tp1 < entry < sl)))
        if (not ok_math) or (rr is None) or (rr < 1.5):
            missing.append("rr_math")
            decision = "no entry"
            entry = sl = tp1 = tp2 = rr = None

    # 9) Mentés: signal.json
    decision_obj = {
        "asset": asset,
        "ok": True,
        "retrieved_at_utc": nowiso(),
        "source": "Twelve Data (lokális JSON)",
        "spot": {"price": spot_price, "utc": spot_utc},
        "signal": decision,
        "probability": int(P),
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": (round(rr,2) if rr else None),
        "leverage": lev,
        "gates": {
            "required": ["bias", "bos5m", "fib79", "atr", "rr_math>=1.5"],
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

    # Egyszerű HTML kivonat
    html = "<!doctype html><meta charset='utf-8'><title>Analysis Summary</title>"
    html += "<h1>Analysis Summary (TD-only)</h1>"
    html += "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>"
    with open(os.path.join(PUBLIC_DIR, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
