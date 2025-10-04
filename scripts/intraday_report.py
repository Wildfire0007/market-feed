#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intraday riport generátor (e_Toro_Ügynök szabálykészlet váz alapján).

Bemenet (repo gyökerében, a fetch lépésből):
  - out_SOL.json
  - out_NSDQ100.json
  - out_GOLD_CFD.json
  (mindegyik tartalmaz: spot, k5m, k1h, k4h — a fetch állítja össze)

Kimenet (report/):
  - analysis_report.md   (emberi olvasásra)
  - summary.csv          (rövid táblázat)
  - (opcionális) a workflow tölti: chart_*.png, fetch_urls.log

Követelmények:
  - Python 3.11
  - pip install pandas numpy
"""

import os, json, math, csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ---- Konstansok / beállítások ----
MAX_RISK_PCT = 1.8  # max kockázat/trade (%)
LEVERAGE = {"SOL": 3.0, "NSDQ100": 3.0, "GOLD_CFD": 2.0}
EU_BUDAPEST = ZoneInfo("Europe/Budapest")


# ---- Segédfüggvények / util ----
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_dt_utc(val):
    """Rugalmas timestamp parser → timezone-aware UTC datetime."""
    if val is None:
        return None
    # epoch (sec/ms) eset
    if isinstance(val, (int, float)):
        ts = int(val)
        if ts > 10_000_000_000:  # ms-ben jöhet
            ts //= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    # ISO string eset
    s = str(val)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def as_df_ohlc(raw) -> pd.DataFrame:
    """
    OHLC idősor normalizálás többféle forrásformátumról.
    Elvárt kimenet: index=UTC datetime, oszlopok: open, high, low, close, volume
    """
    if raw is None:
        return pd.DataFrame()

    # Gyakori elrendezések: {"values":[...]}, {"data":[...]}, vagy közvetlen lista
    arr = None
    if isinstance(raw, dict):
        arr = raw.get("values")
        if not isinstance(arr, list):
            arr = raw.get("data")
    if not isinstance(arr, list):
        if isinstance(raw, list):
            arr = raw
        else:
            return pd.DataFrame()

    rows = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        ts = None
        for k in ("t", "time", "timestamp", "datetime", "date"):
            if k in x:
                ts = _to_dt_utc(x[k])
                break
        if ts is None:
            continue
        try:
            o = float(x.get("o", x.get("open")))
            h = float(x.get("h", x.get("high")))
            l = float(x.get("l", x.get("low")))
            c = float(x.get("c", x.get("close")))
        except Exception:
            # ha hiányos, ugorjuk
            continue

        v_raw = x.get("v", x.get("volume"))
        try:
            v = float(v_raw) if v_raw is not None else float("nan")
        except Exception:
            v = float("nan")
        rows.append([ts, o, h, l, c, v])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.set_index("ts").sort_index()
    # duplikált időbélyegek kiszűrése (utolsó előny)
    df = df[~df.index.duplicated(keep="last")]
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(span=n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(span=n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range egyszerű implementáció."""
    if df.empty:
        return pd.Series(dtype=float)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def bias_from_ema(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 210:
        return "neutral"
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    e200 = ema(df["close"], 200)
    last = df.index[-1]
    try:
        c = df["close"].loc[last]
        if c > e50.loc[last] and e20.loc[last] > e50.loc[last] > e200.loc[last]:
            return "bullish"
        if c < e50.loc[last] and e20.loc[last] < e50.loc[last] < e200.loc[last]:
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def recent_sweep(df: pd.DataFrame, lookback: int = 24):
    """
    Egyszerű sweep-detektálás: utolsó gyertya kilóg-e az előző (lookback) gyertyák csúcs/mélypontja fölé/alá?
    """
    if df.empty or len(df) < lookback + 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-(lookback + 1) : -1]
    hi = prev["high"].max()
    lo = prev["low"].min()
    if last.high > hi:
        return "sweep_high"
    if last.low < lo:
        return "sweep_low"
    return None


def bos_5m(df5: pd.DataFrame, direction: str, lookback: int = 30):
    """
    Nagyon leegyszerűsített BOS:
      - longnál: törje az utolsó swing high-t (záróval)
      - shortnál: törje az utolsó swing low-t (záróval)
    Visszaad: (bool, key_level)
    """
    if df5.empty or len(df5) < lookback + 3:
        return False, None
    window = df5.iloc[-(lookback + 3) : -3]
    last2 = df5.iloc[-2]  # friss, már lezárt gyertya
    if direction == "bullish":
        sh = window["high"].rolling(5).max().iloc[-1]
        return bool(last2.close > sh), float(sh)
    if direction == "bearish":
        sl = window["low"].rolling(5).min().iloc[-1]
        return bool(last2.close < sl), float(sl)
    return False, None


def retrace_79(entry_from: float, entry_to: float, price: float, tol: float = 0.02):
    """
    A 79% Fibonacci retrace szint környékén vagyunk-e? (±2% toleranciával a teljes range-re).
    entry_from → entry_to: a mozgás iránya (pl. BOS szint felé visszahúzódás).
    """
    rng = abs(entry_to - entry_from)
    if rng <= 0:
        return False, None
    fib79 = entry_to + 0.79 * (entry_from - entry_to)
    return (abs(price - fib79) <= tol * rng), fib79


def price_from_spot(spot_obj: dict) -> float:
    for k in ("price", "last", "close", "c", "value"):
        v = spot_obj.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # CoinGecko-szerű alak
    if isinstance(spot_obj, dict) and "usd" in spot_obj:
        try:
            return float(spot_obj["usd"])
        except Exception:
            pass
    return float("nan")


def spot_time_utc(spot_obj: dict):
    for k in ("retrieved_at_utc", "time", "timestamp", "datetime", "last_update"):
        if k in spot_obj:
            dt = _to_dt_utc(spot_obj[k])
            if dt:
                return dt.isoformat()
    return None


def summarize_pl100(side: str, entry: float, tp1: float, tp2: float, sl: float, leverage: float):
    """Egyszerű P/L@$100 becslés a százalékos elmozdulás alapján."""
    vals = [entry, tp1, tp2, sl]
    if any((v is None or isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    if side == "LONG":
        p1 = leverage * 100 * (tp1 - entry) / entry
        p2 = leverage * 100 * (tp2 - entry) / entry
        ps = leverage * 100 * (sl - entry) / entry
    else:
        p1 = leverage * 100 * (entry - tp1) / entry
        p2 = leverage * 100 * (entry - tp2) / entry
        ps = leverage * 100 * (entry - sl) / entry
    return round(p1, 2), round(p2, 2), round(ps, 2)


# ---- Jelzésképző logika (egyszerűsített kapuk) ----
def compute_signal(asset: str, spot_price: float,
                   k5m_df: pd.DataFrame, k1h_df: pd.DataFrame, k4h_df: pd.DataFrame) -> dict:
    res = {
        "asset": asset,
        "spot": spot_price,
        "prob": None,
        "decision": "no entry",
        "reason": [],
        "side": None,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "leverage": LEVERAGE.get(asset, 1.0),
    }

    # 4H → 1H bias összehangolás
    bias4 = bias_from_ema(k4h_df)
    bias1 = bias_from_ema(k1h_df)
    if bias4 == "neutral" or bias1 == "neutral" or bias4 != bias1:
        res["reason"].append(f"Bias alignment hiányos (4H={bias4}, 1H={bias1})")
        return res

    direction = "bullish" if bias4 == "bullish" else "bearish"

    # Sweep (1H/4H egyik idősíkon legalább)
    sw = recent_sweep(k1h_df) or recent_sweep(k4h_df)
    if sw is None:
        res["reason"].append("Nincs 1H/4H sweep")
        return res

    # 5M BOS
    ok_bos, key_level = bos_5m(k5m_df, direction)
    if not ok_bos or key_level is None:
        res["reason"].append("Nincs 5M BOS megerősítés")
        return res

    # 79% retrace a BOS zóna felé (egyszerűsített ellenőrzés)
    last_close = float(k5m_df["close"].iloc[-1])
    ok_fib, fib79 = retrace_79(entry_from=last_close, entry_to=key_level, price=last_close)
    if not ok_fib:
        res["reason"].append("79% fib retrace nincs meg")
        return res

    # SL/TP számítás (nagyon konzervatív)
    atr5 = atr(k5m_df, 14)
    atr_val = float(atr5.iloc[-1]) if len(atr5) else 0.0
    if direction == "bullish":
        entry = last_close
        sl = min(k5m_df["low"].iloc[-10:])  # ~50 perc mélypontja
        if entry <= sl:
            res["reason"].append("SL >= Entry (long) — RR invalid")
            return res
        r = entry - sl
        tp1 = entry + r
        tp2 = entry + 2.2 * r
        side = "LONG"
    else:
        entry = last_close
        sl = max(k5m_df["high"].iloc[-10:])
        if entry >= sl:
            res["reason"].append("SL <= Entry (short) — RR invalid")
            return res
        r = sl - entry
        tp1 = entry - r
        tp2 = entry - 2.2 * r
        side = "SHORT"

    # RR ellenőrzés (TP1 ~1R)
    if side == "LONG":
        rr_ok = (tp1 > entry > sl) and (tp2 > tp1 > entry) and ((tp1 - entry) / (entry - sl) >= 1.0)
    else:
        rr_ok = (tp2 < tp1 < entry < sl) and ((entry - tp1) / (sl - entry) >= 1.0)
    if not rr_ok:
        res["reason"].append("RR feltétel nem teljesül (TP1 ~1R)")
        return res

    # Valószínűség (alap 60% + kis bónusz, ha MACD & RSI is támogat)
    macd_line, signal_line, _ = macd(k1h_df["close"])
    rsi1 = float(rsi(k1h_df["close"]).iloc[-1])
    prob = 60
    if direction == "bullish" and macd_line.iloc[-1] > signal_line.iloc[-1] and rsi1 > 50:
        prob += 6
    if direction == "bearish" and macd_line.iloc[-1] < signal_line.iloc[-1] and rsi1 < 50:
        prob += 6
    prob = min(prob, 75)

    res.update({
        "prob": prob,
        "decision": "enter" if prob >= 60 else "no entry",
        "side": side,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "atr5": atr_val,
    })
    return res


# ---- Riport-összeállítás ----
def make_report_for(asset_json: dict, asset_name: str):
    spot_obj = asset_json.get("spot", {}) or {}
    k5m_df = as_df_ohlc(asset_json.get("k5m"))
    k1h_df = as_df_ohlc(asset_json.get("k1h"))
    k4h_df = as_df_ohlc(asset_json.get("k4h"))

    # Ha nincs 4H, deriváljuk 1H-ból
    if k4h_df.empty and not k1h_df.empty:
        try:
            k4h_df = (
                k1h_df.resample("4H")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
        except Exception:
            k4h_df = pd.DataFrame()

    spot_price = price_from_spot(spot_obj)
    spot_utc = spot_time_utc(spot_obj)

    res = {
        "asset": asset_name,
        "spot_price": spot_price,
        "spot_utc": spot_utc,
        "proxy": None,
        "prob": None,
        "decision": "no entry",
        "details": "",
    }

    if asset_name == "NSDQ100":
        res["proxy"] = "ETF-proxy (QQQ)"
    elif asset_name == "GOLD_CFD":
        res["proxy"] = "XAU/USD proxy"

    if k5m_df.empty or k1h_df.empty or k4h_df.empty or math.isnan(spot_price):
        res["details"] = "Hiányzó adatok (k5m/k1h/k4h/spot)"
        return res, None

    sig = compute_signal(asset_name, spot_price, k5m_df, k1h_df, k4h_df)
    res["prob"] = sig["prob"]
    res["decision"] = "Belépő" if (sig["decision"] == "enter") else "no entry"

    pl = summarize_pl100(sig["side"], sig["entry"], sig["tp1"], sig["tp2"], sig["sl"], sig["leverage"])
    pl_line = ""
    if pl:
        p1, p2, ps = pl
        pl_line = f"P/L@$100: TP1 {p1}$, TP2 {p2}$, SL {ps}$ (lev {sig['leverage']}×)"

    if sig["decision"] == "enter":
        res["details"] = (
            f"[{sig['side']} @ {sig['entry']:.4f}; SL {sig['sl']:.4f}; "
            f"TP1 {sig['tp1']:.4f}; TP2 {sig['tp2']:.4f}; lev {sig['leverage']}×; "
            f"Valószínűség: {sig['prob']}%] {pl_line}"
        )
    else:
        reasons = "; ".join(sig["reason"]) if sig["reason"] else "no entry"
        res["details"] = f"Feltételek: {reasons}"

    return res, sig


def main():
    ensure_dir("report")

    assets = ["SOL", "NSDQ100", "GOLD_CFD"]
    rows = []
    md_lines = ["# Intraday összefoglaló (automatizált)\n"]

    for a in assets:
        path = f"out_{a}.json"
        if not os.path.exists(path):
            rows.append([a, "", "", "", "", "hiányzik out_*.json"])
            md_lines.append(f"## {a}\n- Nincs bemenet: {path}\n")
            continue

        data = load_json(path)
        summary, sig = make_report_for(data, a)

        # Összefoglaló táblához
        spot_str = ""
        if isinstance(summary["spot_price"], (int, float)) and not math.isnan(summary["spot_price"]):
            spot_str = f"{summary['spot_price']:.6f}"

        rows.append([
            summary["asset"],
            spot_str,
            summary["spot_utc"] or "",
            summary.get("proxy") or "Worker JSON (/h/all | /spot/k5m/k1h/k4h fallback)",
            f"{summary.get('prob') or ''}",
            summary["decision"],
        ])

        # Részletes blokk
        md_lines.append(f"## {summary['asset']}")
        src = "Worker JSON (/h/all | /spot/k5m/k1h/k4h fallback)"
        md_lines.append(f"- Spot (USD): **{summary['spot_price']}** • UTC: {summary['spot_utc']} • Forrás: {src}")
        if summary.get("proxy"):
            md_lines.append(f"- Proxy: {summary['proxy']}")
        if summary["decision"] == "Belépő" and sig:
            md_lines.append(f"- {summary['details']}")
        else:
            md_lines.append(f"- Nincs jelzés: {summary['details']}")
        md_lines.append("")

    # Rövid CSV összefoglaló
    with open("report/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Eszköz", "Spot (USD)", "Spot UTC", "Forrás/Proxy", "P(%)", "Döntés"])
        for r in rows:
            w.writerow(r)

    # Táblázat az MD-ben is
    md_lines.append("## Rövid táblázatos összefoglaló\n")
    md_lines.append("| Eszköz | Spot (USD) | Spot UTC | Forrás/Proxy | P(%) | Döntés |")
    md_lines.append("|---|---:|---|---|---:|---|")
    for r in rows:
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")

    # --- Felhasznált URL-ek blokk (ha a fetch lépés logolt) ---
    urls_by_asset = {}
    log_path = "report/fetch_urls.log"
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    try:
                        asset, kind, url = line.split("|", 2)
                    except ValueError:
                        continue
                    urls_by_asset.setdefault(asset, set()).add(url)
        except Exception:
            # log olvasási hiba nem blokkolja a riportot
            pass

    if urls_by_asset:
        md_lines.append("\n## Felhasznált URL-ek\n")
        for a in ["SOL", "NSDQ100", "GOLD_CFD"]:
            if a in urls_by_asset and urls_by_asset[a]:
                md_lines.append(f"**{a}**")
                for u in sorted(urls_by_asset[a]):
                    md_lines.append(f"- {u}")
                md_lines.append("")

    # Riport mentése
    with open("report/analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()
