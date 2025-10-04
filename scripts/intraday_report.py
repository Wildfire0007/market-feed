#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, csv
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MAX_RISK_PCT = 1.8
LEVERAGE = {"SOL": 3.0, "NSDQ100": 3.0, "GOLD_CFD": 2.0}

EU_BUDAPEST = ZoneInfo("Europe/Budapest")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def as_df_ohlc(raw):
    """
    Elfogad többféle formátumot:
      - { "values": [ {"t"/"time"/"timestamp"/"datetime": ..., "o"/"open":..., "h":...,"l":...,"c":...,"v":...}, ...] }
      - [ { ... } ]
    """
    if raw is None:
        return pd.DataFrame()
    arr = raw.get("values", raw if isinstance(raw, list) else raw.get("data", []))
    if not isinstance(arr, list) or len(arr) == 0:
        return pd.DataFrame()

    def to_ts(x):
        for k in ("t","time","timestamp","datetime"):
            if k in x:
                val = x[k]
                if isinstance(val, (int,float)):
                    # másodperc/milliszekundum heur.
                    ts = int(val)
                    if ts > 10_000_000_000:  # ms
                        ts = ts // 1000
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                # ISO
                try:
                    return datetime.fromisoformat(str(val).replace("Z","+00:00")).astimezone(timezone.utc)
                except:
                    pass
        return None

    rows = []
    for x in arr:
        ts = to_ts(x)
        if ts is None:  # próbáljuk "date" kulccsal
            if "date" in x:
                try:
                    ts = datetime.fromisoformat(x["date"].replace("Z","+00:00")).astimezone(timezone.utc)
                except:
                    continue
            else:
                continue
        o = x.get("o", x.get("open", np.nan))
        h = x.get("h", x.get("high", np.nan))
        l = x.get("l", x.get("low", np.nan))
        c = x.get("c", x.get("close", np.nan))
        v = x.get("v", x.get("volume", np.nan))
        rows.append([ts, float(o), float(h), float(l), float(c), float(v) if v is not None else np.nan])

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts").sort_index()
    # távolítsuk el a duplikált időbélyegeket (utolsó nyer)
    df = df[~df.index.duplicated(keep="last")]
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(span=n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(span=n, adjust=False, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def macd(s, fast=12, slow=26, signal=9):
    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(s, n=20, k=2.0):
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k*sd, m - k*sd

def atr(df, n=14):
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def bias_from_ema(df):
    e20, e50, e200 = ema(df["close"],20), ema(df["close"],50), ema(df["close"],200)
    last = df.index[-1]
    try:
        if df["close"].loc[last] > e50.loc[last] and e20.loc[last] > e50.loc[last] > e200.loc[last]:
            return "bullish"
        if df["close"].loc[last] < e50.loc[last] and e20.loc[last] < e50.loc[last] < e200.loc[last]:
            return "bearish"
        return "neutral"
    except:
        return "neutral"

def recent_sweep(df, lookback=24):
    """Egyszerűsített: az utolsó gyertya kilóg-e a megelőző lookback csúcs/mélypont fölé/alá?"""
    if len(df) < lookback + 2: return None
    last = df.iloc[-1]
    prev = df.iloc[-(lookback+1):-1]
    hi = prev["high"].max()
    lo = prev["low"].min()
    if last.high > hi: return "sweep_high"
    if last.low  < lo: return "sweep_low"
    return None

def bos_5m(df5, direction, lookback=30):
    """Nagyon egyszerű BOS: longnál törje az utolsó swing high-t; shortnál swing low-t (záróval)."""
    if len(df5) < lookback + 3: return False, None
    window = df5.iloc[-(lookback+3):-3]
    last2 = df5.iloc[-2]  # friss záró
    if direction == "bullish":
        sh = window["high"].rolling(5).max().iloc[-1]
        return bool(last2.close > sh), float(sh)
    if direction == "bearish":
        sl = window["low"].rolling(5).min().iloc[-1]
        return bool(last2.close < sl), float(sl)
    return False, None

def retrace_79(entry_from, entry_to, price, tol=0.02):
    """Ellenőrizzük, hogy a price közel van-e a 79% fibo retrace szinthez (±2% tolerancia a range-re)."""
    rng = abs(entry_to - entry_from)
    if rng <= 0: return False, None
    fib79 = entry_to + 0.79*(entry_from - entry_to)
    return (abs(price - fib79) <= tol * rng), fib79

def compute_signal(asset, spot, k5m_df, k1h_df, k4h_df):
    result = {
        "asset": asset,
        "spot": spot,
        "prob": None,
        "decision": "no entry",
        "reason": [],
        "side": None,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "leverage": LEVERAGE.get(asset, 1.0)
    }

    # Bias 4H → 1H
    bias4 = bias_from_ema(k4h_df)
    bias1 = bias_from_ema(k1h_df)
    if bias4 == "neutral" or bias1 == "neutral" or bias4 != bias1:
        result["reason"].append(f"Bias alignment hiányos (4H={bias4}, 1H={bias1})")
        return result

    direction = "bullish" if bias4 == "bullish" else "bearish"

    # sweep (1H vagy 4H)
    sw = recent_sweep(k1h_df) or recent_sweep(k4h_df)
    if sw is None:
        result["reason"].append("Nincs 1H/4H sweep")
        return result

    # 5M BOS
    ok_bos, key_level = bos_5m(k5m_df, direction)
    if not ok_bos or key_level is None:
        result["reason"].append("Nincs 5M BOS megerősítés")
        return result

    # 5M ATR + 79% fib retrace (reteszt környéke)
    atr5 = atr(k5m_df, 14).iloc[-1]
    last_close = float(k5m_df["close"].iloc[-1])

    if direction == "bullish":
        # retrace a BOS zóna 79%-áig
        ok_fib, fib79 = retrace_79(entry_from=last_close, entry_to=key_level, price=last_close)
        if not ok_fib:
            result["reason"].append("79% fib retrace nincs meg (long)")
            return result
        entry = last_close
        sl = min(k5m_df["low"].iloc[-10:])  # konzerv: utolsó ~50 perc mélypontja
        if entry <= sl:
            result["reason"].append("SL >= Entry (long) — RR invalid")
            return result
        r = entry - sl
        tp1 = entry + r
        tp2 = entry + 2.2*r
        side = "LONG"
    else:
        ok_fib, fib79 = retrace_79(entry_from=last_close, entry_to=key_level, price=last_close)
        if not ok_fib:
            result["reason"].append("79% fib retrace nincs meg (short)")
            return result
        entry = last_close
        sl = max(k5m_df["high"].iloc[-10:])
        if entry >= sl:
            result["reason"].append("SL <= Entry (short) — RR invalid")
            return result
        r = sl - entry
        tp1 = entry - r
        tp2 = entry - 2.2*r
        side = "SHORT"

    # RR >= 1.5R
    rr_ok = True
    if side == "LONG":
        rr_ok = (tp1 > entry > sl) and (tp2 > tp1 > entry) and ((tp1-entry) / (entry-sl) >= 1.0)  # TP1 ~1R
    else:
        rr_ok = (tp2 < tp1 < entry < sl) and ((entry-tp1) / (sl-entry) >= 1.0)
    if not rr_ok:
        result["reason"].append("RR feltétel nem teljesül (TP1 ~1R)")
        return result

    # Valószínűség (konzerv): alap 60% + kis bónusz, ha MACD & RSI is konzisztens 1H-n
    macd_line, signal_line, hist = macd(k1h_df["close"])
    rsi1 = rsi(k1h_df["close"]).iloc[-1]
    prob = 60
    if direction == "bullish" and macd_line.iloc[-1] > signal_line.iloc[-1] and rsi1 > 50: prob += 6
    if direction == "bearish" and macd_line.iloc[-1] < signal_line.iloc[-1] and rsi1 < 50: prob += 6
    prob = min(prob, 75)

    result.update({
        "prob": prob,
        "decision": "enter" if prob >= 60 else "no entry",
        "side": side,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
    })
    return result

def price_from_spot(spot):
    # több lehetséges kulcs
    for k in ("price","last","close","c","value"):
        v = spot.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    # CoinGecko-stílus?
    if "usd" in spot: 
        try: return float(spot["usd"])
        except: pass
    return float("nan")

def spot_time_utc(spot):
    for k in ("retrieved_at_utc","time","timestamp","datetime","last_update"):
        if k in spot:
            val = spot[k]
            try:
                if isinstance(val, (int,float)):
                    ts = int(val if val < 10_000_000_000 else val//1000)
                    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                return datetime.fromisoformat(str(val).replace("Z","+00:00")).astimezone(timezone.utc).isoformat()
            except:
                continue
    return None

def summarize_pl100(side, entry, tp1, tp2, sl, leverage):
    if any(map(lambda x: x is None or math.isnan(x), [entry, tp1, tp2, sl])): 
        return None
    if side == "LONG":
        p1 = leverage * 100 * (tp1 - entry) / entry
        p2 = leverage * 100 * (tp2 - entry) / entry
        ps = leverage * 100 * (sl  - entry) / entry
    else:
        p1 = leverage * 100 * (entry - tp1) / entry
        p2 = leverage * 100 * (entry - tp2) / entry
        ps = leverage * 100 * (entry - sl ) / entry
    return round(p1,2), round(p2,2), round(ps,2)

def make_report_for(asset_json, asset_name):
    spot_obj = asset_json.get("spot", {}) or {}
    k5m_df = as_df_ohlc(asset_json.get("k5m"))
    k1h_df = as_df_ohlc(asset_json.get("k1h"))
    k4h_df = as_df_ohlc(asset_json.get("k4h"))
    # ha nincs k4h → deriváld 1H-ból
    if k4h_df.empty and not k1h_df.empty:
        k4h_df = k1h_df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    spot_price = price_from_spot(spot_obj)
    spot_utc = spot_time_utc(spot_obj)

    res = {"asset": asset_name, "spot_price": spot_price, "spot_utc": spot_utc,
           "proxy": None, "prob": None, "decision": "no entry", "details": ""}

    if k5m_df.empty or k1h_df.empty or k4h_df.empty or math.isnan(spot_price):
        res["details"] = "Hiányzó adatok (k5m/k1h/k4h/spot)"
        return res, None

    sig = compute_signal(asset_name, spot_price, k5m_df, k1h_df, k4h_df)
    res["prob"] = sig["prob"]
    res["decision"] = "Belépő" if (sig["decision"] == "enter") else "no entry"

    # proxy jelzés
    if asset_name == "NSDQ100":
        res["proxy"] = "ETF-proxy (QQQ)"
    elif asset_name == "GOLD_CFD":
        res["proxy"] = "XAU/USD proxy"

    # PL@100 számítás
    pl = summarize_pl100(sig["side"], sig["entry"], sig["tp1"], sig["tp2"], sig["sl"], sig["leverage"])
    pl_line = ""
    if pl:
        p1, p2, ps = pl
        pl_line = f"P/L@$100: TP1 {p1}$, TP2 {p2}$, SL {ps}$ (lev {sig['leverage']}×)"

    # rövid emberi szöveg
    if sig["decision"] == "enter":
        res["details"] = (f"[{sig['side']} @ {sig['entry']:.4f}; SL {sig['sl']:.4f}; "
                          f"TP1 {sig['tp1']:.4f}; TP2 {sig['tp2']:.4f}; lev {sig['leverage']}×; "
                          f"Valószínűség: {sig['prob']}%] {pl_line}")
    else:
        res["details"] = "Feltételek: " + ("; ".join(sig["reason"]) if sig["reason"] else "no entry")

    return res, sig

def main():
    ensure_dir("report")

    assets = ["SOL","NSDQ100","GOLD_CFD"]
    rows = []
    md_lines = ["# Intraday összefoglaló (automatizált)\n"]

    for a in assets:
        path = f"out_{a}.json"
        if not os.path.exists(path):
            rows.append([a, "", "", "", "", "hiányzik out_*.json"])
            continue
        data = load_json(path)
        summary, sig = make_report_for(data, a)

        rows.append([
            summary["asset"],
            f"{summary['spot_price']:.6f}" if isinstance(summary["spot_price"], (int,float)) and not math.isnan(summary["spot_price"]) else "",
            summary["spot_utc"] or "",
            summary.get("proxy") or "",
            f"{summary.get('prob') or ''}",
            summary["decision"]
        ])

        # rész MD
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

    # CSV összefoglaló
    with open("report/summary.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Eszköz","Spot (USD)","Spot UTC","Forrás/Proxy","P(%)","Döntés"])
        for r in rows:
            w.writerow(r)

    # MD riport
    md_lines.append("## Rövid táblázatos összefoglaló")
    md_lines.append("")
    md_lines.append("| Eszköz | Spot (USD) | Spot UTC | Forrás/Proxy | P(%) | Döntés |")
    md_lines.append("|---|---:|---|---|---:|---|")
    for r in rows:
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")

    with open("report/analysis_report.md","w",encoding="utf-8") as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    main()
