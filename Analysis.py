# -*- coding: utf-8 -*-
"""
Analysis.py — eToro_Ügynök intraday jelzésképzés (alias-alapú, all_<ASSET>.json)
Forrás: Cloudflare Worker alias HTML (<pre> JSON) — /h/all?asset=SOL|NSDQ100|GOLD_CFD
Kimenet:
  public/<ASSET>/decision.json   — jelzés vagy "no entry" okokkal
  public/analysis_summary.json    — összefoglaló minden eszközre

Szabályok (kivonat):
- 4H→1H bias
- 79% Fib (kötelező)
- 5M BOS
- 1M trigger (feedből: signal.trigger_1m = True)  ← KÖTELEZŐ!
- RR ≥ 1.5R, P ≥ 60% → jelzés
"""

import os, re, json, traceback
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

# --- Beállítások
ALIAS_BASE = "https://market-feed-proxy.czipo-agnes.workers.dev"
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
LEV_CAP = {"SOL":3, "NSDQ100":3, "GOLD_CFD":2}

session = requests.Session()
session.headers.update({"User-Agent":"eToro_Ugynok_Analysis/1.2"})

def nowiso(): 
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def save_json(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)

# --- Alias olvasó: <pre> → JSON
def fetch_all_from_alias(asset: str):
    url = f"{ALIAS_BASE}/h/all?asset={asset}"
    r = session.get(url, timeout=25)
    r.raise_for_status()
    if "text/html" not in r.headers.get("content-type","").lower():
        # ritka eset: direkt JSON-t adna
        return r.json(), url
    m = re.search(r"<pre>(.*?)</pre>", r.text, flags=re.S|re.I)
    if not m:
        raise RuntimeError(f"HTML wrapper és nincs <pre>: {url}")
    try:
        data = json.loads(m.group(1))
    except Exception as e:
        raise RuntimeError(f"JSON parse hiba alias <pre>-ből: {e}")
    return data, url

# --- OHLC list -> DataFrame (UTC)
def df_from_k(ktree: dict):
    arr = (ktree or {}).get("ohlc", [])
    if not arr: 
        return pd.DataFrame(columns=["open","high","low","close"])
    ts = pd.to_datetime([x["ts"] for x in arr], utc=True)
    df = pd.DataFrame({
        "open": [float(x["open"]) for x in arr],
        "high": [float(x["high"]) for x in arr],
        "low":  [float(x["low"])  for x in arr],
        "close":[float(x["close"]) for x in arr],
    }, index=ts).sort_index()
    return df

# --- Indikátorok
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi14(c):
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = -d.clip(upper=0).rolling(14).mean()
    rs = up/(dn.replace(0,np.nan)); return 100-(100/(1+rs))
def macd(c, f=12, s=26, sig=9):
    ef, es = ema(c,f), ema(c,s); line = ef-es; signal = ema(line,sig); hist = line-signal; return line, signal, hist
def bb(c, n=20, k=2):
    m = c.rolling(n).mean(); sd = c.rolling(n).std(); return m, m+k*sd, m-k*sd

# --- Fib/BOS/RR/P
def swing_hi_lo(close, lookback=48):
    if len(close) < max(lookback,10): return None
    w = close.iloc[-lookback:]; return float(w.max()), float(w.min())

def fib79(hi, lo):
    rng = hi-lo; return float(lo+0.79*rng), float(hi-0.79*rng)

def bos5m(df5, win=40):
    if df5.empty or len(df5) < win+1: return None
    c = df5["close"]; last = c.iloc[-1]; prev = c.iloc[-(win+1):-1]
    if last > prev.max(): return "bull"
    if last < prev.min(): return "bear"
    return None

def rr(entry, sl, tp1):
    r = abs(entry-sl); 
    if r<=0: return 0.0
    return abs(tp1-entry)/r

def prob_score(bias4, bias1, bos, ema_ok, rsi_ok, macd_ok, near79, bb_ok, rr_ok, trig_ok):
    base = 40
    if bias4 in ("bull","bear"): base += 7
    if bias1 in ("bull","bear"): base += 7
    if bos: base += 10
    if ema_ok: base += 6
    if rsi_ok: base += 4
    if macd_ok: base += 4
    if near79: base += 10
    if bb_ok: base += 3
    if rr_ok: base += 6
    if trig_ok: base += 8       # explicit 1M trigger a feedből
    return int(max(0,min(95,round(base))))

def bias_from(df):
    if df.empty or len(df)<60: return None, {}
    c=df["close"]; e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    line,signal,_=macd(c); r=rsi14(c)
    b=None
    if e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1] and line.iloc[-1]>signal.iloc[-1] and r.iloc[-1]>=50: b="bull"
    elif e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1] and line.iloc[-1]<signal.iloc[-1] and r.iloc[-1]<=50: b="bear"
    return b, {"ema20":float(e20.iloc[-1]),"ema50":float(e50.iloc[-1]),"ema200":float(e200.iloc[-1]),
               "macd":float(line.iloc[-1]),"macd_signal":float(signal.iloc[-1]),"rsi14":float(r.iloc[-1])}

def side_from(b4,b1,bos):
    if b4=="bull" and b1=="bull" and bos=="bull": return "LONG"
    if b4=="bear" and b1=="bear" and bos=="bear": return "SHORT"
    return None

def build_signal(asset, side, spot, atr_proxy):
    if spot is None: return None
    if not atr_proxy or atr_proxy<=0:
        atr_proxy = 0.008*spot if asset=="SOL" else (0.004*spot if asset=="NSDQ100" else 0.0035*spot)
    e=float(spot)
    if side=="LONG":
        sl=max(0.0,e-0.9*atr_proxy); tp1=e+1.0*atr_proxy; tp2=e+2.0*atr_proxy
    else:
        sl=e+0.9*atr_proxy; tp1=e-1.0*atr_proxy; tp2=e-2.0*atr_proxy
    return {"side":side,"entry":round(e,6),"sl":round(sl,6),"tp1":round(tp1,6),"tp2":round(tp2,6),
            "atr1h_proxy":round(float(atr_proxy),6)}

def analyze(asset):
    outdir=os.path.join("public",asset); os.makedirs(outdir,exist_ok=True)
    res={"asset":asset,"ok":True,"retrieved_at_utc":nowiso(),"errors":[]}

    # --- all_<ASSET>.json a Worker aliasról (<pre> → JSON)
    try:
        all_json, used_url = fetch_all_from_alias(asset)
    except Exception as e:
        trace = traceback.format_exc()
        decision_obj={"asset":asset,"ok":False,"error":str(e),"trace":trace,"source":"alias"}
        save_json(os.path.join(outdir,"decision.json"), decision_obj)
        return decision_obj

    # --- mezők
    spot_obj = all_json.get("spot", {}) or {}
    spot = spot_obj.get("price_usd")
    spot_ts = spot_obj.get("last_updated_at")
    spot_src = used_url  # explicit source URL (alias)
    k5m = all_json.get("k5m", {}); k1h = all_json.get("k1h", {}); k4h = all_json.get("k4h", {})
    df5, df1, df4 = df_from_k(k5m), df_from_k(k1h), df_from_k(k4h)

    sig_feed = all_json.get("signal", {}) or {}
    trigger_ok = bool(sig_feed.get("trigger_1m", False))

    # --- biasok
    b4,d4 = bias_from(df4)
    b1,d1 = bias_from(df1)
    bos = bos5m(df5)

    # --- 79% Fib (1H swing) — kötelező közelség
    near79=False; fib={}
    if not df1.empty:
        sw = swing_hi_lo(df1["close"], lookback=48)
        if sw:
            hi,lo = sw
            up79,dn79 = fib79(hi,lo)
            last = df5["close"].iloc[-1] if not df5.empty else df1["close"].iloc[-1]
            tol = 0.0035*float(last)  # ~0.35%
            near79 = abs(float(last)-up79)<=tol or abs(float(last)-dn79)<=tol
            fib={"swing_hi":hi,"swing_lo":lo,"up79":up79,"dn79":dn79,"last":float(last),"tol":float(tol)}

    # --- EMA/RSI/MACD/BB megfelelőség (1H)
    ema_ok=rsi_ok=macd_ok=bb_ok=False
    if not df1.empty:
        c1=df1["close"]; e20,e50,e200=ema(c1,20),ema(c1,50),ema(c1,200)
        r=rsi14(c1); line,signal,_=macd(c1); m,up,lo=bb(c1)
        ema_ok = (e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]) or (e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1])
        rsi_ok = (r.iloc[-1]>=50) or (r.iloc[-1]<=50)
        macd_ok= (line.iloc[-1]-signal.iloc[-1])!=0
        bb_ok  = not (np.isnan(up.iloc[-1]) or np.isnan(lo.iloc[-1]))

    # --- ATR proxy (1H)
    atr_proxy=None
    if not df1.empty and len(df1)>=15:
        hi,lo,cl=df1["high"].values,df1["low"].values,df1["close"].values
        trs=[]
        for i in range(len(cl)):
            if i==0: trs.append(hi[i]-lo[i])
            else: trs.append(max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1])))
        atr_proxy=float(pd.Series(trs).rolling(14).mean().iloc[-1])

    # --- irány + jelzés + RR
    side = side_from(b4,b1,bos)
    sig=None; rr_ok=False; rr_val=0.0
    if side and spot is not None:
        sig=build_signal(asset, side, spot, atr_proxy)
        rr_val = rr(sig["entry"],sig["sl"],sig["tp1"]) if sig else 0.0
        rr_ok = rr_val>=1.5

    # --- P% becslés és végső döntés (1M trigger kötelező!)
    P = prob_score(b4,b1,bos,ema_ok,rsi_ok,macd_ok,near79,bb_ok,rr_ok,trigger_ok)
    decision = "enter" if (side and near79 and rr_ok and P>=60 and sig and trigger_ok) else "no entry"
    reasons=[]
    if side is None: reasons.append("nincs iránykonzisztencia (4H/1H/BOS)")
    if not near79: reasons.append("79% Fib nincs közel (kötelező)")
    if not rr_ok: reasons.append(f"RR < 1.5R ({round(rr_val,2)})")
    if not trigger_ok: reasons.append("1M trigger (feed) = false/hiányzik (kötelező)")
    if P<60: reasons.append(f"P < 60% ({P}%)")

    # --- leverage + P/L@$100
    lev = LEV_CAP.get(asset,2)
    pl = {}
    if sig:
        pl = {
            "TP1": round(100*lev*abs(sig["tp1"]-sig["entry"])/sig["entry"], 2),
            "TP2": round(100*lev*abs(sig["tp2"]-sig["entry"])/sig["entry"], 2),
            "SL":  round(100*lev*abs(sig["sl"] -sig["entry"])/sig["entry"], 2),
        }

    decision_obj = {
        "asset": asset,
        "retrieved_at_utc": nowiso(),
        "spot": {"price_usd": spot, "source": used_url, "retrieved_at_utc": spot_ts},
        "urls": {"alias_all": used_url},
        "diagnostics": {
            "bias_4h": b4, "diag_4h": d4, "bias_1h": b1, "diag_1h": d1,
            "bos_5m": bos, "fib_79": fib, "rr": round(rr_val,3),
            "trigger_1m": trigger_ok, "signal_feed": sig_feed
        },
        "probability_pct": P,
        "leverage_cap": lev,
        "signal": sig,
        "pl_at_100": pl,
        "decision": decision,
        "reasons": reasons,
        "summary_hu": {
            "Spot (USD)": spot,
            "Forrás": used_url,
            "Lekérés (UTC)": spot_ts,
            "Valószínűség": f"{P}%",
            "Döntés": "JELZÉS" if decision=="enter" else "no entry",
            "Javaslat": ({
                "SIDE": sig["side"], "Entry": sig["entry"], "SL": sig["sl"],
                "TP1 (~1R)": sig["tp1"], "TP2 (2R+)": sig["tp2"],
                "Ajánlott leverage": f"{lev}x", "P/L@$100": pl,
                "Lejárat":"Europe/Budapest – intraday"
            } if decision=="enter" else "—"),
            "Indoklás": ("; ".join(reasons) if reasons else "Konfluenciák teljesülnek.")
        }
    }

    save_json(os.path.join(outdir,"decision.json"), decision_obj)
    return decision_obj

def main():
    allout={"generated_at_utc":nowiso(),"assets":{}}
    for a in ASSETS:
        try:
            allout["assets"][a]=analyze(a)
        except Exception as e:
            allout["assets"][a]={"asset":a,"ok":False,"error":str(e),"trace":traceback.format_exc()}
    save_json(os.path.join("public","analysis_summary.json"), allout)

    # Rövid index HTML
    idx="<html><meta charset='utf-8'><title>Analysis</title><body><h1>Analysis</h1><ul>"+ \
        "".join([f"<li><a href='{a}/decision.json'>{a}</a></li>" for a in ASSETS])+ \
        "</ul><p><a href='analysis_summary.json'>analysis_summary.json</a></p></body></html>"
    os.makedirs("public",exist_ok=True)
    with open(os.path.join("public","analysis.html"),"w",encoding="utf-8") as f:
        f.write(idx)

if __name__=="__main__":
    try:
        main(); print("OK")
    except Exception as e:
        traceback.print_exc()
        os.makedirs("public",exist_ok=True)
        save_json(os.path.join("public","analysis_summary.json"),
                  {"ok":False,"error":str(e),"note":"Top-level exception in Analysis.py"})
