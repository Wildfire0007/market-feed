# -*- coding: utf-8 -*-
"""
Analysis.py — eToro_Ügynök intraday jelzésképzés (Worker alias forrás)
Forrás: Cloudflare Worker alias /h/all?asset=SOL|NSDQ100|GOLD_CFD (JSON)
Kimenet:
  public/<ASSET>/decision.json   — jelzés vagy "no entry" okokkal
  public/analysis_summary.json    — összefoglaló minden eszközre
  public/analysis.html            — egyszerű index
Szabályok: 4H→1H bias, 79% Fib (kötelező), 5M BOS, RR≥1.5R, P≥60% → jelzés
"""

import os, json, re, time, traceback
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

# --- Worker ALIAS BÁZIS (Cloudflare)
BASE = "https://market-feed-proxy.czipo-agnes.workers.dev"

# --- Elemzendő eszközök és leverage cap
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
LEV_CAP = {"SOL":3, "NSDQ100":3, "GOLD_CFD":2}

session = requests.Session()
session.headers.update({"User-Agent":"eToro_Ugynok_Analysis/1.2"})

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def fetch_all_from_alias(asset: str) -> dict:
    """/h/all alias hívás cache-busttal (ts=epoch)"""
    url = f"{BASE}/h/all?asset={asset}&ts={int(time.time())}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json(), url

# --- OHLC normalizáló: támogatja az ohlc_utc_ms listát és az ISO-s 'ohlc' listát is
def df_from_ohlc_obj(obj) -> pd.DataFrame:
    if not obj:
        return pd.DataFrame(columns=["open","high","low","close"])
    if "ohlc_utc_ms" in obj and isinstance(obj["ohlc_utc_ms"], list):
        arr = obj["ohlc_utc_ms"]
        if not arr:
            return pd.DataFrame(columns=["open","high","low","close"])
        idx = pd.to_datetime([x[0] for x in arr], unit="ms", utc=True)
        df = pd.DataFrame({
            "open":[float(x[1]) for x in arr],
            "high":[float(x[2]) for x in arr],
            "low" :[float(x[3]) for x in arr],
            "close":[float(x[4]) for x in arr],
        }, index=idx).sort_index()
        return df
    if "ohlc" in obj and isinstance(obj["ohlc"], list):
        arr = obj["ohlc"]
        if not arr:
            return pd.DataFrame(columns=["open","high","low","close"])
        idx = pd.to_datetime([x.get("ts") for x in arr], utc=True)
        df = pd.DataFrame({
            "open":[float(x["open"]) for x in arr],
            "high":[float(x["high"]) for x in arr],
            "low" :[float(x["low"])  for x in arr],
            "close":[float(x["close"]) for x in arr],
        }, index=idx).sort_index()
        return df
    return pd.DataFrame(columns=["open","high","low","close"])

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
    r = abs(entry-sl)
    if r<=0: return 0.0
    return abs(tp1-entry)/r

def prob_score(bias4, bias1, bos, ema_ok, rsi_ok, macd_ok, near79, bb_ok, rr_ok):
    base = 45
    if bias4 in ("bull","bear"): base += 8
    if bias1 in ("bull","bear"): base += 8
    if bos: base += 10
    if ema_ok: base += 7
    if rsi_ok: base += 5
    if macd_ok: base += 5
    if near79: base += 8
    if bb_ok: base += 4
    if rr_ok: base += 8
    return int(max(0,min(95,round(base))))

def bias_from(df: pd.DataFrame):
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

# --- fő logika egy eszközre
def analyze(asset: str) -> dict:
    outdir=os.path.join("public",asset); os.makedirs(outdir,exist_ok=True)
    res={"asset":asset,"ok":True,"retrieved_at_utc":nowiso(),"errors":[]}

    # 1) aliasból mindent egyben
    try:
        all_json, used_url = fetch_all_from_alias(asset)
    except Exception as e:
        msg = {"asset":asset,"ok":False,"error":str(e),"source":"alias"}
        save_json(os.path.join(outdir,"decision.json"), msg)
        return msg

    # 2) parse
    spot_obj = all_json.get("spot") or {}
    spot = spot_obj.get("price_usd")
    spot_src = spot_obj.get("source_url") or used_url
    spot_ts  = spot_obj.get("retrieved_at_utc")

    df5 = df_from_ohlc_obj(all_json.get("k5m"))
    df1 = df_from_ohlc_obj(all_json.get("k1h"))
    df4 = df_from_ohlc_obj(all_json.get("k4h"))

    # 3) bias
    b4,d4=bias_from(df4); b1,d1=bias_from(df1)
    bos=bos5m(df5)

    # 4) 79% fib (1H swing)
    near79=False; fib={}
    if not df1.empty:
        sw=swing_hi_lo(df1["close"],lookback=48)
        if sw:
            hi,lo=sw; up79,dn79=fib79(hi,lo)
            last = df5["close"].iloc[-1] if not df5.empty else df1["close"].iloc[-1]
            tol=0.0035*float(last)  # ~0.35%
            near79= abs(float(last)-up79)<=tol or abs(float(last)-dn79)<=tol
            fib={"swing_hi":hi,"swing_lo":lo,"up79":up79,"dn79":dn79,"last":float(last),"tol":float(tol)}

    # 5) EMA/RSI/MACD/BB megfelelőség (1H)
    ema_ok=rsi_ok=macd_ok=bb_ok=False
    if not df1.empty:
        c1=df1["close"]; e20,e50,e200=ema(c1,20),ema(c1,50),ema(c1,200)
        r=rsi14(c1); line,signal,_=macd(c1); m,up,lo=bb(c1)
        ema_ok = (e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]) or (e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1])
        rsi_ok = (r.iloc[-1]>=50) or (r.iloc[-1]<=50)
        macd_ok= (line.iloc[-1]-signal.iloc[-1])!=0
        bb_ok  = not (np.isnan(up.iloc[-1]) or np.isnan(lo.iloc[-1]))

    # 6) ATR proxy (1H)
    atr_proxy=None
    if not df1.empty and len(df1)>=15:
        hi,lo,cl=df1["high"].values,df1["low"].values,df1["close"].values
        trs=[]
        for i in range(len(cl)):
            if i==0: trs.append(hi[i]-lo[i])
            else: trs.append(max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1])))
        atr_proxy=float(pd.Series(trs).rolling(14).mean().iloc[-1])

    # 7) irány + jelzés + RR
    side=side_from(b4,b1,bos)
    sig=None; rr_ok=False; rr_val=0.0
    if side and spot is not None:
        sig=build_signal(asset,side,spot,atr_proxy)
        rr_val=rr(sig["entry"],sig["sl"],sig["tp1"]) if sig else 0.0
        rr_ok = rr_val>=1.5

    # 8) P% és döntés
    P=prob_score(b4,b1,bos,ema_ok,rsi_ok,macd_ok,near79,bb_ok,rr_ok)
    decision="enter" if (side and near79 and rr_ok and P>=60 and sig) else "no entry"
    reasons=[]
    if side is None: reasons.append("nincs iránykonzisztencia (4H/1H/BOS)")
    if not near79: reasons.append("79% Fib nincs közel (kötelező)")
    if not rr_ok: reasons.append(f"RR < 1.5R ({round(rr_val,2)})")
    if P<60: reasons.append(f"P < 60% ({P}%)")

    # 9) leverage + P/L@$100
    lev=LEV_CAP.get(asset,2)
    pl={}
    if sig:
        pl = {
            "TP1": round(100*lev*abs(sig["tp1"]-sig["entry"])/sig["entry"], 2),
            "TP2": round(100*lev*abs(sig["tp2"]-sig["entry"])/sig["entry"], 2),
            "SL":  round(100*lev*abs(sig["sl"] -sig["entry"])/sig["entry"], 2),
        }

    decision_obj={
        "asset":asset,
        "retrieved_at_utc":nowiso(),
        "spot":{"price_usd":spot,"source":spot_src,"retrieved_at_utc":spot_ts},
        "urls":{
            "alias_all":f"{BASE}/h/all?asset={asset}",
            "alias_k5m":f"{BASE}/h/k5m?asset={asset}",
            "alias_k1h":f"{BASE}/h/k1h?asset={asset}",
            "alias_k4h":f"{BASE}/h/k4h?asset={asset}",
            "alias_spot":f"{BASE}/h/spot?asset={asset}",
        },
        "diagnostics":{"bias_4h":b4,"diag_4h":d4,"bias_1h":b1,"diag_1h":d1,"bos_5m":bos,"fib_79":fib,"rr":round(rr_val,3)},
        "probability_pct":P,
        "leverage_cap":lev,
        "signal":sig,
        "pl_at_100":pl,
        "decision":decision,
        "reasons":reasons,
        "summary_hu":{
            "Spot (USD)":spot,
            "Forrás":spot_src,
            "Lekérés (UTC)":spot_ts,
            "Valószínűség":f"{P}%",
            "Döntés":"JELZÉS" if decision=="enter" else "no entry",
            "Javaslat":({
                "SIDE":sig["side"],"Entry":sig["entry"],"SL":sig["sl"],
                "TP1 (~1R)":sig["tp1"],"TP2 (2R+)":sig["tp2"],
                "Ajánlott leverage":f"{lev}x","P/L@$100":pl,
                "Lejárat":"Europe/Budapest – intraday"
            } if decision=="enter" else "—"),
            "Indoklás":("; ".join(reasons) if reasons else "Konfluenciák teljesülnek.")
        }
    }
    save_json(os.path.join(outdir,"decision.json"),decision_obj)
    return decision_obj

def main():
    allout={"generated_at_utc":nowiso(),"assets":{}}
    for a in ASSETS:
        try:
            allout["assets"][a]=analyze(a)
        except Exception as e:
            allout["assets"][a]={"asset":a,"ok":False,"error":str(e),"trace":traceback.format_exc()}
    save_json(os.path.join("public","analysis_summary.json"),allout)
    # Rövid HTML index
    idx="<html><meta charset='utf-8'><title>Analysis</title><body><h1>Analysis</h1><ul>"+ \
        "".join([f"<li><a href='{a}/decision.json'>{a}</a></li>" for a in ASSETS])+ \
        "</ul><p><a href='analysis_summary.json'>analysis_summary.json</a></p></body></html>"
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
