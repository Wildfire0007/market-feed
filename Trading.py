# -*- coding: utf-8 -*-
"""
Robusztus feed-generátor több tőzsdés (Kraken → Coinbase → OKX) OHLC fallbackkel.
- Binance 451 blokkolás megkerülése alternatív, publikus API-kkal.
- Headless (matplotlib Agg), retry/backoff, minden kimenet public/ alá.
- Hiba esetén is public/status.json készül, a script 0-val lép ki.

Kimenetek (public/):
- spot.json (CoinGecko → fallback Coinbase ticker)  + retrieved_at_utc + age_sec
- klines_5m.json / klines_1h.json / klines_4h.json
- chart_1d.png (napi záróár grafikon – ugyanabból a forrásból)
- signal.json (ATR14 1h minta)
- status.json
- index.html (linklista)
"""

import os, json, math, time, random
import requests
import pandas as pd
import datetime as dt
from datetime import timezone

# Headless plot
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Beállítások --------
COIN_ID = "solana"    # CoinGecko id
SYMBOLS = {
    "kraken":   "SOLUSD",     # Kraken spot pár
    "coinbase": "SOL-USD",    # Coinbase spot termék
    "okx":      "SOL-USDT",   # OKX spot instrument
}
OUTDIR  = "public"
os.makedirs(OUTDIR, exist_ok=True)

def iso(ts_ms_or_s: float) -> str:
    ts_sec = ts_ms_or_s/1000.0 if ts_ms_or_s > 10**12 else float(ts_ms_or_s)
    return dt.datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def save_json(obj, name: str):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def fetch_json_with_retry(url: str, timeout: int = 20, retries: int = 4, base_delay: float = 1.0, headers=None):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            # 429/5xx/451 stb. → retry
            if r.status_code in (429, 451) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            delay = base_delay * (2 ** i) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"Fetch failed after {retries} retries for {url}: {last_err}")

# ---------- Spot: CoinGecko → fallback Coinbase ticker ----------
def fetch_spot():
    # Elsődleges: CoinGecko
    cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={COIN_ID}&vs_currencies=usd&include_last_updated_at=true"
    try:
        d = fetch_json_with_retry(cg_url)
        px = float(d[COIN_ID]["usd"])
        ts = d[COIN_ID].get("last_updated_at")  # epoch sec
        return {"price_usd": px, "last_updated_at": iso(ts), "source": cg_url}
    except Exception as e1:
        # Fallback: Coinbase ticker
        cb_url = f"https://api.exchange.coinbase.com/products/{SYMBOLS['coinbase']}/ticker"
        try:
            d = fetch_json_with_retry(cb_url, headers={"User-Agent":"Mozilla/5.0"})
            # Coinbase: {"price":"xxx","time":"2025-...Z", ...}
            px = float(d["price"])
            # ISO UTC string → hagyjuk úgy, és jelezzük forrásként
            return {"price_usd": px, "last_updated_at": d.get("time",""), "source": cb_url}
        except Exception as e2:
            raise RuntimeError(f"spot fallback failed: CG: {e1} | CB: {e2}")

# ---- Segédfüggvény: többféle last_updated_at formátum parse-olása aware UTC-re ----
def parse_last_updated_utc(s: str):
    if not s:
        return None
    s = str(s).strip()
    try:
        # "YYYY-MM-DD HH:MM:SS UTC" (a mi iso() kimenetünk)
        if s.endswith(" UTC"):
            return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        # Coinbase: "...Z"
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        # ISO "+00:00" formátum
        if "+" in s:
            return dt.datetime.fromisoformat(s)
        # numerikus epoch (biztonsági)
        if s.isdigit():
            return dt.datetime.fromtimestamp(int(s), tz=timezone.utc)
    except Exception:
        pass
    # végső próbálkozás: szabad formátum
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

# ---------- OHLC több tőzsdéről, egységes DF: time,o,h,l,c,v (ASC, UTC) ----------
def klines_from_kraken(interval: str, limit: int) -> pd.DataFrame:
    # Kraken intervallum percben: 1,5,15,30,60,240,1440,10080,21600
    map_int = {"5m":5, "1h":60, "4h":240, "1d":1440}
    iv = map_int[interval]
    url = f"https://api.kraken.com/0/public/OHLC?pair={SYMBOLS['kraken']}&interval={iv}"
    j = fetch_json_with_retry(url)
    # a 'result' kulcs alatt a pár neve dinamikus lehet → vegyük az első listát
    res = j.get("result", {})
    arr = None
    for k,v in res.items():
        if isinstance(v, list):
            arr = v
            break
    if not arr:
        raise RuntimeError("Kraken OHLC missing array")
    df = pd.DataFrame(arr, columns=["t","o","h","l","c","vwap","vol","count"])
    df = df[["t","o","h","l","c","vol"]]
    df["t"] = df["t"].astype(int)
    for col in ["o","h","l","c","vol"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("t").tail(limit)
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","vol"]].rename(columns={"vol":"v"})

def klines_from_coinbase(interval: str, limit: int) -> pd.DataFrame:
    # Coinbase granularity: 60, 300, 900, 3600, 21600, 86400
    map_int = {"5m":300, "1h":3600, "4h":14400, "1d":86400}
    iv = map_int[interval]
    url = f"https://api.exchange.coinbase.com/products/{SYMBOLS['coinbase']}/candles?granularity={iv}"
    j = fetch_json_with_retry(url, headers={"User-Agent":"Mozilla/5.0"})
    # Válasz: list of [time, low, high, open, close, volume] reverse chrono
    df = pd.DataFrame(j, columns=["t","l","h","o","c","v"])
    for col in ["o","h","l","c","v"]:
        df[col] = df[col].astype(float)
    df["t"] = df["t"].astype(int)
    df = df.sort_values("t").tail(limit)
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]]

def klines_from_okx(interval: str, limit: int) -> pd.DataFrame:
    # OKX bar: 1m,3m,5m,15m,30m,1H,4H,1D,1W,1M
    map_int = {"5m":"5m", "1h":"1H", "4h":"4H", "1d":"1D"}
    iv = map_int[interval]
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOLS['okx']}&bar={iv}&limit={min(limit,300)}"
    j = fetch_json_with_retry(url, headers={"User-Agent":"Mozilla/5.0"})
    arr = j.get("data", [])
    if not arr:
        raise RuntimeError("OKX empty data")
    # data: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, ...]
    df = pd.DataFrame(arr, columns=["ts","o","h","l","c","v","vccy","vqq","conf","ign"])
    for col in ["o","h","l","c","v"]:
        df[col] = df[col].astype(float)
    df["ts"] = df["ts"].astype(int)
    df = df.sort_values("ts").tail(limit)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]]

def klines_multi(interval: str, limit: int, status_obj: dict) -> pd.DataFrame:
    providers = []
    # Próbálási sorrend
    for prov in ("kraken","coinbase","okx"):
        try:
            if prov == "kraken":
                df = klines_from_kraken(interval, limit)
            elif prov == "coinbase":
                df = klines_from_coinbase(interval, limit)
            else:
                df = klines_from_okx(interval, limit)
            status_obj[f"klines_{interval}_provider"] = prov
            return df
        except Exception as e:
            providers.append(f"{prov}: {e}")
            continue
    raise RuntimeError("All providers failed → " + " | ".join(providers))

# ---------- ATR és jel ----------
def atr14_from_ohlc(df: pd.DataFrame) -> float:
    d = df.copy()
    d["o"] = d["o"].astype(float); d["h"]=d["h"].astype(float)
    d["l"] = d["l"].astype(float); d["c"]=d["c"].astype(float)
    d["prev_c"] = d["c"].shift(1)
    tr = pd.concat([
        (d["h"] - d["l"]).abs(),
        (d["h"] - d["prev_c"]).abs(),
        (d["l"] - d["prev_c"]).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    return float(atr14.iloc[-1])

status = {"ok": True, "errors": []}

# ---------- Spot ----------
try:
    spot = fetch_spot()

    # --- Frissesség-jelzők hozzáadása a spot.json-hoz ---
    now_utc = dt.datetime.now(timezone.utc)
    spot["retrieved_at_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    lu = parse_last_updated_utc(spot.get("last_updated_at"))
    spot["age_sec"] = int((now_utc - lu).total_seconds()) if lu else None

    save_json(spot, "spot.json")
except Exception as e:
    status["ok"] = False
    status["errors"].append(f"spot: {e}")
    save_json({"status":"Insufficient data (spot)","error":str(e)}, "spot.json")

# ---------- OHLC: 5m / 1h / 4h ----------
try:
    k5m = klines_multi("5m", 864, status)
    k1h = klines_multi("1h", 200, status)
    k4h = klines_multi("4h", 200, status)
    k5m.to_json(os.path.join(OUTDIR, "klines_5m.json"), orient="records", indent=2)
    k1h.to_json(os.path.join(OUTDIR, "klines_1h.json"), orient="records", indent=2)
    k4h.to_json(os.path.join(OUTDIR, "klines_4h.json"), orient="records", indent=2)
except Exception as e:
    status["ok"] = False
    status["errors"].append(f"klines: {e}")
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_5m.json")
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_1h.json")
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_4h.json")

# ---------- 1D chart PNG (ugyanabból a forrásból) ----------
try:
    k1d = klines_multi("1d", 120, status)
    # záróár grafikon
    d = k1d.copy()
    d["date"] = pd.to_datetime(d["time"]).dt.date
    plt.figure(figsize=(10,4))
    plt.plot(d["date"], d["c"].astype(float))
    plt.title("SOL – Close (1D) – multi-exchange feed")
    plt.xlabel("Dátum (UTC)"); plt.ylabel("Ár (USD/USDT)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "chart_1d.png"), dpi=150)
    plt.close()
except Exception as e:
    status["ok"] = False
    status["errors"].append(f"chart_1d: {e}")

# ---------- Jel (ATR14 1h minta) ----------
try:
    # ha 1h hiányos, olvassuk vissza a fájlt (lehet hiba-json)
    k1h_path = os.path.join(OUTDIR, "klines_1h.json")
    if not os.path.exists(k1h_path):
        raise RuntimeError("klines_1h.json missing")
    k1h_df = pd.read_json(k1h_path)
    # hiba-json felismerése
    if isinstance(k1h_df, pd.DataFrame) and "status" in k1h_df.columns:
        raise RuntimeError("No 1h data (error json)")
    for col in ["o","h","l","c"]:
        k1h_df[col] = k1h_df[col].astype(float)
    atr = atr14_from_ohlc(k1h_df)
    sp = json.load(open(os.path.join(OUTDIR, "spot.json"), encoding="utf-8")).get("price_usd", float("nan"))
    entry = float(sp)
    if math.isnan(entry) or math.isnan(atr) or atr <= 0:
        raise RuntimeError("entry/ATR invalid")
    sl  = entry - 2.0*atr
    tp1 = entry + 1.5*atr
    tp2 = entry + 3.0*atr
    lev = 3
    qty = 100.0/entry
    signal = {
        "side":"LONG","entry":round(entry,4),"SL":round(sl,4),
        "TP1":round(tp1,4),"TP2":round(tp2,4),"leverage":lev,
        "P/L@$100":{
            "TP1":round((tp1-entry)*qty,2),
            "TP2":round((tp2-entry)*qty,2),
            "SL": round((sl -entry)*qty,2)
        },
        "notes":"ATR14(1h) MINTA; saját szabályaidat illeszd ide."
    }
    save_json(signal, "signal.json")
except Exception as e:
    status["ok"] = False
    status["errors"].append(f"signal: {e}")
    save_json({"status":"Insufficient data (signal)","error":str(e)}, "signal.json")

# ---------- Státusz + index ----------
status["generated_at_utc"] = iso(time.time())
save_json(status, "status.json")

INDEX_PATH = os.path.join(OUTDIR, "index.html")
html = """<!doctype html><html lang="hu"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>market-feed – files</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;margin:24px;line-height:1.5}code{background:#f4f4f4;padding:2px 6px;border-radius:6px}</style>
</head><body>
<h1>market-feed – public outputs</h1>
<ul>
  <li><a href="./status.json">status.json</a></li>
  <li><a href="./spot.json">spot.json</a></li>
  <li><a href="./klines_5m.json">klines_5m.json</a></li>
  <li><a href="./klines_1h.json">klines_1h.json</a></li>
  <li><a href="./klines_4h.json">klines_4h.json</a></li>
  <li><a href="./signal.json">signal.json</a></li>
  <li><a href="./chart_1d.png">chart_1d.png</a></li>
</ul>
<p>Provider(ek): kraken/coinbase/okx – automatikus fallback. Ha valami hiányzik, nézd meg a <code>status.json</code>-t.</p>
</body></html>
"""
with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Done. Outputs in 'public/'.")
