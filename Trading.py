# -*- coding: utf-8 -*-
"""
Market feed generator for multiple assets (SOL crypto + NSDQ100 index).
- SOL: CoinGecko spot → fallback Coinbase; OHLC: Kraken → Coinbase → OKX
- NSDQ100: Yahoo Finance quote+chart API (spot+OHLC); 4H az 1H-ból aggregálva
- (Opció) chart_1d.png készítés matplotlib-pel; ALAPÉRTELMEZÉS: csak chart_1d.json kimenet
- Retry/backoff; minden kimenet public/<ASSET>/ alá
- status.json összesített állapotot ír (per-asset info-val)
- Minden spot.json tartalmaz: retrieved_at_utc + age_sec

Kimenetek / assetenként (public/<ASSET>/):
- status.json (assetre vonatkozó rész), spot.json, klines_5m.json, klines_1h.json, klines_4h.json,
  signal.json (ATR14 1h minta), chart_1d.json (ÚJ), [opcionális] chart_1d.png, index.html (asset-oldal)
Gyökérben: public/index.html (linkek: SOL, NSDQ100)
"""

import os, json, math, time, random
import requests
import pandas as pd
import datetime as dt
from datetime import timezone
from urllib.parse import quote

# ---------- Beállítások ----------
OUTDIR = "public"
PNG_CHART = False  # <— ALAP: csak JSON. Ha szeretnéd a PNG-t is, állítsd True-ra.

ASSETS = {
    "SOL": {
        "type": "crypto",
        "coingecko_id": "solana",
        "symbols": {
            "kraken":   "SOLUSD",
            "coinbase": "SOL-USD",
            "okx":      "SOL-USDT",
        }
    },
    "NSDQ100": {
        "type": "index",
        "yahoo_symbol": "^NDX",  # URL-ben encode-olni kell
    }
}
os.makedirs(OUTDIR, exist_ok=True)

UA = {"User-Agent":"Mozilla/5.0", "Accept":"application/json,text/html;q=0.9"}

def iso(ts_ms_or_s: float) -> str:
    ts_sec = ts_ms_or_s/1000.0 if ts_ms_or_s > 10**12 else float(ts_ms_or_s)
    return dt.datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def now_utc():
    return dt.datetime.now(timezone.utc)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def fetch_json_with_retry(url: str, timeout: int = 20, retries: int = 4, base_delay: float = 1.0, headers=None):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or UA)
            if r.status_code in (429, 451) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            delay = base_delay * (2 ** i) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"Fetch failed after {retries} retries for {url}: {last_err}")

# ---------- Közös segédek ----------
def parse_last_updated_utc(s: str):
    if not s:
        return None
    s = str(s).strip()
    try:
        if s.endswith(" UTC"):
            return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if "+" in s:
            return dt.datetime.fromisoformat(s)
        if s.isdigit():
            return dt.datetime.fromtimestamp(int(s), tz=timezone.utc)
    except Exception:
        pass
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def df_to_ohlc_json(df: pd.DataFrame, path: str):
    """Elvárt oszlopok: time,o,h,l,c,v (UTC formázott time string)"""
    df.to_json(path, orient="records", indent=2)

# --- PNG csak akkor kell, ha PNG_CHART=True ---
if PNG_CHART:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_chart_png(df_1d: pd.DataFrame, title: str, path: str):
        d = df_1d.copy()
        d["date"] = pd.to_datetime(d["time"]).dt.date
        plt.figure(figsize=(10,4))
        plt.plot(d["date"], d["c"].astype(float))
        plt.title(title)
        plt.xlabel("Dátum (UTC)"); plt.ylabel("Ár")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

# ---------- CRYPTO: SOL ----------
def crypto_fetch_spot(coingecko_id: str, coinbase_symbol: str):
    cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd&include_last_updated_at=true"
    try:
        d = fetch_json_with_retry(cg_url)
        px = float(d[coingecko_id]["usd"])
        ts = d[coingecko_id].get("last_updated_at")
        return {"price_usd": px, "last_updated_at": iso(ts), "source": cg_url}
    except Exception:
        cb_url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/ticker"
        d = fetch_json_with_retry(cb_url)
        px = float(d["price"])
        return {"price_usd": px, "last_updated_at": d.get("time",""), "source": cb_url}

def klines_from_kraken(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    mp = {"5m":5, "1h":60, "4h":240, "1d":1440}
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={mp[interval]}"
    j = fetch_json_with_retry(url)
    res = j.get("result", {})
    arr = None
    for k,v in res.items():
        if isinstance(v, list):
            arr = v; break
    if not arr: raise RuntimeError("Kraken OHLC missing array")
    df = pd.DataFrame(arr, columns=["t","o","h","l","c","vwap","vol","count"])[["t","o","h","l","c","vol"]]
    df["t"] = df["t"].astype(int)
    for col in ["o","h","l","c","vol"]: df[col]=df[col].astype(float)
    df = df.sort_values("t").tail(limit)
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]].rename(columns={"vol":"v"})

def klines_from_coinbase(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    mp = {"5m":300, "1h":3600, "4h":14400, "1d":86400}
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={mp[interval]}"
    j = fetch_json_with_retry(url)
    df = pd.DataFrame(j, columns=["t","l","h","o","c","v"])
    for col in ["o","h","l","c","v"]: df[col]=df[col].astype(float)
    df["t"] = df["t"].astype(int)
    df = df.sort_values("t").tail(limit)
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]]

def klines_from_okx(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    mp = {"5m":"5m", "1h":"1H", "4h":"4H", "1d":"1D"}
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={mp[interval]}&limit={min(limit,300)}"
    j = fetch_json_with_retry(url)
    arr = j.get("data", [])
    if not arr: raise RuntimeError("OKX empty data")
    df = pd.DataFrame(arr, columns=["ts","o","h","l","c","v","vccy","vqq","conf","ign"])
    for col in ["o","h","l","c","v"]: df[col]=df[col].astype(float)
    df["ts"] = df["ts"].astype(int)
    df = df.sort_values("ts").tail(limit)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]]

def crypto_klines_multi(symbols: dict, interval: str, limit: int, asset_status: dict) -> pd.DataFrame:
    providers = []
    for prov in ("kraken","coinbase","okx"):
        try:
            if prov == "kraken":
                df = klines_from_kraken(symbols["kraken"], interval, limit)
            elif prov == "coinbase":
                df = klines_from_coinbase(symbols["coinbase"], interval, limit)
            else:
                df = klines_from_okx(symbols["okx"], interval, limit)
            asset_status[f"klines_{interval}_provider"] = prov
            return df
        except Exception as e:
            providers.append(f"{prov}: {e}")
    raise RuntimeError("All crypto providers failed → " + " | ".join(providers))

# ---------- INDEX: NSDQ100 (Yahoo Finance) ----------
def yahoo_chart(symbol: str, range_str: str, interval: str):
    s_enc = quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s_enc}?range={range_str}&interval={interval}"
    j = fetch_json_with_retry(url)
    if not j or not j.get("chart", {}).get("result"):
        raise RuntimeError("Yahoo: empty result")
    return j["chart"]["result"][0]

def yahoo_to_df(result) -> pd.DataFrame:
    ts = result.get("timestamp", [])
    ind = result.get("indicators", {}).get("quote", [{}])[0]
    o = ind.get("open", []); h = ind.get("high", []); l = ind.get("low", []); c = ind.get("close", []); v = ind.get("volume", [])
    n = min(len(ts), len(o), len(h), len(l), len(c), len(v))
    df = pd.DataFrame({
        "t": ts[:n],
        "o": [float(x) if x is not None else float("nan") for x in o[:n]],
        "h": [float(x) if x is not None else float("nan") for x in h[:n]],
        "l": [float(x) if x is not None else float("nan") for x in l[:n]],
        "c": [float(x) if x is not None else float("nan") for x in c[:n]],
        "v": [float(x) if x is not None else 0.0 for x in v[:n]],
    })
    df = df.dropna(subset=["o","h","l","c"]).sort_values("t")
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time","o","h","l","c","v"]]

def index_fetch_spot(symbol: str):
    def spot_from_chart(sym):
        res = yahoo_chart(sym, "1d", "5m")
        meta = res.get("meta", {})
        candidates = []
        for p_field, t_field, lbl in [
            ("postMarketPrice", "postMarketTime", "postMarket"),
            ("preMarketPrice",  "preMarketTime",  "preMarket"),
            ("regularMarketPrice", "regularMarketTime", "regularMarket"),
        ]:
            px = meta.get(p_field); ts = meta.get(t_field)
            if px is not None and ts:
                try: candidates.append((float(px), int(ts), f"Yahoo chart meta {lbl}"))
                except Exception: pass
        ts_arr = res.get("timestamp", [])
        closes = res.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        if ts_arr and closes and closes[-1] is not None:
            candidates.append((float(closes[-1]), int(ts_arr[-1]), "Yahoo chart last candle (5m)"))
        if not candidates: raise RuntimeError("Yahoo chart: no spot candidates")

        price, ts, src = max(candidates, key=lambda x: x[1])
        ctp = meta.get("currentTradingPeriod", {}).get("regular", {})
        start, end = ctp.get("start"), ctp.get("end")
        now_sec = int(time.time())
        mstate = "REGULAR" if isinstance(start, int) and isinstance(end, int) and start <= now_sec <= end else "CLOSED"

        return {
            "price_usd": float(price),
            "last_updated_at": iso(ts),
            "source": src,
            "market_state": mstate
        }

    try:
        return spot_from_chart(symbol)
    except Exception:
        pass
    try:
        out = spot_from_chart("QQQ"); out["proxy_from"] = "QQQ"; return out
    except Exception:
        pass
    out = spot_from_chart("NQ=F"); out["proxy_from"] = "NQ=F"; return out

def index_fetch_klines(symbol: str, interval: str, limit: int, asset_status: dict) -> pd.DataFrame:
    if interval == "5m":
        res = yahoo_chart(symbol, "5d", "5m"); asset_status["klines_5m_provider"] = "yahoo"; return yahoo_to_df(res).tail(limit)
    elif interval == "1h":
        res = yahoo_chart(symbol, "60d", "60m"); asset_status["klines_1h_provider"] = "yahoo"; return yahoo_to_df(res).tail(limit)
    elif interval == "4h":
        res = yahoo_chart(symbol, "60d", "60m"); asset_status["klines_4h_provider"] = "yahoo(agg)"
        df1h = yahoo_to_df(res)
        if df1h.empty: raise RuntimeError("Yahoo 1h empty")
        d = df1h.copy(); dt_idx = pd.to_datetime(d["time"], utc=True); d = d.set_index(dt_idx)
        agg = {"o":"first","h":"max","l":"min","c":"last","v":"sum"}
        df4h = d.resample("4H").agg(agg).dropna()
        idx = df4h.index
        if getattr(idx, "tz", None) is None: idx = idx.tz_localize("UTC")
        times = pd.to_datetime(idx).strftime("%Y-%m-%d %H:%M:%S UTC")
        df4h_out = df4h.copy(); df4h_out["time"] = times
        return df4h_out.reset_index(drop=True).tail(limit)[["time","o","h","l","c","v"]]
    elif interval == "1d":
        res = yahoo_chart(symbol, "2y", "1d"); asset_status["klines_1d_provider"] = "yahoo"; return yahoo_to_df(res).tail(limit)
    else:
        raise ValueError("Unsupported interval for index")

# ---------- Jel generálás (ATR14 1h minta) ----------
def atr14_from_ohlc(df: pd.DataFrame) -> float:
    d = df.copy()
    for col in ["o","h","l","c"]: d[col] = d[col].astype(float)
    d["prev_c"] = d["c"].shift(1)
    tr = pd.concat([(d["h"]-d["l"]).abs(), (d["h"]-d["prev_c"]).abs(), (d["l"]-d["prev_c"]).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    return float(atr14.iloc[-1])

def build_signal_1h(asset_dir: str, status_obj: dict):
    try:
        k1h_path = os.path.join(asset_dir, "klines_1h.json")
        if not os.path.exists(k1h_path): raise RuntimeError("klines_1h.json missing")
        k1h_df = pd.read_json(k1h_path)
        if not isinstance(k1h_df, pd.DataFrame) or not set(["o","h","l","c"]).issubset(k1h_df.columns):
            raise RuntimeError("No valid 1h OHLC columns")
        for col in ["o","h","l","c"]: k1h_df[col] = k1h_df[col].astype(float)
        atr = atr14_from_ohlc(k1h_df)
        sp = json.load(open(os.path.join(asset_dir, "spot.json"), encoding="utf-8")).get("price_usd", float("nan"))
        entry = float(sp)
        if math.isnan(entry) or math.isnan(atr) or atr <= 0: raise RuntimeError("entry/ATR invalid")
        sl  = entry - 2.0*atr; tp1 = entry + 1.5*atr; tp2 = entry + 3.0*atr; lev = 3; qty = 100.0/entry
        signal = {
            "side":"LONG","entry":round(entry,4),"SL":round(sl,4),
            "TP1":round(tp1,4),"TP2":round(tp2,4),"leverage":lev,
            "P/L@$100":{"TP1":round((tp1-entry)*qty,2),"TP2":round((tp2-entry)*qty,2),"SL": round((sl -entry)*qty,2)},
            "notes":"ATR14(1h) MINTA; igazítsd a saját szabályaidhoz."
        }
        save_json(signal, os.path.join(asset_dir, "signal.json"))
    except Exception as e:
        status_obj["ok"] = False
        status_obj["errors"].append(f"signal: {e}")
        save_json({"status":"Insufficient data (signal)","error":str(e)},
                  os.path.join(asset_dir, "signal.json"))

# ---------- Egy asset teljes buildje ----------
def build_asset(name: str, cfg: dict):
    asset_dir = os.path.join(OUTDIR, name)
    os.makedirs(asset_dir, exist_ok=True)
    astatus = {"ok": True, "errors": []}

    # Spot
    try:
        if cfg["type"] == "crypto":
            spot = crypto_fetch_spot(cfg["coingecko_id"], cfg["symbols"]["coinbase"])
        else:
            spot = index_fetch_spot(cfg["yahoo_symbol"])
        now = now_utc()
        spot["retrieved_at_utc"] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        lu = parse_last_updated_utc(spot.get("last_updated_at"))
        spot["age_sec"] = int((now - lu).total_seconds()) if lu else None
        save_json(spot, os.path.join(asset_dir, "spot.json"))
    except Exception as e:
        astatus["ok"] = False
        astatus["errors"].append(f"spot: {e}")
        save_json({"status":"Insufficient data (spot)","error":str(e)}, os.path.join(asset_dir, "spot.json"))

    # OHLC
    try:
        if cfg["type"] == "crypto":
            k5m = crypto_klines_multi(cfg["symbols"], "5m", 864, astatus)
            k1h = crypto_klines_multi(cfg["symbols"], "1h", 200, astatus)
            k4h = crypto_klines_multi(cfg["symbols"], "4h", 200, astatus)
            k1d = crypto_klines_multi(cfg["symbols"], "1d", 120, astatus)
        else:
            k5m = index_fetch_klines(cfg["yahoo_symbol"], "5m", 864, astatus)
            k1h = index_fetch_klines(cfg["yahoo_symbol"], "1h", 200, astatus)
            k4h = index_fetch_klines(cfg["yahoo_symbol"], "4h", 200, astatus)
            k1d = index_fetch_klines(cfg["yahoo_symbol"], "1d", 120, astatus)

        df_to_ohlc_json(k5m, os.path.join(asset_dir, "klines_5m.json"))
        df_to_ohlc_json(k1h, os.path.join(asset_dir, "klines_1h.json"))
        df_to_ohlc_json(k4h, os.path.join(asset_dir, "klines_4h.json"))

        # ÚJ: 1D JSON mentése (chart_1d.json)
        if isinstance(k1d, pd.DataFrame) and not k1d.empty:
            df_to_ohlc_json(k1d, os.path.join(asset_dir, "chart_1d.json"))
        else:
            astatus["ok"] = False
            astatus["errors"].append("chart_1d.json: no data")

        # (Opció) PNG csak ha kérted
        if PNG_CHART and isinstance(k1d, pd.DataFrame) and not k1d.empty:
            try:
                make_chart_png(k1d, f"{name} – Close (1D)", os.path.join(asset_dir, "chart_1d.png"))
            except Exception as e:
                astatus["ok"] = False
                astatus["errors"].append(f"chart_1d.png: {e}")

    except Exception as e:
        astatus["ok"] = False
        astatus["errors"].append(f"klines: {e}")
        err = {"status":"Insufficient data (klines)","error":str(e)}
        save_json(err, os.path.join(asset_dir, "klines_5m.json"))
        save_json(err, os.path.join(asset_dir, "klines_1h.json"))
        save_json(err, os.path.join(asset_dir, "klines_4h.json"))
        # ha k1d nincs, akkor chart_1d.json is error legyen
        save_json({"status":"Insufficient data (chart_1d)","error":str(e)}, os.path.join(asset_dir, "chart_1d.json"))
        k1d = pd.DataFrame()

    # Jel (ATR14 1h minta)
    build_signal_1h(asset_dir, astatus)

    # Asset-szintű status.json
    astatus["generated_at_utc"] = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    save_json(astatus, os.path.join(asset_dir, "status.json"))

    # Asset index.html
    links = f"""
  <li><a href="./status.json">status.json</a></li>
  <li><a href="./spot.json">spot.json</a></li>
  <li><a href="./klines_5m.json">klines_5m.json</a></li>
  <li><a href="./klines_1h.json">klines_1h.json</a></li>
  <li><a href="./klines_4h.json">klines_4h.json</a></li>
  <li><a href="./signal.json">signal.json</a></li>
  <li><a href="./chart_1d.json">chart_1d.json</a></li>"""
    if PNG_CHART:
        links += '\n  <li><a href="./chart_1d.png">chart_1d.png</a></li>'

    html = f"""<!doctype html><html lang="hu"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{name} – market-feed</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;margin:24px;line-height:1.5}}code{{background:#f4f4f4;padding:2px 6px;border-radius:6px}}</style>
</head><body>
<h1>{name} – public outputs</h1>
<ul>
{links}
</ul>
<p>Források: {('Kraken/Coinbase/OKX (crypto multi)') if ASSETS[name]['type']=='crypto' else 'Yahoo Finance (index)'}</p>
<p><a href="../index.html">« vissza a főoldalra</a></p>
</body></html>"""
    with open(os.path.join(asset_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return astatus

# ---------- Fő futás: minden asset legenerálása ----------
global_status = {"ok": True, "errors": [], "assets": {}}

for asset_name, cfg in ASSETS.items():
    st = build_asset(asset_name, cfg)
    global_status["assets"][asset_name] = st
    if not st.get("ok", False):
        global_status["ok"] = False

global_status["generated_at_utc"] = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
save_json(global_status, os.path.join(OUTDIR, "status.json"))

# Gyökér index.html (főoldal)
root_html = """<!doctype html><html lang="hu"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>market-feed – assets</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;margin:24px;line-height:1.5}ul{line-height:1.8}</style>
</head><body>
<h1>market-feed – assets</h1>
<ul>
  <li><a href="./SOL/index.html">SOL</a></li>
  <li><a href="./NSDQ100/index.html">NSDQ100</a></li>
</ul>
<p>Assetenként külön aloldal: status/spot/klines/signal/chart_1d.json. Összesített állapot: <a href="./status.json">status.json</a>.</p>
</body></html>
"""
with open(os.path.join(OUTDIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(root_html)

print("Done. Outputs in 'public/<ASSET>/' and public/index.html")
