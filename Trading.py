# -*- coding: utf-8 -*-
"""
Market feed generator – GitHub Pages-re ír JSON/PNG kimeneteket.
Források:
- Crypto (SOL): CoinGecko (kulcsmentes) – spot + 1 napos intraday pontokból 5m OHLC
- NSDQ100, GOLD_CFD: Twelve Data (elsődleges, kulcsos) → Finnhub (tartalék, kulcsos) → Stooq 1D fallback

Kimenet assetenként (public/<ASSET>/):
  spot.json, klines_5m.json, klines_1h.json, klines_4h.json, k1d.json, signal.json, chart_1d.png, index.html
Gyökérben: public/status.json, public/index.html, public/run_log.txt (a workflow írja)

Megjegyzés:
- A script SAFE-MODE: hiba esetén is készül index.html és status.json (CI-ben nem dob 1-es exit kódot).
"""

import os, json, time, math, traceback
import datetime as dt
from datetime import timezone

import requests
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Beállítások ----------
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
OUTDIR = "public"
os.makedirs(OUTDIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (MarketFeed/1.0; +https://github.com/)"})

# Kulcsok (GitHub Secrets-ből jönnek a workflow env-vel)
TD_KEY = os.environ.get("TWELVE_DATA_KEY")   # Twelve Data
FH_KEY = os.environ.get("FINNHUB_KEY")       # Finnhub

# Ticker mapping Twelve Data / Finnhub-hoz
PROVIDERS_SYMBOLS = {
    "NSDQ100": {
        "td": {"spot": "NQ=F",   "series": "NQ=F"},  # futures
        "fh": {"spot": "NQ",     "series": "NQ"}     # futures rövid kód
    },
    "GOLD_CFD": {
        "td": {"spot": "XAU/USD","series": "GC=F"},  # arany spot + COMEX futures
        "fh": {"spot": "XAUUSD", "series": "GC"}     # Finnhub spot + futures
    }
}

# CoinGecko ID map kriptókhoz
COINGECKO_IDS = {
    "SOL": "solana",
}

# ---------- Util ----------
def utcnow(): return dt.datetime.now(timezone.utc)
def ts(): return int(utcnow().timestamp())
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def backoff_sleep(attempt): time.sleep(min(2 ** attempt + 0.25, 10))

# ---------- CoinGecko (crypto) ----------
def cg_simple_price(coin_id, vs="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs}"
    for a in range(4):
        r = SESSION.get(url, timeout=15)
        if r.ok:
            return r.json(), url
        backoff_sleep(a)
    raise RuntimeError(f"CoinGecko simple/price error: {r.status_code} {r.text[:200]}")

def cg_market_chart_range(coin_id, start_unix, end_unix, vs="usd"):
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
           f"?vs_currency={vs}&from={start_unix}&to={end_unix}")
    for a in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok:
            return r.json(), url
        backoff_sleep(a)
    raise RuntimeError(f"CoinGecko market_chart/range error: {r.status_code} {r.text[:200]}")

def prices_to_ohlc_from_points(ts_prices):
    # ts_prices: list of [ms, price]
    if not ts_prices:
        return pd.DataFrame(columns=["open","high","low","close"])
    s = pd.Series({pd.to_datetime(ms, unit='ms', utc=True): p for ms, p in ts_prices})
    df = pd.DataFrame({"price": s}).sort_index()
    o = df['price'].resample("5T").first()
    h = df['price'].resample("5T").max()
    l = df['price'].resample("5T").min()
    c = df['price'].resample("5T").last()
    out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()
    return out

# ---------- Twelve Data ----------
def td_quote(symbol):
    if not TD_KEY: raise RuntimeError("TWELVE_DATA_KEY hiányzik (GitHub Secret).")
    base = "https://api.twelvedata.com/quote"
    params = {"symbol": symbol, "format": "JSON", "apikey": TD_KEY}
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=15)
        if r.ok:
            return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"Twelve Data quote error: {r.status_code} {r.text[:200]}")

def td_time_series(symbol, interval="5min", outputsize=390):
    if not TD_KEY: raise RuntimeError("TWELVE_DATA_KEY hiányzik (GitHub Secret).")
    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval, "outputsize": outputsize,
        "timezone": "UTC", "format": "JSON", "apikey": TD_KEY
    }
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=20)
        if r.ok:
            return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"Twelve Data time_series error: {r.status_code} {r.text[:200]}")

# ---------- Finnhub ----------
def fh_quote(symbol):
    if not FH_KEY: raise RuntimeError("FINNHUB_KEY hiányzik (GitHub Secret).")
    base = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": FH_KEY}
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=15)
        if r.ok:
            return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"Finnhub quote error: {r.status_code} {r.text[:200]}")

def fh_candle(symbol, resolution="5", count=390):
    if not FH_KEY: raise RuntimeError("FINNHUB_KEY hiányzik (GitHub Secret).")
    base = "https://finnhub.io/api/v1/stock/candle"
    params = {"symbol": symbol, "resolution": resolution, "count": count, "token": FH_KEY}
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=20)
        if r.ok:
            return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"Finnhub candle error: {r.status_code} {r.text[:200]}")

# ---------- Stooq 1D fallback ----------
def stooq_daily_csv(symbol_encoded):
    # ^NDX → %5Endx ; XAUUSD → xauusd
    url = f"https://stooq.com/q/d/?s={symbol_encoded}"
    r = SESSION.get(url, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Stooq error: {r.status_code}")
    if "text/csv" in r.headers.get("Content-Type","") or r.text.startswith("Date,"):
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return df, url
    raise RuntimeError("Stooq CSV nem közvetlen (HTML jött).")

# ---------- Idősor-aggregálás ----------
def resample_ohlc(df, rule):
    if df.empty: return pd.DataFrame(columns=["open","high","low","close"])
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()

# ---------- ATR + minta-szignál ----------
def atr14_from_ohlc(df_1h):
    if df_1h.empty or len(df_1h) < 15: return None
    hi, lo, cl = df_1h['high'].values, df_1h['low'].values, df_1h['close'].values
    trs = [hi[i]-lo[i] if i==0 else max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])) for i in range(len(cl))]
    return float(pd.Series(trs).rolling(14).mean().iloc[-1])

def make_signal_sample(asset, spot, df_1h):
    atr = atr14_from_ohlc(df_1h)
    if atr is None or spot is None:
        return {"note":"sample only","atr14_1h": None}
    entry = spot
    sl = max(0.0, entry - 0.9*atr)
    tp1 = entry + 1.0*atr
    tp2 = entry + 2.0*atr
    return {
        "side":"LONG","entry":round(entry,5),"sl":round(sl,5),
        "tp1":round(tp1,5),"tp2":round(tp2,5),
        "atr14_1h": round(atr,5),"leverage":"3x (demo)","disclaimer":"DEMO"
    }

# ---------- PNG chart ----------
def save_png_chart(asset, df_5m, outdir_asset):
    if df_5m.empty: return None
    plt.figure(figsize=(8,3))
    plt.plot(df_5m.index, df_5m['close'])
    plt.title(f"{asset} – 1D intraday (close)")
    plt.tight_layout()
    path = os.path.join(outdir_asset, "chart_1d.png")
    plt.savefig(path)
    plt.close()
    return path

# ---------- Fő asset-futás ----------
def run_asset(asset):
    started = utcnow()
    outdir_asset = os.path.join(OUTDIR, asset)
    ensure_dir(outdir_asset)

    result = {
        "asset": asset, "ok": True, "errors": [],
        "retrieved_at_utc": started.isoformat(), "components": {}
    }

    try:
        if asset in COINGECKO_IDS:
            # ---- CRYPTO: CoinGecko
            cid = COINGECKO_IDS[asset]
            sp_json, sp_url = cg_simple_price(cid)
            spot = float(sp_json[cid]["usd"])
            save_json(os.path.join(outdir_asset, "spot.json"), {
                "price_usd": spot, "source_url": sp_url,
                "retrieved_at_utc": utcnow().isoformat(), "age_sec": 0
            })
            result["components"]["spot"] = {"ok": True}

            end_unix, start_unix = ts(), ts()-24*3600
            mc_json, mc_url = cg_market_chart_range(cid, start_unix, end_unix)
            prices = mc_json.get("prices", [])
            df_5m = prices_to_ohlc_from_points(prices)

            df_1h = resample_ohlc(df_5m, "1H")
            df_4h = resample_ohlc(df_1h, "4H")

            if not df_5m.empty:
                day = df_5m.index.normalize()[-1]
                day_df = df_5m[df_5m.index.normalize() == day]
                k1d = {
                    "open": float(day_df['open'].iloc[0]),
                    "high": float(day_df['high'].max()),
                    "low":  float(day_df['low'].min()),
                    "close":float(day_df['close'].iloc[-1])
                }
            else:
                k1d = {}

            def dump_df(df, name, src):
                arr = [
                    [int(ix.value//10**6), float(o), float(h), float(l), float(c)]
                    for ix, (o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())
                ]
                save_json(os.path.join(outdir_asset, f"{name}.json"),
                          {"ohlc_utc_ms": arr, "source_url": src, "retrieved_at_utc": utcnow().isoformat()})
                result["components"][name] = {"rows": len(arr)}

            dump_df(df_5m, "klines_5m", mc_url)
            dump_df(df_1h, "klines_1h", mc_url)
            dump_df(df_4h, "klines_4h", mc_url)
            save_json(os.path.join(outdir_asset, "k1d.json"),
                      {"k1d": k1d, "source": "cg_reconstruct", "retrieved_at_utc": utcnow().isoformat()})
            result["components"]["k1d"] = k1d

            sig = make_signal_sample(asset, spot, df_1h)
            sig["source_note"] = "ATR14(1h) DEMO"
            save_json(os.path.join(outdir_asset, "signal.json"), sig)
            result["components"]["signal"] = sig

            png = save_png_chart(asset, df_5m, outdir_asset)
            result["components"]["chart_png"] = {"path": png}

        elif asset in PROVIDERS_SYMBOLS:
            sym = PROVIDERS_SYMBOLS[asset]
            spot, spot_src = None, None
            df_5m, series_src = pd.DataFrame(), None

            # ---- 1) Twelve Data (elsődleges) ----
            if TD_KEY:
                try:
                    tdq, urlq = td_quote(sym["td"]["spot"])
                    # tdq: {"price": "...", ...}
                    if isinstance(tdq, dict) and tdq.get("price") is not None:
                        spot = float(tdq["price"])
                        spot_src = urlq

                    tds, urls = td_time_series(sym["td"]["series"], interval="5min", outputsize=390)
                    vals = tds.get("values", [])
                    if vals:
                        df = pd.DataFrame(vals)
                        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                        df = df.sort_values("datetime").set_index("datetime")
                        df = df[["open","high","low","close"]].astype(float)
                        df_5m = df
                        series_src = urls
                except Exception as e:
                    result["errors"].append(f"Twelve Data hiba: {e}")

            # ---- 2) Finnhub (tartalék) ----
            if (spot is None or df_5m.empty) and FH_KEY:
                try:
                    fhq, urlq = fh_quote(sym["fh"]["spot"])
                    if isinstance(fhq, dict) and fhq.get("c") not in (None, 0):
                        spot = float(fhq["c"])
                        spot_src = urlq
                    fhc, urls = fh_candle(sym["fh"]["series"], resolution="5", count=390)
                    if fhc.get("s") == "ok":
                        t = pd.to_datetime(pd.Series(fhc["t"]), unit="s", utc=True)
                        df = pd.DataFrame({
                            "open":  fhc["o"],
                            "high":  fhc["h"],
                            "low":   fhc["l"],
                            "close": fhc["c"]
                        }, index=t).astype(float).sort_index()
                        df_5m = df
                        series_src = urls
                except Exception as e:
                    result["errors"].append(f"Finnhub hiba: {e}")

            # ---- 3) Stooq 1D fallback (ha nincs intraday) ----
            if df_5m.empty:
                try:
                    st_sym = "%5Endx" if asset=="NSDQ100" else "xauusd"
                    df_csv, st_url = stooq_daily_csv(st_sym)
                    save_json(os.path.join(outdir_asset, "stooq_daily.json"),
                              {"rows": len(df_csv), "source_url": st_url, "retrieved_at_utc": utcnow().isoformat()})
                    result["components"]["stooq_daily"] = {"rows": len(df_csv)}
                except Exception as e:
                    result["errors"].append(f"Stooq fallback hiba: {e}")

            # spot.json mindenképp
            save_json(os.path.join(outdir_asset, "spot.json"), {
                "price_usd": spot, "source_url": spot_src,
                "retrieved_at_utc": utcnow().isoformat(), "age_sec": 0,
                "ok": spot is not None
            })
            result["components"]["spot"] = {"ok": spot is not None}

            # Aggregálás + mentések (ha nincs df, üres struktúrák is készülnek)
            def dump_or_empty(df, name, src):
                path = os.path.join(outdir_asset, f"{name}.json")
                if df is None or df.empty:
                    save_json(path, {"ohlc_utc_ms": [], "note":"empty"})
                    result["components"][name] = {"rows": 0}
                    return
                arr = [
                    [int(ix.value//10**6), float(o), float(h), float(l), float(c)]
                    for ix, (o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())
                ]
                save_json(path, {"ohlc_utc_ms": arr, "source_url": src, "retrieved_at_utc": utcnow().isoformat()})
                result["components"][name] = {"rows": len(arr)}

            df_1h = resample_ohlc(df_5m, "1H") if not df_5m.empty else pd.DataFrame(columns=["open","high","low","close"])
            df_4h = resample_ohlc(df_1h, "4H") if not df_1h.empty else pd.DataFrame(columns=["open","high","low","close"])

            dump_or_empty(df_5m, "klines_5m", series_src or "unknown")
            dump_or_empty(df_1h, "klines_1h", series_src or "unknown")
            dump_or_empty(df_4h, "klines_4h", series_src or "unknown")

            if not df_5m.empty:
                day = df_5m.index.normalize()[-1]
                day_df = df_5m[df_5m.index.normalize() == day]
                k1d = {
                    "open": float(day_df['open'].iloc[0]),
                    "high": float(day_df['high'].max()),
                    "low":  float(day_df['low'].min()),
                    "close":float(day_df['close'].iloc[-1])
                }
            else:
                k1d = {}
            save_json(os.path.join(outdir_asset, "k1d.json"),
                      {"k1d": k1d, "source": "td/fh_reconstruct", "retrieved_at_utc": utcnow().isoformat()})
            result["components"]["k1d"] = k1d

            sig = make_signal_sample(asset, spot, df_1h)
            sig["source_note"] = "ATR14(1h) DEMO"
            save_json(os.path.join(outdir_asset, "signal.json"), sig)
            result["components"]["signal"] = sig

            png = save_png_chart(asset, df_5m, outdir_asset)
            result["components"]["chart_png"] = {"path": png}

    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        result["trace"] = traceback.format_exc()
    finally:
        # Mini index.html – MINDIG legyen, hogy Pages ne dobjon 404-et
        try:
            errs = ""
            if result.get("errors"):
                errs = "<p><b>Errors:</b><br>" + "<br>".join([str(x) for x in result["errors"]]) + "</p>"
            index_html = f"""<html><head><meta charset="utf-8"><title>{asset} feed</title></head>
<body>
<h1>{asset}</h1>
<p>ok: {result.get("ok", True)}</p>
{errs}
<ul>
  <li><a href="spot.json">spot.json</a></li>
  <li><a href="klines_5m.json">klines_5m.json</a></li>
  <li><a href="klines_1h.json">klines_1h.json</a></li>
  <li><a href="klines_4h.json">klines_4h.json</a></li>
  <li><a href="k1d.json">k1d.json</a></li>
  <li><a href="signal.json">signal.json</a></li>
  <li><a href="chart_1d.png">chart_1d.png</a></li>
</ul>
</body></html>"""
            with open(os.path.join(outdir_asset, "index.html"), "w", encoding="utf-8") as f:
                f.write(index_html)
        except Exception:
            pass

    return result

def main():
    all_status = {"generated_at_utc": utcnow().isoformat(), "assets": {}}
    for a in ASSETS:
        st = run_asset(a)
        all_status["assets"][a] = st
    # gyökér index
    root_index = """<html><head><meta charset="utf-8"><title>Market Feed</title></head>
<body><h1>Market Feed</h1><ul>""" + "".join(
        [f'<li><a href="{a}/index.html">{a}</a></li>' for a in ASSETS]
    ) + "</ul></body></html>"
    with open(os.path.join(OUTDIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(root_index)
    save_json(os.path.join(OUTDIR, "status.json"), all_status)

if __name__ == "__main__":
    try:
        main()
        print("OK")
    except Exception as e:
        # SAFE-MODE: sose bukjon el CI-ben
        traceback.print_exc()
        os.makedirs("public", exist_ok=True)
        with open(os.path.join("public", "status.json"), "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": str(e), "note": "Top-level exception caught. See run_log.txt."}, f, ensure_ascii=False, indent=2)
        # exit 0 – hogy a deploy lépés ettől még lefusson
