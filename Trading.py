# -*- coding: utf-8 -*-
"""
Market feed generator – GitHub Pages-re ír JSON/PNG kimeneteket.

Források:
- Crypto (SOL): CoinGecko (kulcsmentes) – spot + 1 nap pontokból 5m OHLC
- NSDQ100, GOLD_CFD:
    Intraday 5m: Stooq a2 CSV (kulcsmentes) → stabil
    Spot: Twelve Data (kulcsos) → ha nincs, 5m utolsó close fallback
    (Napi CSV fallback már nem kell, mert a2 intraday-t használunk.)

Kimenet assetenként (public/<ASSET>/):
  spot.json, klines_5m.json, klines_1h.json, klines_4h.json, k1d.json, signal.json, chart_1d.png, index.html
Gyökér: public/status.json, public/index.html (a workflow írja run_log.txt-t)

SAFE-MODE: hiba esetén is készül index.html és status.json.
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
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (MarketFeed/1.1; +https://github.com/)"})

# Kulcsok (GitHub Secrets-ből jönnek a workflow env-vel)
TD_KEY = os.environ.get("TWELVE_DATA_KEY")   # Twelve Data

# Jelölések / Stooq kódok
PROVIDERS_SYMBOLS = {
    "NSDQ100": {
        "stooq_intraday": "%5Endx",   # ^NDX → %5Endx
        "td_spot": "NQ=F",            # Twelve Data spot próbálkozás (ha nincs coverage, fallback lesz)
    },
    "GOLD_CFD": {
        "stooq_intraday": "xauusd",   # XAUUSD intraday
        "td_spot": "XAU/USD",         # Twelve Data spot (alternatíva: GC=F futures, de nem kell most)
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
def backoff_sleep(attempt): time.sleep(min(2 ** attempt + 0.25, 10))

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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

# ---------- Twelve Data (spot) ----------
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

# ---------- Stooq intraday a2 CSV (5m) ----------
def stooq_intraday_csv(symbol_encoded, interval="5"):
    """
    Pl. ^NDX 5m: https://stooq.com/q/a2/?s=%5Endx&i=5
        XAUUSD 5m: https://stooq.com/q/a2/?s=xauusd&i=5
    Visszatér: DataFrame index=UTC Datetime, oszlopok: open, high, low, close
    """
    url = f"https://stooq.com/q/a2/?s={symbol_encoded}&i={interval}"
    for a in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok:
            text = r.text.strip()
            # A válasz CSV: Date,Time,Open,High,Low,Close,Volume
            if text.startswith("Date,"):
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                # Dátum+idő -> UTC
                dt_col = pd.to_datetime(df["Date"] + " " + df["Time"], utc=True)
                o = df["Open"].astype(float); h = df["High"].astype(float)
                l = df["Low"].astype(float);  c = df["Close"].astype(float)
                out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=dt_col)
                out = out.sort_index()
                return out, url
        backoff_sleep(a)
    raise RuntimeError(f"Stooq a2 intraday error: {r.status_code if 'r' in locals() else 'n/a'}")

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

            # ---- 1) Intraday 5m: Stooq a2 (kulcsmentes) ----
            df_5m, series_src = pd.DataFrame(), None
            try:
                df_5m, series_src = stooq_intraday_csv(sym["stooq_intraday"], interval="5")
            except Exception as e:
                result["errors"].append(f"Stooq a2 intraday hiba: {e}")

            # ---- 2) Spot: Twelve Data, ha van coverage; különben utolsó 5m close ----
            spot, spot_src = None, None
            if TD_KEY:
                try:
                    tdq, urlq = td_quote(sym["td_spot"])
                    # tdq: {"price": "...", ...}
                    if isinstance(tdq, dict) and tdq.get("price") is not None:
                        spot = float(tdq["price"])
                        spot_src = urlq
                except Exception as e:
                    result["errors"].append(f"Twelve Data spot hiba: {e}")

            if spot is None and not df_5m.empty:
                # Fallback: 5m utolsó close
                spot = float(df_5m['close'].iloc[-1])
                spot_src = series_src + "  (fallback: last 5m close)"

            save_json(os.path.join(outdir_asset, "spot.json"), {
                "price_usd": spot, "source_url": spot_src,
                "retrieved_at_utc": utcnow().isoformat(), "age_sec": 0,
                "ok": spot is not None
            })
            result["components"]["spot"] = {"ok": spot is not None}

            # ---- Aggregálás + mentések (ha nincs df, üres struktúrák) ----
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
                      {"k1d": k1d, "source": "stooq_a2_reconstruct", "retrieved_at_utc": utcnow().isoformat()})
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
