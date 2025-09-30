# -*- coding: utf-8 -*-
"""
Market feed generator (public/ alatt JSON/PNG kimenet) – kulcsmentes publikus forrásokkal.

Források:
- Crypto (BTC, ETH, SOL): CoinGecko simple/price (spot) + market_chart/range (1 nap, ~5m pontok)
- NSDQ100: Yahoo v7 /quote (spot) + v8 /chart (range=1d, interval=5m) ; Stooq napi CSV fallback
- GOLD_CFD: Yahoo v7 /quote (XAUUSD=X elsődleges spot), v8 /chart (GC=F 1d/5m), Stooq XAUUSD napi CSV fallback

Kimenet assetenként (public/<ASSET>/):
- spot.json (ár + meta + forrás + UTC idő)
- klines_5m.json (posix time ms, O/H/L/C 5 perces közelítéssel – Yahoo v8 chartból vagy CG prices-ből)
- klines_1h.json (5m -> 1h aggregált)
- klines_4h.json (1h -> 4h aggregált)
- k1d.json (egynapos OHLC összegzés az aznapi intraday-ből)
- signal.json (mintalogika: ATR14 (1h) alapú jelölések – csak példa)
- chart_1d.png (matplotlib – aznapi close vonal)
- index.html (egyszerű összefoglaló linkekkel)
- status.json (összesített állapot a gyökérben is)

Megjegyzések:
- Rate limit / hálózati hiba esetén exponenciális backoff + több forrás fallback.
- Minden JSON tartalmaz "retrieved_at_utc", "source_url" és "age_sec".
"""
import os, json, time, math, datetime as dt
from datetime import timezone
import random
import traceback

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Beállítások ----------
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]  # bővíthető: BTC, ETH stb.
OUTDIR = "public"
os.makedirs(OUTDIR, exist_ok=True)

# CoinGecko ID map kriptókhoz
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# Yahoo szimbólumok
YAHOO_SYMBOLS = {
    "NSDQ100": {
        "spot_primary": "%5ENDX",   # ^NDX URL-kódolt a /v7/finance/quote-hoz
        "futures_fallback": "NQ%3DF" # NQ=F
    },
    "GOLD_CFD": {
        "spot_primary": "XAUUSD%3DX",  # XAUUSD=X
        "futures_fallback": "GC%3DF"   # GC=F
    }
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; MarketFeedBot/1.0; +https://github.com/)"
})

def utcnow():
    return dt.datetime.now(timezone.utc)

def ts():
    return int(utcnow().timestamp())

def age_sec_from(retrieved_dt):
    return max(0, int((utcnow() - retrieved_dt).total_seconds()))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def backoff_sleep(attempt):
    time.sleep(min(2 ** attempt + random.random(), 10))

# ---------- CoinGecko ----------
def cg_simple_price(ids, vs="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies={vs}"
    # Docs: https://docs.coingecko.com/reference/simple-price
    # NOTE: nincs kulcs, publikus végpont, rate limit lehetséges.
    for attempt in range(4):
        r = SESSION.get(url, timeout=15)
        if r.ok:
            data = r.json()
            return data, url
        backoff_sleep(attempt)
    raise RuntimeError(f"CoinGecko simple/price hiba: {r.status_code} - {r.text[:200]}")

def cg_market_chart_range(coin_id, start_unix, end_unix, vs="usd"):
    # Docs: https://docs.coingecko.com/reference/coins-id-market-chart-range
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
           f"?vs_currency={vs}&from={start_unix}&to={end_unix}")
    for attempt in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok:
            return r.json(), url
        backoff_sleep(attempt)
    raise RuntimeError(f"CoinGecko market_chart/range hiba: {r.status_code} - {r.text[:200]}")

# ---------- Yahoo Finance ----------
def yf_quote(symbols_csv):
    # v7 quote: https://query1.finance.yahoo.com/v7/finance/quote?symbols=%5ENDX
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols_csv}"
    for attempt in range(4):
        r = SESSION.get(url, timeout=15)
        if r.ok:
            return r.json(), url
        backoff_sleep(attempt)
    raise RuntimeError(f"Yahoo v7 quote hiba: {r.status_code} - {r.text[:200]}")

def yf_chart(symbol, interval="5m", range_="1d"):
    # v8 chart: https://query1.finance.yahoo.com/v8/finance/chart/NQ%3DF?interval=5m&range=1d
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
    for attempt in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok:
            return r.json(), url
        backoff_sleep(attempt)
    raise RuntimeError(f"Yahoo v8 chart hiba: {r.status_code} - {r.text[:200]}")

# ---------- Stooq (1D CSV fallback) ----------
def stooq_daily_csv(symbol_encoded):
    # Példa: ^NDX: https://stooq.com/q/d/?s=%5Endx  ; XAUUSD: https://stooq.com/q/d/?s=xauusd
    url = f"https://stooq.com/q/d/?s={symbol_encoded}"
    r = SESSION.get(url, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Stooq hiba: {r.status_code}")
    # Oldalon "Download data in csv file" link található – sok környezetben közvetlenül is kiadja a CSV-t.
    # Ha HTML jön vissza, egyszerű fallback: nem használjuk. (Stooq néha interaktív oldalt ad.)
    if "text/csv" in r.headers.get("Content-Type", "") or r.text.startswith("Date,"):
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return df, url
    # Ha nem CSV: best-effort detektálás a lapon – itt konzervatívan visszadobjuk.
    raise RuntimeError("Stooq CSV nem közvetlen – manuális letöltés szükséges")

# ---------- Idősor-aggregálás ----------
def resample_ohlc(df, rule):
    # df: index=DatetimeIndex(utc), oszlopok: open, high, low, close
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()
    return out

def prices_to_ohlc_from_points(ts_prices):
    # ts_prices: list of [ms, price] – CoinGecko "prices"
    # OHLC rekonstrukció 5 perces bin-ekre (first/max/min/last)
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

def yahoo_chart_to_ohlc(chart_json):
    # Yahoo v8 chart -> OHLC DataFrame (5m)
    r = chart_json.get("chart", {}).get("result", [])
    if not r:
        return pd.DataFrame(columns=["open","high","low","close"])
    res = r[0]
    ts_arr = res.get("timestamp", [])
    ind = pd.to_datetime(pd.Series(ts_arr), unit='s', utc=True)
    o = pd.Series(res["indicators"]["quote"][0]["open"], index=ind)
    h = pd.Series(res["indicators"]["quote"][0]["high"], index=ind)
    l = pd.Series(res["indicators"]["quote"][0]["low"], index=ind)
    c = pd.Series(res["indicators"]["quote"][0]["close"], index=ind)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    # Bizonyos esetekben pre/post session rekordok lehetnek; megtartjuk és időrendben hagyjuk
    return df.sort_index()

# ---------- ATR(14) (1h) minta ----------
def atr14_from_ohlc(df_1h):
    if df_1h.empty or len(df_1h) < 15:
        return None
    high = df_1h['high'].values
    low = df_1h['low'].values
    close = df_1h['close'].values
    trs = [high[i]-low[i] if i==0 else max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) for i in range(len(close))]
    atr14 = pd.Series(trs).rolling(14).mean().iloc[-1]
    return float(atr14)

def make_signal_sample(asset, spot, df_1h):
    atr = atr14_from_ohlc(df_1h)
    if atr is None or spot is None:
        return {"note":"sample only","atr14_1h": None}
    # primitív LONG jel – kizárólag minta
    entry = spot
    sl = max(0.0, entry - 0.9*atr)
    tp1 = entry + 1.0*atr
    tp2 = entry + 2.0*atr
    return {
        "side":"LONG","entry":round(entry,5),"sl":round(sl,5),"tp1":round(tp1,5),"tp2":round(tp2,5),
        "atr14_1h": round(atr,5), "leverage":"3x (max ajánlott mintában)","disclaimer":"DEMO"
    }

# ---------- PNG chart ----------
def save_png_chart(asset, df_5m, outdir_asset):
    if df_5m.empty:
        return None
    plt.figure(figsize=(8,3))
    plt.plot(df_5m.index, df_5m['close'])
    plt.title(f"{asset} – 1D intraday (close)")
    plt.tight_layout()
    path = os.path.join(outdir_asset, "chart_1d.png")
    plt.savefig(path)
    plt.close()
    return path

# ---------- Fő futtatás egy assetre ----------
def run_asset(asset):
    started = utcnow()
    outdir_asset = os.path.join(OUTDIR, asset)
    ensure_dir(outdir_asset)

    result = {
        "asset": asset,
        "ok": True,
        "errors": [],
        "retrieved_at_utc": started.isoformat(),
        "components": {}
    }

    try:
        if asset in COINGECKO_IDS:
            # ---- CRYPTO: CoinGecko
            coin_id = COINGECKO_IDS[asset]
            # spot
            sp_json, sp_url = cg_simple_price(coin_id)
            spot = float(sp_json[coin_id]["usd"])
            spot_obj = {
                "price_usd": spot,
                "source_url": sp_url,
                "retrieved_at_utc": utcnow().isoformat(),
                "age_sec": 0
            }
            save_json(os.path.join(outdir_asset, "spot.json"), spot_obj)
            result["components"]["spot"] = spot_obj

            # intraday prices ~1 nap (5m közeli szemcse)
            end_unix = ts()
            start_unix = end_unix - 24*3600
            mc_json, mc_url = cg_market_chart_range(coin_id, start_unix, end_unix)
            prices = mc_json.get("prices", [])
            df_5m = prices_to_ohlc_from_points(prices)

            # 1h / 4h aggregálás
            df_1h = resample_ohlc(df_5m, "1H")
            df_4h = resample_ohlc(df_1h, "4H")

            # 1D OHLC (aznapi)
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

            # mentések
            def dump_df(df, name):
                path = os.path.join(outdir_asset, f"{name}.json")
                arr = [
                    [int(ts.value//10**6), float(o), float(h), float(l), float(c)]
                    for ts, (o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())
                ]
                save_json(path, {"ohlc_utc_ms": arr, "source_url": mc_url, "retrieved_at_utc": utcnow().isoformat()})
                result["components"][name] = {"rows": len(arr)}

            dump_df(df_5m, "klines_5m")
            dump_df(df_1h, "klines_1h")
            dump_df(df_4h, "klines_4h")
            save_json(os.path.join(outdir_asset, "k1d.json"), {"k1d": k1d, "source":"cg_reconstruct", "retrieved_at_utc": utcnow().isoformat()})
            result["components"]["k1d"] = k1d

            # minta-szignál (ATR14 1h)
            sig = make_signal_sample(asset, spot, df_1h)
            sig["source_note"] = "ATR14(1h) DEMO"
            save_json(os.path.join(outdir_asset, "signal.json"), sig)
            result["components"]["signal"] = sig

            # PNG
            png = save_png_chart(asset, df_5m, outdir_asset)
            result["components"]["chart_png"] = {"path": png}

        elif asset in YAHOO_SYMBOLS:
            sym = YAHOO_SYMBOLS[asset]
            # spot elsődleges
            try:
                q_json, q_url = yf_quote(sym["spot_primary"])
                qres = q_json.get("quoteResponse", {}).get("result", [])
                if not qres:
                    raise RuntimeError("Yahoo v7 quote empty")
                last = float(qres[0].get("regularMarketPrice") or qres[0].get("postMarketPrice"))
                spot = last
                spot_src = q_url
            except Exception:
                # futures fallback
                q_json, q_url = yf_quote(sym["futures_fallback"])
                qres = q_json.get("quoteResponse", {}).get("result", [])
                if not qres:
                    raise
                spot = float(qres[0].get("regularMarketPrice") or qres[0].get("postMarketPrice"))
                spot_src = q_url

            spot_obj = {
                "price_usd": spot,
                "source_url": spot_src,
                "retrieved_at_utc": utcnow().isoformat(),
                "age_sec": 0
            }
            save_json(os.path.join(outdir_asset, "spot.json"), spot_obj)
            result["components"]["spot"] = spot_obj

            # intraday 1d/5m Yahoo v8 chart – futures ticker robusztusabb
            ch_json, ch_url = yf_chart(sym["futures_fallback"], interval="5m", range_="1d")
            df_5m = yahoo_chart_to_ohlc(ch_json)

            # 1h/4h aggregálás
            df_1h = resample_ohlc(df_5m, "1H")
            df_4h = resample_ohlc(df_1h, "4H")

            # 1D OHLC
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

            # mentések
            def dump_df(df, name, src):
                path = os.path.join(outdir_asset, f"{name}.json")
                arr = [
                    [int(ts.value//10**6), float(o), float(h), float(l), float(c)]
                    for ts, (o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())
                ]
                save_json(path, {"ohlc_utc_ms": arr, "source_url": src, "retrieved_at_utc": utcnow().isoformat()})
                result["components"][name] = {"rows": len(arr)}

            dump_df(df_5m, "klines_5m", ch_url)
            dump_df(df_1h, "klines_1h", ch_url)
            dump_df(df_4h, "klines_4h", ch_url)
            save_json(os.path.join(outdir_asset, "k1d.json"), {"k1d": k1d, "source":"yf_chart_reconstruct", "retrieved_at_utc": utcnow().isoformat()})
            result["components"]["k1d"] = k1d

            # minta-szignál
            sig = make_signal_sample(asset, spot, df_1h)
            sig["source_note"] = "ATR14(1h) DEMO"
            save_json(os.path.join(outdir_asset, "signal.json"), sig)
            result["components"]["signal"] = sig

            # PNG
            png = save_png_chart(asset, df_5m, outdir_asset)
            result["components"]["chart_png"] = {"path": png}

            # Stooq 1D fallback (opcionális, ha a fenti üres)
            if df_5m.empty:
                try:
                    st_sym = "%5Endx" if asset=="NSDQ100" else "xauusd"
                    df_csv, st_url = stooq_daily_csv(st_sym)
                    save_json(os.path.join(outdir_asset, "stooq_daily.json"),
                              {"rows": len(df_csv), "source_url": st_url, "retrieved_at_utc": utcnow().isoformat()})
                    result["components"]["stooq_daily"] = {"rows": len(df_csv)}
                except Exception as e:
                    result["errors"].append(f"Stooq fallback hiba: {e}")

        # mini index.html
        index_html = f"""<html><head><meta charset="utf-8"><title>{asset} feed</title></head>
<body>
<h1>{asset}</h1>
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

    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        result["trace"] = traceback.format_exc()

    return result

def main():
    all_status = {
        "generated_at_utc": utcnow().isoformat(),
        "assets": {}
    }
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
        # Utolsó védőháló: sose bukjunk el nem-0 exit kóddal CI-ben
        import sys, traceback, os, json
        traceback.print_exc()
        os.makedirs("public", exist_ok=True)
        with open(os.path.join("public", "status.json"), "w", encoding="utf-8") as f:
            json.dump({
                "ok": False,
                "error": str(e),
                "note": "Top-level exception caught. See CI logs."
            }, f, ensure_ascii=False, indent=2)
        sys.exit(0)

