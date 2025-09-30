# -*- coding: utf-8 -*-
"""
Market feed generator – GitHub Pages-re ír JSON/PNG kimeneteket.

Források:
- SOL (crypto): CoinGecko (kulcsmentes) – spot + 1 nap pontokból 5m OHLC.
- NSDQ100, GOLD_CFD: kizárólag Twelve Data (kulcsos) – intraday 5m + spot.
  Több alternatív ETF tickerrel próbálkozunk (QQQ/QQQM/QLD/TQQQ, GLD/IAU/BAR/SGOL/PHYS),
  és az első működőt használjuk.

SAFE-MODE: hiba esetén is készül index.html és status.json.
"""

import os, json, time, traceback
import datetime as dt
from datetime import timezone

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Alap beállítások ----------
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
OUTDIR = "public"
os.makedirs(OUTDIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (MarketFeed/1.3; +https://github.com/)"})

TD_KEY = os.environ.get("TWELVE_DATA_KEY")   # kötelező nem-kripto eszközökhöz

# Twelve Data ETF mapping – sorrendben próbáljuk
TD_SYMBOLS = {
    "NSDQ100": {
        # stabil ETF proxy-k NASDAQ-100-ra
        "series": ["QQQ", "QQQM", "QLD", "TQQQ", "QQQE"],
        "spot":   ["QQQ", "QQQM", "QLD", "TQQQ", "QQQE"],
    },
    "GOLD_CFD": {
        # CFD-proxy: arany spot FX (XAU/USD). Több névváltozat Twelve Data-hoz.
        # Utolsó fallback: COMEX futures jelölés.
        "series": ["XAU/USD", "XAUUSD", "XAUUSD:FOREX", "XAUUSD:CUR", "GC=F"],
        "spot":   ["XAU/USD", "XAUUSD", "XAUUSD:FOREX", "XAUUSD:CUR", "GC=F"],
    }
}

# CoinGecko ID map kriptókhoz
COINGECKO_IDS = { "SOL": "solana" }

# ---------- Util ----------
def utcnow(): return dt.datetime.now(timezone.utc)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def backoff_sleep(a): time.sleep(min(2**a + 0.25, 8))

# ---------- CoinGecko (SOL) ----------
def cg_simple_price(coin_id, vs="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs}"
    for a in range(4):
        r = SESSION.get(url, timeout=15)
        if r.ok: return r.json(), url
        backoff_sleep(a)
    raise RuntimeError(f"CG simple/price err {r.status_code}")

def cg_market_chart_range(coin_id, start_unix, end_unix, vs="usd"):
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
           f"?vs_currency={vs}&from={start_unix}&to={end_unix}")
    for a in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok: return r.json(), url
        backoff_sleep(a)
    raise RuntimeError(f"CG range err {r.status_code}")

def prices_to_ohlc_5m(ts_prices):
    if not ts_prices: return pd.DataFrame(columns=["open","high","low","close"])
    s = pd.Series({pd.to_datetime(ms, unit='ms', utc=True): p for ms, p in ts_prices})
    df = pd.DataFrame({"price": s}).sort_index()
    o = df['price'].resample("5T").first()
    h = df['price'].resample("5T").max()
    l = df['price'].resample("5T").min()
    c = df['price'].resample("5T").last()
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()

# ---------- Twelve Data helpers ----------
def td_quote(symbol):
    if not TD_KEY: raise RuntimeError("TWELVE_DATA_KEY hiányzik (GitHub Secrets).")
    base = "https://api.twelvedata.com/quote"
    params = {"symbol": symbol, "format": "JSON", "apikey": TD_KEY}
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=15)
        if r.ok: return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"TD quote err {r.status_code}")

def td_time_series(symbol, interval="5min", outputsize=390):
    if not TD_KEY: raise RuntimeError("TWELVE_DATA_KEY hiányzik (GitHub Secrets).")
    base = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "timezone": "UTC", "format": "JSON", "apikey": TD_KEY}
    for a in range(4):
        r = SESSION.get(base, params=params, timeout=20)
        if r.ok: return r.json(), r.url
        backoff_sleep(a)
    raise RuntimeError(f"TD time_series err {r.status_code}")

def try_td_series(symbols):
    """Végigpróbálja a symbols listát; visszaadja az első működő 5m OHLC DataFrame-et + forrás URL + szimbólum."""
    last_err = None
    for sym in symbols:
        try:
            js, url = td_time_series(sym, interval="5min", outputsize=390)
            vals = js.get("values") or js.get("data")
            if not vals:
                if isinstance(js, dict) and js.get("status") == "error":
                    last_err = js.get("message") or js
                continue
            df = pd.DataFrame(vals)
            if not set(["open","high","low","close","datetime"]).issubset(df.columns):
                continue
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").set_index("datetime")
            df = df[["open","high","low","close"]].astype(float)
            if not df.empty:
                return df, url, sym
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Twelve Data time_series: nincs használható adat ({last_err})")

def try_td_quote(symbols):
    """Első működő spot ár + URL + szimbólum. Ha nem jön, vissza (None, None, None)."""
    last_err = None
    for sym in symbols:
        try:
            js, url = td_quote(sym)
            if isinstance(js, dict) and js.get("price") is not None:
                return float(js["price"]), url, sym
            if isinstance(js, dict) and js.get("status") == "error":
                last_err = js.get("message") or js
        except Exception as e:
            last_err = str(e)
            continue
    return None, None, None

# ---------- Technikai segédek ----------
def resample_ohlc(df, rule):
    if df.empty: return pd.DataFrame(columns=["open","high","low","close"])
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()

def atr14_from_ohlc(df_1h):
    if df_1h.empty or len(df_1h) < 15: return None
    hi, lo, cl = df_1h['high'].values, df_1h['low'].values, df_1h['close'].values
    trs = [hi[i]-lo[i] if i==0 else max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])) for i in range(len(cl))]
    return float(pd.Series(trs).rolling(14).mean().iloc[-1])

def make_signal_sample(asset, spot, df_1h):
    atr = atr14_from_ohlc(df_1h)
    if atr is None or spot is None:
        return {"note":"sample only","atr14_1h": None, "source_note":"ATR14(1h) DEMO"}
    entry = spot
    sl = max(0.0, entry - 0.9*atr)
    tp1 = entry + 1.0*atr
    tp2 = entry + 2.0*atr
    return {"side":"LONG","entry":round(entry,5),"sl":round(sl,5),
            "tp1":round(tp1,5),"tp2":round(tp2,5),"atr14_1h":round(atr,5),
            "leverage":"3x (demo)","disclaimer":"DEMO","source_note":"ATR14(1h) DEMO"}

def save_png_chart(asset, df_5m, outdir_asset):
    if df_5m.empty: return None
    plt.figure(figsize=(8,3))
    plt.plot(df_5m.index, df_5m['close'])
    plt.title(f"{asset} – 1D intraday (close)")
    plt.tight_layout()
    path = os.path.join(outdir_asset, "chart_1d.png")
    plt.savefig(path); plt.close()
    return path

# ---------- Fő eszköz-futás ----------
def run_asset(asset):
    started = utcnow()
    out = {"asset": asset, "ok": True, "errors": [], "retrieved_at_utc": started.isoformat(), "components": {}}
    outdir = os.path.join(OUTDIR, asset); ensure_dir(outdir)

    try:
        if asset == "SOL":
            sp, sp_url = cg_simple_price(COINGECKO_IDS["SOL"])
            spot = float(sp["solana"]["usd"])
            save_json(os.path.join(outdir,"spot.json"),
                      {"price_usd":spot,"source_url":sp_url,"retrieved_at_utc":utcnow().isoformat(),"age_sec":0})
            out["components"]["spot"] = {"ok": True}

            now = int(utcnow().timestamp()); start = now - 24*3600
            mc, mc_url = cg_market_chart_range(COINGECKO_IDS["SOL"], start, now)
            df5 = prices_to_ohlc_5m(mc.get("prices", []))
            df1 = resample_ohlc(df5, "1H"); df4 = resample_ohlc(df1, "4H")

            def dump(df, name, src):
                arr = [[int(ix.value//10**6), float(o),float(h),float(l),float(c)]
                       for ix,(o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())]
                save_json(os.path.join(outdir,f"{name}.json"),
                          {"ohlc_utc_ms":arr,"source_url":src,"retrieved_at_utc":utcnow().isoformat()})
                out["components"][name] = {"rows": len(arr)}

            dump(df5,"klines_5m",mc_url); dump(df1,"klines_1h",mc_url); dump(df4,"klines_4h",mc_url)

            if not df5.empty:
                day = df5.index.normalize()[-1]; d = df5[df5.index.normalize()==day]
                k1d = {"open": float(d['open'].iloc[0]), "high": float(d['high'].max()),
                       "low":  float(d['low'].min()),  "close": float(d['close'].iloc[-1])}
            else:
                k1d = {}
            save_json(os.path.join(outdir,"k1d.json"),
                      {"k1d":k1d,"source":"cg_reconstruct","retrieved_at_utc":utcnow().isoformat()})
            out["components"]["k1d"] = k1d

            sig = make_signal_sample(asset, spot, df1); save_json(os.path.join(outdir,"signal.json"), sig)
            save_png_chart(asset, df5, outdir)

        else:
            if not TD_KEY:
                raise RuntimeError("TWELVE_DATA_KEY hiányzik. Add meg GitHub Secrets-ben.")

            # 5m intraday – ETF listáról az első működő
            try:
                df5, src_series, series_sym = try_td_series(TD_SYMBOLS[asset]["series"])
            except Exception as e:
                out["errors"].append(f"Twelve Data 5m hiba: {e}")
                df5, src_series, series_sym = pd.DataFrame(), None, None

            # spot – több ticker; ha nincs, 5m utolsó close
            spot, src_spot, spot_sym = try_td_quote(TD_SYMBOLS[asset]["spot"])
            if spot is None and not df5.empty:
                spot = float(df5['close'].iloc[-1])
                src_spot = (src_series or "unknown") + " (fallback last 5m close)"

            save_json(os.path.join(outdir,"spot.json"),
                      {"price_usd": spot, "source_url": src_spot,
                       "retrieved_at_utc": utcnow().isoformat(), "age_sec": 0, "ok": spot is not None})
            out["components"]["spot"] = {"ok": spot is not None}

            def dump_or_empty(df, name, src):
                p = os.path.join(outdir,f"{name}.json")
                if df is None or df.empty:
                    save_json(p, {"ohlc_utc_ms": [], "note":"empty"})
                    out["components"][name] = {"rows": 0}
                    return
                arr = [[int(ix.value//10**6), float(o),float(h),float(l),float(c)]
                       for ix,(o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())]
                save_json(p, {"ohlc_utc_ms":arr,"source_url":src,"retrieved_at_utc":utcnow().isoformat(),
                              "symbol_used": series_sym})
                out["components"][name] = {"rows": len(arr)}

            df1 = resample_ohlc(df5, "1H") if not df5.empty else pd.DataFrame(columns=["open","high","low","close"])
            df4 = resample_ohlc(df1, "4H") if not df1.empty else pd.DataFrame(columns=["open","high","low","close"])

            dump_or_empty(df5,"klines_5m",src_series or "unknown")
            dump_or_empty(df1,"klines_1h",src_series or "unknown")
            dump_or_empty(df4,"klines_4h",src_series or "unknown")

            if not df5.empty:
                day = df5.index.normalize()[-1]; d = df5[df5.index.normalize()==day]
                k1d = {"open": float(d['open'].iloc[0]), "high": float(d['high'].max()),
                       "low":  float(d['low'].min()),  "close": float(d['close'].iloc[-1])}
            else:
                k1d = {}
            save_json(os.path.join(outdir,"k1d.json"),
                      {"k1d":k1d,"source":"td_reconstruct","retrieved_at_utc":utcnow().isoformat(),
                       "symbol_used": series_sym})

            sig = make_signal_sample(asset, spot, df1); save_json(os.path.join(outdir,"signal.json"), sig)
            if not df5.empty: save_png_chart(asset, df5, outdir)

    except Exception as e:
        out["ok"] = False
        out["errors"].append(str(e))
        out["trace"] = traceback.format_exc()
    finally:
        # mini index MINDIG
        try:
            errs = ""
            if out.get("errors"):
                errs = "<p><b>Errors:</b><br>" + "<br>".join([str(x) for x in out["errors"]]) + "</p>"
            html = f"""<html><head><meta charset="utf-8"><title>{asset} feed</title></head>
<body>
<h1>{asset}</h1>
<p>ok: {out.get("ok", True)}</p>
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
            with open(os.path.join(outdir,"index.html"),"w",encoding="utf-8") as f: f.write(html)
        except Exception:
            pass
    return out

def main():
    status = {"generated_at_utc": utcnow().isoformat(), "assets": {}}
    for a in ASSETS:
        status["assets"][a] = run_asset(a)
    root = "<html><head><meta charset='utf-8'><title>Market Feed</title></head><body><h1>Market Feed</h1><ul>" + \
           "".join([f"<li><a href='{a}/index.html'>{a}</a></li>" for a in ASSETS]) + "</ul></body></html>"
    with open(os.path.join(OUTDIR,"index.html"),"w",encoding="utf-8") as f: f.write(root)
    save_json(os.path.join(OUTDIR,"status.json"), status)

if __name__ == "__main__":
    try:
        main(); print("OK")
    except Exception as e:
        traceback.print_exc()
        os.makedirs("public", exist_ok=True)
        save_json(os.path.join("public","status.json"),
                  {"ok": False, "error": str(e), "note": "Top-level exception caught. See run_log.txt."})

