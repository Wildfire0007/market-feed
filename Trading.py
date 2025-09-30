# -*- coding: utf-8 -*-
"""
Trading.py — Market feed generator GitHub Pages-re.

Források:
- SOL (crypto): CoinGecko — spot + 24h pontokból 5m OHLC (reconstruct)
- NSDQ100 (index-proxy): Twelve Data ETF (QQQ/QQQM/QLD/TQQQ/QQQE) — 5m és spot
- GOLD_CFD (CFD-proxy): Twelve Data XAU/USD — 5m és spot; fallback Stooq a2 (xauusd) 5m

Kimenet (assetenként, public/<ASSET>/):
  spot.json
  klines_5m.json
  klines_1h.json
  klines_4h.json
  klines_1d.json     ← a Worker /k1d ezt szolgálja ki
  signal.json        ← ATR-mintajelzés (DEMO)
  index.html
Gyökér:
  public/status.json
  public/index.html
  public/run_log.txt (stdout/err átirányítva a workflow-ból)

SAFE-MODE: hiba esetén is készül index.html és status.json.
"""

import os, json, time, traceback
import datetime as dt
from datetime import timezone
import requests
import pandas as pd
import numpy as np

# ---------- Alap ----------
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]
OUTDIR = "public"
os.makedirs(OUTDIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MarketFeed/1.4 (+https://github.com/)"})

TD_KEY = os.environ.get("TWELVE_DATA_KEY")  # Twelve Data API kulcs

# Twelve Data ticker-listák (első működő nyer)
TD_SYMBOLS = {
    "NSDQ100": {
        "series": ["QQQ", "QQQM", "QLD", "TQQQ", "QQQE"],
        "spot":   ["QQQ", "QQQM", "QLD", "TQQQ", "QQQE"],
    },
    "GOLD_CFD": {
        "series": ["XAU/USD", "XAUUSD", "XAUUSD:FOREX", "XAUUSD:CUR", "GC=F"],
        "spot":   ["XAU/USD", "XAUUSD", "XAUUSD:FOREX", "XAUUSD:CUR", "GC=F"],
    }
}

# CoinGecko ID-k
CG_IDS = {"SOL": "solana"}

# ---------- Util ----------
def utcnow(): return dt.datetime.now(timezone.utc)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def backoff_sleep(a): time.sleep(min(2**a + 0.25, 8))

# ---------- CoinGecko ----------
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
    # ts_prices = [[ms, price], ...]
    if not ts_prices: return pd.DataFrame(columns=["open","high","low","close"])
    s = pd.Series({pd.to_datetime(ms, unit='ms', utc=True): p for ms, p in ts_prices})
    df = pd.DataFrame({"price": s}).sort_index()
    o = df['price'].resample("5T").first()
    h = df['price'].resample("5T").max()
    l = df['price'].resample("5T").min()
    c = df['price'].resample("5T").last()
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()

# ---------- Twelve Data ----------
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
    """Robusztus 5m OHLC beolvasás Twelve Data-ból (oszlopok kisbetűsítése)."""
    last_err = None
    for sym in symbols:
        try:
            js, url = td_time_series(sym, interval="5min", outputsize=390)
            vals = js.get("values") or js.get("data")
            if not isinstance(vals, list) or not vals:
                if isinstance(js, dict) and js.get("status") == "error":
                    last_err = js.get("message") or js
                else:
                    last_err = f"no values for {sym}"
                continue
            df = pd.DataFrame(vals)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "datetime" in df.columns:
                dtcol = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            elif "date" in df.columns and "time" in df.columns:
                dtcol = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str), utc=True, errors="coerce")
            else:
                last_err = f"missing datetime columns for {sym}"
                continue
            for c in ("open","high","low","close"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            need = {"open","high","low","close"}
            if not need.issubset(set(df.columns)):
                last_err = f"missing ohlc for {sym}"
                continue
            df = df.assign(datetime=dtcol).dropna(subset=["datetime"])
            df = df.sort_values("datetime").set_index("datetime")
            df = df[["open","high","low","close"]].dropna()
            if not df.empty:
                return df, url, sym
            last_err = f"empty df after cleaning for {sym}"
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"TD time_series: nincs használható adat ({last_err})")

def try_td_quote(symbols):
    """Első működő spot ár (float) + URL + szimbólum; None, ha nincs."""
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

# ---------- Stooq a2 (kulcsmentes) — GOLD_CFD fallback ----------
def stooq_intraday_csv(symbol_encoded, interval="5"):
    # pl. XAUUSD 5m: https://stooq.com/q/a2/?s=xauusd&i=5
    url = f"https://stooq.com/q/a2/?s={symbol_encoded}&i={interval}"
    for a in range(4):
        r = SESSION.get(url, timeout=20)
        if r.ok:
            text = r.text.strip()
            if text.startswith("Date,"):
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                dt_col = pd.to_datetime(df["Date"] + " " + df["Time"], utc=True, errors="coerce")
                o = pd.to_numeric(df["Open"], errors="coerce")
                h = pd.to_numeric(df["High"], errors="coerce")
                l = pd.to_numeric(df["Low"], errors="coerce")
                c = pd.to_numeric(df["Close"], errors="coerce")
                out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=dt_col).dropna().sort_index()
                return out, url
        backoff_sleep(a)
    raise RuntimeError(f"Stooq a2 intraday error @ {url}")

# ---------- Idősor-aggregálás ----------
def resample_ohlc(df, rule):
    if df is None or df.empty: return pd.DataFrame(columns=["open","high","low","close"])
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna()

# ---------- ATR-mintajel ----------
def atr14_from_ohlc(df_1h):
    if df_1h.empty or len(df_1h) < 15: return None
    hi, lo, cl = df_1h['high'].values, df_1h['low'].values, df_1h['close'].values
    trs = [hi[i]-lo[i] if i==0 else max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])) for i in range(len(cl))]
    return float(pd.Series(trs).rolling(14).mean().iloc[-1])

def make_signal_sample(asset, spot, df_1h):
    atr = atr14_from_ohlc(df_1h)
    if atr is None or spot is None:
        return {"note":"ATR demo only","atr14_1h": None}
    entry = spot
    sl = max(0.0, entry - 0.9*atr)
    tp1 = entry + 1.0*atr
    tp2 = entry + 2.0*atr
    return {
        "side":"LONG","entry":round(entry,6),"sl":round(sl,6),
        "tp1":round(tp1,6),"tp2":round(tp2,6),
        "atr14_1h": round(atr,6),"leverage":"demo","disclaimer":"DEMO"
    }

# ---------- PNG chart (opcionális, most nem kötelező) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_png_chart(asset, df_5m, outdir_asset):
    if df_5m.empty: return None
    plt.figure(figsize=(8,3))
    plt.plot(df_5m.index, df_5m['close'])
    plt.title(f"{asset} – 1D intraday (close)")
    plt.tight_layout()
    path = os.path.join(outdir_asset, "chart_1d.png")
    plt.savefig(path); plt.close()
    return path

# ---------- Fő futás ----------
def dump_ohlc_as_json(outdir_asset, name, df, src, symbol_used=None):
    path = os.path.join(outdir_asset, f"{name}.json")
    if df is None or df.empty:
        save_json(path, {"ohlc_utc_ms": [], "note": "empty"})
        return 0
    arr = [[int(ix.value//10**6), float(o), float(h), float(l), float(c)]
           for ix,(o,h,l,c) in zip(df.index, df[['open','high','low','close']].to_numpy())]
    payload = {"ohlc_utc_ms": arr, "source_url": src, "retrieved_at_utc": utcnow().isoformat()}
    if symbol_used: payload["symbol_used"] = symbol_used
    save_json(path, payload)
    return len(arr)

def run_asset(asset):
    started = utcnow()
    outdir = os.path.join(OUTDIR, asset); ensure_dir(outdir)
    result = {"asset": asset, "ok": True, "errors": [], "retrieved_at_utc": started.isoformat(), "components": {}}

    try:
        if asset == "SOL":
            # ---- SPOT (CG) ----
            sp, sp_url = cg_simple_price(CG_IDS["SOL"])
            spot = float(sp["solana"]["usd"])
            save_json(os.path.join(outdir,"spot.json"),
                      {"price_usd":spot,"source_url":sp_url,"retrieved_at_utc":utcnow().isoformat(),"age_sec":0})
            result["components"]["spot"] = {"ok": True}

            # ---- 24h pontok → 5m ----
            now = int(utcnow().timestamp()); start = now - 24*3600
            mc, mc_url = cg_market_chart_range(CG_IDS["SOL"], start, now)
            df5 = prices_to_ohlc_5m(mc.get("prices", []))
            df1 = resample_ohlc(df5, "1H"); df4 = resample_ohlc(df1, "4H")

            result["components"]["k5m_rows"] = dump_ohlc_as_json(outdir,"klines_5m",df5,mc_url,"SOLUSD(cg)")
            result["components"]["k1h_rows"] = dump_ohlc_as_json(outdir,"klines_1h",df1,mc_url,"SOLUSD(cg)")
            result["components"]["k4h_rows"] = dump_ohlc_as_json(outdir,"klines_4h",df4,mc_url,"SOLUSD(cg)")

            # ---- 1D rekonstrukció az azonos napon belüli 5m-ből ----
            if not df5.empty:
                day = df5.index.normalize()[-1]; d = df5[df5.index.normalize()==day]
                k1d = {"open": float(d['open'].iloc[0]), "high": float(d['high'].max()),
                       "low": float(d['low'].min()), "close": float(d['close'].iloc[-1])}
            else:
                k1d = {}
            save_json(os.path.join(outdir,"klines_1d.json"),
                      {"k1d":k1d,"source":"cg_reconstruct","retrieved_at_utc":utcnow().isoformat(),"symbol_used":"SOLUSD(cg)"})
            result["components"]["k1d"] = k1d

            # ---- minta szignál ----
            sig = make_signal_sample(asset, spot, df1)
            save_json(os.path.join(outdir,"signal.json"), sig)
            save_png_chart(asset, df5, outdir)

        else:
            # ---- NSDQ100 / GOLD_CFD ----
            df5, src_series, series_sym = pd.DataFrame(), None, None
            td_err = None

            # Első próbálkozás: Twelve Data 5m
            try:
                df5, src_series, series_sym = try_td_series(TD_SYMBOLS[asset]["series"])
            except Exception as e:
                td_err = str(e)
                result["errors"].append(f"Twelve Data 5m hiba: {e}")

            # GOLD_CFD fallback: Stooq a2 xauusd, ha TD nem adott adatot
            if asset == "GOLD_CFD" and (df5 is None or df5.empty):
                try:
                    df5, src_series = stooq_intraday_csv("xauusd", interval="5")
                    series_sym = "XAUUSD(stooq)"
                except Exception as e:
                    result["errors"].append(f"Stooq a2 5m hiba: {e}")

            # ---- SPOT (TD), ha nincs → utolsó 5m close ----
            spot, src_spot, spot_sym = None, None, None
            if TD_KEY:
                s, u, symu = try_td_quote(TD_SYMBOLS[asset]["spot"])
                if s is not None:
                    spot, src_spot, spot_sym = s, u, symu
            if spot is None and not df5.empty:
                spot = float(df5['close'].iloc[-1])
                src_spot = (src_series or "unknown") + " (fallback last 5m close)"

            save_json(os.path.join(outdir,"spot.json"),
                      {"price_usd": spot, "source_url": src_spot,
                       "retrieved_at_utc": utcnow().isoformat(), "age_sec": 0, "ok": spot is not None})
            result["components"]["spot"] = {"ok": spot is not None}

            # ---- Aggregálások ----
            df1 = resample_ohlc(df5, "1H") if not df5.empty else pd.DataFrame(columns=["open","high","low","close"])
            df4 = resample_ohlc(df1, "4H") if not df1.empty else pd.DataFrame(columns=["open","high","low","close"])

            result["components"]["k5m_rows"] = dump_ohlc_as_json(outdir,"klines_5m",df5,src_series or "unknown",series_sym)
            result["components"]["k1h_rows"] = dump_ohlc_as_json(outdir,"klines_1h",df1,src_series or "unknown",series_sym)
            result["components"]["k4h_rows"] = dump_ohlc_as_json(outdir,"klines_4h",df4,src_series or "unknown",series_sym)

            if not df5.empty:
                day = df5.index.normalize()[-1]; d = df5[df5.index.normalize()==day]
                k1d = {"open": float(d['open'].iloc[0]), "high": float(d['high'].max()),
                       "low": float(d['low'].min()),  "close": float(d['close'].iloc[-1])}
            else:
                k1d = {}
            save_json(os.path.join(outdir,"klines_1d.json"),
                      {"k1d":k1d,"source":"series_reconstruct","retrieved_at_utc":utcnow().isoformat(),
                       "symbol_used": series_sym})
            result["components"]["k1d"] = k1d

            sig = make_signal_sample(asset, spot, df1)
            save_json(os.path.join(outdir,"signal.json"), sig)
            if not df5.empty: save_png_chart(asset, df5, outdir)

    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        result["trace"] = traceback.format_exc()
    finally:
        # mini index minden assethez
        try:
            errs = ""
            if result.get("errors"):
                errs = "<p><b>Errors:</b><br>" + "<br>".join([str(x) for x in result["errors"]]) + "</p>"
            html = f"""<html><head><meta charset="utf-8"><title>{asset} feed</title></head>
<body>
<h1>{asset}</h1>
<p>ok: {result.get("ok", True)}</p>
{errs}
<ul>
  <li><a href="spot.json">spot.json</a></li>
  <li><a href="klines_5m.json">klines_5m.json</a></li>
  <li><a href="klines_1h.json">klines_1h.json</a></li>
  <li><a href="klines_4h.json">klines_4h.json</a></li>
  <li><a href="klines_1d.json">klines_1d.json</a></li>
  <li><a href="signal.json">signal.json</a></li>
  <li><a href="chart_1d.png">chart_1d.png</a></li>
</ul>
</body></html>"""
            with open(os.path.join(outdir,"index.html"),"w",encoding="utf-8") as f: f.write(html)
        except Exception:
            pass
    return result

def main():
    status = {"generated_at_utc": utcnow().isoformat(), "assets": {}}
    for a in ASSETS:
        status["assets"][a] = run_asset(a)
    # gyökér index
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
                  {"ok": False, "error": str(e), "note": "Top-level exception caught. See public/run_log.txt"})
