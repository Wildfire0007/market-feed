# -*- coding: utf-8 -*-
"""
Robusztus feed-generátor: CoinGecko spot + Binance OHLC + 1D PNG + ATR-minta
- Headless kompatibilis (matplotlib Agg backend)
- Retry/backoff HTTP hívások
- Minden kimenet a 'public/' mappába kerül
- Hiba esetén is készül 'public/status.json' és a script 0-val lép ki (hogy Pages kimenet legyen)

Kimenetek (public/):
- spot.json
- klines_5m.json
- klines_1h.json
- klines_4h.json
- chart_1d.png
- signal.json
- status.json  (összefoglaló: ok vagy hiba)
"""

import os, json, math, time, random
import requests
import pandas as pd
import datetime as dt
from datetime import timezone

# headless plot
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Beállítások -----------------------------------------------------------
COIN_ID = "solana"
SYMBOL  = "SOLUSDT"
OUTDIR  = "public"

os.makedirs(OUTDIR, exist_ok=True)

# ---- Hasznos függvények ----------------------------------------------------
def iso(ts_ms_or_s: float) -> str:
    ts_sec = ts_ms_or_s/1000.0 if ts_ms_or_s > 10**12 else float(ts_ms_or_s)
    return dt.datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def save_json(obj, name: str):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def fetch_json_with_retry(url: str, timeout: int = 20, retries: int = 4, base_delay: float = 1.0):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code >= 500 or r.status_code == 429:
                # túlterhelés/rate limit -> retry
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            delay = base_delay * (2 ** i) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"Fetch failed after {retries} retries for {url}: {last_err}")

def binance_klines(symbol: str, interval: str, limit: int = 900) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = fetch_json_with_retry(url)
    cols = [
        "open_time","o","h","l","c","v",
        "close_time","qv","n_trades","taker_b","taker_q","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    df["time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["time"] = df["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time", "o", "h", "l", "c", "v"]]

def atr14_from_ohlc(df: pd.DataFrame) -> float:
    d = df.copy()
    d["prev_c"] = d["c"].shift(1)
    tr = pd.concat(
        [
            (d["h"] - d["l"]).abs(),
            (d["h"] - d["prev_c"]).abs(),
            (d["l"] - d["prev_c"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    return float(atr14.iloc[-1])

status = {"ok": True, "errors": []}

try:
    # ---- 1) Spot (CoinGecko) ----------------------------------------------
    spot_url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={COIN_ID}&vs_currencies=usd&include_last_updated_at=true"
    )
    spot_raw = fetch_json_with_retry(spot_url)
    if COIN_ID not in spot_raw or "usd" not in spot_raw[COIN_ID]:
        raise RuntimeError("CoinGecko spot válasz hiányos.")
    spot_price = float(spot_raw[COIN_ID]["usd"])
    ts_field = spot_raw[COIN_ID].get("last_updated_at")
    if ts_field is None:
        raise RuntimeError("CoinGecko timestamp hiányzik.")
    spot_out = {
        "price_usd": spot_price,
        "last_updated_at": iso(ts_field),
        "source": spot_url
    }
    save_json(spot_out, "spot.json")

except Exception as e:
    status["ok"] = False
    status["errors"].append(f"spot: {e}")
    # Írj minimális spotot, hogy a fogyasztó ne dőljön el
    save_json({"status":"Insufficient data (spot)","error":str(e),"source":spot_url}, "spot.json")

try:
    # ---- 2) OHLC (Binance 5m/1h/4h) ---------------------------------------
    k5m = binance_klines(SYMBOL, "5m", 864)
    k1h = binance_klines(SYMBOL, "1h", 200)
    k4h = binance_klines(SYMBOL, "4h", 200)
    k5m.to_json(os.path.join(OUTDIR, "klines_5m.json"), orient="records", indent=2)
    k1h.to_json(os.path.join(OUTDIR, "klines_1h.json"), orient="records", indent=2)
    k4h.to_json(os.path.join(OUTDIR, "klines_4h.json"), orient="records", indent=2)

except Exception as e:
    status["ok"] = False
    status["errors"].append(f"klines: {e}")
    # Minimális hiba-jsonok
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_5m.json")
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_1h.json")
    save_json({"status":"Insufficient data (klines)","error":str(e)}, "klines_4h.json")

try:
    # ---- 3) 1D chart PNG (Binance 1d) -------------------------------------
    url_1d = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1d&limit=120"
    k1d_raw = fetch_json_with_retry(url_1d)
    df1d = pd.DataFrame(
        k1d_raw,
        columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]
    )
    for col in ["o","h","l","c"]:
        df1d[col] = df1d[col].astype(float)
    df1d["date"] = pd.to_datetime(df1d["t"], unit="ms", utc=True).dt.date

    plt.figure(figsize=(10,4))
    plt.plot(df1d["date"], df1d["c"])
    plt.title("SOLUSDT – záróár (1D) – Binance OHLC")
    plt.xlabel("Dátum (UTC)"); plt.ylabel("Ár (USDT)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "chart_1d.png"), dpi=150)
    plt.close()

except Exception as e:
    status["ok"] = False
    status["errors"].append(f"chart_1d: {e}")
    # Hiba esetén nem létfontosságú a PNG, a többi menjen tovább

try:
    # ---- 4) ATR14(1h) minta szignál ---------------------------------------
    # Ha volt klines hiba, itt fallback
    if os.path.exists(os.path.join(OUTDIR, "klines_1h.json")):
        k1h_df = pd.read_json(os.path.join(OUTDIR, "klines_1h.json"))
        if "status" in k1h_df.columns:  # hiba-json
            raise RuntimeError("Nincs 1h OHLC (hiba-json).")
        # konverzió floatokra (ha json->dtype keveredés lenne)
        for col in ["o","h","l","c"]:
            k1h_df[col] = k1h_df[col].astype(float)
        atr = atr14_from_ohlc(k1h_df)
    else:
        raise RuntimeError("klines_1h.json hiányzik.")

    # spot betöltés
    spot_obj = json.load(open(os.path.join(OUTDIR, "spot.json"), encoding="utf-8"))
    entry = float(spot_obj.get("price_usd", "nan"))

    if math.isnan(entry) or math.isnan(atr) or atr <= 0:
        raise RuntimeError("Nem számítható szignál (entry/ATR hibás).")

    sl    = entry - 2.0 * atr
    tp1   = entry + 1.5 * atr
    tp2   = entry + 3.0 * atr
    lev   = 3
    qty   = 100.0 / entry
    pl_tp1 = (tp1 - entry) * qty
    pl_tp2 = (tp2 - entry) * qty
    pl_sl  = (sl  - entry) * qty

    signal = {
        "side": "LONG",
        "entry": round(entry, 4),
        "SL": round(sl, 4),
        "TP1": round(tp1, 4),
        "TP2": round(tp2, 4),
        "leverage": lev,
        "P/L@$100": {
            "TP1": round(pl_tp1, 2),
            "TP2": round(pl_tp2, 2),
            "SL": round(pl_sl, 2)
        },
        "notes": "ATR14(1h) alapú MINTA; saját szabályaid helyére illeszd a saját logikád."
    }
    save_json(signal, "signal.json")

except Exception as e:
    status["ok"] = False
    status["errors"].append(f"signal: {e}")
    save_json({"status":"Insufficient data (signal)","error":str(e)}, "signal.json")

# ---- 5) Összefoglaló státusz -----------------------------------------------
if status["ok"]:
    status["message"] = "OK"
else:
    status["message"] = "Partial data; see 'errors'."
save_json(status, "status.json")

print("Done. Outputs in 'public/'.")