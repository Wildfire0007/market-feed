import json
import math
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timezone
import mplfinance as mpf

# ---- Paraméterek -----------------------------------------------------------
COIN_ID = "solana"
SYMBOL  = "SOLUSDT"

# ---- Segédfüggvények -------------------------------------------------------
def iso(ts_ms_or_s: float) -> str:
    """Unix timestamp (s vagy ms) -> ISO string (UTC, tz-aware)."""
    ts_sec = ts_ms_or_s/1000.0 if ts_ms_or_s > 10**12 else float(ts_ms_or_s)
    return dt.datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fetch_json(url: str, timeout: int = 20):
    """Egyszerű JSON lekérés, hibatűréssel."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---- 1) CoinGecko spot + timestamp -----------------------------------------
spot_url = (
    f"https://api.coingecko.com/api/v3/simple/price"
    f"?ids={COIN_ID}&vs_currencies=usd&include_last_updated_at=true"
)
spot_raw = fetch_json(spot_url)

if COIN_ID not in spot_raw or "usd" not in spot_raw[COIN_ID]:
    raise RuntimeError("Insufficient data: CoinGecko spot válasz hiányos.")

spot_price = float(spot_raw[COIN_ID]["usd"])
if "last_updated_at" not in spot_raw[COIN_ID]:
    raise RuntimeError("Insufficient data: CoinGecko timestamp hiányzik.")

spot_ts_iso = iso(spot_raw[COIN_ID]["last_updated_at"])
spot_out = {
    "price_usd": spot_price,
    "last_updated_at": spot_ts_iso,
    "source": spot_url
}
save_json(spot_out, "spot.json")

# ---- 2) Binance OHLC (5m/1h/4h) --------------------------------------------
def binance_klines(symbol: str, interval: str, limit: int = 900) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = fetch_json(url)
    cols = [
        "open_time","o","h","l","c","v",
        "close_time","qv","n_trades","taker_b","taker_q","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    # típusok
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    # tz-aware idő
    df["time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # emberi olvasásra (string, UTC suffix)
    df["time"] = df["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df[["time", "o", "h", "l", "c", "v"]]

# ~3 nap 5 perces gyertyák (3*24*60/5 = 864)
k5m = binance_klines(SYMBOL, "5m", 864)
k1h = binance_klines(SYMBOL, "1h", 200)
k4h = binance_klines(SYMBOL, "4h", 200)

k5m.to_json("klines_5m.json", orient="records", indent=2)
k1h.to_json("klines_1h.json", orient="records", indent=2)
k4h.to_json("klines_4h.json", orient="records", indent=2)

# ---- 3) 1D chart PNG (záróár a Binance 1D-ből) -----------------------------
k1d_url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1d&limit=120"
k1d_raw = fetch_json(k1d_url)
df1d = pd.DataFrame(
    k1d_raw,
    columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"]
)
for col in ["o","h","l","c"]:
    df1d[col] = df1d[col].astype(float)

# tz-aware -> dátum (UTC)
df1d["date"] = pd.to_datetime(df1d["t"], unit="ms", utc=True).dt.date

plt.figure(figsize=(10, 4))
plt.plot(df1d["date"], df1d["c"])
plt.title("SOLUSDT – záróár (1D) – Binance OHLC")
plt.xlabel("Dátum (UTC)")
plt.ylabel("Ár (USDT)")
plt.tight_layout()
plt.savefig("chart_1d.png", dpi=150)

# ---- 4) Mintaszabály: ATR14 (1h) és LONG szignál generálása ----------------
def atr14_from_ohlc(df: pd.DataFrame) -> float:
    """
    ATR14 számítás 1h gyertyákra.
    df: oszlopok: time, o, h, l, c, v   (float OHLC)
    vissza: utolsó ATR14 (float) vagy NaN
    """
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

atr = atr14_from_ohlc(k1h.rename(columns={"o":"o","h":"h","l":"l","c":"c"}))

# Egyszerű minta: LONG, SL=entry-2*ATR, TP1=entry+1.5*ATR, TP2=entry+3*ATR
# (Valós szabályaid szerint cserélhető!)
if not math.isnan(atr) and atr > 0:
    entry = spot_price             # USD ~ USDT közelítés
    sl    = entry - 2.0 * atr
    tp1   = entry + 1.5 * atr
    tp2   = entry + 3.0 * atr
    lev   = 3                      # példa tőkeáttét
    # P/L $100 poziméretre (egyszerűsített): darabszám * árkülönbség
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
        "notes": "ATR14(1h) alapú MINTA; éles szabályaid helyére illeszd a saját logikád."
    }
else:
    signal = {
        "status": "Insufficient data (ATR14)",
        "reason": "Nem számítható megbízható ATR14 az 1h adatsoron."
    }

def to_ohlc_df(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["time"], utc=True)
    out = out[["Date","o","h","l","c","v"]].rename(
        columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}
    ).set_index("Date")
    return out

# 5m gyertyákból PNG:
ohlc_5m = to_ohlc_df(k5m)
mpf.plot(ohlc_5m.tail(200), type="candle", volume=True, style="yahoo", savefig="candles_5m.png")

# 1h gyertyák:
ohlc_1h = to_ohlc_df(k1h)
mpf.plot(ohlc_1h.tail(200), type="candle", volume=True, style="yahoo", savefig="candles_1h.png")

# 4h gyertyák:
ohlc_4h = to_ohlc_df(k4h)
mpf.plot(ohlc_4h.tail(200), type="candle", volume=True, style="yahoo", savefig="candles_4h.png")

save_json(signal, "signal.json")

print("OK: spot.json, klines_*.json, chart_1d.png, signal.json")