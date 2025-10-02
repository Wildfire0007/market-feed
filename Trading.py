# Trading.py – Feed-generáló script, amely lekérdezi a spot-árakat és gyertyákat, 
# majd JSON fájlokat ír ki a public/<ASSET>/ könyvtárba, valamint összefoglaló all_<ASSET>.json fájlokat.
import os, json, time, pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
import requests

# --------- Beállítások ---------
OUT_DIR = pathlib.Path("public")   # Kimeneti mappa a GitHub Pages-hez
TD_API = os.environ.get("TWELVEDATA_API_KEY", "").strip()  # TwelveData API kulcs (Actions titok)

# Eszközök konfigurációja
ASSETS = {
    "SOL": {
        "source": "binance",
        "binance_symbol": "SOLUSDT",
        "note": "Spot ár Binance-ről (SOL/USDT).",
    },
    "NSDQ100": {
        "source": "twelvedata",
        "td_symbol": "NDX",     # Nasdaq-100 index symbol a TwelveData-on
        "note": "TwelveData (NDX).",
    },
    "GOLD_CFD": {
        "source": "twelvedata",
        "td_symbol": "XAU/USD", # Arany CFD (XAU/USD) TwelveData-nál
        "note": "TwelveData (XAU/USD).",
    },
}

# HTTP idők és ismétlések
TIMEOUT = 15
RETRIES = 3
RETRY_SLEEP = 1.2

# --------- Segédfüggvények ---------
def _http_get_json(url: str, timeout: int = TIMEOUT) -> Dict[str, Any]:
    """HTTP GET, JSON-válasz visszaadása (hiba esetén kivétel)."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _retry(fn, *args, **kwargs):
    """Hibatűrő hívás: több próbálkozás hibára futás esetén."""
    last_err = None
    for _ in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP)
    if last_err:
        raise last_err

def _safe_float(x: Any):
    """Biztonságos float konverzió None esetén."""
    try:
        return float(x)
    except Exception:
        return None

def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    """JSON fájl mentése a megadott path-ra (könyvtár előkészítésével)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------- Árlekérők ---------
def fetch_spot_binance(symbol: str) -> Tuple[float,str]:
    """Lekéri az aktuális spot árat a Binance /ticker/price API-val."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    j = _retry(_http_get_json, url)
    price = _safe_float(j.get("price"))
    if price is None:
        raise RuntimeError(f"Binance nem adott árfolyamot: {j}")
    return price, url

def fetch_spot_twelvedata(symbol: str) -> Tuple[float,str]:
    """Lekéri az aktuális spot árat a TwelveData /price végpontjával (vagy váltakozva a /quote-tal)."""
    if not TD_API:
        raise RuntimeError("Nincs TWELVEDATA_API_KEY beállítva")
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TD_API}"
    j = _retry(_http_get_json, url)
    price = _safe_float(j.get("price"))
    if price is not None:
        return price, url
    # Ha a /price nem adott értéket, próbáljuk a /quote-ot
    url2 = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={TD_API}"
    j = _retry(_http_get_json, url2)
    price = _safe_float(j.get("close") or j.get("last") or j.get("open"))
    if price is None:
        raise RuntimeError(f"TwelveData árlekérdezés sikertelen: {j}")
    return price, url2

# --------- Gyertyaidősor lekérők ---------
def fetch_klines_binance(symbol: str, interval: str, limit: int = 500) -> Tuple[list,str]:
    """Binance gyertyák lekérése adott intervallummal (5m, 1h, 4h)."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = _retry(_http_get_json, url)
    arr = []
    for entry in data:
        # Binance kline formátum: [open_time, open, high, low, close, ...]
        ts = int(entry[0])
        o = _safe_float(entry[1]); h = _safe_float(entry[2])
        l = _safe_float(entry[3]); c = _safe_float(entry[4])
        if None in (o, h, l, c):
            continue
        arr.append([ts, o, h, l, c])
    return arr, url

def fetch_klines_twelvedata(symbol: str, interval: str, outputsize: int = 500) -> Tuple[list,str]:
    """TwelveData OHLC idősor lekérése (interval: pl. '5min','1h','4h')."""
    if not TD_API:
        raise RuntimeError("Nincs TWELVEDATA_API_KEY beállítva")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TD_API}"
    data = _retry(_http_get_json, url)
    arr = []
    vals = data.get("values") or []
    records = []
    for v in vals:
        dt = v.get("datetime", "")
        if not dt: 
            continue
        # Eltávolítjuk a 'T' és 'Z' karaktereket, ha ott vannak
        dt_str = dt.replace('T',' ').replace('Z',' ').strip()
        try:
            dt_obj = datetime.fromisoformat(dt_str)
        except Exception:
            # Ha isoformat nem működik (pl. hiányzó másodperc), használjunk strptime
            try:
                dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        ts = int(dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        o = _safe_float(v.get("open")); h = _safe_float(v.get("high"))
        l = _safe_float(v.get("low")); c = _safe_float(v.get("close"))
        if None in (o, h, l, c):
            continue
        records.append((ts, o, h, l, c))
    # Rendezés idő szerint (felmenő)
    records.sort(key=lambda x: x[0])
    for rec in records:
        arr.append(list(rec))
    return arr, url

# --------- Fő feldolgozás eszközönként ---------
def generate_for_asset(asset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adott eszköz lekérdezése és fájlok írása. 
    
    A public/<asset>/Spot és klines fájlok írása mellett all_<asset>.json 
    generálása is történik a repó gyökerében.
    """
    result = {"asset": asset, "ok": True, "error": None}
    try:
        # Árlekérés és gyertyák beszerzése az adott forrásból
        if cfg.get("source") == "binance":
            sym = cfg["binance_symbol"]
            spot_price, spot_url = fetch_spot_binance(sym)
            k5, url5 = fetch_klines_binance(sym, "5m",  limit=500)
            k1, url1 = fetch_klines_binance(sym, "1h",  limit=500)
            k4, url4 = fetch_klines_binance(sym, "4h",  limit=500)
        elif cfg.get("source") == "twelvedata":
            sym = cfg["td_symbol"]
            spot_price, spot_url = fetch_spot_twelvedata(sym)
            k5, url5 = fetch_klines_twelvedata(sym, "5min", outputsize=500)
            k1, url1 = fetch_klines_twelvedata(sym, "1h",   outputsize=500)
            k4, url4 = fetch_klines_twelvedata(sym, "4h",   outputsize=500)
        else:
            raise ValueError(f"Ismeretlen forrás: {cfg.get('source')}")

        now_iso = datetime.now(timezone.utc).isoformat()

        # Spot adat objektum
        spot_obj = {
            "asset": asset,
            "ok": True,
            "price_usd": spot_price,
            "retrieved_at_utc": now_iso,
            "source": cfg.get("source"),
            "source_url": spot_url
        }
        # Gyertya-idősor objektumok
        k5_obj = {
            "asset": asset,
            "interval": "5min",
            "ohlc_utc_ms": k5,
            "retrieved_at_utc": now_iso,
            "source_url": url5
        }
        k1_obj = {
            "asset": asset,
            "interval": "1h",
            "ohlc_utc_ms": k1,
            "retrieved_at_utc": now_iso,
            "source_url": url1
        }
        k4_obj = {
            "asset": asset,
            "interval": "4h",
            "ohlc_utc_ms": k4,
            "retrieved_at_utc": now_iso,
            "source_url": url4
        }

        # Fájlok írása: public/<asset>/spot.json és klines fájlok
        _write_json(OUT_DIR/asset/"spot.json",       spot_obj)
        print(f"[OK] Írás: {OUT_DIR/asset/'spot.json'}")
        _write_json(OUT_DIR/asset/"klines_5m.json",  k5_obj)
        print(f"[OK] Írás: {OUT_DIR/asset/'klines_5m.json'}")
        _write_json(OUT_DIR/asset/"klines_1h.json",  k1_obj)
        print(f"[OK] Írás: {OUT_DIR/asset/'klines_1h.json'}")
        _write_json(OUT_DIR/asset/"klines_4h.json",  k4_obj)
        print(f"[OK] Írás: {OUT_DIR/asset/'klines_4h.json'}")

        # Összefoglaló all_<asset>.json írása a repó gyökerébe
        all_data = {
            "spot": spot_obj,
            "k5m":  k5_obj,
            "k1h":  k1_obj,
            "k4h":  k4_obj
        }
        all_path = pathlib.Path(f"all_{asset}.json")
        _write_json(all_path, all_data)
        print(f"[OK] Írás: {all_path}")

    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)
        # Hiba esetén próbáljunk meg hibás státuszt is írni
        spot_obj = {"asset": asset, "ok": False, "error": result["error"]}
        try:
            _write_json(OUT_DIR/asset/"spot.json", spot_obj)
            _write_json(pathlib.Path(f"all_{asset}.json"), {"asset": asset, "ok": False, "error": result["error"]})
        except Exception:
            pass

    return result

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset, cfg in ASSETS.items():
        res = generate_for_asset(asset, cfg)
        if res.get("ok"):
            print(f"{asset}: feed elkészült")
        else:
            print(f"{asset}: Hiba történt – {res.get('error')}")

if __name__ == "__main__":
    main()
