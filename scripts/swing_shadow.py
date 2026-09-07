#!/usr/bin/env python3
"""H2 swing arnyek-uzem (SHADOW): Donchian(20) + Chandelier(k=3) a 4h gyertyakon.

NEM ELO KERESKEDES. A fagyott intraday rendszertol fuggetlen. Hiba eseten is
exit 0 (a pipeline-t soha nem toriheti el); a hibak a public/swing_shadow/errors.log-ba kerulnek.
Elojegyzes: claude/hipotezis2_swing_elojegyzes_2026-08-25.md (v1.0).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.webhook_delivery import log_exception as _wh_exc, log_response as _wh_ok

LOGGER = logging.getLogger(__name__)

ASSETS = ["GOLD_CFD", "XAGUSD"]
N_CHANNEL = 20
K_ATR = 3.0
ATR_PERIOD = 14
MAX_BARS_HELD = 90
FLOOR_PCT = {"GOLD_CFD": 0.015, "XAGUSD": 0.025}
ROUND_TRIP_PCT = {"GOLD_CFD": 0.0006, "XAGUSD": 0.0012}
OVERNIGHT_PCT_PER_DAY = 0.0002


def public_dir() -> Path:
    return Path(os.getenv("NOTIFY_PUBLIC_DIR", "public"))


def shadow_dir() -> Path:
    return public_dir() / "swing_shadow"


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def load_bars(asset: str) -> List[Dict[str, Any]]:
    raw = _load_json(public_dir() / asset / "klines_4h.json")
    out: List[Dict[str, Any]] = []
    for v in raw.get("values") or []:
        try:
            out.append(
                {
                    "dt": str(v["datetime"]),
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda b: b["dt"])
    return out


def compute_atr(bars: List[Dict[str, Any]], period: int = ATR_PERIOD) -> List[Optional[float]]:
    """Wilder-fele ATR; az elso `period` indexre None."""
    n = len(bars)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    trs: List[float] = [0.0]
    for i in range(1, n):
        pc = bars[i - 1]["close"]
        b = bars[i]
        trs.append(max(b["high"] - b["low"], abs(b["high"] - pc), abs(b["low"] - pc)))
    out[period] = sum(trs[1 : period + 1]) / period
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + trs[i]) / period  # type: ignore[operator]
    return out


def detect_signal(closes: List[float], i: int, n_channel: int = N_CHANNEL) -> int:
    """+1 long / -1 short / 0. Csak az i ELOTTI n_channel zarast nezi (nincs elorelatas)."""
    if i < n_channel:
        return 0
    window = closes[i - n_channel : i]
    if closes[i] > max(window):
        return 1
    if closes[i] < min(window):
        return -1
    return 0


def _days_between(dt_a: str, dt_b: str) -> int:
    try:
        a = datetime.fromisoformat(dt_a)
        b = datetime.fromisoformat(dt_b)
        return max((b - a).days, 0)
    except ValueError:
        return 0


def net_r(side: int, entry: float, exit_price: float, risk: float, asset: str, days: int) -> float:
    gross = side * (exit_price - entry) / risk
    cost = (ROUND_TRIP_PCT[asset] * entry + OVERNIGHT_PCT_PER_DAY * entry * days) / risk
    return gross - cost


def process_asset(asset: str, bars: List[Dict[str, Any]], st: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Determinisztikus, allapotvezerelt feldolgozas. Csak LEZART gyertyak: az utolso
    (esetleg meg nyitott) gyertyat kihagyja. Egy gyertya pontosan egyszer dolgozodik fel."""
    events: List[Dict[str, Any]] = []
    if len(bars) < 2:
        return st, events
    closes = [b["close"] for b in bars]
    atr = compute_atr(bars)
    last_done = str(st.get("last_bar") or "")
    pos = st.get("position")
    for i in range(len(bars) - 1):  # bars[-1] kihagyva: meg nyithat
        b = bars[i]
        if b["dt"] <= last_done:
            continue
        if pos:
            pos["bars_held"] = int(pos["bars_held"]) + 1
            side = int(pos["side"])
            c = b["close"]
            pos["best_close"] = max(float(pos["best_close"]), c) if side == 1 else min(float(pos["best_close"]), c)
            if atr[i] is not None:
                trail = float(pos["best_close"]) - side * K_ATR * float(atr[i])
                pos["stop"] = max(float(pos["stop"]), trail) if side == 1 else min(float(pos["stop"]), trail)
            reason = None
            if side == 1 and c < float(pos["stop"]):
                reason = "trail_stop"
            elif side == -1 and c > float(pos["stop"]):
                reason = "trail_stop"
            elif int(pos["bars_held"]) >= MAX_BARS_HELD:
                reason = "time_limit"
            if reason:
                days = _days_between(str(pos["entry_dt"]), b["dt"])
                r = net_r(side, float(pos["entry"]), c, float(pos["risk"]), asset, days)
                events.append(
                    {
                        "event": "exit",
                        "asset": asset,
                        "side": "long" if side == 1 else "short",
                        "bar_dt": b["dt"],
                        "price": c,
                        "entry": float(pos["entry"]),
                        "risk": float(pos["risk"]),
                        "stop": float(pos["stop"]),
                        "net_r": round(r, 4),
                        "days_held": days,
                        "reason": reason,
                    }
                )
                pos = None
        else:
            sig = detect_signal(closes, i)
            if sig and atr[i] is not None:
                entry = closes[i]
                risk = max(K_ATR * float(atr[i]), FLOOR_PCT[asset] * entry)
                pos = {
                    "side": sig,
                    "entry": entry,
                    "entry_dt": b["dt"],
                    "risk": risk,
                    "stop": entry - sig * risk,
                    "best_close": entry,
                    "bars_held": 0,
                }
                events.append(
                    {
                        "event": "entry",
                        "asset": asset,
                        "side": "long" if sig == 1 else "short",
                        "bar_dt": b["dt"],
                        "price": entry,
                        "entry": entry,
                        "risk": round(risk, 6),
                        "stop": round(entry - sig * risk, 6),
                        "net_r": "",
                        "days_held": 0,
                        "reason": "donchian_breakout",
                    }
                )
        last_done = b["dt"]
    st = dict(st)
    st["last_bar"] = last_done
    st["position"] = pos
    return st, events


JOURNAL_FIELDS = ["ts_utc", "event", "asset", "side", "bar_dt", "price", "entry", "risk", "stop", "net_r", "days_held", "reason"]


def append_journal(path: Path, events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        for e in events:
            row = {"ts_utc": now}
            row.update({k: e.get(k, "") for k in JOURNAL_FIELDS if k != "ts_utc"})
            w.writerow(row)


def _urls() -> List[str]:
    raw = os.getenv("DISCORD_WEBHOOK_URL_ACTIONABLE") or os.getenv("DISCORD_WEBHOOK_URL", "")
    return [u.strip() for u in raw.replace("\n", ",").split(",") if u.strip()]


def send_cards(events: List[Dict[str, Any]]) -> None:
    if not events or requests is None:
        return
    for e in events:
        if e["event"] == "entry":
            desc = (
                f"**{e['asset']}** {e['side'].upper()} @ {e['price']}\n"
                f"Stop: {e['stop']} (kockazat: {e['risk']})\n"
                f"Szabaly: Donchian(20)/Chandelier(3xATR14), 4h zaras-alapu koveto stop.\n"
                f"NEM ELO — kez ne mozduljon, ez meres."
            )
            title = "🟡 SHADOW swing ENTRY — NEM ÉLŐ"
        else:
            desc = (
                f"**{e['asset']}** {e['side'].upper()} lezarva @ {e['price']}\n"
                f"Eredmeny: {e['net_r']}R (netto, modellkoltseggel) | tartas: {e['days_held']} nap | ok: {e['reason']}\n"
                f"NEM ELO — arnyek-meres."
            )
            title = "🟡 SHADOW swing EXIT — NEM ÉLŐ"
        embed = {"title": title, "description": desc, "color": 0xF1C40F}
        for u in _urls():
            try:
                _wh_ok(LOGGER, "swing_shadow", "actionable", requests.post(u, json={"embeds": [embed]}, timeout=8))
            except Exception as exc:  # pragma: no cover
                _wh_exc(LOGGER, "swing_shadow", "actionable", exc)


def main() -> int:
    try:
        sdir = shadow_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        state_path = sdir / "state.json"
        state = _load_json(state_path)
        all_events: List[Dict[str, Any]] = []
        for asset in ASSETS:
            bars = load_bars(asset)
            st_a, events = process_asset(asset, bars, state.get(asset) or {})
            state[asset] = st_a
            all_events.extend(events)
        append_journal(sdir / "journal.csv", all_events)
        send_cards(all_events)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception:
        try:
            sdir = shadow_dir()
            sdir.mkdir(parents=True, exist_ok=True)
            with (sdir / "errors.log").open("a", encoding="utf-8") as fh:
                fh.write(datetime.now(timezone.utc).isoformat() + "\n" + traceback.format_exc() + "\n")
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
