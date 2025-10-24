#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""intraday_report.py — TD-only riport generátor az új elemzési stratégiához."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis

ASSETS: Iterable[str] = getattr(
    analysis,
    "ASSETS",
    ("EURUSD", "GOLD_CFD", "BTCUSD", "USOIL", "NVDA", "SRTY"),
)
PUBLIC_DIR = os.getenv("PUBLIC_DIR", getattr(analysis, "PUBLIC_DIR", "public"))
REPORT_DIR = os.getenv("REPORT_DIR", "reports")

MISSING_LABELS = {
    "session": "Session",
    "regime": "Regime (EMA21 slope)",
    "bias": "Bias",
    "bos5m": "5m BOS",
    "liquidity(fib|sweep|ema21|retest)": "Liquidity",
    "liquidity(fib_zone|sweep)": "Liquidity",
    "atr": "ATR",
    "tp_min_profit": "TP min. profit",
    "tp1_net>=+1.0%": "TP1 nettó ≥ +1.0%",
    "rr_math>=2.0": "RR≥2.0",
    "rr_math>=1.6": "RR≥1.6",
    "min_stoploss": "Minimum stoploss",
    "momentum(ema9x21)": "Momentum EMA9×21",
    "bos5m|struct_break": "BOS/Structure",
}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def fmt_missing(missing: Iterable[str]) -> str:
    pretty: List[str] = []
    for key in missing:
        label = MISSING_LABELS.get(key)
        if not label and key.startswith("rr_math>="):
            label = "RR≥" + key.split(">=")[-1]
        pretty.append(label or key)
    return ", ".join(dict.fromkeys(pretty))

def gather_signals() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for asset in ASSETS:
        try:
            result = analysis.analyze(str(asset))
        except Exception as exc:  # pragma: no cover — hibatűrő fallback
            result = {
                "asset": str(asset),
                "ok": False,
                "signal": "no entry",
                "probability": 0,
                "reasons": [f"Elemzési hiba: {exc}"],
                "spot": {"price": None, "utc": "-"},
                "gates": {"mode": "-", "missing": []},
            }
        results.append(result)
    return results


def format_signal_md(sig: Dict[str, Any]) -> str:
    asset = sig.get("asset", "?")
    spot = sig.get("spot") or {}
    price = spot.get("price") or spot.get("price_usd")
    spot_utc = spot.get("utc") or spot.get("timestamp") or "-"
    probability = int(sig.get("probability") or 0)
    decision_raw = (sig.get("signal") or "no entry").lower()
    decision = "BUY" if decision_raw == "buy" else "SELL" if decision_raw == "sell" else "no entry"
    gates = sig.get("gates") or {}
    mode = gates.get("mode") or "-"
    missing = list(gates.get("missing") or [])
    missing_line = fmt_missing(missing)
    reasons = sig.get("reasons") or []

    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")
    rr = sig.get("rr")
    lev = sig.get("leverage")

    header = (
        f"### {asset}\n\n"
        f"Spot (USD): **{fmt_num(price)}** • UTC: `{spot_utc}`\n"
        f"Valószínűség: **P = {probability}%**\n"
        "Forrás: Twelve Data (lokális JSON)\n\n"
    )

    if decision == "no entry":
        reason_text = "; ".join(reasons) if reasons else "nincs szignál"
        body_lines = [f"**Állapot:** no entry — {reason_text}"]
        if missing_line:
            body_lines.append(f"Hiányzó kapuk: {missing_line}")
        return header + "\n".join(body_lines) + "\n\n"

    entry_line = (
        f"[{decision} @ {fmt_num(entry)}; SL: {fmt_num(sl)}; TP1: {fmt_num(tp1)}; TP2: {fmt_num(tp2)}; "
        f"mód: {mode}; Ajánlott tőkeáttétel: {fmt_num(lev, 1)}×; RR≈{fmt_num(rr, 2)}]"
    )
    lines = [entry_line]
    if missing_line:
        lines.append(f"Hiányzó kapuk / figyelendő: {missing_line}")
    if reasons:
        lines.append("Indoklás:")
        lines.extend(f"- {r}" for r in reasons)
    return header + "\n".join(lines) + "\n\n"


def write_markdown(signals: List[Dict[str, Any]]) -> None:
    ensure_dir(REPORT_DIR)
    path = os.path.join(REPORT_DIR, "analysis_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Intraday riport (Twelve Data-only)\n\n")
        handle.write(f"Generálva (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n\n")
        for sig in signals:
            handle.write(format_signal_md(sig))
        handle.write("#### Elemzés & döntés checklist\n")
        handle.write("- 4H→1H trend bias + EMA21 rezsim\n")
        handle.write("- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet\n")
        handle.write("- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)\n")
        handle.write("- ATR filter + TP minimum (költség + ATR alapú) + nettó TP1 ≥ +1.0%\n")
        handle.write("- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%; minimum stoploss ≥1%\n")
        handle.write("- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak engedélyezett eszközöknél)\n")

def write_summary_csv(signals: List[Dict[str, Any]]) -> None:
    ensure_dir(REPORT_DIR)
    path = os.path.join(REPORT_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Asset",
            "Signal",
            "P%",
            "Mode",
            "Entry",
            "SL",
            "TP1",
            "TP2",
            "RR",
            "Leverage",
            "Spot",
            "UTC",
            "Missing",
            "Reasons",
        ])
        for sig in signals:
            spot = sig.get("spot") or {}
            writer.writerow([
                sig.get("asset", ""),
                (sig.get("signal") or "no entry").upper(),
                int(sig.get("probability") or 0),
                (sig.get("gates") or {}).get("mode") or "",
                fmt_num(sig.get("entry")),
                fmt_num(sig.get("sl")),
                fmt_num(sig.get("tp1")),
                fmt_num(sig.get("tp2")),
                fmt_num(sig.get("rr"), 2),
                fmt_num(sig.get("leverage"), 1),
                fmt_num(spot.get("price") or spot.get("price_usd")),
                spot.get("utc") or spot.get("timestamp") or "",
                fmt_missing((sig.get("gates") or {}).get("missing") or []),
                " | ".join(sig.get("reasons") or []),
            ])

def main() -> None:
    signals = gather_signals()
    write_markdown(signals)
    write_summary_csv(signals)
    print(f"Kész: {REPORT_DIR}/analysis_report.md, summary.csv")

if __name__ == "__main__":
    main()
