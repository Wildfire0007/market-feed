#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""intraday_report.py — TD-only riport generátor az új elemzési stratégiához."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis

ASSETS: Iterable[str] = getattr(
    analysis,
    "ASSETS",
    ("EURUSD", "GOLD_CFD", "BTCUSD", "USOIL", "NVDA", "XAGUSD"),
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


def spot_display_metadata(signal: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts presentation details for the spot price column.

    The Twelve Data spot feed occasionally lags for hours when markets are
    closed.  ``analysis.analyze`` relaxes freshness violations in those cases to
    avoid blocking the pipeline, but the human-facing reports should not surface
    misleading stale timestamps.  This helper inspects the diagnostics payload
    and hides the spot price/timestamp when a latency breach is detected.
    """

    spot = signal.get("spot") or {}
    diagnostics = signal.get("diagnostics") or {}
    tf_meta = diagnostics.get("timeframes") if isinstance(diagnostics, dict) else {}
    spot_meta = tf_meta.get("spot") if isinstance(tf_meta, dict) else {}

    price = spot.get("price") if spot.get("price") is not None else spot.get("price_usd")
    timestamp = spot.get("utc") or spot.get("timestamp") or ""
    retrieved = spot.get("retrieved_at_utc") or ""

    stale = False
    reason: Optional[str] = None

    if isinstance(spot_meta, dict):
        latency = spot_meta.get("latency_seconds")
        expected = spot_meta.get("expected_max_delay_seconds")
        original_issue = spot_meta.get("original_issue")
        freshness_violation = spot_meta.get("freshness_violation")

        if original_issue:
            reason = str(original_issue)
            stale = True
        if freshness_violation:
            stale = True
            if reason is None:
                reason = "Spot freshness violation"

        try:
            latency_value = float(latency) if latency is not None else None
            expected_value = float(expected) if expected not in (None, 0) else None
        except (TypeError, ValueError):
            latency_value = expected_value = None

        if latency_value is not None and expected_value is not None and latency_value > expected_value:
            stale = True
            if reason is None:
                latency_minutes = int(latency_value // 60)
                limit_minutes = int(expected_value // 60) if expected_value >= 60 else expected_value
                if latency_minutes and isinstance(limit_minutes, int):
                    reason = f"Spot latency {latency_minutes} min exceeds limit {limit_minutes} min"
                else:
                    reason = "Spot latency exceeds freshness limit"

    if stale:
        price = None
        timestamp = ""

    if not timestamp and retrieved:
        timestamp = str(retrieved)

    return {
        "price": price,
        "timestamp": timestamp,
        "stale": stale,
        "reason": reason,
    }


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
    spot_meta = spot_display_metadata(sig)
    price = spot_meta.get("price")
    spot_utc = spot_meta.get("timestamp") or "-"
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

    spot_line = f"Spot (USD): **{fmt_num(price)}** • UTC: `{spot_utc}`"
    if spot_meta.get("stale") and spot_meta.get("reason"):
        spot_line += f" _(stale: {spot_meta['reason']})_"
        
    header = (
        f"### {asset}\n\n"
        f"{spot_line}\n"
        f"Valószínűség: **P = {probability}%**\n"
        "Forrás: Twelve Data (lokális JSON)\n\n"
    )

    extra_reasons = list(reasons)
    if spot_meta.get("stale") and spot_meta.get("reason"):
        stale_reason = spot_meta["reason"]
        if stale_reason not in extra_reasons:
            extra_reasons.append(stale_reason)
    
    if decision == "no entry":
        reason_text = "; ".join(extra_reasons) if extra_reasons else "nincs szignál"
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
    if extra_reasons:
        lines.append("Indoklás:")
        lines.extend(f"- {r}" for r in extra_reasons)
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
            spot_meta = spot_display_metadata(sig)
            reasons = list(sig.get("reasons") or [])
            if spot_meta.get("stale") and spot_meta.get("reason"):
                reason = spot_meta["reason"]
                if reason not in reasons:
                    reasons.append(reason)
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
                fmt_num(spot_meta.get("price")),
                spot_meta.get("timestamp") or "",
                fmt_missing((sig.get("gates") or {}).get("missing") or []),
                " | ".join(reasons),
            ])

def main() -> None:
    signals = gather_signals()
    write_markdown(signals)
    write_summary_csv(signals)
    print(f"Kész: {REPORT_DIR}/analysis_report.md, summary.csv")

if __name__ == "__main__":
    main()
