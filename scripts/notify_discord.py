#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Enterprise Entry Only (v8)
Golyóálló, tiszta belépési jelzések pozíciómenedzsment blokkolás nélkül,
beépített SL/TP Auto-Correctorral és eToro kockázatkezeléssel.
"""
from __future__ import annotations
import importlib
import importlib.util
import json
import os
import sys
import fcntl
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None
BUDAPEST_TZ = ZoneInfo("Europe/Budapest")
from config import analysis_settings as settings
from scripts.reset_notify_state import build_default_state, _default_asset_state
import position_tracker
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
from reports import trade_journal as _trade_journal

DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
ENTRY_COOLDOWN_MINUTES = 30
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
def _webhook_urls(kind: str = "actionable") -> List[str]:
    env = "DISCORD_WEBHOOK_URL_ACTIONABLE" if kind == "actionable" else "DISCORD_WEBHOOK_URL_DIAGNOSTIC"
    raw = os.getenv(env) or os.getenv("DISCORD_WEBHOOK_URL", "")
    return [url.strip() for url in raw.replace("\\n", ",").replace("\n", ",").split(",") if url.strip()]
DISCORD_WEBHOOK_URLS = _webhook_urls("actionable")
DEFAULT_DISCORD_NOTIFY_ASSETS = {"GOLD_CFD", "XAGUSD", "USOIL"}
_DISCORD_NOTIFY_ASSETS_ENV = {p.strip().upper() for p in os.getenv("DISCORD_NOTIFY_ASSETS", "").split(",") if p.strip()}
DISCORD_NOTIFY_ASSETS = (
    _DISCORD_NOTIFY_ASSETS_ENV & DEFAULT_DISCORD_NOTIFY_ASSETS
    if _DISCORD_NOTIFY_ASSETS_ENV
    else set(DEFAULT_DISCORD_NOTIFY_ASSETS)
)

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else BASE_DIR / "public"
if not PUBLIC_DIR.exists() and (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"

NOTIFY_LOCK_PATH = PUBLIC_DIR / ".notify_discord.lock"
COLOR_GREEN, COLOR_RED, COLOR_YELLOW = 0x2ECC71, 0xE74C3C, 0xF1C40F
LIFECYCLE_INBOX_PATH = PUBLIC_DIR / "_position_lifecycle_inbox.jsonl"
LIFECYCLE_STATE_PATH = PUBLIC_DIR / "_position_lifecycle_state.json"

LAST_SENT_RETENTION_DAYS = 14
ASSETS = ["BTCUSD", "XAGUSD", "GOLD_CFD", "USOIL", "NVDA", "EURUSD"]
DEFAULT_ASSET_STATE = {"last_spot_price": None, "last_spot_utc": None}
LOGGER = logging.getLogger(__name__)
NOTIFY_ATTEMPTS = 0
NOTIFY_SUCCESSES = 0
NOTIFY_FAILURES = 0
_WEBHOOK_COOLDOWN_UNTIL: Dict[str, float] = {}
ENTRY_GATE_STATS_PATH = PUBLIC_DIR / "monitoring" / "entry_gate_stats.json"
ENTRY_GATE_LOG_DIR = PUBLIC_DIR / "debug" / "entry_gates"



@dataclass
class EntryAuditRecord:
    asset: str
    intent: str
    decision: str
    setup_grade: str
    stable: bool
    send_kind: str
    should_notify: bool
    manual_state: Dict[str, Any]
    manual_tracking_enabled: bool
    can_write_positions: bool
    state_loaded: bool
    positions_file: str
    gates_missing: List[str]
    notify_reason: Optional[str] = None
    display_stable: bool = False
    dispatch_attempted: Optional[bool] = None
    dispatch_success: Optional[bool] = None
    commit_status: Optional[str] = None

    def _commit_reason(self) -> Optional[str]:
        return self.commit_status

    def log_commit_decision(self) -> None:
        try:
            position_tracker.log_audit_event("entry commit decision", event="OPEN_COMMIT_DECISION", asset=self.asset, commit_reason=self._commit_reason(), commit_result=getattr(self, "commit_result", None))
        except Exception:
            pass
            
            
def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _sanitize_last_sent(asset: str, record: Dict[str, Any], archived: List[Dict[str, Any]], *, now: datetime) -> None:
    record["last_sent_known"] = True
    stamp = _parse_utc(record.get("last_sent"))
    if not stamp:
        if record.get("last_sent"):
            archived.append({"asset": asset, "last_sent": record.get("last_sent"), "reason": "invalid-format"})
        record["last_sent"] = None
        return
    if stamp > now:
        archived.append({"asset": asset, "last_sent": record.get("last_sent"), "reason": "future"})
        record["last_sent"] = None
        return
    if stamp < now - timedelta(days=LAST_SENT_RETENTION_DAYS):
        archived.append({"asset": asset, "last_sent": record.get("last_sent"), "reason": "expired"})
        record["last_sent"] = None


def missing_from_sig(sig: Dict[str, Any]) -> List[str]:
    missing = ((sig.get("gates") or {}).get("missing") or []) if isinstance(sig, dict) else []
    translations = {"precision warning": "Precision figyelmeztetés"}
    return [translations.get(str(item).strip().lower(), str(item)) for item in missing]



def update_asset_send_state(
    record: Dict[str, Any],
    *,
    decision: str,
    now: datetime,
    cooldown_minutes: float,
    mode: Optional[str],
) -> Dict[str, Any]:
    updated = dict(record)
    updated["last_sent"] = to_utc_iso(now)
    updated["last_sent_decision"] = decision
    updated["last_sent_mode"] = mode
    updated["last_sent_known"] = True
    updated["cooldown_until"] = (
        to_utc_iso(now + timedelta(minutes=cooldown_minutes)) if cooldown_minutes and cooldown_minutes > 0 else None
    )
    return updated


def build_pipeline_diag_embed(payload: Dict[str, Any], *, now: datetime) -> Dict[str, Any]:
    trading = payload.get("trading") or {}
    analysis_payload = payload.get("analysis") or {}
    hashes = ((payload.get("artifacts") or {}).get("hashes") or {})
    description = (
        f"Trading→analysis: {trading.get('duration_seconds', 'N/A')}s / "
        f"{analysis_payload.get('duration_seconds', 'N/A')}s\n"
        f"Frissítve: {to_utc_iso(now)}"
    )
    hash_lines = []
    for name, meta in sorted(hashes.items()):
        if isinstance(meta, dict) and meta.get("sha256"):
            hash_lines.append(f"{name}: {str(meta.get('sha256'))[:12]}… ({meta.get('size', 'N/A')} B)")
        else:
            hash_lines.append(f"{name}: hiányzik")
    return {
        "title": "Pipeline diagnosztika",
        "description": description,
        "color": COLOR_YELLOW,
        "fields": [{"name": "Artefakt-hash", "value": "\n".join(hash_lines) or "N/A", "inline": False}],
    }

def _append_lifecycle_entry_event(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass



def _max_concurrent_positions() -> int:
    try:
        value = settings.load_config().get("max_concurrent_positions", 0)
        return max(int(value or 0), 0)
    except Exception:
        return 0


def _open_lifecycle_position_count(path: Optional[Path] = None) -> int:
    state = load_json(path or LIFECYCLE_STATE_PATH)
    positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}
    return sum(
        1
        for pos in positions.values()
        if isinstance(pos, dict) and str(pos.get("status") or "").strip().lower() in {"open", "pending"}
    )


def _record_suppressed_concurrency(asset: str, payload: Dict[str, Any], *, direction: str, entry: Any, sl: Any, tp1: Any, tp2: Any, probability: Any, now_iso: str) -> None:
    _record_suppressed_shadow(asset, payload, direction=direction, entry=entry, sl=sl, tp1=tp1, tp2=tp2, probability=probability, now_iso=now_iso, mode="suppressed_concurrency", reason="Konkurencia-plafon miatt kihagyva")


def _record_suppressed_shadow(asset: str, payload: Dict[str, Any], *, direction: str, entry: Any, sl: Any, tp1: Any, tp2: Any, probability: Any, now_iso: str, mode: str, reason: str) -> None:    
    shadow = dict(payload or {})
    shadow["retrieved_at_utc"] = shadow.get("retrieved_at_utc") or now_iso
    shadow["signal"] = direction
    shadow["probability"] = shadow.get("probability", probability)
    shadow["entry"] = entry
    shadow["sl"] = sl
    shadow["tp1"] = tp1
    shadow["tp2"] = tp2
    shadow["gates"] = {**(shadow.get("gates") or {}), "mode": mode}
    reasons = shadow.get("reasons") if isinstance(shadow.get("reasons"), list) else []
    shadow["reasons"] = [*reasons, reason]
    old_dir, old_file, old_summary = _trade_journal.JOURNAL_DIR, _trade_journal.JOURNAL_FILE, _trade_journal.SUMMARY_FILE
    try:
        _trade_journal.JOURNAL_DIR = PUBLIC_DIR / "journal"
        _trade_journal.JOURNAL_FILE = _trade_journal.JOURNAL_DIR / "trade_journal.csv"
        _trade_journal.SUMMARY_FILE = _trade_journal.JOURNAL_DIR / "summary.json"
        _trade_journal.record_signal_event(asset, shadow)
    finally:
        _trade_journal.JOURNAL_DIR, _trade_journal.JOURNAL_FILE, _trade_journal.SUMMARY_FILE = old_dir, old_file, old_summary


def _entry_gate_passes(data: Dict[str, Any], p_score: Optional[float]) -> bool:
    gates = data.get("gates") if isinstance(data.get("gates"), dict) else {}
    missing = [str(item) for item in (gates.get("missing") or [])]
    critical_missing = [str(item) for item in (gates.get("critical_missing") or data.get("critical_missing") or [])]
    thresholds = data.get("entry_thresholds") if isinstance(data.get("entry_thresholds"), dict) else {}
    threshold = safe_float(thresholds.get("p_score_min_effective") or thresholds.get("p_score_min"))
    score_ok = True if threshold is None else (p_score is not None and p_score >= threshold)
    return score_ok and not missing and not critical_missing
    
def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as h:
            data = json.load(h)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]):
    try:
        with path.open("w", encoding="utf-8") as h:
            json.dump(payload, h, ensure_ascii=False, indent=2)
    except Exception:
        pass


def safe_float(value: Any) -> Optional[float]:
    try:
        res = float(value)
        return res if res == res else None
    except Exception:
        return None


def format_price(price: Any) -> str:
    val = safe_float(price)
    if val is None:
        return "N/A"
    return f"{val:,.1f}" if val > 1000 else f"{val:.5f}"


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_budapest_time(dt: datetime) -> str:
    return dt.astimezone(BUDAPEST_TZ).strftime("%Y-%m-%d %H:%M:%S CET/CEST")


def _asset_emoji(asset: str) -> str:
    key = str(asset or "").upper()
    if key in {"XAU", "XAUUSD", "GOLD_CFD", "GOLDCFD"}:
        return "🟡"
    if key in {"XAG", "XAGUSD", "SILVER", "SILVER_CFD"}:
        return "⚪"
    if key in {"USOIL", "OIL", "BRENT"}:
        return "🛢️"
    return "📌"


def _hu_reason(reason: str) -> str:
    reason_map = {
        "tp1_hit": "TP1 szint elérve.",
        "regime_shift": "Piaci rezsimváltás érzékelve.",
        "momentum_loss": "Lendület gyengül.",
        "structure_break": "Szerkezeti törés.",
        "volatility_spike": "Megugrott volatilitás.",
    }
    return reason_map.get(str(reason or "").strip().lower(), str(reason or "N/A"))


def _format_minutes(minutes: Optional[float]) -> str:
    if minutes is None:
        return "N/A"
    total = max(1, int(round(minutes)))
    if total < 60:
        return f"{total} perc"
    hours, mins = divmod(total, 60)
    return f"{hours}ó {mins}p" if mins else f"{hours} óra"


def _median(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v > 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _percentile(values: List[float], pct: float) -> Optional[float]:
    clean = sorted(v for v in values if v > 0)
    if not clean:
        return None
    idx = max(0, min(len(clean) - 1, int(round((len(clean) - 1) * pct))))
    return clean[idx]


def _load_close_series(asset_dir: Path, filename: str) -> Tuple[List[float], int]:
    payload = load_json(asset_dir / filename)
    rows = payload.get("values") or (payload.get("raw") or {}).get("values") or []
    parsed: List[Tuple[str, float]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        close = safe_float(row.get("close"))
        stamp = str(row.get("datetime") or row.get("timestamp") or "")
        if close is not None and stamp:
            parsed.append((stamp, close))
    parsed.sort(key=lambda item: item[0])
    interval = 5 if "5m" in filename else 1
    return [close for _, close in parsed[-120:]], interval


def _estimate_tp_eta_minutes(asset_dir: Path, direction: str, entry: float, tp1: float) -> Dict[str, Any]:
    closes, interval_minutes = _load_close_series(asset_dir, "klines_1m.json")
    if len(closes) < 12:
        closes, interval_minutes = _load_close_series(asset_dir, "klines_5m.json")
    if len(closes) < 3:
        return {"available": False, "reason": "missing_price_history"}

    favorable: List[float] = []
    absolute: List[float] = []
    for prev, cur in zip(closes, closes[1:]):
        delta = cur - prev
        absolute.append(abs(delta) / interval_minutes)
        if direction == "buy" and delta > 0:
            favorable.append(delta / interval_minutes)
        elif direction == "sell" and delta < 0:
            favorable.append(abs(delta) / interval_minutes)

    distance = abs(tp1 - entry)
    fast_speed = _percentile(favorable, 0.75)
    base_speed = _median(favorable) or _median(absolute)
    conservative_speed = _percentile(favorable, 0.25) or _median(absolute)
    if not base_speed:
        return {"available": False, "reason": "flat_price_history"}

    return {
        "available": True,
        "fast_minutes": distance / fast_speed if fast_speed else None,
        "base_minutes": distance / base_speed,
        "conservative_minutes": distance / conservative_speed if conservative_speed else None,
        "source_interval_minutes": interval_minutes,
    }


def _manual_trade_model_for_asset(asset_name: str, manual_trade_model: Dict[str, Any]) -> Dict[str, Any]:
    model = dict(manual_trade_model)
    overrides = manual_trade_model.get("asset_overrides")
    asset_override = overrides.get(asset_name.upper()) if isinstance(overrides, dict) else None
    if isinstance(asset_override, dict):
        model.update(asset_override)
    leverage_map = getattr(settings, "LEVERAGE", {}) or {}
    asset_leverage = safe_float(leverage_map.get(asset_name.upper()))
    if asset_leverage is not None and "leverage" not in (asset_override or {}):
        model["leverage"] = asset_leverage
    return model


def _stake_margin_usd() -> float:
    profit_target = getattr(settings, "PROFIT_TARGET_CONFIG", {}) or {}
    return safe_float(profit_target.get("margin_usd")) or 100.0


def _stake_multiplier() -> float:
    stake = getattr(settings, "STAKE_CONFIG", {}) or {}
    return safe_float(stake.get("multiplier")) or 1.0


def _stake_amount_usd() -> float:
    return _stake_margin_usd() * _stake_multiplier()


def _fixed_margin_size_units(entry: Optional[float], leverage: Optional[float]) -> Optional[float]:
    if entry is None or entry <= 0 or leverage is None or leverage <= 0:
        return None
    return (_stake_amount_usd() * leverage) / entry


def build_expected_trade_outcome(
    asset_dir: Path,
    asset_name: str,
    data: Dict[str, Any],
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    manual_trade_model: Dict[str, Any],
) -> Dict[str, Any]:
    model = _manual_trade_model_for_asset(asset_name, manual_trade_model)
    stake_amount_usd = _stake_amount_usd()    
    leverage = safe_float(model.get("leverage")) or 20.0
    tp1_close_fraction = safe_float(model.get("tp1_close_fraction")) or 1.0
    min_net_usd = safe_float(model.get("tp1_min_net_usd")) or 10.0
    eta_min = safe_float(model.get("eta_min_minutes")) or 5.0
    eta_max = safe_float(model.get("eta_max_minutes")) or 240.0
    max_chase_r = safe_float(model.get("max_chase_r")) or 0.2
    valid_for = safe_float(model.get("signal_valid_minutes")) or 10.0

    cost_pct = float((settings.ASSET_COST_MODEL.get(asset_name) or {}).get("round_trip_pct", 0.0))
    gross_pct = abs(entry - tp1) / entry if entry else 0.0
    net_pct = gross_pct - cost_pct
    notional = stake_amount_usd * leverage    
    tp1_net_usd = net_pct * notional * tp1_close_fraction
    risk = abs(entry - sl)
    spot = safe_float((data.get("spot") or {}).get("price")) or entry
    chase_r = 0.0
    if risk > 0:
        if direction == "buy" and spot > entry:
            chase_r = (spot - entry) / risk
        elif direction == "sell" and spot < entry:
            chase_r = (entry - spot) / risk

    eta = _estimate_tp_eta_minutes(asset_dir, direction, entry, tp1)
    eta_base = safe_float(eta.get("base_minutes"))
    eta_gate = bool(eta.get("available") and eta_base is not None and eta_min <= eta_base <= eta_max)
    profit_gate = tp1_net_usd >= min_net_usd
    no_chase_gate = chase_r <= max_chase_r

    return {
        "stake_amount_usd": round(stake_amount_usd, 2),        
        "leverage": round(leverage, 2),
        "notional_usd": round(notional, 2),
        "tp1_net_usd": round(tp1_net_usd, 2),
        "min_required_net_usd": round(min_net_usd, 2),
        "tp1_net_pct": round(net_pct, 6),
        "eta_minutes_fast": round(eta["fast_minutes"], 1) if eta.get("fast_minutes") else None,
        "eta_minutes_base": round(eta_base, 1) if eta_base is not None else None,
        "eta_minutes_conservative": round(eta["conservative_minutes"], 1) if eta.get("conservative_minutes") else None,
        "eta_source_interval_minutes": eta.get("source_interval_minutes"),
        "eta_available": bool(eta.get("available")),
        "eta_unavailable_reason": eta.get("reason"),
        "valid_for_minutes": round(valid_for, 1),
        "max_chase_r": round(max_chase_r, 3),
        "current_chase_r": round(chase_r, 3),
        "max_entry_price": round(entry + risk * max_chase_r, 6) if direction == "buy" and risk > 0 else None,
        "min_entry_price": round(entry - risk * max_chase_r, 6) if direction == "sell" and risk > 0 else None,
        "profit_gate_pass": profit_gate,
        "eta_gate_pass": eta_gate,
        "no_chase_pass": no_chase_gate,
        "passes": profit_gate and eta_gate and no_chase_gate,
    }



def _append_notify_event(*args: Any, **kwargs: Any) -> None:
    pass


def _collect_webhook_urls() -> List[str]:
    seen: set[str] = set()
    urls: List[str] = []
    for raw in os.getenv("DISCORD_WEBHOOK_URL", "").replace("\n", ",").split(","):
        url = raw.strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def send_discord_embed(embed: Dict[str, Any]) -> bool:
    global NOTIFY_ATTEMPTS, NOTIFY_SUCCESSES, NOTIFY_FAILURES
    NOTIFY_ATTEMPTS += 1
    if DRY_RUN:
        NOTIFY_SUCCESSES += 1
        return True
    urls = _webhook_urls("actionable") or DISCORD_WEBHOOK_URLS or _collect_webhook_urls()
    ok = False
    for idx, url in enumerate(urls):
        payloads = [{"embeds": [embed]}]
        if len(urls) == 1:
            payloads.append({"content": str(embed.get("title") or embed.get("description") or "Discord alert")[:2000]})
        for payload in payloads:
            try:
                resp = requests.post(url, json=payload, timeout=8) if requests else None
                status = getattr(resp, "status_code", 204)
                ok_status = _webhook_log_response(LOGGER, "notify_discord", "actionable", resp)
                _append_notify_event({"url_index": idx, "status": status})
                if ok_status:
                    NOTIFY_SUCCESSES += 1
                    return True
            except Exception as exc:
                _webhook_log_exception(LOGGER, "notify_discord", "actionable", exc)
                _append_notify_event({"url_index": idx, "error": repr(exc)})
    NOTIFY_FAILURES += 1
    return ok


def post_batches(hook: str, content: str, embeds: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    now = time.time()
    if _WEBHOOK_COOLDOWN_UNTIL.get(hook, 0) > now:
        return {"attempted": False, "success": False, "error": "webhook_cooldown", "batch_results": []}
    batch_results = []
    for i in range(0, len(embeds), batch_size):
        batch = embeds[i:i + batch_size]
        payload = {"content": content, "embeds": batch}
        success = False; status = None; error = None
        for attempt in range(2):
            try:
                resp = requests.post(hook, json=payload, timeout=8) if requests else None
                status = getattr(resp, "status_code", 204)
                _webhook_log_response(LOGGER, "notify_discord", "actionable", resp)
                if status == 429:
                    retry = float(getattr(resp, "headers", {}).get("Retry-After", 1.0) or 1.0)
                    _WEBHOOK_COOLDOWN_UNTIL[hook] = time.time() + retry
                    time.sleep(retry)
                    continue
                if hasattr(resp, "raise_for_status"):
                    resp.raise_for_status()
                success = 200 <= int(status) < 300
                break
            except Exception as exc:
                _webhook_log_exception(LOGGER, "notify_discord", "actionable", exc)
                error = str(exc)
                break
        batch_results.append({"attempted": True, "success": success, "http_status": status, "error": error, "message_id": None, "batch_index": i // batch_size, "embed_count": len(batch)})
    return {"attempted": bool(batch_results), "success": all(b.get("success") for b in batch_results), "http_status": batch_results[-1].get("http_status") if batch_results else None, "error": batch_results[-1].get("error") if batch_results else None, "message_id": None, "batch_results": batch_results}


def _map_batch_results_to_assets(asset_embed_pairs: List[Tuple[str, Dict[str, Any]]], dispatch_result: Dict[str, Any], batch_size: int = 10) -> Dict[str, Dict[str, Any]]:
    results = {}
    batches = dispatch_result.get("batch_results") or []
    for idx, (asset, _embed) in enumerate(asset_embed_pairs):
        br = batches[idx // batch_size] if idx // batch_size < len(batches) else dispatch_result
        results[asset] = dict(br)
    return results


def _finalize_entry_commit(asset: str, pending: Dict[str, Any], dispatch_result: Dict[str, Any], *, manual_positions: Dict[str, Any], tracking_cfg: Dict[str, Any], now_dt: datetime, now_iso: str, cooldown_map: Dict[str, Any], cooldown_default: int, positions_path: str, open_commits_this_run: set) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    import position_tracker
    audit = pending.get("audit")
    if audit:
        audit.dispatch_attempted = bool(dispatch_result.get("attempted"))
        audit.dispatch_success = bool(dispatch_result.get("success"))
    if not dispatch_result.get("attempted"):
        if audit: audit.commit_status = "dispatch_not_attempted"
        return manual_positions, position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt), {"committed": False}
    if not dispatch_result.get("success"):
        if audit: audit.commit_status = "dispatch_failed"
        return manual_positions, position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt), {"committed": False}
    if not pending.get("state_loaded", True):
        if audit: audit.commit_status = "state_not_loaded"
        return manual_positions, position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt), {"committed": False}
    try:
        manual_positions, manual_state, changed, opened = _apply_manual_position_transitions(
            asset=asset, intent=pending.get("intent", "entry"), decision=pending.get("decision", "buy"), setup_grade=pending.get("setup_grade", ""),
            notify_meta=pending.get("notify_meta") or {"should_notify": True}, signal_payload=pending.get("signal_payload") or {},
            manual_tracking_enabled=pending.get("manual_tracking_enabled", True), can_write_positions=pending.get("can_write_positions", True),
            manual_state=pending.get("manual_state_pre") or {}, manual_positions=manual_positions, tracking_cfg=tracking_cfg,
            now_dt=now_dt, now_iso=now_iso, send_kind=pending.get("send_kind"), display_stable=pending.get("display_stable", True),
            missing_list=pending.get("gates_missing") or [], cooldown_map=cooldown_map, cooldown_default=cooldown_default,
        )
        if changed:
            position_tracker.save_positions_atomic(positions_path, manual_positions)
        persisted = position_tracker.load_positions(positions_path, True)
        verified = asset in persisted and position_tracker.compute_state(asset, tracking_cfg, persisted, now_dt).get("has_position")
        manual_state = position_tracker.compute_state(asset, tracking_cfg, persisted, now_dt)
        if audit: audit.commit_status = "commit_ok" if verified else "commit_verify_failed"
        return manual_positions, manual_state, {"committed": True, "verified": bool(verified)}
    except Exception as exc:
        position_tracker.log_audit_event("entry commit failed", event="OPEN_COMMIT_FAILED", asset=asset, exception=repr(exc))
        position_tracker.log_audit_event("entry dispatched but not committed", event="ENTRY_DISPATCHED_BUT_NOT_COMMITTED", asset=asset, exception=repr(exc))
        if audit: audit.commit_status = "commit_exception"
        return manual_positions, position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt), {"committed": False, "error": repr(exc), "exception": repr(exc)}


def build_entry_gate_summary_embed(now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    payload: Dict[str, Any] = {}
    if ENTRY_GATE_STATS_PATH.exists():
        payload = load_json(ENTRY_GATE_STATS_PATH)
    elif ENTRY_GATE_LOG_DIR.exists():
        cutoff = now - timedelta(hours=24)
        for path in sorted(ENTRY_GATE_LOG_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try: row = json.loads(line)
                except Exception: continue
                ts = _parse_utc(row.get("ts_utc") or row.get("timestamp"))                    
                if ts and ts < cutoff:
                    continue
                asset = row.get("asset")
                if asset:
                    payload.setdefault(asset, []).append({"missing": row.get("missing") or row.get("reasons") or [], "precision_hiany": row.get("precision_hiany") or []})
    if not payload:
        return None
    rows=[]
    for asset, items in payload.items():
        rejects=0; reasons=[]
        for item in items if isinstance(items, list) else []:
            missing=(item.get("missing") or [])+(item.get("precision_hiany") or [])
            if missing:
                rejects += 1; reasons.extend(map(str, missing))
        rows.append((rejects, asset, reasons))
    rows.sort(key=lambda item: (-item[0], item[1]))
    value="\n".join(f"• {asset}: {rejects}x blokkolva ({', '.join(reasons[:3]) or 'ok'})" for rejects, asset, reasons in rows[:10])
    return {"title": "Entry gate toplista (24h)", "description": f"session / precision összegzés – {to_utc_iso(now)}", "color": COLOR_YELLOW, "fields": [{"name": "Assetek", "value": value or "N/A", "inline": False}]}


def _coerce_price(value: Any) -> Optional[float]:
    if isinstance(value, str):
        value = value.replace(",", "")
    return safe_float(value)


def _operator_instruction_lines(signal_data: Dict[str, Any], *, size_units: Optional[float] = None, expiry_dt: Optional[datetime] = None) -> List[str]:
    management = signal_data.get("management") if isinstance(signal_data, dict) else None
    instructions = signal_data.get("operator_instructions") if isinstance(signal_data, dict) else None
    if isinstance(management, dict):
        instructions = management.get("operator_instructions") or instructions
    if isinstance(instructions, str):
        lines = [line.strip() for line in instructions.splitlines() if line.strip()]
    elif isinstance(instructions, list):
        lines = [str(line).strip() for line in instructions if str(line).strip()]
    else:
        lines = []
    if isinstance(management, dict):
        lifecycle_cfg = getattr(settings, "POSITION_LIFECYCLE", None) or getattr(settings, "position_lifecycle", None) or {}
        tp1_closes = bool(management["tp1_closes_position"] if "tp1_closes_position" in management else lifecycle_cfg.get("tp1_closes_position", False))
        if tp1_closes:
            lines = [line for line in lines if "TP1 elérésénél" not in line and "TP1 után" not in line]
            lines.insert(0, "TP1 elérésénél zárd a TELJES pozíciót.")
        else:
            partial = safe_float(management.get("partial_close_pct")) or 0.5
            if size_units is not None:
                units_partial = size_units * partial
                lines = [line.replace(f"{partial:.0%}-át manuálisan.", f"{partial:.0%}-át (≈{units_partial:.2f} egység) manuálisan.") for line in lines]
    lines.append("Minimál-tétes protokoll: Amount × 0.1–0.2 az első 10 trade-re.")                
    if expiry_dt is not None:
        lines.append(f"Ha {expiry_dt:%H:%M} UTC-ig nem töltődik a rendszer-limit, lejáratkor ❌ JEL LEJÁRT kártya érkezik — függő megbízásod ekkor töröld.")        
    return lines


def _collect_channel_embeds(*, asset_embeds: Dict[str, Dict[str, Any]], asset_channels: Dict[str, str], watcher_embeds: List[Dict[str, Any]], auto_close_embeds: List[Dict[str, Any]], heartbeat_snapshots: List[Dict[str, Any]], gate_embed: Optional[Dict[str, Any]], pipeline_embed: Optional[Dict[str, Any]]):
    live, management, market_scan = [], [], []
    for asset in ASSETS:
        embed = asset_embeds.get(asset)
        if not embed:
            continue
        channel = asset_channels.get(asset, "live")
        if channel == "management":
            management.append((asset, embed))
        elif channel == "market_scan":
            market_scan.append((asset, embed))
        else:
            live.append((asset, embed))
    for embed in watcher_embeds + auto_close_embeds + heartbeat_snapshots:
        management.append(("_system", embed))
    for embed in (gate_embed, pipeline_embed):
        if embed:
            live.append(("_system", embed))
    return live, management, market_scan



def extract_trade_levels(sig: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    return (_coerce_price(sig.get("entry")), _coerce_price(sig.get("sl")), _coerce_price(sig.get("tp1")), _coerce_price(sig.get("tp2")))


def _apply_manual_position_transitions(
    *, asset: str, intent: str, decision: str, setup_grade: str, notify_meta: Optional[Dict[str, Any]],
    signal_payload: Dict[str, Any], manual_tracking_enabled: bool, can_write_positions: bool,
    manual_state: Dict[str, Any], manual_positions: Dict[str, Any], tracking_cfg: Dict[str, Any],
    now_dt: datetime, now_iso: str, send_kind: Optional[str], display_stable: bool,
    missing_list: List[str], cooldown_map: Dict[str, Any], cooldown_default: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, bool]:
    import position_tracker
    if not manual_tracking_enabled or not can_write_positions:
        return manual_positions, manual_state, False, False
    notify_meta = notify_meta or {}
    if intent == "entry" and decision in {"buy", "sell"}:
        if not send_kind or not display_stable or notify_meta.get("reason") == "cooldown_active" or manual_state.get("has_position") or manual_state.get("pending_active") or manual_state.get("cooldown_active"):
            return manual_positions, manual_state, False, False
        entry, sl, tp1, tp2 = extract_trade_levels(signal_payload)
        position_tracker.log_audit_event(
            "entry open attempt", event="OPEN_ATTEMPT", asset=asset, intent=intent,
            decision=decision, entry_side=decision, setup_grade=setup_grade,
            actionable=True, stable=display_stable, gates_missing=missing_list,
            notify_should_notify=notify_meta.get("should_notify"), notify_reason=notify_meta.get("reason", "actionable"),
            cooldown_until_utc=manual_state.get("cooldown_until_utc"), manual_tracking_enabled=manual_tracking_enabled,
            manual_has_position=manual_state.get("has_position"), manual_cooldown_active=manual_state.get("cooldown_active"),
            entry_level=entry, sl=sl, tp1=tp1, tp2=tp2,
        )
        updated = position_tracker.open_position(asset, "long" if decision == "buy" else "short", entry, sl, tp1, tp2, now_iso, positions=manual_positions)
        return updated, position_tracker.compute_state(asset, tracking_cfg, updated, now_dt), True, True
    if intent == "hard_exit" and manual_state.get("has_position"):
        updated = position_tracker.close_position(asset, "hard_exit", now_iso, int(cooldown_default or 20), manual_positions)
        return updated, position_tracker.compute_state(asset, tracking_cfg, updated, now_dt), True, False
    return manual_positions, manual_state, False, False

def _apply_and_persist_manual_transitions(**kwargs):
    import position_tracker
    manual_positions, manual_state, changed, entry_opened = _apply_manual_position_transitions(
        asset=kwargs["asset"], intent=kwargs.get("intent"), decision=kwargs.get("decision"), setup_grade=kwargs.get("setup_grade"),
        notify_meta=kwargs.get("notify_meta"), signal_payload=kwargs.get("signal_payload") or {},
        manual_tracking_enabled=kwargs.get("manual_tracking_enabled"), can_write_positions=kwargs.get("can_write_positions"),
        manual_state=kwargs.get("manual_state") or {}, manual_positions=kwargs.get("manual_positions") or {},
        tracking_cfg=kwargs.get("tracking_cfg") or {"enabled": True}, now_dt=kwargs.get("now_dt") or datetime.now(timezone.utc),
        now_iso=kwargs.get("now_iso") or to_utc_iso(datetime.now(timezone.utc)), send_kind=kwargs.get("send_kind"),
        display_stable=kwargs.get("display_stable"), missing_list=kwargs.get("missing_list") or {},
        cooldown_map=kwargs.get("cooldown_map") or {}, cooldown_default=int(kwargs.get("cooldown_default") or 20),
    )
    if changed and kwargs.get("positions_path"):
        position_tracker.save_positions_atomic(kwargs.get("positions_path"), manual_positions)
    sig = kwargs.get("sig")
    if isinstance(sig, dict):
        sig["position_state"] = manual_state
    return manual_positions, manual_state, changed, entry_opened, None

def build_mobile_embed_for_asset(
    asset: str,
    state: Dict[str, Any],
    signal_data: Dict[str, Any],
    decision: str,
    mode: str,
    is_stable: bool,
    is_flip: bool,
    is_invalidate: bool,
    *,
    kind: str = "normal",
    manual_positions: Optional[Dict[str, Any]] = None,
    include_manual_position: bool = True,
) -> Dict[str, Any]:
    asset_state = state.setdefault(asset, dict(DEFAULT_ASSET_STATE))
    spot = signal_data.get("spot") or {}
    current = _coerce_price(spot.get("price"))
    previous_source = "state.last_spot_price"
    previous = _coerce_price(asset_state.get("last_spot_price"))
    if previous is None:
        previous = _coerce_price((signal_data.get("notify") or {}).get("state", {}).get("last_spot_price"))
        previous_source = "notify.state.last_spot_price"
    if _coerce_price(spot.get("previous")) is not None:
        previous = _coerce_price(spot.get("previous"))
        previous_source = "signal.spot.previous"
    arrow = "→"
    if current is not None and previous is not None:
        arrow = "↑" if current > previous else "↓" if current < previous else "→"
    if current is not None:
        asset_state["last_spot_price"] = current
        asset_state["last_spot_utc"] = spot.get("utc") or signal_data.get("retrieved_at_utc")
    LOGGER.debug("Price direction resolved", extra={"prev_spot_price_source": previous_source, "prev_spot_price_coerced": previous, "current_spot_price": current, "price_direction": arrow})

    lines = [f"{_asset_emoji(asset)} Eszköz: `{asset}`", f"Spot: `{format_price(current)}` {arrow}"]
    show_levels = kind == "entry" or signal_data.get("intent") == "entry" or str(decision).lower() in {"buy", "sell"}
    if show_levels:
        for label, key in (("Belépő", "entry"), ("SL", "sl"), ("TP1", "tp1"), ("TP2", "tp2")):
            if signal_data.get(key) is not None:
                lines.append(f"{label}: `{format_price(signal_data.get(key))}`")
    manual_state = signal_data.get("position_state") or {}
    tracked = signal_data.get("tracked_levels") or {}
    if include_manual_position and manual_state.get("has_position") and not (manual_positions and isinstance(manual_positions.get(asset), dict) and manual_positions.get(asset, {}).get("status") == "closed"):
        lines.append("Pozíciómenedzsment: aktív")
        for label, key in (("Belépő", "entry"), ("SL", "sl"), ("TP1", "tp1"), ("TP2", "tp2")):
            value = manual_state.get(key, tracked.get(key))
            if value is not None:
                lines.append(f"{label}: `{format_price(value)}`")
    if asset.upper() in DEFAULT_DISCORD_NOTIFY_ASSETS:
        instructions = _operator_instruction_lines(signal_data)
        if instructions:
            lines.append("Kezelési terv:")
            lines.extend(f"• {line}" for line in instructions)
    return {"title": f"{asset} {str(decision).upper()} [{mode}]", "description": "\n".join(lines), "color": COLOR_GREEN if decision == "buy" else COLOR_RED if decision == "sell" else COLOR_YELLOW}



def _position_lifecycle_embed(asset: str, pos: Dict[str, Any], signal_data: Dict[str, Any], now_dt: datetime) -> Optional[Dict[str, Any]]:
    status = str(pos.get("status") or "").lower()
    spot = safe_float((signal_data.get("spot") or {}).get("price"))
    if status == "open":
        order_type = str(pos.get("order_type") or pos.get("orderType") or "Automatikus aktiválás").upper()
        if order_type == "AUTOMATIKUS AKTIVÁLÁS":
            order_type = "Automatikus aktiválás"
        opened = _parse_utc(pos.get("opened_at_utc"))
        opened_txt = opened.astimezone(BUDAPEST_TZ).strftime("%Y-%m-%d %H:%M:%S") if opened else "N/A"
        lines = [
            f"{_asset_emoji(asset)} Eszköz: `{asset}`",
            "Állapot: `Nyitott`",
            f"Spot: `{_lifecycle_price(spot)}`",
            f"Belépő típus: `{order_type}`",
            f"Aktiválva: `{opened_txt}`",
            f"Belépő: `{_lifecycle_price(pos.get('entry'))}`",
            f"SL: `{_lifecycle_price(pos.get('sl'))}`",
            f"TP1: `{_lifecycle_price(pos.get('tp1'))}`",
            f"TP2: `{_lifecycle_price(pos.get('tp2'))}`",
        ]
        return {"title": f"{asset} pozíció aktiválva", "description": "\n".join(lines), "color": COLOR_GREEN}
    if status == "closed":
        lines = [
            f"{_asset_emoji(asset)} Eszköz: `{asset}`",
            "Állapot: `Lezárt`",
            f"Ok: `{pos.get('close_reason') or 'closed'}`",
            f"Spot: `{_lifecycle_price(spot)}`",
            f"SL: `{_lifecycle_price(pos.get('sl'))}`",
        ]
        return {"title": f"{asset} pozíció lezárva", "description": "\n".join(lines), "color": COLOR_YELLOW}
    return None

def _lifecycle_price(value: Any) -> str:
    val = safe_float(value)
    return "N/A" if val is None else f"{val:.2f}"

def _levels_match_direction(direction: str, entry: Optional[float], sl: Optional[float], tp1: Optional[float]) -> bool:
    if None in (entry, sl, tp1):
        return False
    return (sl < entry < tp1) if direction == "buy" else (tp1 < entry < sl)

def _send_hard_exit_embed(asset: str, pos: Dict[str, Any], now_dt: datetime) -> None:
    side = str(pos.get("side") or "").lower()
    side_label = "LONG" if side in {"long", "buy"} else "SHORT" if side in {"short", "sell"} else "N/A"
    embed = {
        "title": f"🔴 AZONNAL ZÁRD A POZÍCIÓT! – {asset}",
        "description": f"{_asset_emoji(asset)} Eszköz: `{asset}`\nOk: `ellentétes belépési jel`",
        "color": COLOR_RED,
        "fields": [{"name": "🎯 Zárandó irány", "value": side_label, "inline": False}],
    }
    send_discord_embed(embed)
    
def check_and_notify() -> None:
    if not PUBLIC_DIR.exists():
        return
    manual_trade_model = settings.MANUAL_TRADE_MODEL or {}
    tp1_min_net_usd = safe_float(manual_trade_model.get("tp1_min_net_usd")) or 10.0
    
    notify_state_path = PUBLIC_DIR / "_notify_state.json"
    notify_state = load_json(notify_state_path)
    notify_changed = False

    for asset_dir in [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]:
        asset_name = asset_dir.name
        if DISCORD_NOTIFY_ASSETS and asset_name.upper() not in DISCORD_NOTIFY_ASSETS:
            continue

        data = load_json(asset_dir / "signal.json")
        if not data:
            continue

        now_dt = datetime.now(timezone.utc)
        positions_path = str(PUBLIC_DIR / "trading.db")
        try:
            positions = position_tracker.load_positions(positions_path, True)
        except Exception:
            positions = {}
        pos = positions.get(asset_name) if isinstance(positions, dict) else None
        asset_state = notify_state.get(asset_name) or {}
        if isinstance(pos, dict) and str(pos.get("status") or "").lower() in {"pending", "open"}:
            if str(pos.get("status") or "").lower() == "pending":
                continue
        if isinstance(pos, dict) and str(pos.get("status") or "").lower() in {"open", "closed"}:
            lifecycle_sig = f"{pos.get('status')}:{pos.get('opened_at_utc') or pos.get('closed_at_utc') or pos.get('close_reason')}"
            if asset_state.get("last_lifecycle_signature") != lifecycle_sig:
                lifecycle_embed = _position_lifecycle_embed(asset_name, pos, data, now_dt)
                if lifecycle_embed:
                    send_discord_embed(lifecycle_embed)
                    asset_state["last_lifecycle_signature"] = lifecycle_sig
                    notify_state[asset_name] = asset_state
                    notify_changed = True

        signal = str(data.get("signal") or "no entry").lower()
        plan = data.get("precision_plan") if isinstance(data.get("precision_plan"), dict) else {}
        if signal == "no entry" and plan.get("trigger_state") == "fire":
            signal = "precision_arming"
        if signal not in {"buy", "sell", "precision_arming"}:
            continue

        entry, sl, tp1, tp2 = safe_float(data.get("entry")), safe_float(data.get("sl")), safe_float(data.get("tp1")), safe_float(data.get("tp2"))
        order_type, direction = str(data.get("order_type") or "MARKET").upper(), signal
        
        if signal == "precision_arming":
            direction = str(plan.get("direction") or "buy").lower()
            order_type = str(plan.get("order_type") or "LIMIT").upper()
            entry = safe_float(plan.get("entry") or entry)
            sl = safe_float(plan.get("stop_loss") or sl)
            tp1 = safe_float(plan.get("take_profit_1") or tp1)
            tp2 = safe_float(plan.get("take_profit_2") or tp2)

        if direction not in {"buy", "sell"}:
            continue

        if not _levels_match_direction(direction, entry, sl, tp1):
            continue

        if isinstance(pos, dict) and str(pos.get("status") or "").lower() == "open":
            current_side = "buy" if str(pos.get("side") or "").lower() in {"long", "buy"} else "sell" if str(pos.get("side") or "").lower() in {"short", "sell"} else None
            if current_side and current_side != direction:
                _send_hard_exit_embed(asset_name, pos, now_dt)
                try:
                    positions = position_tracker.close_position(asset_name, "hard_exit", to_utc_iso(now_dt), ENTRY_COOLDOWN_MINUTES, positions)
                    if not DRY_RUN:
                        position_tracker.save_positions_atomic(positions_path, positions)
                except Exception:
                    pass
            elif current_side:
                continue

        
        expected = build_expected_trade_outcome(asset_dir, asset_name, data, direction, entry, sl, tp1, manual_trade_model)
        tp1_net_usd = safe_float(expected.get("tp1_net_usd")) or 0.0
        if not expected.get("passes") and ((asset_dir / "klines_1m.json").exists() or (asset_dir / "klines_5m.json").exists()):
            continue

        leverage_map = getattr(settings, "LEVERAGE", {}) or {}
        asset_leverage = safe_float(leverage_map.get(asset_name.upper())) or 1.0
        stake_amount_usd = _stake_amount_usd()
        size_units = _fixed_margin_size_units(entry, asset_leverage)
        units_text = f"{size_units:.2f} Egység (Units)" if size_units is not None else "N/A"
        entry_notional_usd = (size_units * entry) if size_units is not None else None
        sl_risk_to_stop_usd = (size_units * abs(entry - sl)) if size_units is not None and sl is not None else None
        
        entry_sig = f"{direction}_{order_type}"

        if asset_state.get("last_entry_signature") == entry_sig and asset_state.get("last_entry_sent_utc"):
            try:
                if now_dt - datetime.fromisoformat(asset_state["last_entry_sent_utc"].replace("Z", "+00:00")) < timedelta(minutes=ENTRY_COOLDOWN_MINUTES):
                    continue
            except Exception:
                pass

        prefix, color = (("🟢", COLOR_GREEN) if direction == "buy" else ("🔴", COLOR_RED))
        side_text = "LONG" if direction == "buy" else "SHORT"
        action_text = "BUY" if direction == "buy" else "SELL"
        asset_emoji = {"GOLD_CFD": "🥇", "XAGUSD": "🥈", "USOIL": "🛢️"}.get(asset_name.upper(), "📊")
        title = f"{prefix} NYISS {side_text} — {asset_emoji} {asset_name} · {action_text} {order_type} @ {format_price(entry)}"       

        p_score = safe_float(data.get("probability_raw") or data.get("probability"))
        if not _entry_gate_passes(data, p_score):
            _record_suppressed_shadow(asset_name, data, direction=direction, entry=entry, sl=sl, tp1=tp1, tp2=tp2, probability=p_score, now_iso=to_utc_iso(now_dt), mode="suppressed_gate_mismatch", reason="Entry gate elutasítás miatt kihagyva")
            continue        
        reasons = "\n".join([f"• {_hu_reason(r)}" for r in (data.get("reasons") or [])[:2]]) or "• Rendszer jelzés"
        reasons_text = f"P Score: **{p_score:.1f}** (Erősség)\n{reasons}" if p_score else reasons
        valid_minutes = safe_float(expected.get('valid_for_minutes'))
        expiry_dt = now_dt + timedelta(minutes=valid_minutes or 0)        
        eta_text = (
            f"Gyors: `{_format_minutes(safe_float(expected.get('eta_minutes_fast')))}`\n"
            f"Normál: `{_format_minutes(safe_float(expected.get('eta_minutes_base')))}`\n"
            f"Konzervatív: `{_format_minutes(safe_float(expected.get('eta_minutes_conservative')))}`\n"
            f"Jel érvényessége: `{_format_minutes(valid_minutes)}`"            
        )
        validity_text = f"⏳ Érvényes eddig: {expiry_dt:%H:%M} UTC ({expiry_dt.astimezone(BUDAPEST_TZ):%H:%M} Budapest)"        
        entry_limit_text = (
            f"Ne nyiss, ha spot > `{format_price(expected.get('max_entry_price'))}`"
            if direction == "buy"
            else f"Ne nyiss, ha spot < `{format_price(expected.get('min_entry_price'))}`"
        )
        _etoro_off = {"GOLD_CFD": 1.5, "XAGUSD": 0.08, "USOIL": 0.10}.get(asset_name.upper())
        _tp1f, _slf = safe_float(tp1), safe_float(sl)
        sl_etoro_text, tp1_etoro_text = "", ""
        if _etoro_off is not None and _tp1f is not None and _slf is not None:
            _sgn = 1.0 if direction == "sell" else -1.0
            _od = f"{'+' if _sgn > 0 else '-'}{_etoro_off:g}"
            sl_etoro_text = f"\nSL eToro (offszet {_od}): `{format_price(_slf + _sgn * _etoro_off)}`"
            tp1_etoro_text = f"\nTP1 eToro (offszet {_od}): `{format_price(_tp1f + _sgn * _etoro_off)}`"        
            
        embed = {
            "title": title,
            "description": f"{_asset_emoji(asset_name)} Eszköz: `{asset_name}`",
            "color": color,
            "fields": [
                {"name": "📊 Árfolyam", "value": f"Spot ár: `{format_price(safe_float((data.get('spot') or {}).get('price')))}`\nBelépő (LIMIT): `{format_price(entry)}`", "inline": False},
                {"name": "⚙️ Paraméterek az eToro-hoz", "value": f"MÉRET: `{units_text}` (~${sl_risk_to_stop_usd:.2f} kockázat SL-ig)\nNotional: `${entry_notional_usd:.2f}`\neToro Amount (X{asset_leverage:g}): `${stake_amount_usd:.2f}`\nSL: `{format_price(sl)}`{sl_etoro_text}\nTP1: `{format_price(tp1)}`{tp1_etoro_text}" + (f"\nTP2: `{format_price(tp2)}`" if tp2 else "") + f"\n{validity_text}", "inline": False},
                {"name": "🎯 Profit cél", "value": f"Várható nettó TP1: `+${tp1_net_usd:.2f}`\nMinimum: `${tp1_min_net_usd:.2f}`\nProfit-cél számítási alap: `${expected.get('notional_usd'):.2f}`", "inline": False},
                {"name": "⏱️ Várható idő TP1-ig", "value": eta_text, "inline": False},
                {"name": "🎯 Belépési pontosság", "value": f"Aktuális chase: `{expected.get('current_chase_r')}R`\n{entry_limit_text}", "inline": False},
                {"name": "💡 Indoklás", "value": reasons_text, "inline": False},
                *([{"name": "🧭 Kezelési terv", "value": "\n".join(f"• {line}" for line in _operator_instruction_lines(data, size_units=size_units, expiry_dt=expiry_dt))[:1024], "inline": False}] if _operator_instruction_lines(data, size_units=size_units, expiry_dt=expiry_dt) else []),
                {"name": "🕒 Időbélyeg", "value": f"`{format_budapest_time(now_dt)}` (Budapest)", "inline": False},
            ],
            "footer": {"text": f"Signal • Budapest: {format_budapest_time(now_dt)} • Várakozás (30 perc csend indítva)"},
        }

        max_concurrent_positions = _max_concurrent_positions()
        if max_concurrent_positions > 0 and _open_lifecycle_position_count() >= max_concurrent_positions:
            _record_suppressed_concurrency(asset_name, data, direction=direction, entry=entry, sl=sl, tp1=tp1, tp2=tp2, probability=p_score, now_iso=to_utc_iso(now_dt))
            continue
        
        sent_result = send_discord_embed(embed)
        if sent_result is False:
            continue

        try:
            manual_state = position_tracker.compute_state(asset_name, {"enabled": True}, positions, now_dt)
            if not manual_state.get("has_position") and not manual_state.get("pending_active"):
                if signal == "precision_arming" and order_type != "MARKET":
                    positions = position_tracker.register_precision_pending_position(asset_name, data, now_dt, positions)
                else:
                    positions = position_tracker.open_position(asset_name, "long" if direction == "buy" else "short", entry, sl, tp1, tp2, to_utc_iso(now_dt), positions=positions)
                if not DRY_RUN:
                    position_tracker.save_positions_atomic(positions_path, positions)
        except Exception:
            pass

        asset_state.update({"last_entry_signature": entry_sig, "last_entry_sent_utc": to_utc_iso(now_dt)})
        notify_state[asset_name] = asset_state
        notify_changed = True
        _append_lifecycle_entry_event(LIFECYCLE_INBOX_PATH, {
            "event": "entry_signal",
            "ts_utc": to_utc_iso(now_dt),
            "asset": asset_name,
            "signal": signal,
            "direction": direction,
            "order_type": order_type,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "entry_signature": entry_sig,
            "expected_trade_outcome": expected,
            "management": data.get("management") if isinstance(data.get("management"), dict) else {},
            "size_units": size_units,    
        })

    if notify_changed and not DRY_RUN:
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        print('usage: notify_discord.py [--help]')
        raise SystemExit(0)    
    if not PUBLIC_DIR.exists():
        sys.exit(0)
    with NOTIFY_LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        check_and_notify()
