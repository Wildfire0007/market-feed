#!/usr/bin/env python3
"""Fetch macroeconomic events and compute asset lockout windows."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


TE_API_URL = (
    "https://api.tradingeconomics.com/calendar?c={key}&initDate={d1}&endDate={d2}&importance=2&f=json"
)
FRED_RELEASE_IDS = {
    "CPI": 10,
    "EMPLOYMENT_SITUATION": 50,
    "GDP": 53,
    "PCE": 54,
}
ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = timezone.utc

DEFAULT_LOCKOUT = {
    3: {"pre_seconds": 900, "post_seconds": 1800},
    2: {"pre_seconds": 600, "post_seconds": 1200},
    1: {"pre_seconds": 0, "post_seconds": 0},
}


@dataclass
class Event:
    id: str
    provider: str
    country: str
    category: str
    event: str
    importance: int
    ts_release_utc: str
    lockout_pre: int
    lockout_post: int

    def to_calendar_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "provider": self.provider,
            "type": "SCHEDULED",
            "country": self.country,
            "category": self.category,
            "event": self.event,
            "importance": self.importance,
            "ts_release_utc": self.ts_release_utc,
            "lockout": {
                "pre_seconds": self.lockout_pre,
                "post_seconds": self.lockout_post,
            },
        }


def utc_today() -> datetime:
    now = datetime.now(UTC_ZONE)
    return datetime(now.year, now.month, now.day, tzinfo=UTC_ZONE)


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "macro-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec - trusted endpoint
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def fetch_te_events(d1: str, d2: str, api_key: str) -> List[Event]:
    url = TE_API_URL.format(key=urllib.parse.quote(api_key), d1=d1, d2=d2)
    try:
        raw = fetch_json(url)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network guard
        raise RuntimeError(f"TradingEconomics HTTP error: {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network guard
        raise RuntimeError(f"TradingEconomics network error: {exc.reason}") from exc

    if not raw:
        raise RuntimeError("TradingEconomics returned empty payload")

    events: List[Event] = []
    for item in raw:
        try:
            calendar_id = str(item.get("CalendarId") or item.get("ID") or item.get("id"))
            country = (item.get("Country") or "").strip().upper() or "UNKNOWN"
            category_raw = (item.get("Category") or "").strip()
            event_name = (item.get("Event") or item.get("Title") or "").strip() or "Unknown Event"
            importance = int(item.get("Importance") or item.get("importance") or 0)
            ts_iso = normalize_te_timestamp(item)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Invalid TE item structure: {item}") from exc

        category = normalize_category(category_raw, event_name)
        lockout = DEFAULT_LOCKOUT.get(importance, DEFAULT_LOCKOUT[1])
        events.append(
            Event(
                id=f"TE:{calendar_id}",
                provider="tradingeconomics",
                country=country,
                category=category,
                event=event_name,
                importance=importance,
                ts_release_utc=ts_iso,
                lockout_pre=lockout["pre_seconds"],
                lockout_post=lockout["post_seconds"],
            )
        )

    return sorted(events, key=lambda e: (e.ts_release_utc, e.id))


def normalize_te_timestamp(item: Dict[str, object]) -> str:
    raw_date = (item.get("Date") or item.get("DateTime") or item.get("date") or "").strip()
    if not raw_date:
        raise ValueError("missing Date in TE item")

    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid TE Date format: {raw_date}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_ZONE)
    else:
        dt = dt.astimezone(UTC_ZONE)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_category(category_raw: str, event_name: str) -> str:
    source = f"{category_raw} {event_name}".lower()
    mapping = [
        ("inflation", "INFLATION"),
        ("cpi", "INFLATION"),
        ("consumer price", "INFLATION"),
        ("pce", "PCE"),
        ("employment", "LABOR"),
        ("jobless", "LABOR"),
        ("unemployment", "LABOR"),
        ("payroll", "LABOR"),
        ("labor", "LABOR"),
        ("gdp", "GDP"),
        ("gross domestic", "GDP"),
        ("pmi", "PMI"),
        ("purchasing managers", "PMI"),
        ("fomc", "FED"),
        ("federal reserve", "FED"),
        ("interest rate", "RATES"),
        ("rate decision", "RATES"),
        ("ecb", "ECB"),
        ("central bank", "RATES"),
        ("opec", "OPEC"),
        ("inventory", "INVENTORIES"),
        ("inventories", "INVENTORIES"),
        ("eia", "ENERGY_EIA"),
        ("energy", "ENERGY_EIA"),
    ]
    for needle, value in mapping:
        if needle in source:
            return value
    normalized = category_raw.strip().upper().replace(" ", "_").replace("-", "_")
    return normalized or "UNKNOWN"


def fetch_fred_events(d1: datetime, d2: datetime, api_key: str) -> List[Event]:
    max_date = d1 + timedelta(days=30)
    events: List[Event] = []
    for label, release_id in FRED_RELEASE_IDS.items():
        url = (
            "https://api.stlouisfed.org/fred/releases/dates?release_id={rid}&file_type=json&api_key={key}"
        ).format(rid=release_id, key=urllib.parse.quote(api_key))
        try:
            raw = fetch_json(url)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network guard
            raise RuntimeError(f"FRED HTTP error for {label}: {exc.code}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network guard
            raise RuntimeError(f"FRED network error for {label}: {exc.reason}") from exc

        release_dates = raw.get("release_dates", []) if isinstance(raw, dict) else []
        for entry in release_dates:
            date_str = entry.get("date") if isinstance(entry, dict) else None
            if not date_str:
                continue
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC_ZONE)
            except ValueError:
                continue
            if date_obj < d1 or date_obj > max_date:
                continue
            ts = combine_et_time(date_obj.date(), fred_release_time(label))
            importance = fred_importance(label, entry)
            lockout = DEFAULT_LOCKOUT.get(importance, DEFAULT_LOCKOUT[1])
            events.append(
                Event(
                    id=f"FRED:{label}:{date_str}",
                    provider="fred",
                    country="US",
                    category=fred_category(label),
                    event=fred_event_name(label, entry),
                    importance=importance,
                    ts_release_utc=ts,
                    lockout_pre=lockout["pre_seconds"],
                    lockout_post=lockout["post_seconds"],
                )
            )

    return sorted(events, key=lambda e: (e.ts_release_utc, e.id))


def combine_et_time(date_obj, time_tuple) -> str:
    hour, minute = time_tuple
    dt_et = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, tzinfo=ET_ZONE)
    dt_utc = dt_et.astimezone(UTC_ZONE)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def fred_release_time(label: str) -> tuple:
    # Default release time 08:30 ET unless stated otherwise.
    return (8, 30)


def fred_category(label: str) -> str:
    return {
        "CPI": "INFLATION",
        "EMPLOYMENT_SITUATION": "LABOR",
        "GDP": "GROWTH",
        "PCE": "PCE",
    }[label]


def fred_event_name(label: str, entry: Dict[str, object]) -> str:
    names = {
        "CPI": "Consumer Price Index (CPI)",
        "EMPLOYMENT_SITUATION": "Employment Situation",
        "GDP": gdp_event_name(entry),
        "PCE": "Personal Income and Outlays (PCE)",
    }
    return names[label]


def gdp_event_name(entry: Dict[str, object]) -> str:
    release_name = (entry.get("release_name") or "GDP") if isinstance(entry, dict) else "GDP"
    return release_name or "GDP"


def fred_importance(label: str, entry: Dict[str, object]) -> int:
    if label in {"CPI", "EMPLOYMENT_SITUATION"}:
        return 3
    if label == "GDP":
        name = (entry.get("release_name") or "").lower()
        if "third" in name:
            return 2
        return 3
    if label == "PCE":
        return 2
    return 1


def load_asset_sensitivity(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_lockout_by_asset(events: Iterable[Event], sensitivity: Dict[str, Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    lockouts: Dict[str, List[Dict[str, object]]] = {asset: [] for asset in sensitivity}
    for event in events:
        preset_key = "HIGH" if event.importance >= 3 else "MED" if event.importance == 2 else "LOW"
        for asset, cfg in sensitivity.items():
            countries = cfg.get("countries", [])
            categories = cfg.get("categories", [])
            presets = cfg.get("presets", {})
            match_country = event.country in countries
            match_category = event.category in categories
            if not (match_country or match_category):
                continue
            preset = presets.get(preset_key)
            if preset:
                pre_seconds = preset.get("pre", event.lockout_pre)
                post_seconds = preset.get("post", event.lockout_post)
            else:
                default_preset = DEFAULT_LOCKOUT.get(event.importance, DEFAULT_LOCKOUT[1])
                pre_seconds = default_preset.get("pre_seconds", event.lockout_pre)
                post_seconds = default_preset.get("post_seconds", event.lockout_post)
            lockouts[asset].append(
                {
                    "id": event.id,
                    "ts_release_utc": event.ts_release_utc,
                    "pre_seconds": pre_seconds,
                    "post_seconds": post_seconds,
                    "label": event.event,
                }
            )
    for entries in lockouts.values():
        entries.sort(key=lambda e: (e["ts_release_utc"], e["id"]))
    return lockouts


def write_json_if_changed(path: str, data: Dict[str, object]) -> bool:
    new_payload = json.dumps(data, indent=2, sort_keys=False) + "\n"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            current = fh.read()
    except FileNotFoundError:
        current = None
    if current == new_payload:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_payload)
    return True


def main() -> int:
    d1_dt = utc_today()
    d2_dt = d1_dt + timedelta(days=1)
    d1_str = d1_dt.strftime("%Y-%m-%d")
    d2_str = d2_dt.strftime("%Y-%m-%d")

    te_key = os.environ.get("TE_KEY", "").strip()
    fred_key = os.environ.get("FRED_API_KEY", "").strip()

    events: List[Event] = []
    te_error: Optional[str] = None
    if te_key:
        try:
            events = fetch_te_events(d1_str, d2_str, te_key)
        except RuntimeError as exc:
            te_error = str(exc)
    else:
        te_error = "TE_KEY not provided"

    if not events:
        if not fred_key:
            raise SystemExit("No macro data fetched: TE unavailable and FRED_API_KEY missing")
        if te_error:
            print(f"TradingEconomics unavailable: {te_error}")
        events = fetch_fred_events(d1_dt, d2_dt, fred_key)

    generated_ts = datetime.now(UTC_ZONE).strftime("%Y-%m-%dT%H:%M:%SZ")
    calendar_payload = {
        "generated_utc": generated_ts,
        "d1": d1_str,
        "d2": d2_str,
        "items": [event.to_calendar_dict() for event in events],
    }
    calendar_path = os.path.join("data", "macro", "calendar.json")
    calendar_changed = write_json_if_changed(calendar_path, calendar_payload)

    sensitivity_path = os.path.join("config", "asset_sensitivity.json")
    sensitivity = load_asset_sensitivity(sensitivity_path)
    lockout_payload = build_lockout_by_asset(events, sensitivity)
    lockout_path = os.path.join("data", "macro", "lockout_by_asset.json")
    lockout_changed = write_json_if_changed(lockout_path, lockout_payload)

    print(
        f"Wrote {len(events)} events, {sum(len(v) for v in lockout_payload.values())} asset lockouts"
    )

    if not (calendar_changed or lockout_changed):
        print("No changes detected in macro outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
