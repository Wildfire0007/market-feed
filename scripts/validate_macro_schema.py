#!/usr/bin/env python3
"""Validate macro calendar and lockout JSON outputs for schema safety."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Dict, List

ISO_FORMATS = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_iso_timestamp(value: str) -> bool:
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return True
        except ValueError:
            continue
    return False


def _validate_calendar_item(item: Dict[str, object], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    required = [
        "id",
        "provider",
        "type",
        "country",
        "category",
        "event",
        "importance",
        "ts_release_utc",
        "lockout",
    ]
    for key in required:
        if key not in item:
            errors.append(f"{path}: items[{idx}] missing {key}")
    if not isinstance(item.get("id"), str):
        errors.append(f"{path}: items[{idx}].id must be string")
    if not isinstance(item.get("provider"), str):
        errors.append(f"{path}: items[{idx}].provider must be string")
    if not isinstance(item.get("type"), str):
        errors.append(f"{path}: items[{idx}].type must be string")
    if not isinstance(item.get("country"), str):
        errors.append(f"{path}: items[{idx}].country must be string")
    if not isinstance(item.get("category"), str):
        errors.append(f"{path}: items[{idx}].category must be string")
    if not isinstance(item.get("event"), str):
        errors.append(f"{path}: items[{idx}].event must be string")
    if not isinstance(item.get("importance"), int):
        errors.append(f"{path}: items[{idx}].importance must be int")
    ts_value = item.get("ts_release_utc")
    if not isinstance(ts_value, str) or not _is_iso_timestamp(ts_value):
        errors.append(f"{path}: items[{idx}].ts_release_utc must be ISO timestamp")
    lockout = item.get("lockout")
    if not isinstance(lockout, dict):
        errors.append(f"{path}: items[{idx}].lockout must be object")
    else:
        if not isinstance(lockout.get("pre_seconds"), int):
            errors.append(f"{path}: items[{idx}].lockout.pre_seconds must be int")
        if not isinstance(lockout.get("post_seconds"), int):
            errors.append(f"{path}: items[{idx}].lockout.post_seconds must be int")
    return errors


def validate_calendar(path: str) -> List[str]:
    payload = _load_json(path)
    errors: List[str] = []
    for key in ["generated_utc", "d1", "d2", "items"]:
        if key not in payload:
            errors.append(f"{path}: missing {key}")
    if not isinstance(payload.get("generated_utc"), str) or not _is_iso_timestamp(
        payload.get("generated_utc", "")
    ):
        errors.append(f"{path}: generated_utc must be ISO timestamp")
    if not isinstance(payload.get("d1"), str):
        errors.append(f"{path}: d1 must be string")
    if not isinstance(payload.get("d2"), str):
        errors.append(f"{path}: d2 must be string")
    items = payload.get("items")
    if not isinstance(items, list):
        errors.append(f"{path}: items must be list")
        return errors
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{path}: items[{idx}] must be object")
            continue
        errors.extend(_validate_calendar_item(item, path, idx))
    return errors


def _validate_lockout_entry(entry: Dict[str, object], path: str, asset: str, idx: int) -> List[str]:
    errors: List[str] = []
    required = ["id", "ts_release_utc", "pre_seconds", "post_seconds", "label"]
    for key in required:
        if key not in entry:
            errors.append(f"{path}: {asset}[{idx}] missing {key}")
    if not isinstance(entry.get("id"), str):
        errors.append(f"{path}: {asset}[{idx}].id must be string")
    ts_value = entry.get("ts_release_utc")
    if not isinstance(ts_value, str) or not _is_iso_timestamp(ts_value):
        errors.append(f"{path}: {asset}[{idx}].ts_release_utc must be ISO timestamp")
    if not isinstance(entry.get("pre_seconds"), int):
        errors.append(f"{path}: {asset}[{idx}].pre_seconds must be int")
    if not isinstance(entry.get("post_seconds"), int):
        errors.append(f"{path}: {asset}[{idx}].post_seconds must be int")
    if not isinstance(entry.get("label"), str):
        errors.append(f"{path}: {asset}[{idx}].label must be string")
    return errors


def validate_lockout(path: str) -> List[str]:
    payload = _load_json(path)
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"{path}: root must be object"]
    for asset, entries in payload.items():
        if not isinstance(entries, list):
            errors.append(f"{path}: {asset} must be list")
            continue
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{path}: {asset}[{idx}] must be object")
                continue
            errors.extend(_validate_lockout_entry(entry, path, asset, idx))
    return errors


def main() -> int:
    calendar_errors = validate_calendar("data/macro/calendar.json")
    lockout_errors = validate_lockout("data/macro/lockout_by_asset.json")
    errors = calendar_errors + lockout_errors
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print("Macro schema validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
