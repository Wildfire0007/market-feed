import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
JOURNAL_FILE = Path(
    os.getenv("PRECISION_MONITOR_JOURNAL", str(PUBLIC_DIR / "journal" / "trade_journal.csv"))
)
MONITOR_DIR = Path(os.getenv("PRECISION_MONITOR_DIR", str(PUBLIC_DIR / "monitoring")))
SUMMARY_PATH = Path(
    os.getenv("PRECISION_MONITOR_SUMMARY", str(MONITOR_DIR / "precision_gates.json"))
)
DAILY_PATH = Path(
    os.getenv("PRECISION_MONITOR_DAILY", str(MONITOR_DIR / "precision_gates_daily.csv"))
)
ASSET_SUMMARY_PATH = Path(
    os.getenv(
        "PRECISION_MONITOR_ASSET_SUMMARY",
        str(MONITOR_DIR / "precision_gates_by_asset.csv"),
    )
)
DEFAULT_LOOKBACK_DAYS = int(os.getenv("PRECISION_MONITOR_LOOKBACK_DAYS", "7"))
SETTINGS_PATH = Path(
    os.getenv("ANALYSIS_SETTINGS_PATH", "config/analysis_settings.json")
)

PRECISION_PATTERNS: Dict[str, str] = {
    "precision_score": "precision_score>=65",
    "precision_flow_alignment": "precision_flow_alignment",
    "precision_trigger_sync": "precision_trigger_sync",
}

NO_ENTRY_SIGNALS = {"no entry", "no-entry", "no_entry"}
TRADE_SIGNALS = {"buy", "sell"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_active_assets(
    settings_path: Path = SETTINGS_PATH,
) -> Optional[Tuple[Set[str], List[str]]]:
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None
    except json.JSONDecodeError:
        return None
    assets = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(assets, list):
        return None
    ordered_assets: List[str] = []
    for asset in assets:
        name = str(asset).strip().upper()
        if not name:
            continue
        if name not in ordered_assets:
            ordered_assets.append(name)
    active_assets = set(ordered_assets)
    if not ordered_assets:
        return None
    return active_assets, ordered_assets


def _safe_pct(numerator: int, denominator: int) -> Optional[float]:
    if denominator:
        return round(float(numerator) / float(denominator), 4)
    return None


def _filter_to_active_assets(
    df: pd.DataFrame, active_assets: Optional[Set[str]]
) -> pd.DataFrame:
    if not active_assets or df.empty:
        return df
    asset_series = df.get("asset")
    if asset_series is None:
        return df
    mask = asset_series.astype(str).str.upper().isin(active_assets)
    return df[mask].copy()


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    working = df.copy()
    working["asset"] = working.get("asset", "").fillna("").astype(str).str.upper()
    working["signal"] = working.get("signal", "").fillna("").astype(str)
    working["signal_normalized"] = working["signal"].str.strip().str.lower()
    working["is_trade"] = working["signal_normalized"].isin(TRADE_SIGNALS)
    working["is_no_entry"] = working["signal_normalized"].isin(NO_ENTRY_SIGNALS)
    working["is_no_entry_flag"] = working["is_no_entry"].astype(int)

    notes_series = working.get("notes")
    if notes_series is None:
        working["_notes"] = ""
    else:
        working["_notes"] = notes_series.fillna("").astype(str)

    flag_columns: Dict[str, str] = {}
    for key, pattern in PRECISION_PATTERNS.items():
        column_name = f"{key}_flag"
        working[column_name] = working["_notes"].str.contains(pattern, case=False, na=False)
        working[column_name] = working[column_name].astype(int)
        flag_columns[key] = column_name

    precision_columns = [flag_columns[key] for key in PRECISION_PATTERNS]
    working["precision_any_flag"] = (
        working[precision_columns].sum(axis=1) > 0
    ).astype(int)
    working["precision_no_entry_flag"] = (
        (working["precision_any_flag"] == 1) & (working["is_no_entry_flag"] == 1)
    ).astype(int)

    working["analysis_timestamp"] = pd.to_datetime(
        working.get("analysis_timestamp"), errors="coerce", utc=True
    )
    working["analysis_day"] = working["analysis_timestamp"].dt.strftime("%Y-%m-%d")

    return working, flag_columns


def _summarise_dataframe(df: pd.DataFrame, flag_columns: Dict[str, str]) -> Dict[str, Any]:
    total_signals = int(len(df))
    precision_blocked = int(df["precision_any_flag"].sum()) if total_signals else 0
    no_entry_total = int(df["is_no_entry_flag"].sum()) if total_signals else 0
    blocked_no_entry = int(df["precision_no_entry_flag"].sum()) if total_signals else 0

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for key, column in flag_columns.items():
        count = int(df[column].sum()) if total_signals else 0
        count_no_entry = (
            int(df.loc[df["is_no_entry_flag"] == 1, column].sum()) if no_entry_total else 0
        )
        metrics[key] = {
            "count": count,
            "share_of_signals": _safe_pct(count, total_signals),
            "share_of_precision_blocked": _safe_pct(count, precision_blocked),
            "share_of_no_entry": _safe_pct(count_no_entry, no_entry_total),
        }

    return {
        "total_signals": total_signals,
        "no_entry_signals": no_entry_total,
        "precision_blocked": precision_blocked,
        "precision_blocked_share": _safe_pct(precision_blocked, total_signals),
        "precision_blocked_no_entry": blocked_no_entry,
        "precision_blocked_no_entry_share": _safe_pct(blocked_no_entry, no_entry_total),
        "metrics": metrics,
    }


def _summarise_assets(
    df: pd.DataFrame,
    flag_columns: Dict[str, str],
    active_assets: Optional[Set[str]] = None,
    ordered_assets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, pd.DataFrame] = {}
    if not df.empty:
        for asset, group in df.groupby("asset", dropna=False):
            name = str(asset).strip().upper()
            grouped[name if name else "UNKNOWN"] = group

    results: List[Dict[str, Any]] = []

    def _summarise(asset_name: str, frame: pd.DataFrame) -> Dict[str, Any]:
        summary = _summarise_dataframe(frame, flag_columns)
        summary["asset"] = asset_name
        return summary

    if ordered_assets:
        empty_like = df.iloc[0:0]
        for asset_name in ordered_assets:
            frame = grouped.get(asset_name, empty_like)
            results.append(_summarise(asset_name, frame))
        return results

    for asset_name, group in grouped.items():
        normalised = asset_name if asset_name else "UNKNOWN"
        if active_assets and normalised not in active_assets:
            continue
        results.append(_summarise(normalised, group))

    results.sort(key=lambda item: item["precision_blocked"], reverse=True)
    return results


def _build_asset_rows(
    df: pd.DataFrame,
    flag_columns: Dict[str, str],
    active_assets: Optional[Set[str]] = None,
    ordered_assets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, pd.DataFrame] = {}
    if not df.empty:
        for asset, group in df.groupby("asset", dropna=False):
            name = str(asset).strip().upper()
            grouped[name if name else "UNKNOWN"] = group

    def _frame_for(asset_name: str) -> pd.DataFrame:
        frame = grouped.get(asset_name)
        if frame is not None:
            return frame
        return df.iloc[0:0]

    def _build(asset_name: str, frame: pd.DataFrame) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "asset": asset_name,
            "total_signals": int(len(frame)),
            "no_entry_signals": int(frame["is_no_entry_flag"].sum()),
            "precision_blocked": int(frame["precision_any_flag"].sum()),
            "precision_blocked_no_entry": int(frame["precision_no_entry_flag"].sum()),
        }
        for key, column in flag_columns.items():
            row[key] = int(frame[column].sum())
        return row

    rows: List[Dict[str, Any]] = []
    if ordered_assets:
        for asset_name in ordered_assets:
            rows.append(_build(asset_name, _frame_for(asset_name)))
        return rows

    for asset_name, frame in grouped.items():
        normalised = asset_name if asset_name else "UNKNOWN"
        if active_assets and normalised not in active_assets:
            continue
        rows.append(_build(normalised, frame))
        
    rows.sort(key=lambda item: item["precision_blocked"], reverse=True)
    return rows


def _build_daily_rows(df: pd.DataFrame, flag_columns: Dict[str, str]) -> List[Dict[str, Any]]:
    daily_df = df.dropna(subset=["analysis_day"])
    if daily_df.empty:
        return []
    grouped = (
        daily_df.groupby("analysis_day")
        .agg(
            total_signals=("analysis_day", "size"),
            no_entry_signals=("is_no_entry_flag", "sum"),
            precision_blocked=("precision_any_flag", "sum"),
            precision_blocked_no_entry=("precision_no_entry_flag", "sum"),
            **{key: (column, "sum") for key, column in flag_columns.items()},
        )
        .reset_index()
        .sort_values("analysis_day")
    )
    records: List[Dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        converted: Dict[str, Any] = {}
        for key, value in row.items():
            if key == "analysis_day":
                converted[key] = value
            else:
                converted[key] = int(value)
        records.append(converted)
    return records


def update_precision_gate_report(
    journal_path: Optional[Path] = None,
    monitor_dir: Optional[Path] = None,
    lookback_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> Optional[Path]:
    journal_file = Path(journal_path or JOURNAL_FILE)
    if not journal_file.exists():
        return None
    try:
        df = pd.read_csv(journal_file)
    except Exception:
        return None
    if df.empty:
        return None

    prepared, flag_columns = _prepare_dataframe(df)
    active_assets_info = _load_active_assets()
    active_asset_set: Optional[Set[str]] = None
    active_asset_order: Optional[List[str]] = None
    if active_assets_info:
        active_asset_set, active_asset_order = active_assets_info
        prepared = _filter_to_active_assets(prepared, active_asset_set)
        
    reference_time = now or _now()
    lookback = lookback_days if lookback_days is not None else DEFAULT_LOOKBACK_DAYS
    if lookback is not None and lookback > 0:
        cutoff = reference_time - timedelta(days=float(lookback))
        prepared_recent = prepared[prepared["analysis_timestamp"].notna()]
        prepared_recent = prepared_recent[prepared_recent["analysis_timestamp"] >= cutoff]
    else:
        prepared_recent = prepared

    if active_asset_set:
        prepared_recent = _filter_to_active_assets(prepared_recent, active_asset_set)
        
    monitor_path = Path(monitor_dir or MONITOR_DIR)
    monitor_path.mkdir(parents=True, exist_ok=True)

    total_summary = _summarise_dataframe(prepared, flag_columns)
    recent_summary = _summarise_dataframe(prepared_recent, flag_columns)
    asset_summaries = _summarise_assets(
        prepared,
        flag_columns,
        active_assets=active_asset_set,
        ordered_assets=active_asset_order,
    )
    daily_records = _build_daily_rows(prepared, flag_columns)

    payload = {
        "generated_utc": reference_time.isoformat(),
        "lookback_days": lookback,
        "total": total_summary,
        "recent": recent_summary,
        "assets": asset_summaries,
        "daily": daily_records,
    }

    summary_target = Path(
        SUMMARY_PATH if monitor_dir is None else monitor_path / SUMMARY_PATH.name
    )
    with summary_target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    daily_target = Path(DAILY_PATH if monitor_dir is None else monitor_path / DAILY_PATH.name)
    if daily_records:
        pd.DataFrame(daily_records).to_csv(daily_target, index=False)
    elif daily_target.exists():
        daily_target.unlink()

    asset_rows = _build_asset_rows(
        prepared,
        flag_columns,
        active_assets=active_asset_set,
        ordered_assets=active_asset_order,
    )
    asset_target = Path(
        ASSET_SUMMARY_PATH if monitor_dir is None else monitor_path / ASSET_SUMMARY_PATH.name
    )
    pd.DataFrame(asset_rows).to_csv(asset_target, index=False)

    return summary_target


__all__ = ["update_precision_gate_report"]
