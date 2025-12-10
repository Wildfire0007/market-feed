"""Generate a Markdown summary of configured trading session windows per asset."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import analysis_settings

LOCAL_TZ = ZoneInfo("Europe/Budapest")


Window = Tuple[int, int, int, int]


def _minute_to_hhmm(minute: int) -> str:
    minute = max(0, min(minute, 23 * 60 + 59))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _format_window(window: Window) -> str:
    sh, sm, eh, em = window
    return f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d} UTC"


def _format_window_local(window: Window, ref_day: date, tz: ZoneInfo) -> str:
    sh, sm, eh, em = window
    start_utc = datetime.combine(ref_day, time(sh, sm, tzinfo=timezone.utc))
    end_utc = datetime.combine(ref_day, time(eh, em, tzinfo=timezone.utc))
    start_local = start_utc.astimezone(tz)
    end_local = end_utc.astimezone(tz)
    return f"{start_local.strftime('%H:%M')}–{end_local.strftime('%H:%M')} {tz.key}"


def _format_daily_breaks(breaks: Iterable[Tuple[int, int]], ref_day: date, tz: ZoneInfo) -> List[str]:
    formatted: List[str] = []
    for start, end in breaks:
        start_dt = datetime.combine(ref_day, time(0, 0, tzinfo=timezone.utc)) + timedelta(minutes=start)
        end_dt = datetime.combine(ref_day, time(0, 0, tzinfo=timezone.utc)) + timedelta(minutes=end)
        formatted.append(
            f"{_minute_to_hhmm(start)}–{_minute_to_hhmm(end)} UTC"
            f" ({start_dt.astimezone(tz).strftime('%H:%M')}–{end_dt.astimezone(tz).strftime('%H:%M')} {tz.key})"
        )
    return formatted


def _format_weekdays(weekdays: Sequence[int]) -> str:
    names = ["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"]
    return ", ".join(names[idx] for idx in weekdays if 0 <= idx < len(names))


def build_report(ref_day: Optional[date] = None) -> str:
    """Return a Markdown report of trading session settings per asset."""

    cfg = analysis_settings.load_config()
    assets: List[str] = cfg.get("assets", [])
    windows: Mapping[str, Mapping[str, List[Window]]] = cfg.get("session_windows_utc", {})
    time_rules: Mapping[str, Mapping[str, object]] = cfg.get("session_time_rules", {})
    weekdays_cfg: Mapping[str, Sequence[int]] = cfg.get("session_weekdays", {})

    ref_day = ref_day or date.today()

    lines: List[str] = ["# Kereskedési időablakok eszközönként", ""]
    lines.append(f"Referencia nap az időzóna-konverzióhoz: {ref_day.isoformat()}")
    lines.append("")

    for asset in assets:
        lines.append(f"## {asset}")
        asset_windows = windows.get(asset, {}) if isinstance(windows, Mapping) else {}
        entry_windows: Optional[List[Window]] = asset_windows.get("entry") if isinstance(asset_windows, Mapping) else None
        monitor_windows: Optional[List[Window]] = asset_windows.get("monitor") if isinstance(asset_windows, Mapping) else None

        if entry_windows:
            lines.append("- Belépési ablakok:")
            for window in entry_windows:
                lines.append(f"  - {_format_window(window)} ({_format_window_local(window, ref_day, LOCAL_TZ)})")
        else:
            lines.append("- Belépési ablakok: nincs megadva (teljes nap)")

        if monitor_windows:
            lines.append("- Monitorozási ablakok:")
            for window in monitor_windows:
                lines.append(f"  - {_format_window(window)} ({_format_window_local(window, ref_day, LOCAL_TZ)})")
        else:
            lines.append("- Monitorozási ablakok: nincs megadva (teljes nap)")

        rules = time_rules.get(asset, {}) if isinstance(time_rules, Mapping) else {}
        sunday_open = rules.get("sunday_open_minute") if isinstance(rules, Mapping) else None
        friday_close = rules.get("friday_close_minute") if isinstance(rules, Mapping) else None
        daily_breaks = rules.get("daily_breaks") if isinstance(rules, Mapping) else None

        if sunday_open is not None:
            lines.append(f"- Vasárnapi nyitás (perc éjféltől): {_minute_to_hhmm(int(sunday_open))} UTC")
        if friday_close is not None:
            lines.append(f"- Pénteki zárás (perc éjféltől): {_minute_to_hhmm(int(friday_close))} UTC")

        breaks_fmt = _format_daily_breaks(daily_breaks or [], ref_day, LOCAL_TZ)
        if breaks_fmt:
            lines.append("- Napi szünetek:")
            for br in breaks_fmt:
                lines.append(f"  - {br}")
        else:
            lines.append("- Napi szünetek: nincs megadva")

        weekdays = weekdays_cfg.get(asset, []) if isinstance(weekdays_cfg, Mapping) else []
        if weekdays:
            lines.append(f"- Engedélyezett napok: {_format_weekdays(weekdays)}")
        else:
            lines.append("- Engedélyezett napok: nincs korlátozás")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "session_windows.md",
        help="Output Markdown file (default: reports/session_windows.md)",
    )
    parser.add_argument(
        "--ref-date",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Reference date (YYYY-MM-DD) for timezone conversion",
    )
    args = parser.parse_args()

    report = build_report(ref_day=args.ref_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
