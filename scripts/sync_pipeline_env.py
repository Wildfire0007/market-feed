#!/usr/bin/env python3
"""Ensure the pipeline runner uses the in-repo code and artefacts."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.reset_notify_state import normalise_notify_state_file

LOGGER = logging.getLogger("pipeline_env")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _purge_bytecode(module_name: str, repo_root: Path) -> None:
    source = repo_root / f"{module_name}.py"
    if source.exists():
        try:
            cache_path_str = importlib.util.cache_from_source(str(source))
        except Exception:
            cache_path_str = None
        if cache_path_str:
            cache_path = Path(cache_path_str)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except FileNotFoundError:
                    pass
            parent = cache_path.parent
            if parent.name == "__pycache__":
                try:
                    parent.rmdir()
                except OSError:
                    pass
    legacy = repo_root / f"{module_name}.pyc"
    if legacy.exists():
        legacy.unlink()
    importlib.invalidate_caches()
    if module_name in sys.modules:
        del sys.modules[module_name]


def _gather_git_metadata(repo_root: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return metadata

    if not commit:
        return metadata

    metadata["commit"] = commit

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        branch = ""
    if branch and branch != "HEAD":
        metadata["branch"] = branch

    try:
        describe = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        describe = ""
    if describe and describe != commit:
        metadata["describe"] = describe

    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        status = ""
    if status:
        metadata["dirty"] = True

    return metadata


def _verify_module(module_name: str, repo_root: Path) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Could not locate module '{module_name}' on PYTHONPATH")
    module_path = Path(spec.origin).resolve()
    if repo_root not in module_path.parents and module_path != repo_root / f"{module_name}.py":
        raise RuntimeError(
            f"Module '{module_name}' resolved to unexpected location: {module_path}"
        )
    return module_path


def _allow_missing_models_flag(args_allow: bool) -> bool:
    if args_allow:
        return True
    env_flag = os.getenv("ALLOW_MISSING_MODELS", "").strip().lower()
    return env_flag in {"1", "true", "yes", "on"}


def _check_models(public_dir: Path, allow_missing: bool) -> None:
    os.environ.setdefault("PUBLIC_DIR", str(public_dir))
    from config.analysis_settings import ASSETS  # noqa: WPS433 - runtime import
    from ml_model import missing_model_artifacts  # noqa: WPS433 - runtime import

    missing = missing_model_artifacts(ASSETS)
    if missing:
        assets = ", ".join(sorted(missing.keys()))
        if allow_missing:
            LOGGER.warning("Missing ML model artefacts (ignored): %s", assets)
            return
        raise SystemExit(f"Missing ML model artefacts: {assets}")
    LOGGER.info("Validated ML model artefacts for %d assets", len(ASSETS))


def _write_build_metadata(
    public_dir: Path,
    metadata: Dict[str, Any],
    module_path: Path,
    repo_root: Path,
    module_name: str,
) -> Path:
    monitor_dir = public_dir / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "module": module_name,
        "module_path": str(module_path),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "git": metadata,
        "sys_path": sys.path[:8],
    }
    path = monitor_dir / "build_info.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _extract_bid_ask(frame: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bid_keys = ("bid", "best_bid", "bid_price", "b")
    ask_keys = ("ask", "best_ask", "ask_price", "a")
    bid_val: Optional[float] = None
    ask_val: Optional[float] = None
    for key in bid_keys:
        if key in frame and frame.get(key) is not None:
            try:
                bid_val = float(frame[key])
            except (TypeError, ValueError):
                bid_val = None
            if bid_val is not None:
                break
    for key in ask_keys:
        if key in frame and frame.get(key) is not None:
            try:
                ask_val = float(frame[key])
            except (TypeError, ValueError):
                ask_val = None
            if ask_val is not None:
                break
    return bid_val, ask_val


def _mark_snapshot_stale(
    payload: Dict[str, Any],
    reason: str,
    *,
    now: datetime,
    age_seconds: Optional[float] = None,
) -> bool:
    changed = False
    if payload.get("ok") is not False:
        payload["ok"] = False
        changed = True
    if payload.get("stale_reason") != reason:
        payload["stale_reason"] = reason
        changed = True
    payload["stale_marked_at_utc"] = now.isoformat()
    payload["suggested_refresh"] = "now"
    if age_seconds is not None:
        payload["stale_age_seconds"] = float(age_seconds)
    snapshot_utc = payload.pop("utc", None)
    if snapshot_utc is not None:
        payload["stale_snapshot_utc"] = snapshot_utc
        changed = True
    retrieved_utc = payload.pop("retrieved_at_utc", None)
    if retrieved_utc is not None:
        payload["stale_snapshot_retrieved_at_utc"] = retrieved_utc
        changed = True
    for field in ("price", "frames", "statistics", "forced", "force_reason"):
        if field in payload:
            payload.pop(field, None)
            changed = True
    return changed


def _sanitize_spot_snapshots(public_dir: Path, *, now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    updated = 0
    for snapshot_path in public_dir.glob("*/spot_realtime.json"):
        if not snapshot_path.is_file():
            continue
        payload = _load_json(snapshot_path)
        if not isinstance(payload, dict) or not payload:
            continue
        forced = bool(payload.get("forced")) or payload.get("stale_reason") in {"missing_bid_ask", "older_than_300s"}        
        reason: Optional[str] = None
        frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
        last_frame: Dict[str, Any] = frames[-1] if frames else {}
        bid_val, ask_val = _extract_bid_ask(last_frame)
        if forced and (bid_val is None or ask_val is None):
            reason = "missing_bid_ask"
        snapshot_ts = _parse_iso_timestamp(payload.get("utc") or payload.get("retrieved_at_utc"))
        age_seconds: Optional[float] = None
        if snapshot_ts is not None:
            age_seconds = (now - snapshot_ts).total_seconds()
            if age_seconds < 0:
                age_seconds = 0.0
        if snapshot_ts is None:
            reason = reason or "missing_timestamp"
        elif age_seconds is not None and age_seconds > 300:
            reason = reason or "older_than_300s"
        if forced is False and reason not in {"missing_timestamp", "older_than_300s"}:
            continue
        if not reason:
            continue
        if _mark_snapshot_stale(payload, reason, now=now, age_seconds=age_seconds):
            with snapshot_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            updated += 1
    return updated


def _sanitize_order_flow_summary_files(public_dir: Path) -> int:
    updated = 0
    summary_paths: List[Path] = []
    root_summary = public_dir / "analysis_summary.json"
    if root_summary.exists():
        summary_paths.append(root_summary)
    for path in public_dir.glob("*/analysis_summary.json"):
        if path.is_file():
            summary_paths.append(path)

    for path in summary_paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        assets = payload.get("assets")
        if not isinstance(assets, dict):
            continue
        changed = False
        for meta in assets.values():
            if not isinstance(meta, dict):
                continue
            metrics = meta.get("order_flow_metrics")
            if not isinstance(metrics, dict):
                continue
            status = str(metrics.get("status") or "").strip()
            if not status:
                values = [metrics.get("imbalance"), metrics.get("pressure"), metrics.get("delta_volume")]
                if all(value in (None, 0, 0.0) for value in values):
                    metrics["status"] = "volume_unavailable"
                    metrics["pressure"] = None
                    metrics["delta_volume"] = None
                    changed = True
            elif status in {"volume_unavailable", "unavailable"}:
                for key in ("pressure", "delta_volume", "aggressor_ratio"):
                    if metrics.get(key) not in (None, 0, 0.0):
                        metrics[key] = None
                        changed = True
        if changed:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            updated += 1
    return updated


def _prune_archive_snapshots(root: Path, retention: int) -> None:
    if retention <= 0 or not root.exists():
        return
    snapshots: List[Path] = [candidate for candidate in root.iterdir() if candidate.is_dir()]
    snapshots.sort(key=lambda candidate: candidate.name, reverse=True)
    for obsolete in snapshots[retention:]:
        shutil.rmtree(obsolete, ignore_errors=True)


def _archive_analysis_summaries(
    public_dir: Path,
    *,
    now: Optional[datetime] = None,
    retention: int = 7,
) -> int:
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive_root = public_dir / "archive" / "analysis_summary"

    summary_paths: List[Path] = []
    root_summary = public_dir / "analysis_summary.json"
    if root_summary.exists():
        summary_paths.append(root_summary)
    for path in public_dir.glob("*/analysis_summary.json"):
        if path.is_file():
            summary_paths.append(path)

    archived = 0
    for path in summary_paths:
        relative = path.relative_to(public_dir)
        target_dir = archive_root / timestamp / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / relative.name
        try:
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(path), str(target_path))
            archived += 1
        except Exception:
            LOGGER.warning("Failed to archive %s", path)

    _prune_archive_snapshots(archive_root, retention)
    return archived


def _archive_stale_feature_monitors(
    public_dir: Path,
    *,
    now: Optional[datetime] = None,
    max_age_hours: float = 36.0,
    retention: int = 5,
) -> int:
    monitor_dir = public_dir / "ml_features" / "monitoring"
    if not monitor_dir.exists():
        return 0

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0.0, float(max_age_hours)))
    archive_root = public_dir / "archive" / "feature_monitor"
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    archived = 0
    for json_path in monitor_dir.glob("*_monitor.json"):
        if not json_path.is_file():
            continue
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = None
        generated = None
        if isinstance(payload, dict):
            generated = _parse_iso_timestamp(payload.get("generated_utc"))
        if generated is not None and generated >= cutoff:
            continue

        relative = json_path.relative_to(public_dir)
        target_dir = archive_root / timestamp / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_json = target_dir / relative.name
        try:
            if target_json.exists():
                target_json.unlink()
            shutil.move(str(json_path), str(target_json))
            archived += 1
        except Exception:
            LOGGER.warning("Failed to archive %s", json_path)
            continue

        csv_path = json_path.with_suffix(".csv")
        if csv_path.exists():
            rel_csv = csv_path.relative_to(public_dir)
            csv_dir = archive_root / timestamp / rel_csv.parent
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_target = csv_dir / rel_csv.name
            try:
                if csv_target.exists():
                    csv_target.unlink()
                shutil.move(str(csv_path), str(csv_target))
            except Exception:
                LOGGER.warning("Failed to archive %s", csv_path)

    _prune_archive_snapshots(archive_root, retention)
    return archived


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _maybe_reset_notify_state(
    public_dir: Path,
    *,
    now: Optional[datetime] = None,
    stale_hours: float = 72.0,
):
    from scripts.reset_dashboard_state import reset_notify_state_file

    now = now or datetime.now(timezone.utc)
    state_path = public_dir / "_notify_state.json"
    normalise_result = normalise_notify_state_file(
        path=state_path,
        now=now,
        reset_counts=True,
        reason="pipeline_cleanup_normalise",
    )
    if normalise_result.changed:
        LOGGER.info(
            "notify_state_normalised",
            extra={
                "path": str(state_path),
                "assets_reset": normalise_result.reset_assets,
                "cleared_counts": normalise_result.cleared_counts,
            },
        )
    payload = _load_json(state_path)
    latest = None
    if isinstance(payload, dict):
        meta = payload.get("_meta")
        if isinstance(meta, dict):
            latest = _parse_iso_timestamp(meta.get("last_heartbeat_utc")) or latest
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            for key in ("last_sent", "last_sent_utc", "last_notification_utc", "timestamp"):
                candidate = _parse_iso_timestamp(value.get(key))
                if candidate and (latest is None or candidate > latest):
                    latest = candidate

    if latest is not None and latest >= now - timedelta(hours=float(stale_hours)):
        return None

    return reset_notify_state_file(
        path=state_path,
        now=now,
        backup_dir=public_dir / "monitoring" / "reset_backups",
        reason="pipeline_cleanup",
    )


def _maybe_reset_status(
    public_dir: Path,
    *,
    now: Optional[datetime] = None,
    stale_hours: float = 36.0,
):
    from scripts.reset_dashboard_state import reset_status_file

    now = now or datetime.now(timezone.utc)
    status_path = public_dir / "status.json"
    payload = _load_json(status_path)
    generated = None
    if isinstance(payload, dict):
        generated = _parse_iso_timestamp(payload.get("generated_utc"))
    if generated is not None and generated >= now - timedelta(hours=float(stale_hours)):
        return None

    return reset_status_file(
        path=status_path,
        now=now,
        backup_dir=public_dir / "monitoring" / "reset_backups",
        reason="pipeline_cleanup",
    )


def _prune_anchor_state(
    public_dir: Path,
    *,
    now: Optional[datetime] = None,
    max_age_hours: float = 72.0,
):
    from scripts.reset_dashboard_state import reset_anchor_state_file

    now = now or datetime.now(timezone.utc)
    return reset_anchor_state_file(
        max_age_hours=max_age_hours,
        now=now,
        backup_dir=public_dir / "monitoring" / "reset_backups",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronise pipeline runtime state")
    parser.add_argument("--public-dir", default=os.getenv("PUBLIC_DIR", "public"))
    parser.add_argument("--module", default="analysis", help="Module to validate on PYTHONPATH")
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Do not abort when ML model artefacts are missing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise (warnings and errors only)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.quiet:
        LOGGER.setLevel(logging.WARNING)

    repo_root = _repo_root()
    LOGGER.info("Pipeline repo root: %s", repo_root)
    LOGGER.info("Python executable: %s", sys.executable)

    _ensure_repo_on_path(repo_root)
    _purge_bytecode(args.module, repo_root)

    try:
        module_path = _verify_module(args.module, repo_root)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Using %s module from %s", args.module, module_path)

    git_meta = _gather_git_metadata(repo_root)
    if git_meta:
        dirty = " (dirty)" if git_meta.get("dirty") else ""
        LOGGER.info("Repository commit: %s%s", git_meta.get("commit"), dirty)
        if git_meta.get("describe"):
            LOGGER.info("Repository describe: %s", git_meta.get("describe"))
        if git_meta.get("branch"):
            LOGGER.info("Repository branch: %s", git_meta.get("branch"))

    public_dir = Path(args.public_dir).expanduser().resolve()
    LOGGER.info("Public artefact directory: %s", public_dir)

    allow_missing = _allow_missing_models_flag(args.allow_missing_models)
    try:
        _check_models(public_dir, allow_missing)
    except SystemExit as exc:
        LOGGER.error("%s", exc)
        return 2

    now = datetime.now(timezone.utc)
    stale_snapshots = _sanitize_spot_snapshots(public_dir, now=now)
    if stale_snapshots:
        LOGGER.info("Marked %d realtime snapshot(s) as stale", stale_snapshots)

    sanitized_summaries = _sanitize_order_flow_summary_files(public_dir)
    if sanitized_summaries:
        LOGGER.info("Normalised order-flow metrics in %d summary file(s)", sanitized_summaries)
        
    archived = _archive_analysis_summaries(public_dir, now=now)
    if archived:
        LOGGER.info("Archived %d analysis summary snapshot(s)", archived)

    monitor_archived = _archive_stale_feature_monitors(public_dir, now=now)
    if monitor_archived:
        LOGGER.info("Archived %d stale feature monitor snapshot(s)", monitor_archived)

    notify_reset = _maybe_reset_notify_state(public_dir, now=now)
    if notify_reset and getattr(notify_reset, "changed", False):
        LOGGER.info("%s", getattr(notify_reset, "message", "reset notify state"))

    status_reset = _maybe_reset_status(public_dir, now=now)
    if status_reset and getattr(status_reset, "changed", False):
        LOGGER.info("%s", getattr(status_reset, "message", "reset status"))

    anchor_reset = _prune_anchor_state(public_dir, now=now)
    if anchor_reset and (getattr(anchor_reset, "removed", 0) or getattr(anchor_reset, "changed", False)):
        LOGGER.info("%s", getattr(anchor_reset, "message", "refreshed anchor state"))

    build_info_path = _write_build_metadata(public_dir, git_meta, module_path, repo_root, args.module)
    LOGGER.info("Wrote build metadata to %s", build_info_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
