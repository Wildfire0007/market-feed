#!/usr/bin/env python3
"""Ensure the pipeline runner uses the in-repo code and artefacts."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

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

    build_info_path = _write_build_metadata(public_dir, git_meta, module_path, repo_root, args.module)
    LOGGER.info("Wrote build metadata to %s", build_info_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
