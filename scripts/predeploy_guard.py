#!/usr/bin/env python3
"""Validate analysis outputs before deployment.

The guard ensures ``public/status.json`` reports a healthy analysis state
prior to deploying or notifying, with an escape hatch for rollback flows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is importable when executed from the scripts directory
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
from logging_utils import ensure_json_stream_handler

LOGGER = logging.getLogger("market_feed.predeploy")
ensure_json_stream_handler(LOGGER, static_fields={"component": "predeploy"})


DEFAULT_STATUS_PATH = Path("public/status.json")
# Keep the hash manifest under version control so it cannot be overwritten by
# the downloaded analysis artefact. The previous location under ``public/`` was
# wiped before the pre-deploy step, which meant the validation was reading an
# outdated manifest from the artefact rather than the repository.
DEFAULT_HASH_MANIFEST = Path("config/pipeline/hash_manifest.json")


class StatusValidationError(RuntimeError):
    """Raised when the status snapshot fails validation."""


class StatusLoader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
                return payload if isinstance(payload, dict) else {}
        except FileNotFoundError:
            raise StatusValidationError(f"status file missing: {self.path}")
        except json.JSONDecodeError as exc:
            raise StatusValidationError(f"status file invalid JSON: {exc}")
        except Exception as exc:
            raise StatusValidationError(f"status file could not be read: {exc}")


class StatusValidator:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def ensure_ok(self) -> None:
        ok_value = self.payload.get("ok")
        if ok_value is not True:
            raise StatusValidationError(f"status.ok must be true, got {ok_value!r}")

    def ensure_assets(self) -> None:
        assets = self.payload.get("assets")
        if not isinstance(assets, dict) or not assets:
            raise StatusValidationError("status.assets must be a non-empty object")

    def validate(self) -> None:
        self.ensure_ok()
        self.ensure_assets()


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashValidator:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise StatusValidationError(f"hash manifest missing: {self.manifest_path}")
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise StatusValidationError(f"hash manifest invalid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "files" not in payload:
            raise StatusValidationError("hash manifest must contain 'files' map")
        return payload

    def validate(self) -> None:
        payload = self.load_manifest()
        files = payload.get("files") or {}
        if not isinstance(files, dict) or not files:
            raise StatusValidationError("hash manifest files map is empty")
        mismatches: Dict[str, str] = {}
        missing: Dict[str, str] = {}
        for rel_path, meta in files.items():
            expected = None
            if isinstance(meta, dict):
                expected = meta.get("sha256")
            if not expected or not isinstance(expected, str):
                raise StatusValidationError(f"hash manifest missing sha256 for {rel_path}")
            candidate = Path(rel_path)
            if not candidate.is_absolute():
                candidate = _REPO_ROOT / candidate
            if not candidate.exists():
                missing[str(rel_path)] = "missing"
                continue
            actual = _compute_sha256(candidate)
            if actual != expected:
                mismatches[str(rel_path)] = actual
        if missing:
            raise StatusValidationError(f"hash targets missing: {sorted(missing)}")
        if mismatches:
            raise StatusValidationError(
                "hash mismatch for "
                + ", ".join(
                    f"{path} (expected {files[path]['sha256']}, actual {digest})"  # type: ignore[index]
                    for path, digest in sorted(mismatches.items())
                )
            )


class GuardConfig:
    def __init__(self, *, skip: bool, status_path: Path, hash_manifest: Path) -> None:
        self.skip = skip
        self.status_path = status_path
        self.hash_manifest = hash_manifest

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GuardConfig":
        env_skip = os.getenv("PREDEPLOY_SKIP_FOR_ROLLBACK")
        skip = args.allow_rollback or (env_skip not in (None, "", "0", "false", "False"))
        return cls(
            skip=bool(skip),
            status_path=Path(args.status_path),
            hash_manifest=Path(args.hash_manifest),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-path",
        default=str(DEFAULT_STATUS_PATH),
        help="Path to the status.json artefact",
    )
    parser.add_argument(
        "--hash-manifest",
        default=str(DEFAULT_HASH_MANIFEST),
        help="Path to the expected hash manifest",
    )
    parser.add_argument(
        "--allow-rollback",
        action="store_true",
        help="Skip validation when running in rollback/cleanup mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = GuardConfig.from_args(args)

    LOGGER.info(
        "predeploy_guard_start",
        extra={
            "status_path": str(cfg.status_path),
            "skip": cfg.skip,
        },
    )

    if cfg.skip:
        LOGGER.info("predeploy_guard_skipped", extra={"reason": "rollback"})
        return 0

    loader = StatusLoader(cfg.status_path)
    try:
        payload = loader.load()
        validator = StatusValidator(payload)
        validator.validate()
        hash_validator = HashValidator(cfg.hash_manifest)
        hash_validator.validate()
    except StatusValidationError as exc:
        LOGGER.error("predeploy_guard_failed", extra={"error": str(exc)})
        return 1

    LOGGER.info(
        "predeploy_guard_ok",
        extra={
            "status_ok": payload.get("ok"),
            "asset_count": len(payload.get("assets") or {}),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
