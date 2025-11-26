#!/usr/bin/env python3
"""Validate analysis outputs before deployment.

The guard ensures ``public/status.json`` reports a healthy analysis state
prior to deploying or notifying, with an escape hatch for rollback flows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from logging_utils import ensure_json_stream_handler

LOGGER = logging.getLogger("market_feed.predeploy")
ensure_json_stream_handler(LOGGER, static_fields={"component": "predeploy"})


DEFAULT_STATUS_PATH = Path("public/status.json")


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


class GuardConfig:
    def __init__(self, *, skip: bool, status_path: Path) -> None:
        self.skip = skip
        self.status_path = status_path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GuardConfig":
        env_skip = os.getenv("PREDEPLOY_SKIP_FOR_ROLLBACK")
        skip = args.allow_rollback or (env_skip not in (None, "", "0", "false", "False"))
        return cls(skip=bool(skip), status_path=Path(args.status_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-path",
        default=str(DEFAULT_STATUS_PATH),
        help="Path to the status.json artefact",
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
