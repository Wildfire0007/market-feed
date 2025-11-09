"""Shared logging helpers for structured JSON logging."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class JsonFormatter(logging.Formatter):
    """Formatter that emits log records as JSON strings."""

    _RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def __init__(
        self,
        *,
        static_fields: Optional[Mapping[str, Any]] = None,
        timestamp_field: str = "timestamp",
    ) -> None:
        super().__init__()
        self._static_fields = dict(static_fields or {})
        self._timestamp_field = timestamp_field

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = dict(self._static_fields)
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload[self._timestamp_field] = timestamp.isoformat().replace("+00:00", "Z")
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["message"] = record.getMessage()

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in self._RESERVED_ATTRS:
                continue
            if key in payload:
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False)


def ensure_json_file_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int = logging.INFO,
    static_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    """Attach a JSON file handler to ``logger`` pointing to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in logger.handlers:
        if getattr(handler, "_pipeline_log", False):
            return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter(static_fields=static_fields))
    handler._pipeline_log = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    if logger.level > level:
        logger.setLevel(level)
    else:
        logger.setLevel(level)
    logger.propagate = False


def ensure_json_stream_handler(
    logger: logging.Logger,
    *,
    level: int = logging.INFO,
    static_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    """Ensure ``logger`` emits JSON logs to stdout/stderr."""

    for handler in logger.handlers:
        if getattr(handler, "_json_stream", False):
            break
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter(static_fields=static_fields))
        handler._json_stream = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    if logger.level > level:
        logger.setLevel(level)
    else:
        logger.setLevel(level)
    logger.propagate = False


__all__ = [
    "JsonFormatter",
    "ensure_json_file_handler",
    "ensure_json_stream_handler",
]
