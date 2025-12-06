"""Lightweight JSON Schema validator stub for offline tests.

This module implements a minimal subset of ``jsonschema.validate`` sufficient
for the repository's schema checks without external dependencies.
"""
from __future__ import annotations

from typing import Any, Dict, List


class ValidationError(Exception):
    """Raised when a payload does not satisfy the expected schema."""


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _validate(instance: Any, schema: Dict[str, Any], path: str = "") -> None:
    expected_type = schema.get("type")
    if expected_type:
        py_type = _TYPE_MAP.get(expected_type)
        if py_type and not isinstance(instance, py_type):
            raise ValidationError(f"Expected {expected_type} at {path or '<root>'}")

    required: List[str] = schema.get("required") or []
    if required and isinstance(instance, dict):
        for key in required:
            if key not in instance:
                raise ValidationError(f"Missing required key '{key}' at {path or '<root>'}")

    properties: Dict[str, Any] = schema.get("properties") or {}
    if properties and isinstance(instance, dict):
        for key, subschema in properties.items():
            if key in instance:
                _validate(instance[key], subschema, path=f"{path}/{key}" if path else key)

    items_schema = schema.get("items")
    if items_schema and isinstance(instance, list):
        for idx, item in enumerate(instance):
            _validate(item, items_schema, path=f"{path}[{idx}]")


def validate(instance: Any, schema: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
    """Validate ``instance`` against a minimal JSON schema dictionary."""

    _validate(instance, schema)
