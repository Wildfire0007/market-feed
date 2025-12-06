import json
from pathlib import Path

import jsonschema


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_status_schema_accepts_public_snapshot(tmp_path):
    schema = _load(Path("config/status_schema.json"))
    payload = {
        "ok": True,
        "generated_utc": "2024-01-01T00:00:00Z",
        "assets": {"BTCUSD": {"ok": True}},
        "notes": [{"type": "reset", "message": "ok", "reset_utc": "2024-01-01T00:00:00Z"}],
    }
    jsonschema.validate(instance=payload, schema=schema)

    invalid = {"ok": True}
    try:
        jsonschema.validate(instance=invalid, schema=schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("schema should reject missing assets")


def test_pipeline_timing_schema(tmp_path):
    schema = _load(Path("config/monitoring_schema.json"))
    sample = _load(Path("public/monitoring/pipeline_timing.json"))
    jsonschema.validate(instance=sample, schema=schema)

    broken = {"updated_utc": "2024-01-01T00:00:00Z", "trading": {}}
    try:
        jsonschema.validate(instance=broken, schema=schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("schema should flag missing trading.start")
