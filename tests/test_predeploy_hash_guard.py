import json
from pathlib import Path

import pytest

from scripts.predeploy_guard import HashValidator, StatusValidationError, _compute_sha256


def test_hash_validator_accepts_matching_manifest(tmp_path):
    target = tmp_path / "critical.txt"
    target.write_text("alpha", encoding="utf-8")
    digest = _compute_sha256(target)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {str(target): {"sha256": digest}}}), encoding="utf-8")

    validator = HashValidator(manifest)
    validator.validate()  # no exception


def test_hash_validator_rejects_mismatch(tmp_path):
    target = tmp_path / "critical.txt"
    target.write_text("alpha", encoding="utf-8")
    digest = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {str(target): {"sha256": digest}}}), encoding="utf-8")

    validator = HashValidator(manifest)
    with pytest.raises(StatusValidationError):
        validator.validate()
