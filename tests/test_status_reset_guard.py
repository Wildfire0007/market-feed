import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predeploy_guard import StatusValidationError, StatusValidator


def test_status_reset_payload_blocks_guard():
    payload = {
        "ok": False,
        "status": "reset",
        "assets": {},
    }
    validator = StatusValidator(payload)
    with pytest.raises(StatusValidationError):
        validator.validate()
