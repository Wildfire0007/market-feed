import logging
from datetime import datetime, timezone

import analysis


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_precision_gate_log_captures_missing_ofi():
    handler = ListHandler()
    logger = logging.getLogger("analysis.precision_test")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        analysis._emit_precision_gate_log(
            "NVDA",
            "can_enter_core",
            False,
            "ofi_missing",
            order_flow_metrics={},
            tick_order_flow=None,
            latency_seconds={"k1m": "720"},
            precision_plan={
                "order_flow_strength": None,
                "trigger_state": "idle",
                "trigger_ready": False,
                "score": 41.2,
            },
            logger=logger,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            extra={"missing": ["ofi"]},
        )
    finally:
        logger.removeHandler(handler)

    assert handler.records, "A precision lognak legalább egy rekordot kellett írnia"
    record = handler.records[-1]
    assert record.asset == "NVDA"
    assert record.gate == "can_enter_core"
    assert record.decision is False
    assert record.reason_code == "ofi_missing"
    assert record.ofi_present is False
    assert record.ofi_age_seconds == 720
    assert record.ofi_window_minutes == analysis.OFI_Z_LOOKBACK
    assert record.precision_score == 41.2
    assert record.missing == ["ofi"]
    assert record.timestamp_utc.startswith("2024-01-01 12:00")
    assert record.timestamp_cet.startswith("2024-01-01")
