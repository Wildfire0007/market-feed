import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_precision_summary_log_includes_block_reasons():
    handler = ListHandler()
    logger = logging.getLogger("analysis.precision.summary")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        analysis._log_precision_gate_summary(
            "BTCUSD",
            {
                "flow_ready": False,
                "trigger_ready": False,
                "score_ready": False,
                "flow_blockers": ["imbalance -0.40", "latency>600"],
            },
            ["imbalance -0.40", "latency>600"],
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    assert handler.records, "Precision összefoglaló log nem íródott ki"
    record = handler.records[-1]
    assert getattr(record, "precision_block_reason", "") == "imbalance -0.40,latency>600"
    assert record.flow_ready is False
    assert record.trigger_ready is False


def test_precision_gate_log_handles_dst_transition():
    handler = ListHandler()
    logger = logging.getLogger("analysis.precision.dst")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        analysis._emit_precision_gate_log(
            "BTCUSD",
            "precision_flow",
            False,
            "flow_blocked",
            order_flow_metrics={"imbalance_z": 0.0},
            tick_order_flow={"window_minutes": 30, "source": "test"},
            latency_seconds={"k1m": 300},
            precision_plan={
                "order_flow_strength": 0.0,
                "trigger_state": "standby",
                "trigger_ready": False,
                "score": 41.2,
            },
            logger=logger,
            timestamp=datetime(2024, 3, 31, 1, 30, tzinfo=timezone.utc),
        )
    finally:
        logger.removeHandler(handler)

    assert handler.records, "Precision log hiányzik DST vizsgálathoz"
    record = handler.records[-1]
    assert record.timestamp_utc.startswith("2024-03-31 01:30")
    assert record.timestamp_cet.startswith("2024-03-31 03:30"), record.timestamp_cet
    assert record.ofi_window_minutes == 30
