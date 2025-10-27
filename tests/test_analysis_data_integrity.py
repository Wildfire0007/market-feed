import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis


class CriticalStalenessTests(unittest.TestCase):
    def test_srty_minute_is_not_critical_when_relaxed(self) -> None:
        session_meta = {"open": True, "within_monitor_window": True, "entry_open": True}
        stale_timeframes = {"k1m": True, "k5m": False, "k1h": False, "k4h": False}
        latencies = {"k1m": 1020, "k5m": 0, "k1h": 0, "k4h": 0}

        flags, reasons = analysis.classify_critical_staleness("SRTY", stale_timeframes, latencies, session_meta)

        self.assertFalse(flags.get("k1m"))
        self.assertTrue(all("k1m" not in reason for reason in reasons))

    def test_spy_minute_is_critical_when_stale(self) -> None:
        session_meta = {"open": True, "within_monitor_window": True, "entry_open": True}
        stale_timeframes = {"k1m": True, "k5m": False, "k1h": False, "k4h": False}
        latencies = {"k1m": 720, "k5m": 0, "k1h": 0, "k4h": 0}

        flags, reasons = analysis.classify_critical_staleness("SPY", stale_timeframes, latencies, session_meta)

        self.assertTrue(flags.get("k1m"))
        self.assertTrue(any("k1m" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
