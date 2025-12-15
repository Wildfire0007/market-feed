"""Quick sanity checks for core-only data gap handling."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import compute_data_integrity_status


def scenario_core_only_k1m_stale() -> bool:
    """Spot + k5m healthy, k1m stale → core should run, precision disabled."""

    critical_flags = {"k1m": True, "k5m": False, "k1h": False, "k4h": False}
    latency_seconds = {"k1m": 720, "k5m": 120, "k1h": 600, "k4h": 2400}
    core_ok, precision_ok, precision_disabled, reason_map = compute_data_integrity_status(
        spot_stale=False, critical_flags=critical_flags, latency_seconds=latency_seconds
    )
    assert core_ok is True
    assert precision_ok is False
    assert precision_disabled is True
    assert "k1m" in reason_map
    return True


def scenario_core_gap_on_k5m() -> bool:
    """Spot or k5m stale should still hard-stop via data_gap."""

    critical_flags = {"k1m": False, "k5m": True, "k1h": False, "k4h": False}
    latency_seconds = {"k1m": 60, "k5m": 720, "k1h": 600, "k4h": 2400}
    core_ok, precision_ok, precision_disabled, reason_map = compute_data_integrity_status(
        spot_stale=True, critical_flags=critical_flags, latency_seconds=latency_seconds
    )
    assert core_ok is False
    assert precision_ok is False
    assert precision_disabled is False
    assert "k5m" in reason_map
    return True


def main() -> None:
    results = {
        "core_only_k1m_stale": scenario_core_only_k1m_stale(),
        "core_gap_on_k5m": scenario_core_gap_on_k5m(),
    }
    for name, passed in results.items():
        print(f"{name}: {'ok' if passed else 'failed'}")


if __name__ == "__main__":
    main()
