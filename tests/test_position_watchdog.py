from scripts import position_watchdog


def test_can_write_positions_supports_watchdog_writer() -> None:
    assert position_watchdog._can_write_positions("watchdog", False) is True
    assert position_watchdog._can_write_positions("notify", False) is False
