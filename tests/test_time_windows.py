import pytest

from analysis import in_any_window_utc


class TestInAnyWindowUtc:
    def test_wrap_window_crosses_midnight(self):
        windows = [(23, 0, 1, 0)]

        assert in_any_window_utc(windows, 23, 0) is True
        assert in_any_window_utc(windows, 23, 30) is True
        assert in_any_window_utc(windows, 0, 0) is True
        assert in_any_window_utc(windows, 0, 59) is True
        assert in_any_window_utc(windows, 1, 0) is True
        assert in_any_window_utc(windows, 1, 1) is False
        assert in_any_window_utc(windows, 2, 0) is False

    def test_non_wrap_window(self):
        windows = [(9, 0, 17, 0)]

        assert in_any_window_utc(windows, 8, 59) is False
        assert in_any_window_utc(windows, 9, 0) is True
        assert in_any_window_utc(windows, 17, 0) is True
        assert in_any_window_utc(windows, 17, 1) is False
