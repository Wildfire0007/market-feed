from datetime import datetime as _datetime, timezone as _timezone, timedelta as _timedelta
from functools import wraps
from unittest.mock import patch


def _parse(dt_string: str, tz_offset: int = 0):
    ts = _datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        tz = _timezone(_timedelta(hours=tz_offset)) if tz_offset is not None else None
        ts = ts.replace(tzinfo=tz)
    return ts


def freeze_time(dt_string: str, tz_offset: int = 0):
    frozen_dt = _parse(dt_string, tz_offset=tz_offset)
    frozen_utc = frozen_dt.astimezone(_timezone.utc)

    class FrozenDateTime(_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return frozen_dt
            return frozen_dt.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return frozen_utc.replace(tzinfo=None)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with patch("datetime.datetime", FrozenDateTime):
                return func(*args, **kwargs)

        return wrapper

    return decorator

__all__ = ["freeze_time"]
