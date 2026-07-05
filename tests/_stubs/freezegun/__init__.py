from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from datetime import timezone as _timezone
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

    class FrozenTime:
        def __init__(self):
            self._patch = patch("datetime.datetime", FrozenDateTime)

        def __enter__(self):
            self._patch.__enter__()
            return self
            
        def __exit__(self, exc_type, exc, tb):
            return self._patch.__exit__(exc_type, exc, tb)

        def __call__(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapper

    return FrozenTime()
    
__all__ = ["freeze_time"]
