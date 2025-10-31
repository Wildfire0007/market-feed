from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Trading


def test_normalize_symbol_attempts_generates_crypto_variants() -> None:
    cfg = {"symbol": "BTC/USD", "exchange": "CRYPTO"}

    attempts = Trading._normalize_symbol_attempts(cfg)

    assert attempts[0] == ("BTC/USD", "CRYPTO")
    assert ("BTC/USD", None) in attempts
    assert ("BTCUSD", "CRYPTO") in attempts
    assert ("BTCUSD", None) in attempts
    assert ("BTC/USD:CRYPTO", None) in attempts


def test_normalize_symbol_attempts_respects_alt_symbols() -> None:
    cfg = {
        "symbol": "ABC/DEF",
        "exchange": "TEST",
        "alt": [
            "XYZ/DEF",
            {"symbol": "ABCDEF", "exchange": "ALT"},
            ("QQQ",),
        ],
    }

    attempts = Trading._normalize_symbol_attempts(cfg)

    assert ("XYZ/DEF", "TEST") in attempts
    assert ("ABCDEF", "ALT") in attempts
    assert ("QQQ", "TEST") in attempts


def test_xagusd_attempts_cover_physical_metal_exchange() -> None:
    cfg = Trading.ASSETS["XAGUSD"]

    attempts = Trading._normalize_symbol_attempts(cfg)

    assert attempts[0] == ("XAG/USD", "COMMODITY")
    assert ("XAG/USD", None) in attempts
    assert all(symbol != "XAGUSD" for symbol, _ in attempts)


def test_skip_flag_removes_configured_variants() -> None:
    cfg = {
        "symbol": "AAA/BBB",
        "exchange": "TEST",
        "alt": [
            {"symbol": "CCC/DDD", "skip": True, "note": "bad"},
            {"symbol": "AAA", "exchange": "TEST"},
        ],
    }

    attempts = Trading._normalize_symbol_attempts(cfg)

    assert ("CCC/DDD", "TEST") not in attempts
    assert ("AAA", "TEST") in attempts


def test_global_bad_symbol_cache_short_circuits_attempts() -> None:
    Trading._reset_global_symbol_failure_cache()

    attempts = [("BAD/USD", "FOREX"), ("GOOD/USD", None)]
    calls = {"bad": 0, "good": 0}

    def fetch(symbol: str, exchange: str | None) -> dict[str, object]:
        if symbol == "BAD/USD":
            calls["bad"] += 1
            raise Trading.TDError("invalid symbol", status_code=400)
        calls["good"] += 1
        return {"ok": True, "latency_seconds": 0.1}

    first = Trading.try_symbols(
        attempts,
        fetch,
        freshness_limit=None,
        attempt_memory=Trading.AttemptMemory(),
    )

    assert first["ok"]
    assert calls == {"bad": 1, "good": 1}

    second = Trading.try_symbols(
        attempts,
        fetch,
        freshness_limit=None,
        attempt_memory=Trading.AttemptMemory(),
    )

    assert second["ok"]
    assert calls == {"bad": 1, "good": 2}
