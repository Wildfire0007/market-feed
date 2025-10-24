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
