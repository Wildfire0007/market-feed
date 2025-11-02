import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis


def test_initialize_overrides_weekend_missing_data() -> None:
    """Simulate weekend gap: overrides stay usable even with no prior data."""

    entry_meta = {"profile": "baseline"}

    xag_overrides, usoil_overrides, eurusd_overrides = analysis._initialize_asset_overrides(
        entry_meta, "USOIL"
    )

    assert xag_overrides == {}
    assert eurusd_overrides == {}
    assert entry_meta["usoil_overrides"] is usoil_overrides

    # Weekend scenario: no intraday data populated before usage.
    usoil_overrides.setdefault("momentum_stop", {})
    usoil_overrides["momentum_stop"].setdefault("atr_multiplier", None)


def test_initialize_overrides_weekday_missing_data() -> None:
    """Simulate weekday feed drop: EURUSD overrides can be populated lazily."""

    entry_meta = {"profile": "baseline"}

    xag_overrides, usoil_overrides, eurusd_overrides = analysis._initialize_asset_overrides(
        entry_meta, "EURUSD"
    )

    assert xag_overrides == {}
    assert usoil_overrides == {}
    assert entry_meta["eurusd_overrides"] is eurusd_overrides

    # Weekday scenario: initialize fields even though data input is missing.
    eurusd_overrides.setdefault("vwap_alignment", {})
    eurusd_overrides["vwap_alignment"].setdefault("distance", None)
