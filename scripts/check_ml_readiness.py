#!/usr/bin/env python3
"""Utility to verify ML scoring readiness for all configured assets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON: Tuple[int, int] = (3, 9)

from config.analysis_settings import ASSETS
from ml_model import inspect_model_artifact, runtime_dependency_issues


def python_version_status(min_version: Tuple[int, int] = MIN_PYTHON) -> dict:
    """Return status metadata describing whether the interpreter meets requirements."""

    version_info = sys.version_info
    current_tuple = (version_info[0], version_info[1], version_info[2])
    required_tuple = (min_version[0], min_version[1])
    is_supported = current_tuple[:2] >= required_tuple

    return {
        "status": "ok" if is_supported else "outdated",
        "current": f"{current_tuple[0]}.{current_tuple[1]}.{current_tuple[2]}",
        "recommended": f">= {required_tuple[0]}.{required_tuple[1]}",
    }


def _normalise_assets(assets: Sequence[str]) -> List[str]:
    return sorted({asset.upper() for asset in assets})


def _format_status(status: str) -> str:
    if status == "ok":
        return "OK"
    return status.upper()


def run_diagnostics(assets: Iterable[str]) -> int:
    exit_code = 0

    version_meta = python_version_status()
    if version_meta["status"] != "ok":
        exit_code = 1
        print("⚠️  Python verzió figyelmeztetés:")
        print(
            "  - Aktuális interpreter: {current} (ajánlott {recommended}).".format(
                **version_meta
            )
        )
        print("  → Helyi ellenőrzés: futtasd a `python --version` parancsot ugyanabban a környezetben.")
        print(
            "  → GitHub Actions ellenőrzés: az Actions fülön nyisd meg a legutóbbi pipeline-t, "
            "és a `Set up Python` lépés logjában ellenőrizd, hogy legalább 3.9-es verzió fut."
        )
        print(
            "    Ha régebbi verzió látszik, módosítsd a workflow fájlban a `python-version` mezőt "
            "(pl. `.github/workflows/<workflow>.yml`)."
        )
        print()

    issues = runtime_dependency_issues()
    if issues:
        exit_code = 1
        print("⚠️  Függőségi problémák észlelve:")
        for name, detail in issues.items():
            print(f"  - {name}: {detail}")
        print("  → Telepítési lépések:")
        print("    1. Aktiváld ugyanazt a virtuális környezetet, amit a jelgenerátor is használ.")
        print("    2. Futtasd: python -m pip install -r requirements.txt")
        print(
            "    3. Ellenőrizd: python - <<'PY'\\nimport sklearn, joblib\\nprint(sklearn.__version__, joblib.__version__)\\nPY"
        )
        print()

    print("ML modell diagnosztika:")
    for asset in assets:
        info = inspect_model_artifact(asset)
        status = info.get("status", "unknown")
        if status not in {"ok", "placeholder"}:
            exit_code = 1
        status_label = _format_status(status)
        print(f"- {asset}: {status_label}")
        extra = {
            key: value
            for key, value in info.items()
            if key not in {"asset", "status"} and value is not None
        }
        if extra:
            formatted = json.dumps(extra, ensure_ascii=False, indent=2)
            for line in formatted.splitlines():
                print(f"    {line}")

        if status == "missing":
            print("    Teendő:")
            print("      1. Másold a betanított modellt ebbe az útvonalba: {path}".format(path=info["path"]))
            print(
                "      2. A fájl méretének legalább néhány MB-nak kell lennie (ls -lh {path}).".format(
                    path=info["path"]
                )
            )
            print("      3. Ha nincs modell, futtasd le a tréning pipeline-t vagy kérd le az artefaktot a tárolóból.")
        elif status == "dependency_unavailable":
            print("    Teendő: lásd a fenti függőség telepítési lépéseket, majd futtasd újra a scriptet.")
        elif status == "load_error":
            print("    Teendő:")
            print("      1. Valószínűleg sérült a pickle: töltsd fel újra a modellt a {path} helyre.".format(path=info["path"]))
            print(
                "      2. Ellenőrizd, hogy ugyanazzal a scikit-learn verzióval készült, mint amit a requirements.txt ír elő."
            )
        elif status == "placeholder":
            print("    Figyelmeztetés: placeholder modell fájl észlelve.")
            print(
                "    Töltsd fel a tényleges GradientBoostingClassifier pickle-t ide: {path}".format(
                    path=info["path"]
                )
            )
            if info.get("detail"):
                print(f"    Részletek: {info['detail']}")
            print("    Lépésről lépésre javítás:")
            print("      1. Telepítsd a függőségeket: python -m pip install -r requirements.txt")
            print("      2. Generáld újra a feature CSV-t: USE_ML=0 python analysis.py")
            print(
                "      3. Címkézd fel a `public/ml_features/BTCUSD_features.csv` fájl sorait és mentsd `public/ml_features/BTCUSD_labelled.csv` néven."
            )
            print(
                "      4. Taníts modellt: python scripts/train_models.py --asset BTCUSD --dataset public/ml_features/BTCUSD_labelled.csv"
            )
            print(
                "      5. Ellenőrizd: python scripts/check_ml_readiness.py BTCUSD (a státusz legyen OK)."
            )
            print(
                "      6. Ha GitHubon néznéd meg a részletes útmutatót, kattints a docs/ml_model_training_hu.md fájlban a `Raw` gombra."
            )
        elif status == "type_mismatch":
            print("    Teendő: győződj meg róla, hogy GradientBoostingClassifier példányt tartalmaz a pickle.")
        elif status not in {"ok", "unknown"}:
            print("    Teendő: ellenőrizd a fenti részleteket és javítsd a problémát, majd futtasd újra.")

        if info.get("size_warning"):
            print("    Figyelmeztetés a fájlméretre:")
            print("      1. Ellenőrizd, hogy a fájl nem placeholder: ls -lh {path}".format(path=info["path"]))
            print(
                "      2. Ha a méret <1 MB, töltsd fel újra a teljes .pkl modellt a fenti helyre."
            )

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets",
        nargs="*",
        help="Csak a felsorolt eszközöket ellenőrzi (alapértelmezés: összes konfigurált)",
    )
    args = parser.parse_args(argv)
    assets = _normalise_assets(args.assets or ASSETS)
    return run_diagnostics(assets)


if __name__ == "__main__":
    sys.exit(main())
