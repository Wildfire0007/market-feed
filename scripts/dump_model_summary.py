diff --git a/scripts/dump_model_summary.py b/scripts/dump_model_summary.py
new file mode 100755
index 0000000000000000000000000000000000000000..cb6223595905eb19b0ab1e7dcb289eb567689d62
--- /dev/null
 b/scripts/dump_model_summary.py
@@ -0,0 1,165 @@
#!/usr/bin/env python3
"""Emit a JSON summary for trained gradient boosting models.

The live pipeline stores gradient boosting classifiers as Joblib pickles under
``public/models`` so they can be updated without changing the application
runtime.  These pickles are binary blobs which are inconvenient to review in
code reviews.  This helper loads the model and serialises its public metadata to
JSON so humans can quickly inspect the hyper-parameters and feature importances
without having to unpickle the artefact manually.

Example
-------
```
$ scripts/dump_model_summary.py --asset XAGUSD \
      --output public/models/XAGUSD_model_summary.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from joblib import load

import sklearn

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_model import MODEL_DIR, MODEL_FEATURES


def _floatify(sequence: Iterable[Any]) -> List[float]:
    return [float(value) for value in sequence]


def _sequence_summary(values: Iterable[Any]) -> Dict[str, Any]:
    numbers = _floatify(values)
    if not numbers:
        return {}
    head = numbers[:5]
    tail = numbers[-5:] if len(numbers) > 5 else numbers
    return {
        "count": len(numbers),
        "first": numbers[0],
        "last": numbers[-1],
        "min": min(numbers),
        "max": max(numbers),
        "mean": mean(numbers),
        "head": head,
        "tail": tail,
    }


def _feature_importances(estimator: Any) -> List[Dict[str, Any]]:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []
    values = list(map(float, importances))
    if len(values) != len(MODEL_FEATURES):
        # The estimator might have been trained with an outdated feature list.
        names: List[str] = [f"feature_{idx}" for idx in range(len(values))]
    else:
        names = MODEL_FEATURES
    return [
        {"feature": name, "importance": value}
        for name, value in zip(names, values)
    ]


def build_summary(asset: str, estimator: Any) -> Dict[str, Any]:
    params = estimator.get_params()
    tracked_params = {
        key: params[key]
        for key in sorted(
            {
                "learning_rate",
                "loss",
                "max_depth",
                "max_features",
                "min_samples_leaf",
                "min_samples_split",
                "n_estimators",
                "random_state",
                "subsample",
            }
        )
        if key in params
    }
    summary: Dict[str, Any] = {
        "asset": asset.upper(),
        "estimator_class": type(estimator).__name__,
        "estimator_module": type(estimator).__module__,
        "sklearn_version": sklearn.__version__,
        "parameters": tracked_params,
        "n_features_in": getattr(estimator, "n_features_in_", None),
        "n_estimators_trained": len(getattr(estimator, "estimators_", [])),
        "feature_importances": _feature_importances(estimator),
    }

    train_score = getattr(estimator, "train_score_", None)
    if train_score is not None:
        summary["train_loss"] = _sequence_summary(train_score)

    oob_improvement = getattr(estimator, "oob_improvement_", None)
    if oob_improvement is not None:
        summary["oob_improvement"] = _sequence_summary(oob_improvement)

    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        summary["classes"] = list(map(int, classes))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. XAGUSD)")
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Directory containing <asset>_gbm.pkl (default: public/models)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write JSON summary (stdout if omitted)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = args.asset.upper()
    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{asset}_gbm.pkl"
    if not model_path.exists():
        raise SystemExit(f"Model artefact not found: {model_path}")

    estimator = load(model_path)
    summary = build_summary(asset, estimator)

    payload = json.dumps(summary, indent=args.indent, sort_keys=True)
    if args.output:
        args.output.write_text(payload  "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
