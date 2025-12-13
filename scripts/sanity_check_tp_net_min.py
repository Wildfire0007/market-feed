import json
from pathlib import Path
from typing import Any, Dict, List


def find_tp_net_min(data: Any, path: List[str]) -> List[str]:
    occurrences: List[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + [key]
            if key == "tp_net_min":
                occurrences.append("/".join(new_path))
            occurrences.extend(find_tp_net_min(value, new_path))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            occurrences.extend(find_tp_net_min(value, path + [f"[{idx}]"]))
    return occurrences


def main() -> None:
    settings_path = Path(__file__).resolve().parent.parent / "config" / "analysis_settings.json"
    with settings_path.open("r", encoding="utf-8") as f:
        settings: Dict[str, Any] = json.load(f)

    risk_templates = settings.get("risk_templates", {})
    occurrences = find_tp_net_min(risk_templates, ["risk_templates"])
    if occurrences:
        raise SystemExit(
            "Found disallowed tp_net_min entries under risk_templates: " + ", ".join(occurrences)
        )

    assets = settings.get("assets", [])
    default_value = settings.get("tp_net_min_default")
    asset_overrides = settings.get("tp_net_min", {})
    effective = {asset: asset_overrides.get(asset, default_value) for asset in assets}

    print("tp_net_min_default:", default_value)
    print("tp_net_min overrides:")
    for asset, value in asset_overrides.items():
        print(f"  {asset}: {value}")
    print("\nEffective tp_net_min values:")
    for asset in assets:
        print(f"  {asset}: {effective[asset]}")


if __name__ == "__main__":
    main()
