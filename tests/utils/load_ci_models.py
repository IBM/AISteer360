from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent / "ci_models.yaml"


def get_models() -> dict[str, str]:
    with open(_YAML_PATH, "r", encoding="utf-8") as filepath:
        return yaml.safe_load(filepath)["models"]
