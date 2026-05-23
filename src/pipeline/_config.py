"""Load pipeline configuration."""

import json
from pathlib import Path
from typing import Any, Dict

_DEFAULT = Path("configs/pipeline.json")


def load_pipeline_config(path: str = "configs/pipeline.json") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)
