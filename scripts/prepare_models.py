"""Download only the three required community face model assets at image-build time."""
from __future__ import annotations

import sys
from pathlib import Path

repo = Path("/opt/openbiometrics")
sys.path.insert(0, str(repo / "engine"))
from openbiometrics.runtime.registry import ModelRegistry  # noqa: E402

MODELS_DIR = repo / "engine" / "models"
REQUIRED_MODELS = ("yunet", "sface", "antispoofing")
registry = ModelRegistry(models_dir=MODELS_DIR)
for model_name in REQUIRED_MODELS:
    info = next((item for item in registry.list_models() if item.name == model_name), None)
    if info is None:
        raise SystemExit(f"Required model '{model_name}' is missing from the pinned model catalog")
    if info.tier != "community":
        raise SystemExit(f"Refusing to download non-community model tier for '{model_name}'")
    print(f"Preparing {model_name}: tier={info.tier}, catalog_license={info.license}, role={info.role}")
    path = registry.ensure_model(model_name)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Model download produced an empty file: {model_name}")
print("All required face model files are present.")
