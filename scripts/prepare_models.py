"""Prepare pinned community models for the biometric service Docker image."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

repo = Path("/opt/openbiometrics")
sys.path.insert(0, str(repo / "engine"))
from openbiometrics.runtime.registry import ModelRegistry  # noqa: E402

MODELS_DIR = repo / "engine" / "models"
REQUIRED_REGISTRY_MODELS = ("yunet", "sface")

# Public release pinned by immutable SHA-256. Upstream's configured
# openbiometrics/models/antispoofing.onnx URL currently returns HTTP 401.
ANTISPOOFING_URL = (
    "https://github.com/facenox/face-antispoof-onnx/"
    "releases/download/v1.0.0/best_model.onnx"
)
ANTISPOOFING_SHA256 = "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6"
ANTISPOOFING_SIZE = 1_912_594


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_antispoofing_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / "antispoofing.onnx"
    if destination.is_file():
        digest = sha256_file(destination)
        if digest == ANTISPOOFING_SHA256 and destination.stat().st_size == ANTISPOOFING_SIZE:
            print(f"Using verified antispoofing model: sha256={digest}")
            return
        destination.unlink()

    temporary = MODELS_DIR / "antispoofing.onnx.part"
    if temporary.exists():
        temporary.unlink()
    try:
        print(f"Downloading pinned MiniFASNet V2 SE model from {ANTISPOOFING_URL}")
        urllib.request.urlretrieve(ANTISPOOFING_URL, str(temporary))
        actual_size = temporary.stat().st_size
        actual_digest = sha256_file(temporary)
        if actual_size != ANTISPOOFING_SIZE or actual_digest != ANTISPOOFING_SHA256:
            raise RuntimeError(
                "Anti-spoof model integrity check failed: "
                f"size={actual_size}, sha256={actual_digest}"
            )
        temporary.replace(destination)
        print(f"Verified antispoofing model: sha256={actual_digest}, size={actual_size}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


registry = ModelRegistry(models_dir=MODELS_DIR)
for model_name in REQUIRED_REGISTRY_MODELS:
    info = next((item for item in registry.list_models() if item.name == model_name), None)
    if info is None:
        raise SystemExit(f"Required model '{model_name}' is missing from the pinned model catalog")
    if info.tier != "community":
        raise SystemExit(f"Refusing to download non-community model tier for '{model_name}'")
    print(f"Preparing {model_name}: tier={info.tier}, catalog_license={info.license}, role={info.role}")
    path = registry.ensure_model(model_name)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Model download produced an empty file: {model_name}")

ensure_antispoofing_model()
print("All required face model files are present and the anti-spoof model hash is verified.")
