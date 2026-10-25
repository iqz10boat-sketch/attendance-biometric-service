FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIOMETRIC_DATA_DIR=/data \
    OPENBIOMETRICS_API_PATH=/opt/openbiometrics/api \
    PYTHONPATH=/opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/openbiometrics
RUN git init . \
    && git remote add origin https://github.com/openbm/openbiometrics.git \
    && git fetch --depth 1 origin ab09d469327684d630b7383119dd4544253a03c9 \
    && git checkout --detach FETCH_HEAD
# Fix upstream dynamic-version paths rejected by modern setuptools.
RUN python - <<'PY'
from pathlib import Path

root = Path("/opt/openbiometrics")
version = (root / "VERSION").read_text(encoding="utf-8").strip()

if not version:
    raise SystemExit("Pinned OpenBiometrics VERSION file is empty")

for package in ("engine", "api"):
    package_dir = root / package
    pyproject = package_dir / "pyproject.toml"

    if not pyproject.is_file():
        raise SystemExit(f"Missing upstream packaging metadata: {pyproject}")

    source = pyproject.read_text(encoding="utf-8")
    old_reference = 'version = {file = "../VERSION"}'
    new_reference = 'version = {file = "VERSION"}'

    if source.count(old_reference) != 1:
        raise SystemExit(
            f"Expected exactly one parent-relative version reference in {pyproject}"
        )

    (package_dir / "VERSION").write_text(
        version + "\n", encoding="utf-8"
    )
    pyproject.write_text(
        source.replace(old_reference, new_reference, 1),
        encoding="utf-8",
    )

print(f"Patched pinned upstream package metadata to version {version}.")
PY

COPY patch_openbiometrics.py /opt/openbiometrics/patch_openbiometrics.py
COPY scripts/prepare_models.py /opt/openbiometrics/prepare_models.py
COPY requirements.txt /tmp/adapter-requirements.txt
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r /tmp/adapter-requirements.txt \
    && pip install --no-cache-dir --timeout 180 --retries 10 "/opt/openbiometrics/engine[liveness]" "/opt/openbiometrics/api" \
    && python /opt/openbiometrics/patch_openbiometrics.py \
    && python /opt/openbiometrics/prepare_models.py

WORKDIR /data
RUN mkdir -p /data/watchlists /data/collections \
    && useradd --system --uid 10001 --create-home biometric \
    && chown -R biometric:biometric /data /opt/openbiometrics
USER biometric

COPY --chown=biometric:biometric biometric_service /opt/app/biometric_service
EXPOSE 10000
CMD ["sh", "-c", "uvicorn biometric_service.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers ${BIOMETRIC_WORKERS:-1} --no-access-log"]
