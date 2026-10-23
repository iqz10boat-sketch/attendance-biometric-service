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

COPY patch_openbiometrics.py /opt/openbiometrics/patch_openbiometrics.py
COPY scripts/prepare_models.py /opt/openbiometrics/prepare_models.py
COPY requirements.txt /tmp/adapter-requirements.txt
RUN pip install --no-cache-dir -r /tmp/adapter-requirements.txt \
    && pip install --no-cache-dir "/opt/openbiometrics/engine[liveness]" "/opt/openbiometrics/api" \
    && python /opt/openbiometrics/patch_openbiometrics.py \
    && python /opt/openbiometrics/prepare_models.py

WORKDIR /data
RUN mkdir -p /data/watchlists /data/collections \
    && useradd --system --uid 10001 --create-home biometric \
    && chown -R biometric:biometric /data /opt/openbiometrics
USER biometric

COPY --chown=biometric:biometric biometric_service /opt/app/biometric_service
EXPOSE 10000
CMD ["sh", "-c", "uvicorn biometric_service.main:app --host 0.0.0.0 --port ${PORT:-10000} --no-access-log"]
