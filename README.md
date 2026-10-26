# Attendance Biometric Service

A standalone Render-hosted biometric service for the Attendance-System backend. It wraps a pinned OpenBiometrics engine behind an API-key-protected compatibility API matching the current `backend/src/modules/Biometric/LocalBiometricProvider.js` contract.

> **Deployment status:** The source implements the compatibility adapter and Render blueprint, but the service is not considered production-ready until the Docker build/model loading, authenticated Render health checks, real enrollment/matching/deletion, and physical-device liveness flow have been verified. Do not infer a live Render deployment from a passing unit test.

## Architecture

- OpenBiometrics is pinned to source commit `ab09d469327684d630b7383119dd4544253a03c9` in `Dockerfile`.
- The Docker build downloads only the YuNet face detector, SFace recognizer and MiniFASNet anti-spoof model (skipping the unneeded age/gender model) and uses CPU inference. It disables optional demographics, document, person-detection, video and event/webhook modules.
- The upstream FastAPI app is initialized in-process but is **not mounted on the public app**. Only the `/v1/*` compatibility endpoints are exposed.
- Every `/v1/*` endpoint requires `Authorization: Bearer <BIOMETRIC_API_KEY>`. `/healthz` is the only public endpoint, for Render's health check, and reports `503` until models and the API key are ready.
- Enrolled face embeddings and watchlist indexes persist on the Render disk under `/data/watchlists`; collection metadata and an identity registry persist under `/data/collections` and `/data/registry.sqlite3`. The service does not intentionally persist enrollment/verification image bytes.
- Liveness fails closed: if the engine does not produce an explicit boolean decision and a numeric score from `0.0` to `1.0`, the API returns `status: unavailable`; the Attendance backend must reject that frame. The adapter does not promise that passive PAD defeats all photo, screen, replay or mask attacks.

## API contract

All endpoints below except `/healthz` require the bearer API key.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Render health check; no secrets or model details |
| `GET` | `/v1/health` | Authenticated ready/auth-enabled check |
| `GET` | `/v1/collections` | List collection metadata |
| `POST` | `/v1/collections` | Create collection with `{ "id": "students", "name": "Attendance students", "threshold": 0.4 }` |
| `GET` | `/v1/collections/{collection_id}` | Read collection metadata |
| `POST` | `/v1/collections/{collection_id}/persons` | Multipart `id`, `external_id`, `review_mode=strict`, `images` |
| `POST` | `/v1/collections/{collection_id}/search` | Multipart `image`, `limit`, `threshold`; returns matches and evaluated liveness |
| `POST` | `/v1/detect` | Multipart `image`, `max_faces`, `collection_id`; returns five-point landmarks and per-frame liveness |
| `DELETE` | `/v1/collections/{collection_id}/persons/{person_id}` | Delete template; repeated delete is idempotent |

The enrollment adapter first invokes the upstream detector and requires exactly one face with an explicit positive passive-liveness result before adding the recognition embedding. It also serializes enrollment/deletion in this single-process service and prevents duplicate enrollment IDs in the local registry.

## Required Render setup

Use the `render.yaml` blueprint or create a **Docker Web Service** manually with:

- **Name:** `attendance-biometric-service`
- **Region:** Singapore (use the same region as the Attendance backend when available)
- **Plan:** Standard/paid CPU instance (the CPU face models need real memory; do not use a free, sleeping instance for attendance)
- **Disk:** 5 GB mounted at `/data`
- **Health check path:** `/healthz`
- **Auto deploy:** Off until the initial deployment has passed smoke tests
- **Secret environment variable:** `BIOMETRIC_API_KEY` set to a newly generated random secret of at least 32 characters
- `BIOMETRIC_DATA_DIR=/data`
- `BIOMETRIC_FACE_DETECTION_THRESHOLD=0.60` (valid range: `0.50`–`0.90`)

A Render persistent disk is necessary: without it, redeploys/restarts can erase watchlists and stored identity vectors. This deployment targets one running instance; do not scale horizontally while using a single attached disk and a local SQLite registry.

The face-detection threshold is a precision/recall trade-off. The default `0.60` is higher than YuNet's upstream `0.50` default to suppress weaker detections that can cause false multi-face prompts. It does not ignore credible additional faces or prove accuracy on every camera. Validate with consenting test subjects on the actual device before operational use; a higher threshold can miss genuine faces in poor lighting or at a distance.

The Docker build needs outbound network access to fetch the pinned upstream source and model files. A successful image build is not proof the model loads; verify `/healthz` and authenticated `/v1/health` after deployment.

## Connect the existing Attendance backend

In the existing Render backend service, set:

```text
BIOMETRIC_BASE_URL=https://<actual-biometric-service-host>
BIOMETRIC_API_KEY=<exact-same-secret-as-biometric-service>
BIOMETRIC_COLLECTION_ID=students
BIOMETRIC_TIMEOUT_MS=10000
BIOMETRIC_MATCH_THRESHOLD=0.40
```

Use the actual Render hostname; do not include a trailing path, query, or credentials in `BIOMETRIC_BASE_URL`. Keep the API key only in Render environment variables—never in the React, Expo, or browser bundle. Redeploy the backend after the environment is updated. Do not redirect production traffic before the collection bootstrap and matching/deletion tests pass.

## Local adapter tests

Adapter contract tests use a mocked engine; they test auth, collection persistence, route mapping, duplicate protection, liveness fail-closed behavior and deletion. They do **not** prove real image recognition, passive PAD performance, or physical-device attendance correctness.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m compileall -q biometric_service tests
pytest -q
```

For a full model-backed smoke test, build/run the Docker image with a real secret and persistent local volume. The `scripts/smoke.py` script checks authenticated health, creates/reads the collection, and can submit one **consenting test subject's** image to `/v1/detect` when given `--image`. Do not use actual student images for initial engineering smoke tests.

```bash
python scripts/smoke.py --base-url https://<actual-biometric-service-host> --api-key-env BIOMETRIC_API_KEY --collection students
python scripts/smoke.py --base-url https://<actual-biometric-service-host> --api-key-env BIOMETRIC_API_KEY --collection students --image ./consenting-test-subject.jpg
```

The command reads the secret from the named environment variable; don't put the key directly in shell history or commit it to Git. The second command is an engineering smoke test, not a spoof-resistance certification.

## Production gates before real students

1. Successful clean Render build and restart with persistent disk; `/healthz` and `/v1/health` both ready.
2. Enroll one consenting test subject and verify with a separate capture; wrong-subject match must be rejected by the Attendance backend.
3. Demonstrate negative and unavailable liveness outcomes fail closed; test still-photo, printed photo, screen replay and video replay using a controlled test plan. Do not infer PAD quality from a single pass.
4. Verify delete, retry, restart persistence, and absence of raw image persistence. Confirm how backups and disk disposal will securely delete templates.
5. Calibrate false-accept/false-reject threshold on representative consenting test data and review demographic performance before operational use.
6. Run the actual mobile QR → liveness → identity match flow and load test on the target Render plan. Mocked tests are not evidence of real biometric accuracy.

## Privacy and limitations

Face embeddings are sensitive biometric data even when raw images are not stored. Restrict dashboard access, rotate the API key when needed, protect the Render disk, define retention/deletion policy, obtain appropriate consent, and review applicable Indian privacy/education requirements before collecting student data. Do not log uploaded image content, face embeddings, API keys, or student identifiers. Use only an approved model/license set and get a supervised pilot sign-off before enabling production biometric acceptance.

See `NOTICE.md` for upstream source and model-license cautions.
