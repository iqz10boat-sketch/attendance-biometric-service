"""Authenticated compatibility API for the Attendance-System biometric provider.

The upstream OpenBiometrics FastAPI app is started in-process but never mounted
on the public ASGI application. Only the endpoints below are exposed publicly.
"""
from __future__ import annotations

import asyncio
import cv2
import hmac
import json
import os
import re
import secrets
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from .liveness_challenges import initialize_liveness_tables, router as liveness_challenge_router
from .process_lock import watchlist_file_lock

SERVICE_VERSION = "0.1.0"
DEFAULT_DATA_DIR = Path(os.getenv("BIOMETRIC_DATA_DIR", str(Path.cwd() / ".data")))
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
BEARER = HTTPBearer(auto_error=False, scheme_name="BiometricServiceKey")


class CollectionCreate(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(default="Attendance students", min_length=1, max_length=120)
    threshold: float = Field(default=0.4, gt=0.0, le=1.0)


def _safe_collection_id(value: str) -> str:
    if not COLLECTION_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid collection id")
    return value


def _safe_person_id(value: str) -> str:
    if not ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid person id")
    return value


def _api_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _api_key() -> str:
    return os.getenv("BIOMETRIC_API_KEY", "")


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(BEARER),
) -> None:
    expected = _api_key()
    if len(expected) < 32 or credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def _collection_file(data_dir: Path, collection_id: str) -> Path:
    return data_dir / "collections" / f"{_safe_collection_id(collection_id)}.json"


def _read_collection(data_dir: Path, collection_id: str) -> dict[str, Any] | None:
    path = _collection_file(data_dir, collection_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Collection metadata is unavailable")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{secrets.token_hex(6)}.tmp")
    tmp_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp_path, path)


def _init_registry(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS persons (
            collection_id TEXT NOT NULL,
            person_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (collection_id, person_id)
        )"""
    )
    db.commit()
    return db


def _liveness_payload(face: dict[str, Any] | None) -> dict[str, Any]:
    raw = (face or {}).get("liveness")
    if not isinstance(raw, dict):
        return {"status": "unavailable", "is_live": False, "live_score": None}
    is_live = raw.get("is_live")
    score = raw.get("score")
    score_ok = isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= float(score) <= 1.0
    evaluated = isinstance(is_live, bool) and score_ok
    return {
        "status": "ok" if evaluated else "unavailable",
        "is_live": is_live if isinstance(is_live, bool) else False,
        "live_score": float(score) if score_ok else None,
    }


def _face_landmarks(face: dict[str, Any]) -> Any:
    detection = face.get("detection") if isinstance(face, dict) else None
    return detection.get("landmarks") if isinstance(detection, dict) else None


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(os.getenv("BIOMETRIC_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "collections").mkdir(parents=True, exist_ok=True)
    (data_dir / "watchlists").mkdir(parents=True, exist_ok=True)
    if len(_api_key()) < 32:
        raise RuntimeError("BIOMETRIC_API_KEY must be configured with at least 32 characters")

    # Bound native CPU threading because each Uvicorn worker owns an independent model stack.
    try:
        opencv_threads = int(os.getenv("BIOMETRIC_OPENCV_THREADS", "1"))
    except ValueError as exc:
        raise RuntimeError("BIOMETRIC_OPENCV_THREADS must be a positive integer") from exc
    if not 1 <= opencv_threads <= 64:
        raise RuntimeError("BIOMETRIC_OPENCV_THREADS must be between 1 and 64")
    cv2.setNumThreads(opencv_threads)

    # Render's Docker image places the pinned upstream repository here. Import it
    # dynamically so unit tests can exercise the adapter without downloading models.
    upstream_api = os.getenv("OPENBIOMETRICS_API_PATH", "/opt/openbiometrics/api")
    if upstream_api not in sys.path:
        sys.path.insert(0, upstream_api)
    try:
        from app.main import app as engine_app
        from app import deps as engine_deps
    except Exception as exc:  # fail startup rather than serving a fake/partial engine
        raise RuntimeError("OpenBiometrics engine could not be imported") from exc

    engine_context = engine_app.router.lifespan_context(engine_app)
    await engine_context.__aenter__()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=engine_app),
        base_url="http://openbiometrics.internal",
        timeout=httpx.Timeout(20.0, connect=3.0),
    )
    try:
        kernel = engine_deps.get_kernel()
        pipeline = getattr(kernel, "face", None)
        engine_status = {
            "detector_loaded": getattr(pipeline, "_detector", None) is not None,
            "recognizer_loaded": getattr(pipeline, "_recognizer", None) is not None,
            "passive_liveness_loaded": getattr(pipeline, "_liveness", None) is not None,
            "active_liveness_loaded": getattr(kernel, "liveness", None) is not None,
        }
        # Strict readiness: fail closed if a critical model/component is missing.
        app.state.engine_status = engine_status
        app.state.data_dir = data_dir
        app.state.engine_lock = asyncio.Lock()
        app.state.watchlist_fallback_lock = asyncio.Lock()
        async with watchlist_file_lock(
            data_dir, shared=False, fallback_lock=app.state.watchlist_fallback_lock
        ):
            app.state.registry = _init_registry(data_dir / "registry.sqlite3")
            initialize_liveness_tables(app.state.registry)
        app.state.liveness_lock = asyncio.Lock()
        app.state.engine_client = client
        app.state.started_at = time.monotonic()
        if not all(engine_status.values()):
            missing = [name for name, ready in engine_status.items() if not ready]
            raise RuntimeError("Required biometric components did not load: " + ", ".join(missing))
        yield
    finally:
        await client.aclose()
        try:
            engine_deps.shutdown_services()
        except Exception:
            pass
        await engine_context.__aexit__(None, None, None)
        registry = getattr(app.state, "registry", None)
        if registry is not None:
            registry.close()


app = FastAPI(
    title="Attendance Biometric API",
    version=SERVICE_VERSION,
    description="API-key-protected attendance biometric adapter around OpenBiometrics.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.include_router(
    liveness_challenge_router,
    prefix="/v1/liveness",
    dependencies=[Depends(require_api_key)],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", incoming) else uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    if exc.status_code == 401:
        response = _api_error(401, "UNAUTHORIZED", "A valid bearer API key is required")
        if exc.headers:
            response.headers.update(exc.headers)
        return response
    messages = {
        400: "The request was invalid",
        404: "The requested resource was not found",
        409: "The requested resource conflicts with existing state",
        413: "The uploaded image exceeds the maximum size",
        415: "Only JPEG, PNG, and WebP images are accepted",
        422: "The request did not pass validation",
        503: "The biometric service is temporarily unavailable",
    }
    message = messages.get(exc.status_code, "The request could not be completed")
    # Keep safe, intentionally authored validation messages, but do not leak upstream internals.
    if isinstance(exc.detail, str) and exc.detail in {
        "Invalid collection id", "Invalid person id", "Invalid threshold", "Invalid result limit",
        "Collection not found", "Image is required", "Invalid image content type",
        "Collection metadata is unavailable", "Enrollment requires one live face",
        "Person is already enrolled", "Image must pass an evaluated liveness check",
    }:
        message = exc.detail
    return _api_error(exc.status_code, "REQUEST_REJECTED", message)


@app.get("/healthz", include_in_schema=False)
async def healthz(request: Request):
    key_configured = len(_api_key()) >= 32
    status = getattr(request.app.state, "engine_status", {})
    ready = key_configured and bool(status) and all(status.values())
    payload = {"status": "ok" if ready else "unavailable", "ready": ready}
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/v1/health", dependencies=[Depends(require_api_key)])
async def provider_health(request: Request):
    status = getattr(request.app.state, "engine_status", {})
    ready = bool(status) and all(status.values())
    if not ready:
        raise HTTPException(status_code=503, detail="Required biometric components are unavailable")
    return {"status": "ready", "auth_enabled": True, "version": SERVICE_VERSION}


@app.get("/v1/collections", dependencies=[Depends(require_api_key)])
async def list_collections(request: Request):
    root = request.app.state.data_dir / "collections"
    collections = []
    for path in sorted(root.glob("*.json")):
        try:
            collections.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return {"collections": collections}


@app.post("/v1/collections", status_code=201, dependencies=[Depends(require_api_key)])
async def create_collection(payload: CollectionCreate, request: Request):
    collection_id = _safe_collection_id(payload.id)
    path = _collection_file(request.app.state.data_dir, collection_id)
    async with request.app.state.engine_lock, watchlist_file_lock(
        request.app.state.data_dir,
        shared=False,
        fallback_lock=_watchlist_fallback_lock(request),
    ):
        if path.exists():
            raise HTTPException(status_code=409, detail="Collection already exists")
        collection = {"collection": {"id": collection_id, "name": payload.name, "threshold": payload.threshold}}
        _write_json_atomic(path, collection)
    return collection


@app.get("/v1/collections/{collection_id}", dependencies=[Depends(require_api_key)])
async def get_collection(collection_id: str, request: Request):
    collection_id = _safe_collection_id(collection_id)
    result = _read_collection(request.app.state.data_dir, collection_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return result


async def _read_image(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Invalid image content type")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Image is required")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    return data


def _watchlist_fallback_lock(request: Request) -> asyncio.Lock:
    """Return the per-app lock used when OS flock is unavailable (e.g. Windows tests)."""
    lock = getattr(request.app.state, "watchlist_fallback_lock", None)
    if lock is None:
        # Assignment is atomic with respect to this event loop: there is no await here.
        lock = asyncio.Lock()
        request.app.state.watchlist_fallback_lock = lock
    return lock


def _client(request: Request) -> httpx.AsyncClient:
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Biometric engine is not ready")
    return engine_client


async def _engine_failure(response: httpx.Response) -> None:
    # Keep upstream model/stack details private, while preserving basic 4xx rejection.
    if response.status_code >= 500:
        raise HTTPException(status_code=503, detail="Biometric engine failed to process the request")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Collection not found")
    if response.status_code == 409:
        raise HTTPException(status_code=409, detail="Person is already enrolled")
    if response.status_code == 400:
        raise HTTPException(status_code=422, detail="The image was rejected by the biometric engine")
    raise HTTPException(status_code=422, detail="The biometric engine rejected the request")


@app.post("/v1/collections/{collection_id}/persons", dependencies=[Depends(require_api_key)])
async def enroll_person(
    collection_id: str,
    request: Request,
    id: str = Form(...),
    external_id: str = Form(...),
    review_mode: str = Form("strict"),
    images: UploadFile = File(...),
):
    collection_id = _safe_collection_id(collection_id)
    person_id = _safe_person_id(external_id or id)
    _safe_person_id(id)
    if not _read_collection(request.app.state.data_dir, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if review_mode != "strict":
        raise HTTPException(status_code=422, detail="The request did not pass validation")
    image_bytes = await _read_image(images)
    client = _client(request)
    async with request.app.state.engine_lock, watchlist_file_lock(
        request.app.state.data_dir,
        shared=False,
        fallback_lock=_watchlist_fallback_lock(request),
    ):
        existing = request.app.state.registry.execute(
            "SELECT 1 FROM persons WHERE collection_id=? AND person_id=?", (collection_id, person_id)
        ).fetchone()
        # Reconcile against the engine's persisted metadata too, in case the
        # small registry was restored independently of its FAISS watchlist.
        watchlist_metadata = request.app.state.data_dir / "watchlists" / f"{collection_id}.json"
        watchlist_has_person = False
        if watchlist_metadata.is_file():
            try:
                identities = json.loads(watchlist_metadata.read_text(encoding="utf-8"))
                watchlist_has_person = any(
                    isinstance(item, dict) and item.get("id") == person_id
                    for item in identities
                ) if isinstance(identities, list) else False
            except (OSError, json.JSONDecodeError):
                raise HTTPException(status_code=503, detail="Collection metadata is unavailable")
        if existing or watchlist_has_person:
            raise HTTPException(status_code=409, detail="Person is already enrolled")
        # First perform an explicit passive PAD assessment. Never enroll an image
        # when the engine cannot positively evaluate its liveness.
        detected = await client.post(
            "/api/v1/detect", files={"image": (images.filename or "enrollment.jpg", image_bytes, images.content_type or "image/jpeg")}
        )
        if detected.status_code >= 400:
            await _engine_failure(detected)
        detect_data = detected.json()
        detected_faces = detect_data.get("faces") if isinstance(detect_data, dict) else None
        if not isinstance(detected_faces, list) or len(detected_faces) != 1:
            raise HTTPException(status_code=422, detail="Enrollment requires one live face")
        live = _liveness_payload(detected_faces[0])
        if live["status"] != "ok" or live["is_live"] is not True:
            raise HTTPException(status_code=422, detail="Image must pass an evaluated liveness check")

        enrolled = await client.post(
            "/api/v1/watchlist/enroll",
            data={"identity_id": person_id, "label": person_id, "watchlist_name": collection_id},
            files={"image": (images.filename or "enrollment.jpg", image_bytes, images.content_type or "image/jpeg")},
        )
        if enrolled.status_code >= 400:
            await _engine_failure(enrolled)
        enrolled_data = enrolled.json()
        if enrolled_data.get("identity_id") != person_id or not isinstance(enrolled_data.get("face"), dict):
            # Roll back engine state if its response does not match our request.
            await client.delete(f"/api/v1/watchlist/{person_id}", params={"watchlist_name": collection_id})
            raise HTTPException(status_code=503, detail="Biometric engine returned an unexpected enrollment response")
        try:
            request.app.state.registry.execute(
                "INSERT INTO persons(collection_id, person_id) VALUES (?, ?)", (collection_id, person_id)
            )
            request.app.state.registry.commit()
        except sqlite3.Error:
            await client.delete(f"/api/v1/watchlist/{person_id}", params={"watchlist_name": collection_id})
            request.app.state.registry.rollback()
            raise HTTPException(status_code=503, detail="Enrollment state could not be saved")
    return {
        "person": {"id": person_id, "external_id": person_id},
        "faces": [enrolled_data["face"]],
        "rejected_images": [],
        "request_id": _request_id(request),
    }


@app.post("/v1/collections/{collection_id}/search", dependencies=[Depends(require_api_key)])
async def search_person(
    collection_id: str,
    request: Request,
    image: UploadFile = File(...),
    limit: int = Form(5),
    threshold: float = Form(0.4),
):
    collection_id = _safe_collection_id(collection_id)
    if not _read_collection(request.app.state.data_dir, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if not 1 <= limit <= 25:
        raise HTTPException(status_code=422, detail="Invalid result limit")
    if not 0.0 < threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Invalid threshold")
    image_bytes = await _read_image(image)
    async with watchlist_file_lock(
        request.app.state.data_dir,
        shared=True,
        fallback_lock=_watchlist_fallback_lock(request),
    ):
        response = await _client(request).post(
            "/api/v1/identify",
            data={"watchlist_name": collection_id, "top_k": limit, "threshold": threshold},
            files={"image": (image.filename or "verification.jpg", image_bytes, image.content_type or "image/jpeg")},
        )
        if response.status_code >= 400:
            await _engine_failure(response)
        data = response.json()
    face = data.get("face")
    if not isinstance(face, dict):
        raise HTTPException(status_code=503, detail="Biometric engine returned an invalid search response")
    engine_matches = data.get("matches") if isinstance(data.get("matches"), list) else []
    matches = []
    for match in engine_matches:
        if not isinstance(match, dict):
            continue
        external_id = match.get("label") or match.get("identity_id")
        if not isinstance(external_id, str) or not ID_RE.fullmatch(external_id):
            continue
        score = match.get("similarity")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            continue
        matches.append({
            "person": {"id": match.get("identity_id", external_id), "external_id": external_id},
            "similarity": float(score),
        })
    return {
        "matches": matches,
        "searched_face": {"liveness": _liveness_payload(face)},
        "request_id": _request_id(request),
    }


@app.post("/v1/detect", dependencies=[Depends(require_api_key)])
async def detect_faces(
    request: Request,
    image: UploadFile = File(...),
    max_faces: int = Form(2),
    collection_id: str = Form("students"),
):
    collection_id = _safe_collection_id(collection_id)
    if not _read_collection(request.app.state.data_dir, collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if not 1 <= max_faces <= 10:
        raise HTTPException(status_code=422, detail="Invalid result limit")
    image_bytes = await _read_image(image)
    response = await _client(request).post(
        "/api/v1/detect", files={"image": (image.filename or "liveness-frame.jpg", image_bytes, image.content_type or "image/jpeg")}
    )
    if response.status_code >= 400:
        await _engine_failure(response)
    data = response.json()
    engine_faces = data.get("faces") if isinstance(data, dict) else None
    if not isinstance(engine_faces, list):
        raise HTTPException(status_code=503, detail="Biometric engine returned an invalid detect response")
    faces = []
    # Do not truncate multi-face detections to make them look valid. Attendance's
    # caller requires exactly one face and will reject zero or multiple faces.
    for face in engine_faces:
        if not isinstance(face, dict):
            continue
        faces.append({
            "liveness": _liveness_payload(face),
            "landmarks": _face_landmarks(face),
        })
    return {"faces": faces, "request_id": _request_id(request)}


@app.delete("/v1/collections/{collection_id}/persons/{person_id}", dependencies=[Depends(require_api_key)])
async def delete_person(collection_id: str, person_id: str, request: Request):
    collection_id = _safe_collection_id(collection_id)
    person_id = _safe_person_id(person_id)
    if not _read_collection(request.app.state.data_dir, collection_id):
        # Treat a deleted collection as already removed for idempotent cleanup.
        return {"deleted": True, "already_missing": True, "request_id": _request_id(request)}
    async with request.app.state.engine_lock, watchlist_file_lock(
        request.app.state.data_dir,
        shared=False,
        fallback_lock=_watchlist_fallback_lock(request),
    ):
        response = await _client(request).delete(
            f"/api/v1/watchlist/{person_id}", params={"watchlist_name": collection_id}
        )
        if response.status_code not in (200, 404):
            await _engine_failure(response)
        row = request.app.state.registry.execute(
            "SELECT 1 FROM persons WHERE collection_id=? AND person_id=?", (collection_id, person_id)
        ).fetchone()
        request.app.state.registry.execute(
            "DELETE FROM persons WHERE collection_id=? AND person_id=?", (collection_id, person_id)
        )
        request.app.state.registry.commit()
    return {"deleted": True, "already_missing": response.status_code == 404 or row is None, "request_id": _request_id(request)}
