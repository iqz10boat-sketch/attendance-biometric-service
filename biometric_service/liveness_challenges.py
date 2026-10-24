"""Pilot challenge-bound liveness sessions for the Attendance Biometric API.

Important: this is a defence-in-depth pilot, not a certified PAD solution. The
current passive model accepted a screen-photo test; therefore these routes must
not be treated as production approval. Session state is stored in SQLite and is
suitable only for a single service instance with persistent storage.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import sqlite3
import time
import uuid
from statistics import median
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
SESSION_TTL_SECONDS = 45
PROOF_TTL_SECONDS = 30
MIN_FRAME_INTERVAL_SECONDS = 0.20
MIN_TURN_RATIO = 0.12
CENTER_TOLERANCE = 0.10
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
router = APIRouter()


class StartLivenessSession(BaseModel):
    subject_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    attendance_session_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    device_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)


class ConsumeLivenessProof(BaseModel):
    proof_token: str = Field(min_length=32, max_length=256)
    subject_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    attendance_session_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    device_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)


def initialize_liveness_tables(db: sqlite3.Connection) -> None:
    """Create the short-lived challenge state table in the existing registry DB."""
    db.execute(
        """CREATE TABLE IF NOT EXISTS liveness_sessions (
            session_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            attendance_session_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            nonce_hash TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            current_step INTEGER NOT NULL DEFAULT 0,
            observations_json TEXT NOT NULL DEFAULT '[]',
            frame_hashes_json TEXT NOT NULL DEFAULT '[]',
            last_frame_at REAL,
            proof_hash TEXT,
            proof_expires_at REAL,
            result_json TEXT
        )"""
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_liveness_sessions_expiry "
        "ON liveness_sessions(status, expires_at)"
    )
    db.commit()


def _challenge_options() -> list[list[str]]:
    left_first = ["center", "turn_left", "center", "turn_right", "center"]
    right_first = ["center", "turn_right", "center", "turn_left", "center"]
    return [left_first, right_first]


def challenge_steps() -> list[str]:
    """Return a fresh randomized left/right sequence with neutral checkpoints."""
    return list(secrets.choice(_challenge_options()))


def _step_payload(steps: list[str]) -> list[dict[str, Any]]:
    prompts = {
        "center": "Look straight at the camera and keep your head level.",
        "turn_left": "Turn your nose toward the LEFT edge of the screen.",
        "turn_right": "Turn your nose toward the RIGHT edge of the screen.",
    }
    return [
        {"step_index": index, "action": action, "prompt": prompts[action]}
        for index, action in enumerate(steps)
    ]


def normalized_nose_position(landmarks: Any) -> float:
    """Return nose x-offset from eye midpoint, normalized by inter-eye distance.

    Expected landmark order is [eye-1, eye-2, nose, mouth-1, mouth-2], as used
    by the pinned detector. This metric is only a coarse yaw proxy, not pose
    estimation or anti-spoof proof; it must be calibrated with real captures.
    """
    if not isinstance(landmarks, (list, tuple)) or len(landmarks) < 3:
        raise ValueError("Face landmarks are unavailable")
    points: list[tuple[float, float]] = []
    try:
        for point in landmarks[:3]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError("Invalid face landmarks")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Invalid face landmarks")
            points.append((x, y))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid face landmarks") from exc

    eye1, eye2, nose = points
    eye_distance = math.hypot(eye2[0] - eye1[0], eye2[1] - eye1[1])
    if eye_distance < 8.0:
        raise ValueError("Face landmarks are too small")
    eye_mid_x = (eye1[0] + eye2[0]) / 2.0
    ratio = (nose[0] - eye_mid_x) / eye_distance
    if not math.isfinite(ratio) or abs(ratio) > 1.5:
        raise ValueError("Face landmarks are invalid")
    return ratio


def validate_motion_sequence(steps: list[str], ratios: list[float]) -> tuple[bool, str]:
    """Validate left/right motion and return-to-center across a five-frame set.

    Thresholds are conservative starting values for local calibration, not
    empirically validated operating points. Production use remains prohibited.
    """
    if len(steps) != 5 or len(ratios) != 5:
        return False, "wrong_frame_count"
    if steps.count("center") != 3 or steps.count("turn_left") != 1 or steps.count("turn_right") != 1:
        return False, "invalid_challenge"
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in ratios):
        return False, "invalid_pose_measurements"

    centers = [ratios[i] for i, action in enumerate(steps) if action == "center"]
    baseline = median(centers)
    if max(centers) - min(centers) > CENTER_TOLERANCE:
        return False, "center_not_stable"

    left_index = steps.index("turn_left")
    right_index = steps.index("turn_right")
    # This detector/capture path reports leftward screen motion as positive.
    # Keep camera image orientation consistent between capture and detection.
    if ratios[left_index] < baseline + MIN_TURN_RATIO:
        return False, "left_turn_not_detected"
    if ratios[right_index] > baseline - MIN_TURN_RATIO:
        return False, "right_turn_not_detected"
    return True, "motion_challenge_passed"


def _row(db: sqlite3.Connection, session_id: str) -> tuple[Any, ...] | None:
    return db.execute(
        """SELECT session_id, subject_id, attendance_session_id, device_id,
                  nonce_hash, steps_json, status, created_at, expires_at,
                  current_step, observations_json, frame_hashes_json, last_frame_at,
                  proof_hash, proof_expires_at, result_json
           FROM liveness_sessions WHERE session_id=?""",
        (session_id,),
    ).fetchone()


def _hash(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _get_landmarks(face: dict[str, Any]) -> Any:
    detection = face.get("detection")
    if isinstance(detection, dict):
        return detection.get("landmarks")
    return face.get("landmarks")


def _get_live(face: dict[str, Any]) -> tuple[bool, float] | None:
    raw = face.get("liveness")
    if not isinstance(raw, dict) or not isinstance(raw.get("is_live"), bool):
        return None
    score = raw.get("score", raw.get("live_score"))
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return raw["is_live"], score


def _mark_rejected(db: sqlite3.Connection, session_id: str, reason: str) -> None:
    db.execute(
        "UPDATE liveness_sessions SET status='rejected', result_json=? "
        "WHERE session_id=? AND status='pending'",
        (json.dumps({"reason": reason}), session_id),
    )
    db.commit()


def _load_pending_session(request: Request, session_id: str) -> tuple[Any, ...]:
    if not re_full_session_id(session_id):
        raise HTTPException(status_code=404, detail="Liveness session not found")
    db = request.app.state.registry
    row = _row(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Liveness session not found")
    if row[6] != "pending":
        raise HTTPException(status_code=409, detail="Liveness session is not active")
    if time.time() >= row[8]:
        db.execute("UPDATE liveness_sessions SET status='expired' WHERE session_id=?", (session_id,))
        db.commit()
        raise HTTPException(status_code=410, detail="Liveness session expired")
    return row


def re_full_session_id(value: str) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 64 and all(c in "0123456789abcdef-" for c in value.lower())


@router.post("/sessions", status_code=201)
async def start_liveness_session(payload: StartLivenessSession, request: Request):
    db: sqlite3.Connection = request.app.state.registry
    now = time.time()
    db.execute(
        """DELETE FROM liveness_sessions
           WHERE (status IN ('consumed','rejected','expired') AND created_at < ?)
              OR (status NOT IN ('verified') AND expires_at < ?)
              OR (status='verified' AND proof_expires_at IS NOT NULL AND proof_expires_at < ?)""",
        (now - 3600, now, now),
    )
    steps = challenge_steps()
    session_id = uuid.uuid4().hex
    nonce = secrets.token_urlsafe(32)
    expires_at = now + SESSION_TTL_SECONDS
    db.execute(
        """INSERT INTO liveness_sessions
           (session_id, subject_id, attendance_session_id, device_id, nonce_hash,
            steps_json, status, created_at, expires_at, current_step,
            observations_json, frame_hashes_json)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0, '[]', '[]')""",
        (
            session_id,
            payload.subject_id,
            payload.attendance_session_id,
            payload.device_id,
            _hash(nonce),
            json.dumps(steps),
            now,
            expires_at,
        ),
    )
    db.commit()
    return {
        "session_id": session_id,
        "nonce": nonce,
        "steps": _step_payload(steps),
        "expires_at": expires_at,
        "expires_in_seconds": SESSION_TTL_SECONDS,
        "pilot_only": True,
        "production_approved": False,
    }


@router.post("/sessions/{session_id}/frames")
async def submit_liveness_frame(
    session_id: str,
    request: Request,
    nonce: str = Form(...),
    step_index: int = Form(..., ge=0, le=4),
    image: UploadFile = File(...),
):
    # Serialize all challenge state transitions so concurrent frame posts cannot
    # both advance the same step. It is initialized in the main app lifespan.
    async with request.app.state.liveness_lock:
        db: sqlite3.Connection = request.app.state.registry
        row = _load_pending_session(request, session_id)
        if not hmac.compare_digest(_hash(nonce), row[4]):
            _mark_rejected(db, session_id, "invalid_nonce")
            raise HTTPException(status_code=401, detail="Invalid liveness session token")
        if step_index != row[9]:
            _mark_rejected(db, session_id, "out_of_order_frame")
            raise HTTPException(status_code=409, detail="Liveness frames must be submitted in challenge order")
        steps: list[str] = json.loads(row[5])
        if step_index >= len(steps):
            _mark_rejected(db, session_id, "too_many_frames")
            raise HTTPException(status_code=409, detail="All challenge frames were already submitted")
        now = time.time()
        if row[12] is not None and now - float(row[12]) < MIN_FRAME_INTERVAL_SECONDS:
            raise HTTPException(status_code=429, detail="Frames are arriving too quickly; try again after a short pause")
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            _mark_rejected(db, session_id, "invalid_content_type")
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are accepted")
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes:
            _mark_rejected(db, session_id, "empty_image")
            raise HTTPException(status_code=400, detail="Image is required")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            _mark_rejected(db, session_id, "image_too_large")
            raise HTTPException(status_code=413, detail="The uploaded image exceeds the maximum size")

        image_hash = _hash(image_bytes)
        previous_hashes: list[str] = json.loads(row[11])
        if image_hash in previous_hashes:
            _mark_rejected(db, session_id, "repeated_frame")
            raise HTTPException(status_code=422, detail="Repeated frames are not accepted")

        client = getattr(request.app.state, "engine_client", None)
        engine_lock = getattr(request.app.state, "engine_lock", None)
        if client is None or engine_lock is None:
            raise HTTPException(status_code=503, detail="Biometric engine is not ready")
        async with engine_lock:
            response = await client.post(
                "/api/v1/detect",
                files={"image": (image.filename or "liveness-frame.jpg", image_bytes, image.content_type)},
            )
        if response.status_code >= 500:
            raise HTTPException(status_code=503, detail="Biometric engine failed to process the frame")
        if response.status_code >= 400:
            _mark_rejected(db, session_id, "engine_rejected_frame")
            raise HTTPException(status_code=422, detail="The image was rejected by the biometric engine")
        data = response.json()
        faces = data.get("faces") if isinstance(data, dict) else None
        if not isinstance(faces, list) or len(faces) != 1 or not isinstance(faces[0], dict):
            _mark_rejected(db, session_id, "face_count_not_one")
            raise HTTPException(status_code=422, detail="Exactly one face is required in every challenge frame")
        live = _get_live(faces[0])
        if live is None or live[0] is not True:
            _mark_rejected(db, session_id, "liveness_not_positive")
            raise HTTPException(status_code=422, detail="The passive liveness model did not positively evaluate this frame")
        try:
            nose_ratio = normalized_nose_position(_get_landmarks(faces[0]))
        except ValueError:
            _mark_rejected(db, session_id, "landmarks_unavailable")
            raise HTTPException(status_code=422, detail="Face landmarks could not be evaluated")

        observations: list[dict[str, Any]] = json.loads(row[10])
        observations.append({
            "step_index": step_index,
            "action": steps[step_index],
            "nose_ratio": nose_ratio,
            "passive_live_score": live[1],
        })
        previous_hashes.append(image_hash)
        next_step = step_index + 1
        db.execute(
            """UPDATE liveness_sessions
               SET current_step=?, observations_json=?, frame_hashes_json=?, last_frame_at=?
               WHERE session_id=? AND status='pending' AND current_step=?""",
            (next_step, json.dumps(observations), json.dumps(previous_hashes), now, session_id, step_index),
        )
        db.commit()
        return {
            "session_id": session_id,
            "step_index": step_index,
            "accepted": True,
            "steps_remaining": len(steps) - next_step,
            "next_step": _step_payload(steps)[next_step] if next_step < len(steps) else None,
            "pilot_only": True,
            "production_approved": False,
        }


@router.post("/sessions/{session_id}/complete")
async def complete_liveness_session(session_id: str, request: Request, nonce: str = Form(...)):
    async with request.app.state.liveness_lock:
        db: sqlite3.Connection = request.app.state.registry
        row = _load_pending_session(request, session_id)
        if not hmac.compare_digest(_hash(nonce), row[4]):
            _mark_rejected(db, session_id, "invalid_nonce")
            raise HTTPException(status_code=401, detail="Invalid liveness session token")
        steps: list[str] = json.loads(row[5])
        observations: list[dict[str, Any]] = json.loads(row[10])
        if row[9] != len(steps) or len(observations) != len(steps):
            _mark_rejected(db, session_id, "incomplete_challenge")
            raise HTTPException(status_code=422, detail="The full liveness challenge must be completed")
        ratios = [float(item["nose_ratio"]) for item in observations]
        passed, reason = validate_motion_sequence(steps, ratios)
        if not passed:
            _mark_rejected(db, session_id, reason)
            raise HTTPException(status_code=422, detail="The liveness motion challenge was not completed")

        proof_token = secrets.token_urlsafe(32)
        proof_expires_at = time.time() + PROOF_TTL_SECONDS
        result = {
            "motion_challenge_passed": True,
            "frames_evaluated": len(observations),
            "lowest_passive_live_score": min(float(item["passive_live_score"]) for item in observations),
            "pilot_only": True,
            "production_approved": False,
        }
        db.execute(
            """UPDATE liveness_sessions SET status='verified', proof_hash=?,
               proof_expires_at=?, result_json=? WHERE session_id=? AND status='pending'""",
            (_hash(proof_token), proof_expires_at, json.dumps(result), session_id),
        )
        db.commit()
        return {
            "session_id": session_id,
            "status": "pilot_challenge_passed",
            "proof_token": proof_token,
            "proof_expires_at": proof_expires_at,
            **result,
        }


@router.post("/proofs/consume")
async def consume_liveness_proof(payload: ConsumeLivenessProof, request: Request):
    """Consume a one-time proof, bound to the original student/session/device.

    This route is intended only for trusted backend-to-backend use. Its result is
    explicitly marked pilot-only and must not be treated as production approval.
    """
    async with request.app.state.liveness_lock:
        db: sqlite3.Connection = request.app.state.registry
        now = time.time()
        row = db.execute(
            """SELECT session_id, subject_id, attendance_session_id, device_id,
                      status, proof_expires_at, result_json
               FROM liveness_sessions WHERE proof_hash=?""",
            (_hash(payload.proof_token),),
        ).fetchone()
        if row is None or row[5] is None or float(row[5]) <= now:
            raise HTTPException(status_code=401, detail="Invalid or expired liveness proof")
        if (
            row[1] != payload.subject_id
            or row[2] != payload.attendance_session_id
            or row[3] != payload.device_id
        ):
            raise HTTPException(status_code=401, detail="Invalid or expired liveness proof")
        if row[4] == "consumed":
            raise HTTPException(status_code=409, detail="Liveness proof has already been consumed")
        if row[4] != "verified":
            raise HTTPException(status_code=401, detail="Invalid or expired liveness proof")
        updated = db.execute(
            "UPDATE liveness_sessions SET status='consumed' WHERE session_id=? AND status='verified' AND proof_hash=?",
            (row[0], _hash(payload.proof_token)),
        )
        db.commit()
        if updated.rowcount != 1:
            raise HTTPException(status_code=409, detail="Liveness proof has already been consumed")
        return {
            "consumed": True,
            "session_id": row[0],
            "subject_id": row[1],
            "attendance_session_id": row[2],
            "device_id": row[3],
            "result": json.loads(row[6] or "{}"),
            "pilot_only": True,
            "production_approved": False,
        }
