from __future__ import annotations

import sqlite3

import pytest

from biometric_service.liveness_challenges import (
    challenge_steps,
    initialize_liveness_tables,
    normalized_nose_position,
    validate_motion_sequence,
)


def test_challenge_has_center_and_both_randomized_turns():
    for _ in range(20):
        steps = challenge_steps()
        assert len(steps) == 5
        assert steps[0] == steps[2] == steps[4] == "center"
        assert steps.count("turn_left") == 1
        assert steps.count("turn_right") == 1
        assert steps in (
            ["center", "turn_left", "center", "turn_right", "center"],
            ["center", "turn_right", "center", "turn_left", "center"],
        )


def test_normalized_nose_position_uses_inter_eye_distance():
    landmarks = [[40, 20], [80, 20], [50, 50], [40, 70], [80, 70]]
    assert normalized_nose_position(landmarks) == pytest.approx(-0.25)


def test_motion_sequence_passes_for_opposing_turns_and_stable_center():
    steps = ["center", "turn_left", "center", "turn_right", "center"]
    ratios = [0.00, 0.20, 0.02, -0.22, -0.01]
    assert validate_motion_sequence(steps, ratios) == (True, "motion_challenge_passed")


def test_motion_sequence_rejects_static_screen_photo():
    steps = ["center", "turn_left", "center", "turn_right", "center"]
    ratios = [-0.01, 0.00, 0.01, -0.01, 0.00]
    passed, reason = validate_motion_sequence(steps, ratios)
    assert not passed
    assert reason in {"left_turn_not_detected", "right_turn_not_detected"}


def test_motion_sequence_rejects_wrong_directions():
    steps = ["center", "turn_left", "center", "turn_right", "center"]
    ratios = [0.0, -0.2, 0.0, 0.2, 0.0]
    passed, reason = validate_motion_sequence(steps, ratios)
    assert not passed
    assert reason in {"left_turn_not_detected", "right_turn_not_detected"}


def test_motion_sequence_rejects_unstable_center():
    steps = ["center", "turn_left", "center", "turn_right", "center"]
    ratios = [0.0, -0.2, 0.15, 0.2, 0.0]
    passed, reason = validate_motion_sequence(steps, ratios)
    assert not passed
    assert reason == "center_not_stable"


def test_motion_sequence_rejects_wrong_frame_count():
    assert validate_motion_sequence(["center"], [0.0]) == (False, "wrong_frame_count")


def test_initialize_liveness_tables_is_idempotent():
    db = sqlite3.connect(":memory:")
    initialize_liveness_tables(db)
    initialize_liveness_tables(db)
    columns = {row[1] for row in db.execute("PRAGMA table_info(liveness_sessions)")}
    assert {"session_id", "nonce_hash", "status", "proof_hash", "proof_expires_at"} <= columns
    db.close()


def test_consumed_proof_replay_returns_conflict():
    import asyncio
    import hashlib
    import time
    import uuid
    from types import SimpleNamespace

    from fastapi import HTTPException
    from biometric_service.liveness_challenges import (
        ConsumeLivenessProof,
        consume_liveness_proof,
    )

    db = sqlite3.connect(":memory:")
    initialize_liveness_tables(db)

    now = time.time()
    token = "replay-regression-proof-token-000000000000"
    proof_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    db.execute(
        """INSERT INTO liveness_sessions (
            session_id, subject_id, attendance_session_id, device_id,
            nonce_hash, steps_json, status, created_at, expires_at,
            current_step, observations_json, frame_hashes_json,
            proof_hash, proof_expires_at, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            uuid.uuid4().hex, "student-1", "attendance-1", "device-1",
            "unused-nonce-hash", "[]", "verified", now, now + 45,
            5, "[]", "[]", proof_hash, now + 30, "{}",
        ),
    )
    db.commit()

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                registry=db,
                liveness_lock=asyncio.Lock(),
            )
        )
    )
    payload = ConsumeLivenessProof(
        proof_token=token,
        subject_id="student-1",
        attendance_session_id="attendance-1",
        device_id="device-1",
    )

    async def exercise():
        first = await consume_liveness_proof(payload, request)
        assert first["consumed"] is True

        with pytest.raises(HTTPException) as exc:
            await consume_liveness_proof(payload, request)

        assert exc.value.status_code == 409

    try:
        asyncio.run(exercise())
    finally:
        db.close()
