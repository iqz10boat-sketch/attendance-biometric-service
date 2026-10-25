"""Robust multi-scale YuNet detection for inconsistent mobile camera framing.

Only face localization uses scaled copies of the image. Facial landmarks are
mapped back to the original pixels before the aligned 112x112 recognition crop
is created. This module does not make identity or liveness decisions.
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

MAX_DETECTOR_LONGEST_SIDE = 1280
_SCALE_FACTORS = (1.0, 0.75, 0.5, 1.25, 1.5)
# Confidence controls only whether a second detector pass is worth the latency;
# it is not a face-match score and never authorizes attendance.
_EARLY_EXIT_CONFIDENCE = 0.85
_DUPLICATE_IOU = 0.65
_DUPLICATE_CENTER_FRACTION = 0.20


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_same_face(left: Any, right: Any) -> bool:
    """Suppress a repeated detection only when overlap and center both agree."""
    a = np.asarray(left.bbox, dtype=np.float32)
    b = np.asarray(right.bbox, dtype=np.float32)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    if _iou(a, b) < _DUPLICATE_IOU:
        return False
    ac = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    bc = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    size = max(float(a[2] - a[0]), float(a[3] - a[1]), float(b[2] - b[0]), float(b[3] - b[1]), 1.0)
    return float(np.linalg.norm(ac - bc)) <= size * _DUPLICATE_CENTER_FRACTION


def _deduplicate_faces(faces: list[Any], max_faces: int) -> list[Any]:
    faces.sort(key=lambda face: float(face.confidence), reverse=True)
    kept: list[Any] = []
    for candidate in faces:
        if not any(_is_same_face(candidate, existing) for existing in kept):
            kept.append(candidate)
        if max_faces > 0 and len(kept) >= max_faces:
            break
    return kept


def detect_yunet_multiscale(
    *,
    detector: Any,
    image: np.ndarray,
    max_faces: int,
    detected_face_cls: type,
    align_face: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> list[Any]:
    """Detect faces on bounded image scales, retaining original-frame geometry.

    A high-confidence detector pass exits immediately for latency. If a pass
    returns only weak candidates, alternate down/up-scaled copies are inspected;
    the best result is chosen after cross-scale duplicate suppression. Detection
    never accepts a student identity: the caller must still run liveness and
    identity matching on the aligned face.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        return []
    if detector is None:
        return []

    source_h, source_w = image.shape[:2]
    if source_h < 1 or source_w < 1:
        return []

    base_scale = min(1.0, MAX_DETECTOR_LONGEST_SIDE / float(max(source_h, source_w)))
    input_sizes: list[tuple[int, int]] = []
    seen_sizes: set[tuple[int, int]] = set()
    for factor in _SCALE_FACTORS:
        scale = min(1.75, max(0.25, base_scale * factor))
        size = (max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale))))
        if size not in seen_sizes:
            seen_sizes.add(size)
            input_sizes.append(size)

    candidates: list[Any] = []
    for input_w, input_h in input_sizes:
        if (input_w, input_h) == (source_w, source_h):
            detector_image = image
        else:
            scale_x = input_w / float(source_w)
            scale_y = input_h / float(source_h)
            interpolation = cv2.INTER_CUBIC if max(scale_x, scale_y) > 1.0 else cv2.INTER_AREA
            detector_image = cv2.resize(image, (input_w, input_h), interpolation=interpolation)

        try:
            detector.setInputSize((input_w, input_h))
            _, raw = detector.detect(detector_image)
        except cv2.error:
            # Camera-specific input sizes can fail; continue at another scale.
            continue

        if raw is None or len(raw) == 0:
            continue

        source_scale_x = source_w / float(input_w)
        source_scale_y = source_h / float(input_h)
        scale_faces: list[Any] = []
        for detection in raw:
            values = np.asarray(detection, dtype=np.float32).reshape(-1)
            if values.size < 15 or not np.isfinite(values[:15]).all():
                continue
            x, y, box_w, box_h = (float(value) for value in values[:4])
            confidence = float(values[14])
            if box_w <= 0.0 or box_h <= 0.0 or not 0.0 <= confidence <= 1.0:
                continue

            bbox = np.array([
                x * source_scale_x, y * source_scale_y,
                (x + box_w) * source_scale_x, (y + box_h) * source_scale_y,
            ], dtype=np.float32)
            # YuNet order: right eye, left eye, nose, right mouth, left mouth.
            landmarks = np.array([
                [values[6] * source_scale_x, values[7] * source_scale_y],
                [values[4] * source_scale_x, values[5] * source_scale_y],
                [values[8] * source_scale_x, values[9] * source_scale_y],
                [values[12] * source_scale_x, values[13] * source_scale_y],
                [values[10] * source_scale_x, values[11] * source_scale_y],
            ], dtype=np.float32)
            if not np.isfinite(bbox).all() or not np.isfinite(landmarks).all():
                continue
            try:
                aligned = align_face(image, landmarks)
            except (cv2.error, ValueError, IndexError):
                continue
            if not isinstance(aligned, np.ndarray) or aligned.size == 0:
                continue
            scale_faces.append(detected_face_cls(
                bbox=bbox, landmarks=landmarks, confidence=confidence, aligned=aligned
            ))

        candidates.extend(scale_faces)
        # Common clear captures take one pass. Weak results trigger fallback
        # scales; this does not alter any downstream match/liveness threshold.
        if scale_faces and max(float(face.confidence) for face in scale_faces) >= _EARLY_EXIT_CONFIDENCE:
            break

    return _deduplicate_faces(candidates, max_faces)
