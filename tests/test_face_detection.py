from dataclasses import dataclass

import numpy as np

from biometric_service.face_detection import detect_yunet_multiscale


@dataclass
class Face:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    aligned: np.ndarray


class FakeYuNet:
    def __init__(self, detections_by_size=None, failing_sizes=None):
        self.detections_by_size = detections_by_size or {}
        self.failing_sizes = set(failing_sizes or ())
        self.calls = []
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = tuple(size)

    def detect(self, image):
        h, w = image.shape[:2]
        self.calls.append((w, h))
        if (w, h) in self.failing_sizes:
            import cv2
            raise cv2.error("simulated unsupported size")
        return 1, self.detections_by_size.get((w, h))


def raw_face(x=60, y=40, box_w=100, box_h=120, confidence=0.95):
    # YuNet: bbox, right eye, left eye, nose, right mouth, left mouth, score.
    return np.array([[
        x, y, box_w, box_h, x + box_w * .70, y + box_h * .25,
        x + box_w * .30, y + box_h * .25, x + box_w * .50, y + box_h * .50,
        x + box_w * .70, y + box_h * .75, x + box_w * .30, y + box_h * .75,
        confidence,
    ]], dtype=np.float32)


def run(image, detector, max_faces=0, aligned_sources=None):
    return detect_yunet_multiscale(
        detector=detector,
        image=image,
        max_faces=max_faces,
        detected_face_cls=Face,
        align_face=lambda original, landmarks: (
            aligned_sources.append((original.shape, landmarks.copy())) if aligned_sources is not None else None
        ) or np.zeros((112, 112, 3), dtype=np.uint8),
    )


def test_retries_other_sizes_and_maps_face_geometry_to_original_frame():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeYuNet({(320, 240): raw_face()})
    aligned_sources = []
    faces = run(image, detector, aligned_sources=aligned_sources)
    assert detector.calls[:3] == [(640, 480), (480, 360), (320, 240)]
    assert len(faces) == 1
    np.testing.assert_allclose(faces[0].bbox, [120, 80, 320, 320])
    assert aligned_sources[0][0] == (480, 640, 3)
    assert faces[0].aligned.shape == (112, 112, 3)


def test_large_camera_image_is_capped_and_no_face_fails_closed():
    image = np.zeros((1600, 3200, 3), dtype=np.uint8)
    detector = FakeYuNet()
    assert run(image, detector, max_faces=1) == []
    assert detector.calls[0] == (1280, 640)
    assert all(max(width, height) <= 2240 for width, height in detector.calls)
    assert len(detector.calls) == 5


def test_max_faces_is_applied_after_confidence_sorting():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rows = np.vstack([raw_face(confidence=0.75), raw_face(x=300, confidence=0.95)])
    faces = run(image, FakeYuNet({(640, 480): rows}), max_faces=1)
    assert len(faces) == 1
    assert abs(faces[0].confidence - 0.95) < 1e-6


def test_weak_close_face_detection_retries_and_replaces_duplicate_with_stronger_scale():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    weak = raw_face(x=60, y=40, box_w=100, box_h=120, confidence=0.42)
    # Same geometry at 0.75x, correctly expressed in the detector's scaled pixels.
    strong = raw_face(x=45, y=30, box_w=75, box_h=90, confidence=0.96)
    detector = FakeYuNet({(640, 480): weak, (480, 360): strong})
    faces = run(image, detector)
    assert detector.calls[:2] == [(640, 480), (480, 360)]
    assert len(faces) == 1
    assert abs(faces[0].confidence - 0.96) < 1e-6
    np.testing.assert_allclose(faces[0].bbox, [60, 40, 160, 160])


def test_upsampled_scale_can_find_a_distant_small_face():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # The fourth attempted size is an upsample. The output is mapped back to
    # the source image, not left in the larger detector coordinate system.
    face = raw_face(x=100, y=100, box_w=125, box_h=150, confidence=0.91)
    detector = FakeYuNet({(800, 600): face})
    faces = run(image, detector)
    assert detector.calls[:4] == [(640, 480), (480, 360), (320, 240), (800, 600)]
    assert len(faces) == 1
    np.testing.assert_allclose(faces[0].bbox, [80, 80, 180, 200])


def test_distinct_faces_are_not_collapsed_as_cross_scale_duplicates():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rows = np.vstack([
        raw_face(x=50, y=70, box_w=100, box_h=120, confidence=0.96),
        raw_face(x=330, y=70, box_w=100, box_h=120, confidence=0.93),
    ])
    faces = run(image, FakeYuNet({(640, 480): rows}))
    assert len(faces) == 2


def test_detector_size_error_falls_back_to_another_scale():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeYuNet(
        {(480, 360): raw_face(confidence=0.94)},
        failing_sizes={(640, 480)},
    )
    faces = run(image, detector)
    assert detector.calls[:2] == [(640, 480), (480, 360)]
    assert len(faces) == 1
