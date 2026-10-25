"""Apply checked runtime patches to the pinned OpenBiometrics source/wheel."""
from __future__ import annotations

import sysconfig
from pathlib import Path

UPSTREAM_ROOT = Path("/opt/openbiometrics")


def replace_exactly_once(path: Path, replacements: dict[str, str]) -> None:
    source = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        count = source.count(old)
        if count != 1:
            raise SystemExit(
                f"Expected exactly one upstream source block in {path}, found {count}: {old[:80]!r}"
            )
        source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8")


# Trim optional modules from the upstream API startup configuration.
api_main = UPSTREAM_ROOT / "api" / "app" / "main.py"
replace_exactly_once(
    api_main,
    {
        """            ctx_id=-1,  # CPU on Mac, set to 0 for GPU
        ),""": """            ctx_id=-1,
            det_thresh=0.35,
            enable_demographics=False,
            enable_quality=True,
        ),""",
        """        document=DocumentConfig(
            enabled=True,""": """        document=DocumentConfig(
            enabled=False,""",
        """        person=PersonConfig(
            enabled=True,""": """        person=PersonConfig(
            enabled=False,""",
        """        video=VideoConfig(
            enabled=True,""": """        video=VideoConfig(
            enabled=False,""",
        """        events=EventsConfig(
            enabled=True,
            webhooks_enabled=True,""": """        events=EventsConfig(
            enabled=False,
            webhooks_enabled=False,""",
    },
)

# The wheel is installed before this script runs. Patch the installed package,
# because runtime imports it from site-packages rather than the clone's engine/.
site_packages = Path(sysconfig.get_paths()["purelib"])
engine_package = site_packages / "openbiometrics"
liveness_path = engine_package / "core" / "liveness.py"
pipeline_path = engine_package / "core" / "pipeline.py"
detector_path = engine_package / "core" / "detector.py"

# YuNet's single full-resolution pass is brittle across phone camera outputs and
# near/far framing. Delegate to the tested multi-scale helper while keeping the
# aligned recognition crop and every downstream liveness/match check intact.
replace_exactly_once(
    detector_path,
    {
        '''    def _detect_yunet(self, image: np.ndarray, max_faces: int) -> list[DetectedFace]:
        """YuNet detection via OpenCV DNN."""
        h, w = image.shape[:2]
        self._yunet.setInputSize((w, h))
        _, raw = self._yunet.detect(image)

        if raw is None:
            return []

        results = []
        for det in raw:
            # YuNet output: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #                 nose_x, nose_y, right_mouth_x, right_mouth_y,
            #                 left_mouth_x, left_mouth_y, confidence]
            x, y, bw, bh = det[0:4]
            confidence = float(det[14])

            bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)

            # YuNet landmark order: right_eye, left_eye, nose, right_mouth, left_mouth
            # Our order: left_eye, right_eye, nose, left_mouth, right_mouth
            landmarks = np.array(
                [
                    [det[6], det[7]],   # left_eye
                    [det[4], det[5]],   # right_eye
                    [det[8], det[9]],   # nose
                    [det[12], det[13]], # left_mouth
                    [det[10], det[11]], # right_mouth
                ],
                dtype=np.float32,
            )

            aligned = _align_face(image, landmarks)
            results.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=landmarks,
                    confidence=confidence,
                    aligned=aligned,
                )
            )

        results.sort(key=lambda f: f.confidence, reverse=True)
        if max_faces > 0:
            results = results[:max_faces]
        return results
''': '''    def _detect_yunet(self, image: np.ndarray, max_faces: int) -> list[DetectedFace]:
        """Detect faces at several bounded scales for close-up and distant captures."""
        from biometric_service.face_detection import detect_yunet_multiscale

        return detect_yunet_multiscale(
            detector=self._yunet,
            image=image,
            max_faces=max_faces,
            detected_face_cls=DetectedFace,
            align_face=_align_face,
        )
''',
    },
)

replace_exactly_once(
    liveness_path,
    {
        """        # Softmax over [spoof, live] logits
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()

        live_score = float(probs[1])
""": """        # Pinned MiniFASNet V2 SE outputs logits in [real, spoof] order.
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()

        live_score = float(probs[0])
""",
        '''    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize and normalize for MiniFASNet."""
        img = cv2.resize(face_crop, self.input_size)
        img = img.astype(np.float32)
        img = (img / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img.astype(np.float32)
''': '''    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocess RGB input like the pinned MiniFASNet V2 SE release."""
        if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3 or face_crop.shape[2] != 3:
            raise ValueError("Liveness input must be a non-empty RGB image")

        target_w, target_h = self.input_size
        source_h, source_w = face_crop.shape[:2]
        ratio = min(target_w / source_w, target_h / source_h)
        scaled_w = max(1, int(source_w * ratio))
        scaled_h = max(1, int(source_h * ratio))
        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        img = cv2.resize(face_crop, (scaled_w, scaled_h), interpolation=interpolation)

        delta_w = target_w - scaled_w
        delta_h = target_h - scaled_h
        if delta_w < 0 or delta_h < 0:
            raise RuntimeError("Liveness model input size is incompatible with preprocessing")
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        if top or bottom or left or right:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_REFLECT_101
            )
        if img.shape[:2] != (target_h, target_w):
            raise RuntimeError("Liveness preprocessing produced an unexpected image size")

        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img[np.newaxis, ...].astype(np.float32)
''',
    },
)

replace_exactly_once(
    pipeline_path,
    {
        """            # Liveness
            if self._liveness is not None:
                result.is_live, result.liveness_score = self._liveness.check(face.aligned)
""": """            # Liveness: this MiniFASNet model needs a context-expanded RGB crop,
            # not the 112x112 recognition-alignment crop.
            if self._liveness is not None:
                x1, y1, x2, y2 = (float(value) for value in face.bbox)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w <= 0.0 or box_h <= 0.0:
                    raise RuntimeError("Face detector returned an invalid bounding box for liveness")

                source_h, source_w = image.shape[:2]
                side = max(1, int(max(box_w, box_h) * 1.5))
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                left = int(center_x - side / 2.0)
                top = int(center_y - side / 2.0)
                right = left + side
                bottom = top + side

                crop = image[max(0, top):min(source_h, bottom), max(0, left):min(source_w, right)]
                if crop.size == 0:
                    raise RuntimeError("Could not extract a liveness crop")
                crop = cv2.copyMakeBorder(
                    crop,
                    max(0, -top),
                    max(0, bottom - source_h),
                    max(0, -left),
                    max(0, right - source_w),
                    cv2.BORDER_REFLECT_101,
                )
                if crop.shape[0] != side or crop.shape[1] != side:
                    raise RuntimeError("Liveness crop dimensions were invalid")
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result.is_live, result.liveness_score = self._liveness.check(crop_rgb)
""",
    },
)


# The pinned upstream kernel references model-selection fields which its
# PipelineConfig dataclass omitted. Add the fields and pass configured values.
pipeline_config_path = engine_package / "core" / "pipeline.py"
replace_exactly_once(
    pipeline_config_path,
    {
        """    quality_gate: bool = False  # Skip recognition if quality fails
""": """    quality_gate: bool = False  # Skip recognition if quality fails

    detector: str = "auto"
    recognizer: str = "auto"
    demographics_model: str = "auto"
"""
    },
)

kernel_path = engine_package / "kernel.py"
replace_exactly_once(
    kernel_path,
    {
        """            enable_quality=face_cfg.enable_quality,
            quality_gate=face_cfg.quality_gate,
        )""": """            enable_quality=face_cfg.enable_quality,
            quality_gate=face_cfg.quality_gate,
            det_thresh=face_cfg.det_thresh,
            det_size=face_cfg.det_size,
            detector=face_cfg.detector,
            recognizer=face_cfg.recognizer,
            demographics_model=face_cfg.demographics_model,
        )"""
    },
)


# The community SFace recognizer emits 128-dimensional embeddings, while the
# pinned upstream watchlist defaults to 512-dimensional ArcFace embeddings.
# Size an empty FAISS index from the actual vector, guard mismatches before
# FAISS asserts, and preserve non-empty indexes if they use another dimension.
watchlist_path = engine_package / "watchlist" / "store.py"
replace_exactly_once(
    watchlist_path,
    {
        "    EMBEDDING_DIM = 512": "    EMBEDDING_DIM = 128",
        """        embedding = embedding.reshape(1, -1).astype(np.float32)
        with self._lock:
            self._index.add(embedding)
            self._identities.append(
""": """        embedding = embedding.reshape(1, -1).astype(np.float32)
        if embedding.shape[1] < 1:
            raise ValueError("Face embedding must not be empty")
        with self._lock:
            if embedding.shape[1] != self._index.d:
                if self._index.ntotal == 0:
                    # An empty index has no templates to lose; use the model's
                    # actual output dimension before inserting the first vector.
                    self._index = faiss.IndexFlatIP(int(embedding.shape[1]))
                else:
                    raise ValueError(
                        f"Face embedding dimension mismatch: index expects {self._index.d}, "
                        f"received {embedding.shape[1]}; existing templates were preserved"
                    )
            self._index.add(embedding)
            self._identities.append(
""",
        """        embedding = embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.size)
""": """        embedding = embedding.reshape(1, -1).astype(np.float32)
        if embedding.shape[1] != self._index.d:
            raise ValueError(
                f"Face query dimension mismatch: index expects {self._index.d}, "
                f"received {embedding.shape[1]}"
            )
        k = min(top_k, self.size)
""",
        """            all_vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self.size * self.EMBEDDING_DIM
            ).reshape(-1, self.EMBEDDING_DIM).copy()
""": """            index_dimension = self._index.d
            all_vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self.size * index_dimension
            ).reshape(-1, index_dimension).copy()
""",
        """            self._index = faiss.read_index(str(index_path))
            with open(meta_path) as f:
                self._identities = json.load(f)
""": """            loaded_index = faiss.read_index(str(index_path))
            with open(meta_path) as f:
                loaded_identities = json.load(f)
            if not isinstance(loaded_identities, list) or loaded_index.ntotal != len(loaded_identities):
                raise ValueError(
                    "Persisted watchlist index and identity metadata are inconsistent; "
                    "refusing to overwrite biometric templates"
                )
            self._index = loaded_index
            self._identities = loaded_identities
""",
    },
)

print("Applied CPU-only API configuration, pinned MiniFASNet preprocessing, and recognizer-compatible FAISS watchlists.")


# BIOMETRIC_CAPACITY_PATCH_V1: bounded CPU threading for multi-worker attendance inference.
# This is applied during the Docker build to the pinned installed OpenBiometrics package.
import os as _biometric_capacity_os

_session_path = engine_package / "runtime" / "session.py"
replace_exactly_once(
    _session_path,
    {
        """from pathlib import Path

import numpy as np
import onnxruntime as ort
""": """from pathlib import Path

import os as _biometric_capacity_os
import numpy as np
import onnxruntime as ort
""",
        """        self._session = ort.InferenceSession(self._model_path, providers=self._providers)
""": """        # BIOMETRIC_CAPACITY_PATCH_V1: tune per-process ONNX CPU usage to avoid worker oversubscription.
        try:
            intra_threads = int(_biometric_capacity_os.getenv("BIOMETRIC_ORT_INTRA_THREADS", "1"))
        except ValueError as exc:
            raise RuntimeError("BIOMETRIC_ORT_INTRA_THREADS must be a positive integer") from exc
        if not 1 <= intra_threads <= 64:
            raise RuntimeError("BIOMETRIC_ORT_INTRA_THREADS must be between 1 and 64")
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=session_options,
            providers=self._providers,
        )
"""
    },
)

# BIOMETRIC_KERNEL_SYNTAX_GUARD_V2
# Normalize only known identical duplicates in the single PipelineConfig call.
# Unexpected multiplicities or values fail closed; no thresholds are changed.
_kernel_source = kernel_path.read_text(encoding="utf-8")
_kernel_lines = _kernel_source.splitlines()
_call_starts = [
    i for i, line in enumerate(_kernel_lines)
    if "pipeline_config = PipelineConfig(" in line
]
if len(_call_starts) != 1:
    raise SystemExit(
        "Expected exactly one PipelineConfig construction in patched kernel.py; "
        f"found {len(_call_starts)}"
    )
_call_start = _call_starts[0]
_call_end = next(
    (i for i in range(_call_start + 1, len(_kernel_lines))
     if _kernel_lines[i].strip() == ")" and len(_kernel_lines[i]) - len(_kernel_lines[i].lstrip()) == 8),
    None,
)
if _call_end is None:
    raise SystemExit("Could not locate the end of PipelineConfig call in kernel.py")

# These exact assignments were found duplicated during build-time validation.
# Remove one later copy only when both lines are textually identical.
for _target in (
    "det_thresh=face_cfg.det_thresh,",
    "det_size=face_cfg.det_size,",
):
    _target_lines = [
        i for i in range(_call_start + 1, _call_end)
        if _kernel_lines[i].strip() == _target
    ]
    if len(_target_lines) == 2:
        _kernel_lines.pop(_target_lines[1])
        _call_end -= 1
    elif len(_target_lines) != 1:
        raise SystemExit(
            "Unexpected PipelineConfig assignment; refusing to guess: "
            f"{_target!r} occurred {len(_target_lines)} times"
        )

_kernel_source = "\n".join(_kernel_lines) + "\n"
kernel_path.write_text(_kernel_source, encoding="utf-8")

# Compile the actual installed files that will be imported at runtime.
_validation_paths = [kernel_path, pipeline_config_path, liveness_path, pipeline_path]
if "_session_path" in globals():
    _validation_paths.append(_session_path)
for _patched_path in _validation_paths:
    try:
        compile(_patched_path.read_text(encoding="utf-8"), str(_patched_path), "exec")
    except SyntaxError as _exc:
        raise SystemExit(
            f"Patched upstream source failed syntax validation: {_patched_path}: {_exc}"
        ) from _exc

print("Validated patched upstream kernel, pipeline, liveness and ONNX session syntax.")

# START: patched upstream syntax guard
def _validate_patched_upstream_sources():
    import compileall
    import sysconfig
    from pathlib import Path

    engine_root = (
        Path(sysconfig.get_paths()["purelib"]) / "openbiometrics"
    )
    kernel_file = engine_root / "kernel.py"

    if not kernel_file.is_file():
        raise SystemExit(
            f"Patched kernel not found: {kernel_file}"
        )

    raw = kernel_file.read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.decode("utf-8").splitlines()

    starts = [
        i for i, line in enumerate(lines)
        if "pipeline_config = PipelineConfig(" in line
    ]
    if len(starts) != 1:
        raise SystemExit(
            "Expected exactly one PipelineConfig constructor; "
            f"found {len(starts)}"
        )

    start = starts[0]
    start_indent = len(lines[start]) - len(lines[start].lstrip())

    ends = [
        i for i in range(start + 1, len(lines))
        if lines[i].strip() == ")"
        and len(lines[i]) - len(lines[i].lstrip()) <= start_indent
    ]
    if not ends:
        raise SystemExit(
            "Could not locate the end of PipelineConfig constructor"
        )

    end = ends[0]
    target = "det_thresh=face_cfg.det_thresh,"
    positions = [
        i for i in range(start + 1, end)
        if lines[i].strip() == target
    ]

    if len(positions) == 2:
        del lines[positions[1]]
        kernel_file.write_bytes(
            newline.join(line.encode("utf-8") for line in lines)
            + newline
        )
        print("Removed one duplicate kernel det_thresh argument.")
    elif len(positions) == 1:
        print("Kernel det_thresh argument is already unique.")
    else:
        raise SystemExit(
            "Expected one or two kernel det_thresh arguments; "
            f"found {len(positions)}. Refusing to guess."
        )

    for root in (
        engine_root,
        Path("/opt/openbiometrics/api/app"),
    ):
        if root.is_dir() and not compileall.compile_dir(
            str(root), quiet=1, force=True
        ):
            raise SystemExit(
                f"Syntax compilation failed under {root}"
            )

    print("Patched upstream Python source compiled successfully.")

_validate_patched_upstream_sources()
# END: patched upstream syntax guard
