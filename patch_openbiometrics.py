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
