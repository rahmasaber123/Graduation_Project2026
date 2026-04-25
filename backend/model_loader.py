"""
NBIS Model Loader
=================
Rebuilds the MobileNetV2 + triplet-head architecture in-process and loads
weights only. Avoids keras.models.load_model() entirely, so the file is
immune to Keras/TF version drift between Colab and local.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# Pillow 10+ compatibility
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS


class L2Normalize(keras.layers.Layer):
    """L2-normalize along axis=1. Keeps embeddings on the unit hypersphere."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()


class NBISSystem:
    """In-memory NBIS identification system. Loaded once at API startup."""

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.model: keras.Model | None = None
        self.index: faiss.Index | None = None
        self.mapping: list[dict] = []
        self.threshold: float = 0.0
        self.prep_config: dict = {}
        self.img_size: tuple[int, int] = (224, 224)
        self.ready: bool = False

    # ─────────────────────────────────────────────────────────────────────
    def _build_model(self, img_size: tuple[int, int], embedding_dim: int) -> Model:
        """Rebuild the exact architecture from the notebook."""
        base = MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights=None,           # weights come from the checkpoint below
        )
        inputs  = keras.Input(shape=(*img_size, 3), name='fingerprint_input')
        x       = base(inputs, training=False)
        x       = layers.GlobalAveragePooling2D()(x)
        x       = layers.Dense(256, activation='relu')(x)
        x       = layers.BatchNormalization()(x)
        x       = layers.Dropout(0.3)(x)
        x       = layers.Dense(embedding_dim)(x)
        outputs = L2Normalize(name='l2_embedding')(x)
        return Model(inputs, outputs, name='embedding_network')

    # ─────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load all artifacts from disk. Called once at startup."""
        d = self.artifacts_dir
        if not d.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {d}")

        # 1. Preprocessing config
        with open(d / "nbis_preprocessing_config.json") as f:
            self.prep_config = json.load(f)
        self.img_size = tuple(self.prep_config["img_size"])

        # 2. Model — rebuild architecture + load weights only (version-safe)
        self.model = self._build_model(
            self.img_size, int(self.prep_config["embedding_dim"])
        )
        self.model.load_weights(d / "nbis_embedding_model.keras")

        # Warm up the TF graph so the first real request is fast
        dummy = np.zeros((1, *self.img_size, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

        # 3. FAISS index
        self.index = faiss.read_index(str(d / "nbis_faiss.index"))

        # 4. Mapping
        with open(d / "nbis_index_mapping.pkl", "rb") as f:
            self.mapping = pickle.load(f)

        # 5. Threshold
        with open(d / "nbis_threshold.json") as f:
            self.threshold = float(json.load(f)["eer_threshold"])

        self.ready = True

    # ─────────────────────────────────────────────────────────────────────
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize(self.img_size, _LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    # ─────────────────────────────────────────────────────────────────────
    def identify(self, image_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("NBIS system not loaded. Call load() first.")

        arr   = self.preprocess(image_bytes)
        batch = np.expand_dims(arr, 0).astype(np.float32)

        embedding = self.model.predict(batch, verbose=0)
        emb = embedding.astype(np.float32).copy()
        faiss.normalize_L2(emb)

        k = max(1, min(top_k, self.index.ntotal))
        scores, indices = self.index.search(emb, k)
        top_score = float(scores[0][0])
        top_idx   = int(indices[0][0])

        top_k_results = [
            {
                "rank"       : r + 1,
                "subject_id" : self.mapping[int(indices[0][r])]["subject_id"],
                "finger"     : self.mapping[int(indices[0][r])].get("finger"),
                "hand"       : self.mapping[int(indices[0][r])].get("hand"),
                "identity_id": self.mapping[int(indices[0][r])].get("identity_id"),
                "score"      : float(scores[0][r]),
            }
            for r in range(k)
        ]

        if top_score >= self.threshold:
            m = self.mapping[top_idx]
            return {
                "status"      : "MATCH",
                "confidence"  : round(top_score * 100, 2),
                "similarity"  : top_score,
                "threshold"   : self.threshold,
                "subject_id"  : m["subject_id"],
                "hand"        : m.get("hand"),
                "finger"      : m.get("finger"),
                "identity_id" : m.get("identity_id"),
                "parent_id"   : m.get("parent_id"),
                "parent_name" : m.get("full_name"),
                "parent_phone": m.get("phone"),
                "parent_email": m.get("email"),
                "city"        : m.get("city"),
                "top_k"       : top_k_results,
            }
        else:
            return {
                "status"    : "NO_MATCH",
                "confidence": round(top_score * 100, 2),
                "similarity": top_score,
                "threshold" : self.threshold,
                "subject_id": None,
                "identity_id": None,
                "message"   : "Similarity below threshold — identity could not be verified.",
                "top_k"     : top_k_results,
            }

    # ─────────────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        return {
            "ready"        : self.ready,
            "model_loaded" : self.model is not None,
            "index_ready"  : self.index is not None and self.index.ntotal > 0,
            "index_size"   : int(self.index.ntotal) if self.index else 0,
            "embedding_dim": int(self.prep_config.get("embedding_dim", 0)),
            "img_size"     : list(self.img_size),
            "threshold"    : self.threshold,
            "artifacts_dir": str(self.artifacts_dir),
        }