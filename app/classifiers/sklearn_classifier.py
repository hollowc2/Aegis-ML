"""
app/classifiers/sklearn_classifier.py
======================================
Phase 1 classifier — TF-IDF + Logistic Regression (or Random Forest).

Extremely lightweight: <50 MB RAM, <5 ms inference latency.
Model is serialised with joblib and loaded once at startup.

Async wrapper: uses asyncio.to_thread so the sync sklearn inference
doesn't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SklearnClassifier:
    """
    Wraps a trained scikit-learn Pipeline (TF-IDF → classifier).

    Expected pipeline steps:
        - "tfidf": TfidfVectorizer
        - "clf":   LogisticRegression or RandomForestClassifier

    The pipeline must have been trained with classes [0=benign, 1=malicious].
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._pipeline: Pipeline | None = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the serialised model from disk. Called once during app startup."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"scikit-learn model not found at {path}. "
                "Run 'python -m training.phase1_sklearn.train' first."
            )
        self._pipeline = joblib.load(path)
        self._loaded = True
        logger.info("scikit-learn classifier loaded from %s", path)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ──────────────────────────────────────────────────────────────

    async def predict(self, text: str) -> dict[str, Any]:
        """
        Async prediction wrapper.
        Runs sync sklearn inference in a thread pool to avoid blocking the loop.

        Returns:
            {
                "label":          "benign" | "malicious",
                "malicious_prob": float,
                "benign_prob":    float,
            }
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("SklearnClassifier is not loaded. Call .load() first.")

        return await asyncio.to_thread(self._predict_sync, text)

    def _predict_sync(self, text: str) -> dict[str, Any]:
        """Synchronous inference — called inside a thread pool worker."""
        pipeline = self._pipeline
        assert pipeline is not None

        # predict_proba returns shape (n_samples, n_classes)
        proba: np.ndarray = pipeline.predict_proba([text])[0]

        # Classes are [0=benign, 1=malicious] by sklearn convention
        classes = pipeline.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}

        benign_idx = class_to_idx.get(0, 0)
        malicious_idx = class_to_idx.get(1, 1)

        benign_prob = float(proba[benign_idx])
        malicious_prob = float(proba[malicious_idx])

        label = "malicious" if malicious_prob > benign_prob else "benign"

        return {
            "label": label,
            "malicious_prob": malicious_prob,
            "benign_prob": benign_prob,
        }
