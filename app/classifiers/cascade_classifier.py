"""
app/classifiers/cascade_classifier.py
=======================================
Two-stage cascade classifier: sklearn fast path → ONNX/HF slow path.

For inputs where sklearn is highly confident (prob ≥ high_threshold or
prob ≤ low_threshold), the verdict is returned immediately without invoking
the heavier ONNX/HF model.  Only inputs in the ambiguous "grey zone" are
escalated.

Measured on the external benchmark with the default 0.05/0.95 band (see
docs/BENCHMARKS.md): ~77% of prompts are settled by sklearn alone and ~23% are
escalated, with recall/FPR matching the transformer on its own. Hard,
out-of-distribution prompts escalate far more often (~60% of the holdout set).

The block threshold is the slow model's tuned threshold: grey-zone prompts are
scored by the slow model alone, and fast-path sklearn scores sit outside the
[low, high] band (beyond any threshold in it), so one threshold serves both.

The thresholds are tunable via CASCADE_SK_LOW_THRESHOLD /
CASCADE_SK_HIGH_THRESHOLD environment variables (see config.py).

Predict return dict is identical to individual classifiers plus a diagnostic
"stage" key ("sklearn" | "onnx") for audit logging / metrics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CascadeClassifier:
    """
    Combines a fast sklearn classifier with a slower ONNX/HF classifier.

    Parameters
    ----------
    sklearn_clf    : SklearnClassifier
        Fast first-stage classifier (~1.5 ms/request).
    slow_clf       : ONNXClassifier | HFClassifier
        Accurate slow-path classifier (~18 ms on GPU, ~40 ms INT8 on CPU).
    low_threshold  : float
        sklearn malicious_prob ≤ this → benign fast path (default 0.05).
    high_threshold : float
        sklearn malicious_prob ≥ this → malicious fast path (default 0.95).

    Both classifiers must be loaded before calling predict().
    """

    def __init__(
        self,
        sklearn_clf,
        slow_clf,
        low_threshold: float = 0.05,
        high_threshold: float = 0.95,
        slow_clf_label: str = "onnx",
    ) -> None:
        self.sklearn_clf = sklearn_clf
        self.slow_clf = slow_clf
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.slow_clf_label = slow_clf_label

    @property
    def threshold(self) -> float | None:
        """Tuned threshold of the slow model (falls back to sklearn's)."""
        for clf in (self.slow_clf, self.sklearn_clf):
            tuned = getattr(clf, "threshold", None)
            if isinstance(tuned, (int, float)):
                return float(tuned)
        return None

    def is_loaded(self) -> bool:
        return self.sklearn_clf.is_loaded() and self.slow_clf.is_loaded()

    async def predict(self, text: str) -> dict[str, Any]:
        """
        Run the two-stage cascade.

        Returns:
            {
                "label":          "benign" | "malicious",
                "malicious_prob": float,
                "benign_prob":    float,
                "stage":          "sklearn" | "onnx",  # diagnostic only
            }
        """
        if not self.is_loaded():
            raise RuntimeError("CascadeClassifier: one or both classifiers are not loaded.")

        # ── Stage 1: fast sklearn check ────────────────────────────────────────
        sk_result = await self.sklearn_clf.predict(text)
        sk_prob: float = sk_result["malicious_prob"]

        if sk_prob >= self.high_threshold:
            logger.debug(
                "Cascade fast-path MALICIOUS (sklearn=%.3f ≥ %.2f)",
                sk_prob,
                self.high_threshold,
            )
            return {**sk_result, "stage": "sklearn"}

        if sk_prob <= self.low_threshold:
            logger.debug(
                "Cascade fast-path benign (sklearn=%.3f ≤ %.2f)",
                sk_prob,
                self.low_threshold,
            )
            return {**sk_result, "stage": "sklearn"}

        # ── Stage 2: grey zone → escalate to ONNX/HF ──────────────────────────
        logger.debug(
            "Cascade escalating to ONNX (sklearn=%.3f in grey zone [%.2f, %.2f])",
            sk_prob,
            self.low_threshold,
            self.high_threshold,
        )
        slow_result = await self.slow_clf.predict(text)

        # The slow model decides the grey zone on its own score. Its threshold
        # is the cascade's threshold, and the two models' scores are on
        # different scales (an INT8 graph's tuned threshold can be ~0.3 while
        # sklearn's is ~0.55), so mixing them — e.g. max(sk, slow) — would let
        # middling sklearn scores block. Strong sklearn signals are already
        # honoured by the high_threshold fast path above.
        return {**slow_result, "stage": self.slow_clf_label}
