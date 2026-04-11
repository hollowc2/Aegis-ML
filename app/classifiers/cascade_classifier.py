"""
app/classifiers/cascade_classifier.py
=======================================
Two-stage cascade classifier: sklearn fast path → ONNX/HF slow path.

For inputs where sklearn is highly confident (prob ≥ high_threshold or
prob ≤ low_threshold), the verdict is returned immediately without invoking
the heavier ONNX/HF model.  Only inputs in the ambiguous "grey zone" are
escalated.

Expected traffic split (empirical):
    ~85-90%  cleared by sklearn  → ~0.3 ms average
    ~10-15%  escalated to ONNX   → ~3-8 ms for those samples
    Blended average latency      → ~0.5-1.5 ms

The thresholds are tunable via CASCADE_SK_LOW_THRESHOLD /
CASCADE_SK_HIGH_THRESHOLD environment variables (see config.py).

Predict return dict is identical to individual classifiers plus a diagnostic
"stage" key ("sklearn" | "onnx") for audit logging / metrics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CascadeClassifier:
    """
    Combines a fast sklearn classifier with a slower ONNX/HF classifier.

    Parameters
    ----------
    sklearn_clf    : SklearnClassifier
        Fast first-stage classifier (~0.3 ms/sample).
    slow_clf       : ONNXClassifier | HFClassifier
        Accurate slow-path classifier (~3-15 ms/sample).
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
            raise RuntimeError(
                "CascadeClassifier: one or both classifiers are not loaded."
            )

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

        # Take the maximum malicious probability from both stages so that a
        # strong sklearn signal is never cancelled by a less-accurate slow
        # classifier (e.g. INT8-quantised ONNX dropping a clear attack below
        # the final threshold).
        slow_prob: float = slow_result.get("malicious_prob", 0.0)
        if sk_prob > slow_prob:
            logger.debug(
                "Cascade: sklearn signal (%.3f) > slow clf (%.3f) — using sklearn prob",
                sk_prob,
                slow_prob,
            )
            combined_prob = sk_prob
            combined_benign = 1.0 - sk_prob
            combined_label = "malicious" if combined_prob >= 0.5 else "benign"
            return {
                **slow_result,
                "malicious_prob": combined_prob,
                "benign_prob": combined_benign,
                "label": combined_label,
                "stage": f"sklearn_override({self.slow_clf_label})",
            }

        return {**slow_result, "stage": self.slow_clf_label}
