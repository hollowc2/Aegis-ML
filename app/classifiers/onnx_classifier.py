"""
app/classifiers/onnx_classifier.py
====================================
Phase 2b classifier — ONNX Runtime inference on the exported DeBERTa model.

Exposes an identical async predict() interface to HFClassifier and
SklearnClassifier so it can be used as a drop-in replacement or as the
slow-path of the CascadeClassifier.

Provider preference (first available wins):
    ROCMExecutionProvider → MIGraphXExecutionProvider → CUDAExecutionProvider
    → CPUExecutionProvider

ORT applies graph-level optimisations (constant folding, operator fusion,
memory planning) at session load, giving significant speedup over the
PyTorch pipeline even on CPU.

This module imports torch/optimum lazily so the base service can start
without the ML stack installed (CLASSIFIER_TYPE=sklearn).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Providers tried in preference order; first one present in ORT build is used.
_PROVIDER_PREFERENCE = [
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


class ONNXClassifier:
    """
    Wraps a fine-tuned model exported to ONNX via Hugging Face Optimum.

    The model directory must contain:
        - model.onnx           (produced by export_onnx.py)
        - tokenizer files      (tokenizer.json / spiece.model)
        - config.json          with id2label: {"0": "BENIGN", "1": "MALICIOUS"}
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._session = None
        self._tokenizer = None
        self._id2label: dict = {}
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the ONNX model.  Heavy imports are deferred to here so the rest
        of the app doesn't pay the import cost when CLASSIFIER_TYPE=sklearn.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime not installed. Run:\n"
                "  pip install onnxruntime\n"
                "AMD ROCm users: pip install onnxruntime-rocm"
            ) from exc

        from transformers import AutoTokenizer
        import json

        path = Path(self.model_path)
        onnx_file = path / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_file}. "
                "Run: python -m training.phase2_hf.export_onnx"
            )

        available_providers = ort.get_available_providers()
        provider = next(
            (p for p in _PROVIDER_PREFERENCE if p in available_providers),
            "CPUExecutionProvider",
        )
        logger.info(
            "Loading ONNX classifier from %s using %s", path, provider
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_opts,
            providers=[provider],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))

        # Load id2label from config.json
        config_file = path / "config.json"
        with open(config_file) as f:
            cfg = json.load(f)
        self._id2label: dict = cfg.get("id2label", {"0": "BENIGN", "1": "MALICIOUS"})

        self._loaded = True
        logger.info("ONNX classifier loaded successfully (provider: %s).", provider)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ──────────────────────────────────────────────────────────────

    async def predict(self, text: str) -> dict[str, Any]:
        """
        Async prediction wrapper.

        Returns:
            {
                "label":          "benign" | "malicious",
                "malicious_prob": float,
                "benign_prob":    float,
            }
        """
        if not self._loaded or self._session is None:
            raise RuntimeError("ONNXClassifier is not loaded. Call .load() first.")

        return await asyncio.to_thread(self._predict_sync, text)

    def _predict_sync(self, text: str) -> dict[str, Any]:
        """Synchronous ORT inference — called inside a thread pool worker."""
        import numpy as np
        from scipy.special import softmax

        assert self._session is not None and self._tokenizer is not None

        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Only pass inputs the model graph actually expects
        input_names = {inp.name for inp in self._session.get_inputs()}
        feed = {k: v for k, v in inputs.items() if k in input_names}

        logits = self._session.run(["logits"], feed)[0][0]  # shape: (num_labels,)
        probs = softmax(logits).tolist()

        label_to_prob = {
            str(self._id2label[k]).upper(): probs[int(k)]
            for k in self._id2label
        }

        malicious_prob = label_to_prob.get("MALICIOUS", label_to_prob.get("LABEL_1", 0.0))
        benign_prob = label_to_prob.get("BENIGN", label_to_prob.get("LABEL_0", 1.0 - malicious_prob))
        label = "malicious" if malicious_prob >= 0.5 else "benign"

        return {
            "label": label,
            "malicious_prob": malicious_prob,
            "benign_prob": benign_prob,
        }
