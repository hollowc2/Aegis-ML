"""
app/classifiers/onnx2_classifier.py
=====================================
Phase 3 (HF2 Ultra) ONNX Runtime classifier.

Serves the exported ONNX model (FP32 or INT8-quantised).

Returns the same extended predict dict as HF2Classifier (backward-compatible).
Temperature scaling is applied to binary_logits before softmax.
TextPreprocessor (Unicode normalization + invisible char detection) runs
before every inference call.

Provider preference:
  FP32 model: ROCMExecutionProvider → MIGraphXExecutionProvider →
              CUDAExecutionProvider → CPUExecutionProvider
  INT8 model: CPUExecutionProvider only
  (AMD ROCm onnxruntime does not support INT8 kernel variants)

ONNX graph output names:
  binary_logits  — shape (batch, 2)
  threat_logits  — shape (batch, 7)

Lazy imports: onnxruntime/transformers only imported when .load() is called.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROVIDER_PREFERENCE = [
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


class ONNX2Classifier:
    """
    Wraps the exported HF2 ONNX model for runtime inference.

    The model directory must contain:
        model.onnx OR model_int8.onnx   — ONNX graph(s)
        tokenizer files                 — tokenizer.json / spiece.model
        config.json                     — with temperature_scaling + aegis_* metadata
    """

    def __init__(self, model_path: str, use_int8: bool = True) -> None:
        self.model_path = model_path
        self.use_int8 = use_int8
        self._session = None
        self._tokenizer = None
        self._temperature: float = 1.0
        self._input_names: set[str] = set()
        self._output_names: list[str] = []
        self._threat_categories: list[str] = []
        self._loaded = False
        self._preprocessor = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the ONNX model.  Heavy imports deferred to here."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime not installed. Run:\n"
                "  uv sync --extra hf2\n"
                "AMD ROCm users: pip install onnxruntime-rocm"
            ) from exc

        from transformers import AutoTokenizer
        from app.classifiers.text_preprocessor import TextPreprocessor

        path = Path(self.model_path)

        # Select model file (INT8 preferred, fallback to FP32)
        int8_file = path / "model_int8.onnx"
        fp32_file = path / "model.onnx"
        if self.use_int8 and int8_file.exists():
            onnx_file = int8_file
            force_cpu = True
        elif fp32_file.exists():
            onnx_file = fp32_file
            force_cpu = False
        else:
            raise FileNotFoundError(
                f"ONNX model not found in {path}. "
                "Run: python -m training.phase3_hf2.export_onnx"
            )

        # Provider selection — INT8 only works on CPU
        if force_cpu:
            provider = "CPUExecutionProvider"
            logger.info("INT8 model: using CPUExecutionProvider")
        else:
            available = ort.get_available_providers()
            provider = next(
                (p for p in _PROVIDER_PREFERENCE if p in available),
                "CPUExecutionProvider",
            )

        logger.info("Loading ONNX2 classifier from %s using %s", onnx_file, provider)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_opts,
            providers=[provider],
        )

        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))

        # Load config for temperature scaling and threat category labels
        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self._temperature = float(cfg.get("temperature_scaling", 1.0))
            id2threat: dict = cfg.get("aegis_id2threat", {})
            if id2threat:
                max_id = max(int(k) for k in id2threat)
                self._threat_categories = [
                    id2threat.get(str(i), f"unknown_{i}") for i in range(max_id + 1)
                ]
        if not self._threat_categories:
            # Fallback to default order
            from training.phase3_hf2.model import THREAT_CATEGORIES
            self._threat_categories = THREAT_CATEGORIES

        self._preprocessor = TextPreprocessor()
        self._loaded = True
        logger.info(
            "ONNX2 classifier loaded (provider=%s, T=%.4f, int8=%s).",
            provider, self._temperature, force_cpu,
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ──────────────────────────────────────────────────────────────

    async def predict(self, text: str) -> dict[str, Any]:
        """Async prediction wrapper — returns extended predict dict."""
        if not self._loaded or self._session is None:
            raise RuntimeError("ONNX2Classifier is not loaded. Call .load() first.")
        return await asyncio.to_thread(self._predict_sync, text)

    def _predict_sync(self, text: str) -> dict[str, Any]:
        """Synchronous ORT inference — called inside a thread pool worker."""
        import numpy as np
        from scipy.special import softmax

        assert self._session is not None and self._tokenizer is not None
        assert self._preprocessor is not None

        # ── Preprocessing ─────────────────────────────────────────────────────
        cleaned_text, preprocess_flags = self._preprocessor.preprocess(text)

        # ── Tokenise ──────────────────────────────────────────────────────────
        inputs = self._tokenizer(
            cleaned_text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding=True,
        )
        feed = {k: v for k, v in inputs.items() if k in self._input_names}

        # ── Inference ─────────────────────────────────────────────────────────
        outputs = self._session.run(self._output_names, feed)

        # Parse outputs by name
        output_map = dict(zip(self._output_names, outputs))
        binary_logits = output_map.get("binary_logits", outputs[0])[0]   # (2,)
        threat_logits = output_map.get("threat_logits", outputs[1] if len(outputs) > 1 else outputs[0])[0]

        # Apply temperature scaling to binary logits
        binary_probs = softmax(binary_logits / self._temperature).tolist()
        threat_probs_arr = softmax(threat_logits).tolist()

        malicious_prob = float(binary_probs[1])
        benign_prob = float(binary_probs[0])
        label = "malicious" if malicious_prob >= 0.5 else "benign"

        # Build threat_category_probs dict
        threat_probs_dict = {
            self._threat_categories[i]: float(threat_probs_arr[i])
            for i in range(len(self._threat_categories))
        }
        threat_category = max(threat_probs_dict, key=threat_probs_dict.get)

        return {
            "label": label,
            "malicious_prob": malicious_prob,
            "benign_prob": benign_prob,
            "threat_category_probs": threat_probs_dict,
            "threat_category": threat_category,
            "classifier_stage": "onnx2",
            "preprocessing_flags": preprocess_flags,
        }
