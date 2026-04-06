"""
app/classifiers/hf_classifier.py
=================================
Phase 2 classifier — fine-tuned DistilBERT / DeBERTa-v3-small with optional
4-bit quantisation (bitsandbytes) for low VRAM usage.

Falls back gracefully to CPU if no GPU is available.
Async wrapper: runs torch inference in a thread pool.

This module imports torch/transformers lazily so the base service can start
without the heavy ML stack installed (using CLASSIFIER_TYPE=sklearn).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HFClassifier:
    """
    Wraps a fine-tuned HuggingFace sequence classification model.

    The model directory must contain:
        - config.json
        - tokenizer files
        - model weights (model.safetensors or adapter_model.safetensors for QLoRA)

    Label mapping expected in config.json:
        id2label: {"0": "BENIGN", "1": "MALICIOUS"}
    """

    def __init__(self, model_path: str, use_4bit: bool = False) -> None:
        self.model_path = model_path
        self.use_4bit = use_4bit
        self._pipeline = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @staticmethod
    def _probe_device(torch) -> int:
        """
        Return GPU device index (0) if CUDA/ROCm is available AND functional,
        otherwise -1 (CPU).  A quick kernel probe catches broken HIP setups
        (e.g. PyTorch built for a different GPU architecture) before they
        silently trigger the fail-secure block on every inference call.
        """
        if not torch.cuda.is_available():
            return -1
        try:
            t = torch.zeros(4).cuda()
            torch.isfinite(t)  # Exercises a GPU kernel; fails fast on bad HIP builds
            return 0
        except RuntimeError as exc:
            logger.warning(
                "GPU detected but not functional (%s) — falling back to CPU. "
                "Set HSA_OVERRIDE_GFX_VERSION or reinstall PyTorch with matching "
                "ROCm support to enable GPU inference.",
                exc,
            )
            return -1

    def load(self) -> None:
        """
        Load the fine-tuned model.  Heavy imports are deferred to here
        so the rest of the app doesn't pay the import cost when using sklearn.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HuggingFace model not found at {path}. "
                "Run 'python -m training.phase2_hf.train' first."
            )

        device = self._probe_device(torch)
        device_label = "CUDA/ROCm" if device == 0 else "CPU"
        logger.info("Loading HF classifier from %s on %s", path, device_label)

        # Optional 4-bit quantisation for GPU inference (reduces VRAM ~4x)
        model_kwargs: dict[str, Any] = {}
        if self.use_4bit and device == 0:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("4-bit quantisation enabled for HF inference")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — 4-bit inference disabled."
                )

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), dtype=torch.float32, **model_kwargs
        )

        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,   # Return all class scores
            truncation=True,
            max_length=512,
        )
        self._loaded = True
        logger.info("HF classifier loaded successfully.")

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
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("HFClassifier is not loaded. Call .load() first.")

        return await asyncio.to_thread(self._predict_sync, text)

    def _predict_sync(self, text: str) -> dict[str, Any]:
        """Synchronous inference — called inside a thread pool worker."""
        pipeline = self._pipeline
        assert pipeline is not None

        # HF pipeline returns: [[{"label": "BENIGN", "score": 0.95}, {"label": "MALICIOUS", ...}]]
        results: list[list[dict]] = pipeline(text)
        scores = {r["label"].upper(): r["score"] for r in results[0]}

        malicious_prob = scores.get("MALICIOUS", scores.get("LABEL_1", 0.0))
        benign_prob = scores.get("BENIGN", scores.get("LABEL_0", 1.0 - malicious_prob))

        label = "malicious" if malicious_prob >= 0.5 else "benign"

        return {
            "label": label,
            "malicious_prob": float(malicious_prob),
            "benign_prob": float(benign_prob),
        }
