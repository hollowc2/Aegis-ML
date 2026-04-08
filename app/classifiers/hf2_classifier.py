"""
app/classifiers/hf2_classifier.py
===================================
Phase 3 (HF2 Ultra) runtime classifier — loads and serves AegisMTModel.

Returns an extended predict dict (backward-compatible with Phase 1/2):

    {
        # Existing keys (all classifiers)
        "label":          "benign" | "malicious",
        "malicious_prob": float,
        "benign_prob":    float,
        # Extended keys (hf2 only)
        "threat_category_probs": {
            "prompt_injection": float,
            "jailbreak": float,
            "data_exfiltration": float,
            "canary_leak": float,
            "pii_leak": float,
            "harmful_content": float,
            "none": float,
        },
        "threat_category":    str,    # argmax of threat_category_probs
        "classifier_stage":   "hf2",
        "preprocessing_flags": dict,  # from TextPreprocessor
    }

The temperature scaling scalar T* is read from config.json and applied
before converting logits to probabilities.

TextPreprocessor (Unicode normalization + invisible char detection) runs
internally before every inference call.

Lazy imports: torch/transformers are only imported when .load() is called,
so the rest of the service starts without the ML stack.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HF2Classifier:
    """
    Wraps a trained AegisMTModel for runtime inference.

    The model directory must contain:
        model.pt        — PyTorch state dict (from AegisMTModel.save_pretrained)
        config.json     — config with temperature_scaling + aegis_* metadata
        tokenizer files — tokenizer.json / spiece.model
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._module = None
        self._tokenizer = None
        self._temperature: float = 1.0
        self._device = None
        self._loaded = False
        self._preprocessor = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @staticmethod
    def _probe_device(torch) -> int:
        """Return GPU device index (0) if available and functional, else -1 (CPU)."""
        if not torch.cuda.is_available():
            return -1
        try:
            t = torch.zeros(4).cuda()
            torch.isfinite(t)
            return 0
        except RuntimeError as exc:
            logger.warning(
                "GPU detected but not functional (%s) — falling back to CPU.", exc
            )
            return -1

    def load(self) -> None:
        """Load the AegisMTModel.  Heavy imports deferred to here."""
        import torch
        from transformers import AutoTokenizer
        from training.phase3_hf2.model import AegisMTModel
        from app.classifiers.text_preprocessor import TextPreprocessor

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HF2 model not found at {path}. "
                "Run 'python -m training.phase3_hf2.train' first."
            )

        device_id = self._probe_device(torch)
        device_label = "CUDA/ROCm" if device_id == 0 else "CPU"
        logger.info("Loading HF2 classifier from %s on %s", path, device_label)

        aegis_model, temperature = AegisMTModel.from_pretrained(path)
        self._module = aegis_model.module
        self._temperature = temperature

        device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        self._module.to(device)
        self._module.eval()
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._preprocessor = TextPreprocessor()
        self._loaded = True
        logger.info(
            "HF2 classifier loaded successfully (T=%.4f).", self._temperature
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ──────────────────────────────────────────────────────────────

    async def predict(self, text: str) -> dict[str, Any]:
        """
        Async prediction wrapper.

        Returns the extended predict dict (see module docstring).
        """
        if not self._loaded or self._module is None:
            raise RuntimeError("HF2Classifier is not loaded. Call .load() first.")

        return await asyncio.to_thread(self._predict_sync, text)

    def _predict_sync(self, text: str) -> dict[str, Any]:
        """Synchronous inference — called inside a thread pool worker."""
        import torch
        import torch.nn.functional as F
        from training.phase3_hf2.model import ID2THREAT, THREAT_CATEGORIES

        assert self._module is not None
        assert self._tokenizer is not None
        assert self._preprocessor is not None

        # ── Preprocessing (Unicode normalization + invisible char detection) ──
        cleaned_text, preprocess_flags = self._preprocessor.preprocess(text)

        # ── Tokenise ──────────────────────────────────────────────────────────
        inputs = self._tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = self._module(**inputs)

        binary_logits = outputs.binary_logits
        threat_logits = outputs.threat_logits

        # Apply temperature scaling to binary logits
        binary_probs = F.softmax(binary_logits / self._temperature, dim=-1)[0]
        threat_probs = F.softmax(threat_logits, dim=-1)[0]

        malicious_prob = float(binary_probs[1].item())
        benign_prob = float(binary_probs[0].item())
        label = "malicious" if malicious_prob >= 0.5 else "benign"

        # Build threat_category_probs dict
        threat_probs_dict = {
            THREAT_CATEGORIES[i]: float(threat_probs[i].item())
            for i in range(len(THREAT_CATEGORIES))
        }
        threat_category = max(threat_probs_dict, key=threat_probs_dict.get)

        return {
            "label": label,
            "malicious_prob": malicious_prob,
            "benign_prob": benign_prob,
            "threat_category_probs": threat_probs_dict,
            "threat_category": threat_category,
            "classifier_stage": "hf2",
            "preprocessing_flags": preprocess_flags,
        }
