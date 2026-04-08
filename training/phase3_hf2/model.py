"""
training/phase3_hf2/model.py
==============================
AegisMTModel — multi-task DeBERTa-v3-base classifier.

Architecture:
    DeBERTa-v3-base encoder (183M params)
              │
       [CLS] hidden (768-dim)
              │
      ┌───────┴──────────────┐
      ▼                      ▼
  Dropout(0.1)           Dropout(0.1)
      │                      │
  Linear(768, 2)         Linear(768, 7)
  binary_logits          threat_logits

Task 1 (primary):  binary classification — 0=BENIGN, 1=MALICIOUS
Task 2 (auxiliary): 7-way threat category — prompt_injection, jailbreak,
                    data_exfiltration, canary_leak, pii_leak,
                    harmful_content, none

Loss = CE(binary) + alpha * CE(threat, ignore_index=-100)
    where alpha=0.3 and ignore_index=-100 means "unknown category" is
    excluded from the threat loss (allows HF-sourced data without labels).

Post-training: temperature scaling scalar T* is stored in config.json
    under the key "temperature_scaling" and applied at inference time.

Serialisation:
    save_pretrained(path) — saves model.pt + config.json + tokeniser files
    from_pretrained(path) — loads model, config, and tokeniser

This module uses lazy imports (torch/transformers) so the rest of the
service can run without the heavy ML stack (CLASSIFIER_TYPE=sklearn).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ordered list of threat category labels — index = class id
THREAT_CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "data_exfiltration",
    "canary_leak",
    "pii_leak",
    "harmful_content",
    "none",
]

THREAT2ID = {cat: i for i, cat in enumerate(THREAT_CATEGORIES)}
ID2THREAT = {i: cat for i, cat in enumerate(THREAT_CATEGORIES)}

BINARY_ID2LABEL = {0: "BENIGN", 1: "MALICIOUS"}
BINARY_LABEL2ID = {"BENIGN": 0, "MALICIOUS": 1}

# Default temperature scalar (no calibration applied)
DEFAULT_TEMPERATURE = 1.0

# Auxiliary task loss weight
THREAT_LOSS_ALPHA = 0.3


class AegisMTModel:
    """
    Thin wrapper around a two-headed DeBERTa-v3-base PyTorch module.

    Keeps the nn.Module definition inside this class so torch is only imported
    when explicitly instantiated.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        dropout: float = 0.1,
        n_threat_classes: int = len(THREAT_CATEGORIES),
    ) -> None:
        self.base_model_name_or_path = base_model_name_or_path
        self.dropout = dropout
        self.n_threat_classes = n_threat_classes
        self._module = None  # lazy; set after load_pretrained or build

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> "_AegisMTModule":
        """
        Instantiate the nn.Module from a pre-trained base model.
        Called during training to create a fresh model.
        """
        import torch
        from transformers import AutoModel, AutoConfig

        logger.info("Building AegisMTModel from %s", self.base_model_name_or_path)
        module = _AegisMTModule(
            base_model_name_or_path=self.base_model_name_or_path,
            dropout=self.dropout,
            n_threat_classes=self.n_threat_classes,
        )
        self._module = module
        return module

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_pretrained(
        self,
        output_dir: str | Path,
        tokenizer=None,
        temperature_scaling: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """
        Save model weights, config, and optionally the tokeniser.

        Files produced:
            model.pt        — PyTorch state dict
            config.json     — DeBERTa config + Aegis metadata
            tokenizer.*     — if tokenizer is provided
        """
        import torch

        assert self._module is not None, "Call build() or from_pretrained() first."
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self._module.state_dict(), output_dir / "model.pt")

        # Save config (merge DeBERTa config with our metadata)
        base_config = self._module.deberta.config.to_dict()
        base_config.update({
            "model_type": "aegis_mt",
            "aegis_n_threat_classes": self.n_threat_classes,
            "aegis_threat2id": THREAT2ID,
            "aegis_id2threat": {str(k): v for k, v in ID2THREAT.items()},
            "aegis_binary_id2label": {str(k): v for k, v in BINARY_ID2LABEL.items()},
            "aegis_binary_label2id": BINARY_LABEL2ID,
            "temperature_scaling": temperature_scaling,
            "dropout": self.dropout,
        })
        with open(output_dir / "config.json", "w") as f:
            json.dump(base_config, f, indent=2)

        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))

        logger.info(
            "AegisMTModel saved to %s (temperature=%.4f)", output_dir, temperature_scaling
        )

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path
    ) -> tuple["AegisMTModel", float]:
        """
        Load a saved AegisMTModel.

        Returns:
            (wrapper, temperature_scalar)
        """
        import torch

        model_dir = Path(model_dir)
        config_file = model_dir / "config.json"
        weights_file = model_dir / "model.pt"

        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        if not weights_file.exists():
            raise FileNotFoundError(f"model.pt not found in {model_dir}")

        with open(config_file) as f:
            cfg = json.load(f)

        n_threat = cfg.get("aegis_n_threat_classes", len(THREAT_CATEGORIES))
        dropout = cfg.get("dropout", 0.1)
        temperature = float(cfg.get("temperature_scaling", DEFAULT_TEMPERATURE))

        wrapper = cls(
            base_model_name_or_path=str(model_dir),
            dropout=dropout,
            n_threat_classes=n_threat,
        )
        module = _AegisMTModule(
            base_model_name_or_path=str(model_dir),
            dropout=dropout,
            n_threat_classes=n_threat,
            load_base_weights=True,  # load DeBERTa weights from saved dir
        )

        state = torch.load(str(weights_file), map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=False)
        wrapper._module = module

        logger.info(
            "AegisMTModel loaded from %s (temperature=%.4f)", model_dir, temperature
        )
        return wrapper, temperature

    @property
    def module(self) -> "_AegisMTModule":
        assert self._module is not None, "Model not built or loaded."
        return self._module


class _AegisMTModule:
    """
    The actual nn.Module.  Defined as a nested plain class to defer the
    'import torch / nn.Module' until instantiation time.
    """

    def __new__(
        cls,
        base_model_name_or_path: str,
        dropout: float,
        n_threat_classes: int,
        load_base_weights: bool = True,
    ):
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoConfig

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                if load_base_weights:
                    self.deberta = AutoModel.from_pretrained(
                        base_model_name_or_path,
                        dtype=torch.float32,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    config = AutoConfig.from_pretrained(base_model_name_or_path)
                    self.deberta = AutoModel.from_config(config)

                hidden = self.deberta.config.hidden_size  # 768 for deberta-v3-base
                self.drop = nn.Dropout(dropout)
                self.binary_head = nn.Linear(hidden, 2)
                self.threat_head = nn.Linear(hidden, n_threat_classes)

            def forward(
                self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                binary_labels=None,
                threat_labels=None,
            ):
                outputs = self.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                # [CLS] token representation
                cls_rep = self.drop(outputs.last_hidden_state[:, 0, :])
                binary_logits = self.binary_head(cls_rep)
                threat_logits = self.threat_head(cls_rep)

                loss = None
                if binary_labels is not None:
                    import torch.nn.functional as F
                    binary_loss = F.cross_entropy(
                        binary_logits,
                        binary_labels,
                        label_smoothing=0.1,
                    )
                    threat_loss = torch.tensor(0.0, device=binary_logits.device)
                    if threat_labels is not None:
                        # ignore_index=-100 skips samples with unknown category
                        threat_loss = F.cross_entropy(
                            threat_logits,
                            threat_labels,
                            ignore_index=-100,
                        )
                    loss = binary_loss + THREAT_LOSS_ALPHA * threat_loss

                return _ForwardOutput(
                    loss=loss,
                    binary_logits=binary_logits,
                    threat_logits=threat_logits,
                )

        instance = _Inner()
        return instance


class _ForwardOutput:
    """Simple output container compatible with HuggingFace Trainer."""

    def __init__(self, loss, binary_logits, threat_logits):
        self.loss = loss
        self.binary_logits = binary_logits
        self.threat_logits = threat_logits

    def __getitem__(self, key):
        # Allow Trainer to unpack as a tuple: (loss, logits)
        if key == 0:
            return self.loss
        if key == 1:
            return self.binary_logits
        raise IndexError(key)
