"""
training/phase2_hf/export_onnx.py
===================================
Export the fine-tuned DeBERTa-v3-small classifier to ONNX format using
Hugging Face Optimum, then save the tokenizer alongside it.

ONNX Runtime applies graph-level optimisations (constant folding, operator
fusion, memory planning) at session load time — no separate post-export step
is required.  Simply load the exported directory with ONNXClassifier.

Usage:
    python -m training.phase2_hf.export_onnx
    python -m training.phase2_hf.export_onnx --model-dir models/hf_classifier \\
                                              --output-dir models/hf_classifier_onnx

Dependencies (install via pyproject.toml [onnx] extra):
    pip install "optimum[onnxruntime]"
    # AMD ROCm users: pip install onnxruntime-rocm  (instead of onnxruntime)

Notes:
    - DeBERTa-v3 uses fp32 — no precision flags needed.
    - Export runs on CPU; GPU is not needed for export.
    - The exported model is ~same size as the PyTorch weights but runs faster
      at inference due to ORT graph optimisation passes.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def export(model_dir: str, output_dir: str) -> None:
    # ROCm GFX version override must be set before any torch/hip import
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Optimum is not installed. Run:\n"
            "  pip install 'optimum[onnxruntime]'\n"
            "AMD ROCm users: pip install onnxruntime-rocm"
        ) from exc

    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    out_path = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"HuggingFace model not found at {model_path}. "
            "Run 'python -m training.phase2_hf.train' first."
        )

    logger.info("Exporting %s → ONNX (this may take 30–60 s) ...", model_path)

    # Optimum handles the full export: traces the model, converts to ONNX,
    # and preserves the label mapping from config.json.
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        export=True,
        provider="CPUExecutionProvider",  # export always on CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    out_path.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    exported_onnx = out_path / "model.onnx"
    size_mb = exported_onnx.stat().st_size / 1_048_576 if exported_onnx.exists() else 0

    logger.info("Export complete.")
    logger.info("  Output dir : %s", out_path)
    logger.info("  model.onnx : %.1f MB", size_mb)
    logger.info(
        "  Next step  : set CLASSIFIER_TYPE=cascade (or onnx) and "
        "ONNX_MODEL_PATH=%s",
        out_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Export fine-tuned HuggingFace classifier to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        default="models/hf_classifier",
        help="Path to the trained HuggingFace model directory (default: models/hf_classifier)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/hf_classifier_onnx",
        help="Destination directory for the ONNX export (default: models/hf_classifier_onnx)",
    )
    args = parser.parse_args()
    export(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
