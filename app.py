"""
app.py — HuggingFace Spaces entry point for Aegis-ML.

Startup sequence
----------------
1. sklearn model: load from models/sklearn_classifier.joblib.
   If missing, auto-train from public HF datasets (~60 s).

2. ONNX2 model: load from models/hf2_classifier_onnx/.
   If missing, download from the HF model repo (hollowc2/aegis-ml-classifier).
   If the download fails (repo not yet created, no network), the Space still
   launches with sklearn only — onnx2 falls back to keyword heuristics in the UI.

HF Spaces runs this file and serves the module-level `demo` object.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SKLEARN_PATH  = Path(os.getenv("SKLEARN_MODEL_PATH",  "models/sklearn_classifier.joblib"))
ONNX2_DIR     = Path(os.getenv("ONNX2_MODEL_PATH",    "models/hf2_classifier_onnx"))
HF_MODEL_REPO = os.getenv("AEGIS_MODEL_REPO",         "billybitcoin/aegis-ml-classifier")

# ── 1. sklearn model ──────────────────────────────────────────────────────────

if not SKLEARN_PATH.exists():
    logger.info("sklearn model not found — training from scratch (~60 s)...")
    try:
        SKLEARN_PATH.parent.mkdir(parents=True, exist_ok=True)
        from training.data.prepare_dataset import main as prep
        from training.phase1_sklearn.train import main as train_sklearn
        prep()
        train_sklearn()
        logger.info("sklearn model saved to %s", SKLEARN_PATH)
    except Exception as exc:
        logger.warning("Auto-training failed (%s) — demo falls back to keyword heuristics.", exc)
else:
    logger.info("sklearn model found at %s", SKLEARN_PATH)

# ── 2. ONNX2 model ────────────────────────────────────────────────────────────

_onnx2_ready = (ONNX2_DIR / "model_int8.onnx").exists()

if not _onnx2_ready:
    logger.info("ONNX2 model not found — downloading from %s ...", HF_MODEL_REPO)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_MODEL_REPO,
            local_dir=str(ONNX2_DIR),
            repo_type="model",
            ignore_patterns=["model.onnx", "model.onnx.data"],  # INT8 only; skip FP32
        )
        _onnx2_ready = (ONNX2_DIR / "model_int8.onnx").exists()
        if _onnx2_ready:
            logger.info("ONNX2 model downloaded to %s", ONNX2_DIR)
        else:
            logger.warning("Download completed but model_int8.onnx not found in %s", ONNX2_DIR)
    except Exception as exc:
        logger.warning(
            "Could not download ONNX2 model (%s). "
            "Classifier selector will show onnx2 but it will fall back to keyword heuristics. "
            "Upload the model with: huggingface-cli upload %s models/hf2_classifier_onnx/ --repo-type model",
            exc,
            HF_MODEL_REPO,
        )
else:
    logger.info("ONNX2 model found at %s", ONNX2_DIR)

# ── 3. Build and expose the Gradio demo ───────────────────────────────────────

from demo.gradio_ui import build_ui  # noqa: E402

demo = build_ui(onnx2_available=_onnx2_ready)

if __name__ == "__main__":
    demo.launch()
