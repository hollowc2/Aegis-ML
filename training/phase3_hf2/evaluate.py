"""
training/phase3_hf2/evaluate.py
================================
Phase 3 (HF2 Ultra) evaluation script.

Evaluation modes:
  Default: standard test-set evaluation (binary classification metrics)
  --adversarial: per-category FNR/FPR across all 15 attack categories
  --calibration: print reliability diagram data (calibration curve)
  --onnx2:       evaluate the INT8 ONNX model on CPU

Metrics reported:
  - ROC-AUC, F1, Precision, Recall
  - False Positive Rate (FPR), False Negative Rate (FNR)
  - Per-threat-category accuracy (for both binary and threat heads)
  - Latency per sample (ms)

Run:
    python -m training.phase3_hf2.evaluate
    python -m training.phase3_hf2.evaluate --adversarial
    python -m training.phase3_hf2.evaluate --calibration
    python -m training.phase3_hf2.evaluate --onnx2
    python -m training.phase3_hf2.evaluate --cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = Path("data/combined_dataset.csv")
ADVERSARIAL_CSV = Path("data/adversarial_eval.csv")
HF2_MODEL_DIR = Path("models/hf2_classifier")
ONNX2_MODEL_DIR = Path("models/hf2_classifier_onnx")


# ── PyTorch evaluation ────────────────────────────────────────────────────────

def evaluate(
    threshold: float = 0.70,
    force_cpu: bool = False,
    model_dir: Path | None = None,
) -> dict:
    """Standard test-set evaluation with the PyTorch AegisMTModel."""
    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    from training.phase3_hf2.model import AegisMTModel, THREAT2ID

    model_dir = model_dir or HF2_MODEL_DIR
    if not model_dir.exists():
        logger.error("Model not found at %s — run training first.", model_dir)
        return {}

    aegis_model, temperature = AegisMTModel.from_pretrained(model_dir)
    module = aegis_model.module
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Device
    device_str = "cpu"
    if not force_cpu and torch.cuda.is_available():
        try:
            torch.zeros(4).cuda()
            device_str = "cuda:0"
        except RuntimeError:
            pass
    device = torch.device(device_str)
    module.to(device)
    module.eval()
    logger.info("Evaluating on %s  (temperature=%.4f)", device_str, temperature)

    # Load test split
    if not DATASET_CSV.exists():
        logger.error("Dataset not found at %s — run prepare_dataset first.", DATASET_CSV)
        return {}

    df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    _, X_test, _, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.15, stratify=df["label"].tolist(), random_state=42,
    )

    probs, preds, latencies = [], [], []
    for text in X_test:
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = module(**inputs)
        import torch.nn.functional as F
        binary_probs = F.softmax(out.binary_logits / temperature, dim=-1)
        mal_prob = binary_probs[0, 1].item()
        latencies.append((time.perf_counter() - t0) * 1000)
        probs.append(mal_prob)
        preds.append(1 if mal_prob >= threshold else 0)

    labels_arr = np.array(y_test)
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)

    cm = confusion_matrix(labels_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = roc_auc_score(labels_arr, probs_arr)
    f1 = f1_score(labels_arr, preds_arr)
    avg_latency = float(np.mean(latencies))

    results = {
        "model": str(model_dir),
        "threshold": threshold,
        "temperature": temperature,
        "n_test": len(y_test),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "avg_latency_ms": round(avg_latency, 3),
    }
    print("\n── Phase 3 HF2 Test Results ────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<22} {v}")
    print(classification_report(labels_arr, preds_arr, target_names=["benign", "malicious"]))
    return results


# ── Adversarial evaluation ────────────────────────────────────────────────────

def evaluate_adversarial(
    threshold: float = 0.70,
    force_cpu: bool = False,
    model_dir: Path | None = None,
) -> None:
    """Per-category FNR/FPR evaluation on the full 15-category adversarial set."""
    import torch
    from transformers import AutoTokenizer
    from training.phase3_hf2.model import AegisMTModel

    model_dir = model_dir or HF2_MODEL_DIR

    if not ADVERSARIAL_CSV.exists():
        logger.info("Adversarial eval CSV not found — generating...")
        from training.data.adversarial_eval import main as adv_main
        adv_main()

    df = pd.read_csv(ADVERSARIAL_CSV)
    aegis_model, temperature = AegisMTModel.from_pretrained(model_dir)
    module = aegis_model.module
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    device_str = "cpu"
    if not force_cpu and torch.cuda.is_available():
        try:
            torch.zeros(4).cuda()
            device_str = "cuda:0"
        except RuntimeError:
            pass
    device = torch.device(device_str)
    module.to(device)
    module.eval()

    import torch.nn.functional as F
    probs = []
    for text in df["text"].tolist():
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = module(**inputs)
        p = F.softmax(out.binary_logits / temperature, dim=-1)[0, 1].item()
        probs.append(p)

    df["pred_prob"] = probs
    df["pred"] = (df["pred_prob"] >= threshold).astype(int)

    print(f"\n── Adversarial Evaluation (threshold={threshold}) ──────────────")
    print(f"{'Category':<35} {'N':>5} {'FNR%':>8} {'FPR%':>8}")
    print("─" * 60)

    categories = df["category"].unique()
    for cat in sorted(categories):
        subset = df[df["category"] == cat]
        n = len(subset)
        if subset["label"].iloc[0] == 1:
            # Malicious category: compute FNR
            fn = (subset["pred"] == 0).sum()
            rate = fn / n * 100
            print(f"  {cat:<33} {n:>5} {rate:>7.1f}%")
        else:
            # Benign category: compute FPR
            fp = (subset["pred"] == 1).sum()
            rate = fp / n * 100
            print(f"  {cat:<33} {n:>5} {'---':>8} {rate:>7.1f}%")

    # Overall
    mal = df[df["label"] == 1]
    ben = df[df["label"] == 0]
    overall_fnr = (mal["pred"] == 0).sum() / len(mal) * 100
    overall_fpr = (ben["pred"] == 1).sum() / len(ben) * 100
    print("─" * 60)
    print(f"  {'OVERALL (malicious FNR)':<33} {len(mal):>5} {overall_fnr:>7.1f}%")
    print(f"  {'OVERALL (benign FPR)':<33} {len(ben):>5} {'---':>8} {overall_fpr:>7.1f}%")


# ── Calibration curve ─────────────────────────────────────────────────────────

def evaluate_calibration(
    force_cpu: bool = False,
    model_dir: Path | None = None,
    n_bins: int = 10,
) -> None:
    """
    Print reliability diagram data (mean predicted probability vs. fraction positive
    per bin).  A well-calibrated model produces a near-diagonal curve.
    """
    import torch
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split
    from training.phase3_hf2.model import AegisMTModel

    model_dir = model_dir or HF2_MODEL_DIR
    aegis_model, temperature = AegisMTModel.from_pretrained(model_dir)
    module = aegis_model.module
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    device_str = "cpu"
    if not force_cpu and torch.cuda.is_available():
        try:
            torch.zeros(4).cuda()
            device_str = "cuda:0"
        except RuntimeError:
            pass
    device = torch.device(device_str)
    module.to(device)
    module.eval()

    df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
    _, X_test, _, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.15, stratify=df["label"].tolist(), random_state=42,
    )

    import torch.nn.functional as F
    probs = []
    for text in X_test:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = module(**inputs)
        p = F.softmax(out.binary_logits / temperature, dim=-1)[0, 1].item()
        probs.append(p)

    probs_arr = np.array(probs)
    labels_arr = np.array(y_test, dtype=float)

    print(f"\n── Calibration Curve (temperature={temperature:.4f}) ─────────────")
    print(f"{'Bin':>6}  {'Mean Pred Prob':>15}  {'Fraction Positive':>18}  {'Count':>6}")
    print("─" * 52)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs_arr >= lo) & (probs_arr < hi)
        count = mask.sum()
        if count == 0:
            continue
        mean_pred = probs_arr[mask].mean()
        frac_pos = labels_arr[mask].mean()
        print(f"  [{lo:.1f}-{hi:.1f}]  {mean_pred:>15.4f}  {frac_pos:>18.4f}  {count:>6}")


# ── ONNX2 evaluation ──────────────────────────────────────────────────────────

def evaluate_onnx2(
    threshold: float = 0.70,
    use_int8: bool = True,
    model_dir: Path | None = None,
) -> dict:
    """Evaluate the INT8 ONNX model on CPU."""
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    model_dir = model_dir or ONNX2_MODEL_DIR
    if not model_dir.exists():
        logger.error("ONNX2 model dir not found at %s", model_dir)
        return {}

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return {}

    from transformers import AutoTokenizer
    import json as _json

    onnx_file = model_dir / ("model_int8.onnx" if use_int8 else "model.onnx")
    if not onnx_file.exists():
        onnx_file = model_dir / "model.onnx"
        logger.info("INT8 model not found — falling back to FP32.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    with open(model_dir / "config.json") as f:
        cfg = _json.load(f)
    temperature = float(cfg.get("temperature_scaling", 1.0))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_file), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in session.get_outputs()]
    input_names = {i.name for i in session.get_inputs()}

    from scipy.special import softmax as scipy_softmax

    df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
    _, X_test, _, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.15, stratify=df["label"].tolist(), random_state=42,
    )

    probs, latencies = [], []
    for text in X_test:
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512, padding=True)
        feed = {k: v for k, v in inputs.items() if k in input_names}
        outs = session.run(output_names, feed)
        # First output is binary_logits
        binary_logits = outs[0][0] / temperature
        p = scipy_softmax(binary_logits)[1]
        latencies.append((time.perf_counter() - t0) * 1000)
        probs.append(float(p))

    labels_arr = np.array(y_test)
    preds_arr = (np.array(probs) >= threshold).astype(int)

    cm = confusion_matrix(labels_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = roc_auc_score(labels_arr, np.array(probs))
    f1 = f1_score(labels_arr, preds_arr)

    results = {
        "model": str(onnx_file),
        "threshold": threshold,
        "temperature": temperature,
        "n_test": len(y_test),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 3),
    }
    print("\n── Phase 3 ONNX2 Test Results (CPU) ────────────────────")
    for k, v in results.items():
        print(f"  {k:<22} {v}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 HF2 classifier")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--adversarial", action="store_true", help="Per-category adversarial eval")
    parser.add_argument("--calibration", action="store_true", help="Print calibration curve data")
    parser.add_argument("--onnx2", action="store_true", help="Evaluate INT8 ONNX model on CPU")
    parser.add_argument("--model-dir", type=Path, default=None, help="Override model directory")
    args = parser.parse_args()

    if args.onnx2:
        evaluate_onnx2(threshold=args.threshold, model_dir=args.model_dir)
    elif args.adversarial:
        evaluate_adversarial(
            threshold=args.threshold, force_cpu=args.cpu, model_dir=args.model_dir
        )
    elif args.calibration:
        evaluate_calibration(force_cpu=args.cpu, model_dir=args.model_dir)
    else:
        evaluate(threshold=args.threshold, force_cpu=args.cpu, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
