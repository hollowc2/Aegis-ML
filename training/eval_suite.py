"""
training/eval_suite.py
=======================
One evaluation harness shared by every classifier phase, so results are
comparable and reported the same way.

Evaluation sets:
  test         — in-distribution split of data/combined_dataset.csv (split == "test")
  external     — third-party test splits never trained on (data/external_eval.csv)
  holdout      — hand-written attacks + hard negatives (data/holdout_eval.csv)
  adversarial  — generator-derived regression suite (data/adversarial_eval.csv);
                 shares vocabulary with synthetic_gen.py, so treat as optimistic

Texts are passed through the serving-path TextPreprocessor before scoring, so
the numbers match what the API sees with ENABLE_TEXT_PREPROCESSING=true.

Usage from a training script:
    from training.eval_suite import pick_threshold, run_suite
    threshold = pick_threshold(y_val, p_val, max_fpr=0.02)
    run_suite(score_fn, threshold, name="sklearn")

Standalone (any serving classifier type, loaded exactly as the API loads it):
    python -m training.eval_suite --model sklearn [--threshold 0.6]
    python -m training.eval_suite --model onnx2
    python -m training.eval_suite --model cascade2 --threshold 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.classifiers.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

DATASET_CSV = Path("data/combined_dataset.csv")
EVAL_SETS: dict[str, tuple[Path, str]] = {
    # name: (path, column to break results down by)
    "external": (Path("data/external_eval.csv"), "source"),
    "holdout": (Path("data/holdout_eval.csv"), "category"),
    "adversarial": (Path("data/adversarial_eval.csv"), "category"),
}
REPORTS_DIR = Path("reports")

ScoreFn = Callable[[Sequence[str]], np.ndarray]

_preprocessor = TextPreprocessor()


def preprocess(texts: Sequence[str]) -> list[str]:
    return [_preprocessor.preprocess(str(t))[0] for t in texts]


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_CSV)
    if "split" not in df.columns:
        raise ValueError(f"{DATASET_CSV} has no split column — rerun training.data.prepare_dataset")
    return df[df["split"] == split].reset_index(drop=True)


def dataset_fingerprint() -> dict:
    digest = hashlib.sha256(DATASET_CSV.read_bytes()).hexdigest()[:16]
    return {"path": str(DATASET_CSV), "sha256_16": digest, "rows": int(len(pd.read_csv(DATASET_CSV)))}


def pick_threshold(y: Sequence[int], p: np.ndarray, max_fpr: float = 0.02) -> float:
    """Lowest threshold whose false-positive rate on (y, p) is at most *max_fpr*."""
    y_arr = np.asarray(y)
    benign = np.sort(p[y_arr == 0])
    if len(benign) == 0:
        return 0.5
    # Allow at most floor(max_fpr * n) benign scores at or above the threshold.
    allowed = int(np.floor(max_fpr * len(benign)))
    idx = len(benign) - allowed - 1
    threshold = float(benign[idx]) + 1e-6 if idx >= 0 else 0.0
    return float(min(max(threshold, 0.05), 0.95))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & neg).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    out = {
        "n": int(len(y)), "n_malicious": int(pos.sum()), "n_benign": int(neg.sum()),
        "recall": recall,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall and tp else 0.0,
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if pos.any() and neg.any() else float("nan")
    return out


def _breakdown(df: pd.DataFrame, by: str, threshold: float) -> dict:
    """Per-group recall (malicious groups) or FPR (benign groups)."""
    out = {}
    for group, g in df.groupby(by):
        flagged = g["p"] >= threshold
        row: dict = {"n": int(len(g))}
        if (g["label"] == 1).any():
            row["recall"] = float(flagged[g["label"] == 1].mean())
        if (g["label"] == 0).any():
            row["fpr"] = float(flagged[g["label"] == 0].mean())
        out[str(group)] = row
    return out


def run_suite(
    score_fn: ScoreFn,
    threshold: float,
    name: str,
    extra: dict | None = None,
    write: bool = True,
) -> dict:
    """Score every evaluation set, print a summary, and write reports/<name>.json."""
    report: dict = {
        "model": name,
        "threshold": threshold,
        "dataset": dataset_fingerprint(),
        "sets": {},
        **(extra or {}),
    }

    frames = {"test": (load_split("test"), "source")}
    for set_name, (path, by) in EVAL_SETS.items():
        if path.exists():
            frames[set_name] = (pd.read_csv(path), by)
        else:
            logger.warning("Evaluation set %s missing (%s) — skipped", set_name, path)

    for set_name, (df, by) in frames.items():
        texts = preprocess(df["text"].astype(str).tolist())
        start = time.perf_counter()
        df = df.assign(p=np.asarray(score_fn(texts), dtype=float))
        elapsed = time.perf_counter() - start
        metrics = binary_metrics(df["label"].to_numpy(), df["p"].to_numpy(), threshold)
        metrics["ms_per_example_batched"] = elapsed / len(df) * 1000
        metrics["breakdown"] = _breakdown(df, by, threshold)
        report["sets"][set_name] = metrics

    _print_report(report)
    if write:
        REPORTS_DIR.mkdir(exist_ok=True)
        path = REPORTS_DIR / f"{name}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", path)
    return report


def _fmt(x: float) -> str:
    return "  —  " if x != x else f"{x * 100:5.1f}"  # NaN check


def _print_report(report: dict) -> None:
    print(f"\n=== {report['model']}  (threshold={report['threshold']:.3f}) ===")
    print(f"{'set':<12}{'n':>7}{'AUC':>8}{'recall%':>9}{'FPR%':>7}{'F1%':>7}")
    for set_name, m in report["sets"].items():
        auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] == m["roc_auc"] else "  —  "
        print(f"{set_name:<12}{m['n']:>7}{auc:>8}{_fmt(m['recall']):>9}{_fmt(m['fpr']):>7}{_fmt(m['f1']):>7}")
    for set_name in ("external", "holdout"):
        if set_name not in report["sets"]:
            continue
        print(f"\n{set_name} breakdown (recall% on attacks / FPR% on benign):")
        for group, row in report["sets"][set_name]["breakdown"].items():
            parts = []
            if "recall" in row:
                parts.append(f"recall {_fmt(row['recall'])}")
            if "fpr" in row:
                parts.append(f"FPR {_fmt(row['fpr'])}")
            print(f"  {group:<28} n={row['n']:<5} " + "  ".join(parts))


# ── Standalone loaders for already-trained models ────────────────────────────

def load_classifier(classifier_type: str):
    """Load a classifier through the API's own loader.

    "cascade2_hf2" is evaluation-only: the cascade2 logic with the PyTorch HF2
    model as the slow path, so the FP32 cascade can be measured on a GPU (the
    FP32 ONNX graph is validated to match PyTorch at export time).
    """
    from app.config import Settings
    from app.main import _load_classifier

    if classifier_type == "cascade2_hf2":
        from app.classifiers.cascade_classifier import CascadeClassifier

        settings = Settings()
        clf = CascadeClassifier(
            sklearn_clf=_load_classifier(Settings(classifier_type="sklearn")),
            slow_clf=_load_classifier(Settings(classifier_type="hf2")),
            low_threshold=settings.cascade_sk_low_threshold,
            high_threshold=settings.cascade_sk_high_threshold,
            slow_clf_label="hf2",
        )
    else:
        clf = _load_classifier(Settings(classifier_type=classifier_type))
    if not clf.is_loaded():
        raise RuntimeError(f"{classifier_type} classifier failed to load — see warnings above")
    return clf


def classifier_score_fn(classifier_type_or_clf, concurrency: int = 8) -> ScoreFn:
    """Score texts with a serving classifier (given by type name or instance)."""
    clf = (load_classifier(classifier_type_or_clf)
           if isinstance(classifier_type_or_clf, str) else classifier_type_or_clf)

    async def _score(texts: Sequence[str]) -> list[float]:
        sem = asyncio.Semaphore(concurrency)

        async def one(text: str) -> float:
            async with sem:
                return float((await clf.predict(text))["malicious_prob"])

        return await asyncio.gather(*(one(t) for t in texts))

    return lambda texts: np.asarray(asyncio.run(_score(list(texts))))


def _default_threshold(classifier_type: str) -> float | None:
    """Threshold tuned at training time, read from the model's metadata."""
    from app.config import Settings

    settings = Settings(classifier_type=classifier_type)
    candidates = {
        "sklearn": Path(settings.sklearn_model_path).with_suffix(".json"),
        "hf2": Path(settings.hf2_model_path) / "config.json",
        "onnx2": Path(settings.onnx2_model_path) / "config.json",
    }
    path = candidates.get(classifier_type)
    if path is None or not path.exists():
        return None
    meta = json.loads(path.read_text())
    if classifier_type == "onnx2" and settings.onnx2_use_int8 and (path.parent / "model_int8.onnx").exists():
        return meta.get("aegis_threshold_int8", meta.get("aegis_threshold"))
    return meta.get("threshold", meta.get("aegis_threshold"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Run the shared evaluation suite")
    parser.add_argument("--model", default="sklearn",
                        choices=["sklearn", "hf", "onnx", "cascade", "hf2", "onnx2", "cascade2",
                                 "cascade2_hf2"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Defaults to the value tuned at training time (as the API does)")
    parser.add_argument("--name", default=None, help="Report name (default: --model)")
    args = parser.parse_args()

    from app.classifiers import resolve_threshold

    clf = load_classifier(args.model)
    threshold = resolve_threshold(args.threshold, clf)
    run_suite(classifier_score_fn(clf), threshold, name=args.name or args.model)


if __name__ == "__main__":
    main()
