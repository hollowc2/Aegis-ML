"""
training/phase1_sklearn/train.py
==================================
Phase 1 training — TF-IDF (word + character n-grams) + Logistic Regression.

  - Uses the fixed train/val/test split written by prepare_dataset.py
    (grouped by near-duplicate family, so CV-style leakage is avoided)
  - Word n-grams capture phrasing; char_wb n-grams make the model robust to
    leetspeak, spacing tricks, and partial matches across languages
  - Hyperparameters are chosen on the val split by ROC-AUC
  - The decision threshold is chosen on val to keep FPR under --max-fpr
  - Evaluates on every set in training.eval_suite and writes reports/sklearn.json
  - Saves models/sklearn_classifier.joblib plus a .json sidecar (threshold, params)

Run:
    python -m training.phase1_sklearn.train
    python -m training.phase1_sklearn.train --loso       # + leave-one-source-out check
    python -m training.phase1_sklearn.train --model rf   # Random Forest (comparison only)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from training.eval_suite import (
    DATASET_CSV,
    dataset_fingerprint,
    pick_threshold,
    preprocess,
    run_suite,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
SKLEARN_MODEL_PATH = MODELS_DIR / "sklearn_classifier.joblib"

ModelType = Literal["lr", "rf"]

# Sources large enough to be meaningful when held out entirely.
LOSO_SOURCES = ["xtram1", "spml", "gandalf", "multilingual_pi", "itw_jailbreak", "deepset", "synthetic"]


def load_splits() -> dict[str, pd.DataFrame]:
    if not DATASET_CSV.exists():
        logger.info("Dataset not found — generating now...")
        from training.data.prepare_dataset import main as prep_main
        prep_main()
    df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
    if "split" not in df.columns:
        raise ValueError("Dataset has no split column — rerun training.data.prepare_dataset")
    df["text"] = preprocess(df["text"].astype(str).tolist())
    df["label"] = df["label"].astype(int)
    splits = {name: g.reset_index(drop=True) for name, g in df.groupby("split")}
    for name, g in splits.items():
        logger.info("%-5s %6d rows (benign=%d malicious=%d)",
                    name, len(g), int((g.label == 0).sum()), int((g.label == 1).sum()))
    return splits


def build_pipeline(
    model_type: ModelType = "lr",
    C: float = 4.0,  # noqa: N803 — mirrors LogisticRegression(C=...)
    word_max_features: int = 100_000,
    char_max_features: int = 200_000,
) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9,
            sublinear_tf=True, max_features=word_max_features,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.9,
            sublinear_tf=True, max_features=char_max_features,
        )),
    ])
    if model_type == "lr":
        clf = LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
    return Pipeline([("features", features), ("clf", clf)])


def _proba(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    return pipeline.predict_proba(texts)[:, list(pipeline.classes_).index(1)]


def select_model(model_type: ModelType, train: pd.DataFrame, val: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Small grid search scored on the held-out val split."""
    if model_type == "lr":
        grid = {"C": [1.0, 4.0, 16.0], "char_max_features": [100_000, 300_000]}
    else:
        grid = {"char_max_features": [50_000]}  # RF is a comparison baseline only

    best: tuple[float, Pipeline | None, dict] = (-1.0, None, {})
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        pipe = build_pipeline(model_type, **params).fit(train.text.tolist(), train.label)
        auc = roc_auc_score(val.label, _proba(pipe, val.text.tolist()))
        logger.info("  %s → val AUC %.4f", params, auc)
        if auc > best[0]:
            best = (auc, pipe, params)
    logger.info("Best params %s (val AUC %.4f)", best[2], best[0])
    assert best[1] is not None
    return best[1], best[2]


def leave_one_source_out(params: dict, splits: dict[str, pd.DataFrame]) -> dict:
    """Train without one source at a time and score that source — a cross-domain estimate."""
    full = pd.concat(splits.values(), ignore_index=True)
    results = {}
    for source in LOSO_SOURCES:
        held = full[full.source == source]
        rest = full[(full.source != source) & (full.split == "train")]
        if held.empty:
            continue
        pipe = build_pipeline("lr", **params).fit(rest.text.tolist(), rest.label)
        p = _proba(pipe, held.text.tolist())
        row = {"n": int(len(held)), "recall@0.5": float((p[held.label == 1] >= 0.5).mean())
               if (held.label == 1).any() else None}
        if held.label.nunique() == 2:
            row["roc_auc"] = float(roc_auc_score(held.label, p))
            row["fpr@0.5"] = float((p[held.label == 0] >= 0.5).mean())
        results[source] = row
        logger.info("LOSO %-16s %s", source, row)
    return results


def train(model_type: ModelType = "lr", max_fpr: float = 0.02, loso: bool = False) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]

    logger.info("Selecting hyperparameters on val split...")
    pipeline, params = select_model(model_type, train_df, val_df)

    p_val = _proba(pipeline, val_df.text.tolist())
    threshold = pick_threshold(val_df.label, p_val, max_fpr=max_fpr)
    logger.info("Threshold %.3f keeps val FPR ≤ %.1f%%", threshold, max_fpr * 100)

    extra: dict = {"params": params, "sklearn_version": sklearn.__version__, "max_val_fpr": max_fpr}
    if loso and model_type == "lr":
        extra["leave_one_source_out"] = leave_one_source_out(params, splits)

    name = "sklearn" if model_type == "lr" else "sklearn_rf"
    run_suite(lambda texts: _proba(pipeline, list(texts)), threshold, name=name, extra=extra)

    if model_type != "lr":
        logger.info("RF is a comparison baseline; not overwriting %s", SKLEARN_MODEL_PATH)
        return
    joblib.dump(pipeline, SKLEARN_MODEL_PATH, compress=3)
    meta = {"threshold": round(threshold, 4), "params": params,
            "sklearn_version": sklearn.__version__, "dataset": dataset_fingerprint()}
    SKLEARN_MODEL_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("Model saved to %s (%.1f MB)", SKLEARN_MODEL_PATH,
                SKLEARN_MODEL_PATH.stat().st_size / 1e6)
    logger.info("Recommended: CONFIDENCE_THRESHOLD=%.2f", threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1 scikit-learn classifier")
    parser.add_argument("--model", choices=["lr", "rf"], default="lr")
    parser.add_argument("--max-fpr", type=float, default=0.02,
                        help="FPR budget on the val split used to pick the threshold")
    parser.add_argument("--loso", action="store_true", help="Run leave-one-source-out evaluation")
    args = parser.parse_args()
    train(model_type=args.model, max_fpr=args.max_fpr, loso=args.loso)


if __name__ == "__main__":
    main()
