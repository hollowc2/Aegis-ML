"""
training/phase1_sklearn/train.py
==================================
Phase 1 training — TF-IDF + Logistic Regression (primary) and
Random Forest (secondary, for comparison in Phase 3 notebook).

Features:
  - Loads the combined dataset prepared by prepare_dataset.py
  - Trains with cross-validation and hyperparameter grid search
  - Aggressively tunes to hit <5% false-positive rate on benign prompts
  - Saves the best model to models/sklearn_classifier.joblib
  - Prints a full classification report + confusion matrix

Run:
    python -m training.phase1_sklearn.train
    python -m training.phase1_sklearn.train --model rf   # Random Forest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = Path("data/combined_dataset.csv")
MODELS_DIR = Path("models")
SKLEARN_MODEL_PATH = MODELS_DIR / "sklearn_classifier.joblib"

ModelType = Literal["lr", "rf"]


def load_data() -> tuple[list[str], list[int]]:
    """Load the combined dataset. Runs prepare_dataset.py if CSV not found."""
    if not DATASET_CSV.exists():
        logger.info("Dataset not found — generating now...")
        from training.data.prepare_dataset import main as prep_main
        prep_main()

    df = pd.read_csv(DATASET_CSV)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    logger.info("Loaded %d samples (benign=%d  malicious=%d)",
                len(df), (df["label"] == 0).sum(), (df["label"] == 1).sum())
    return df["text"].tolist(), df["label"].tolist()


def build_pipeline(model_type: ModelType = "lr") -> Pipeline:
    """Build a TF-IDF → classifier pipeline."""
    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 3),       # unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,        # log-scaled TF for better balance
        max_features=50_000,
    )

    if model_type == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
    else:  # rf
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def tune_threshold(
    pipeline: Pipeline,
    X_val: list[str],
    y_val: list[int],
    target_fpr: float = 0.05,
) -> float:
    """
    Find the lowest threshold where FPR (false-positive rate on benign prompts)
    stays at or below *target_fpr*.

    Returns the optimal threshold (default 0.70 if no threshold achieves the target).
    """
    proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_arr = np.array(y_val)

    best_threshold = 0.70
    best_fpr = 1.0

    for threshold in np.arange(0.30, 0.96, 0.01):
        preds = (proba >= threshold).astype(int)
        # FP = predicted malicious when actually benign
        benign_mask = y_val_arr == 0
        fp = ((preds == 1) & benign_mask).sum()
        tn = ((preds == 0) & benign_mask).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if fpr <= target_fpr:
            best_threshold = float(threshold)
            best_fpr = fpr
            break  # take the lowest threshold that satisfies the constraint

    logger.info(
        "Threshold tuning → optimal threshold=%.2f  FPR=%.2f%%",
        best_threshold, best_fpr * 100,
    )
    return best_threshold


def train(model_type: ModelType = "lr") -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # ── Grid search ────────────────────────────────────────────────────────────
    pipeline = build_pipeline(model_type)

    if model_type == "lr":
        param_grid = {
            "tfidf__ngram_range": [(1, 2), (1, 3)],
            "tfidf__max_features": [30_000, 50_000],
            "clf__C": [0.1, 1.0, 10.0],
        }
    else:
        param_grid = {
            "tfidf__ngram_range": [(1, 2), (1, 3)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 20],
        }

    logger.info("Running GridSearchCV (5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",       # optimise for F1 (balances precision + recall)
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_inner, y_train_inner)

    best_pipeline = grid_search.best_estimator_
    logger.info("Best params: %s", grid_search.best_params_)
    logger.info("Best CV F1:  %.4f", grid_search.best_score_)

    # ── Threshold tuning ───────────────────────────────────────────────────────
    optimal_threshold = tune_threshold(best_pipeline, X_val, y_val, target_fpr=0.05)

    # ── Evaluate on test set ───────────────────────────────────────────────────
    proba_test = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= optimal_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, proba_test)
    report = classification_report(y_test, y_pred, target_names=["benign", "malicious"])
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n%s", "=" * 50)
    logger.info("TEST SET RESULTS (threshold=%.2f)", optimal_threshold)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("\nClassification Report:\n%s", report)
    logger.info("Confusion Matrix:\n%s", cm)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    logger.info("False Positive Rate (FPR): %.2f%%", fpr * 100)
    logger.info("False Negative Rate (FNR): %.2f%%", fnr * 100)

    if fpr > 0.05:
        logger.warning(
            "FPR=%.2f%% exceeds target of 5%%. Consider raising the threshold.",
            fpr * 100,
        )
    else:
        logger.info("FPR target (<5%%) achieved!")

    # ── Save model ─────────────────────────────────────────────────────────────
    joblib.dump(best_pipeline, SKLEARN_MODEL_PATH)
    logger.info("Model saved to %s", SKLEARN_MODEL_PATH)
    logger.info("Optimal threshold: %.2f  (set CONFIDENCE_THRESHOLD=%.2f in .env)",
                optimal_threshold, optimal_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1 scikit-learn classifier")
    parser.add_argument(
        "--model", choices=["lr", "rf"], default="lr",
        help="Classifier type: lr=LogisticRegression (default), rf=RandomForest",
    )
    args = parser.parse_args()
    train(model_type=args.model)


if __name__ == "__main__":
    main()
