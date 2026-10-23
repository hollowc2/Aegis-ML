"""sklearn helpers shared by the round-2 runners (train/val only)."""

from __future__ import annotations

import numpy as np


def proba(pipeline, texts: list[str]) -> np.ndarray:
    return pipeline.predict_proba(texts)[:, list(pipeline.classes_).index(1)]


def pick_threshold_strict(y, p: np.ndarray, max_fpr: float = 0.02) -> dict:
    """Lowest threshold with val FPR ≤ max_fpr, without the release helper's 0.95 cap.

    ``training.eval_suite.pick_threshold`` clamps to [0.05, 0.95]; the upper
    clamp can silently break the budget. Here the 0.05 floor is kept (it can
    only lower FPR), the cap is dropped, and the achieved FPR is returned.
    """
    y = np.asarray(y)
    benign = np.sort(p[y == 0])
    allowed = int(np.floor(max_fpr * len(benign)))
    idx = len(benign) - allowed - 1
    threshold = float(benign[idx]) + 1e-6 if idx >= 0 else 0.0
    threshold = max(threshold, 0.05)
    fpr = float((p[y == 0] >= threshold).mean())
    return {"threshold": threshold, "val_fpr": fpr, "budget_met": fpr <= max_fpr}
