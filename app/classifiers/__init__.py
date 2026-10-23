"""Classifier backends — scikit-learn (Phase 1) and transformer models (Phases 2/3)."""

# Used when neither CONFIDENCE_THRESHOLD nor the model's metadata gives a threshold.
DEFAULT_THRESHOLD = 0.70


def resolve_threshold(override: float | None, classifier: object) -> float:
    """CONFIDENCE_THRESHOLD if set, else the threshold tuned for *classifier*."""
    if override is not None:
        return float(override)
    tuned = getattr(classifier, "threshold", None)
    return float(tuned) if isinstance(tuned, (int, float)) else DEFAULT_THRESHOLD
