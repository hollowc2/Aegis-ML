"""Does window-by-window scanning inflate false positives on long documents?

Validation data only. Long benign documents are built by joining benign rows of
the released dataset's ``val`` split (paragraph-separated) up to a target
length; buried-attack documents are the same benign documents with one ``val``
malicious row of at most ``SEGMENT_OVERLAP`` characters inserted at a paragraph
boundary, so serving's overlapping windows always contain it whole. No new
attack text is written. Every document goes through the real input guard.

Short ``val`` benign rows are scored the same way as the reference false
positive rate, so the multiplicity effect is the difference between them.

``--segment-chars``/``--overlap`` patch the guard's window size for this run
only (the overlap must stay at least 400 so inserted attacks fit one window).

Only aggregates are written (reports/round2/long_docs_<name>.json); the
threshold sweep is descriptive, not a selection.

Run (one model at a time):
    taskset -c 0-3 uv run python -m training.round2.long_docs --model sklearn --name sklearn
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random

import numpy as np
import pandas as pd

from training.round2.common import (
    ORIGINAL_DATASET,
    ORIGINAL_SHA16,
    REPORTS,
    cap_threads,
    rate,
    require_sha,
    write_json,
)

logger = logging.getLogger(__name__)
LENGTHS = (2_500, 5_000, 10_000, 20_000, 40_000)
BENIGN_PER_LENGTH = 200
BURIED_PER_LENGTH = 100
SEED = 20261023
MAX_ATTACK_CHARS = 400  # serving's default overlap: always inside one window


def compose(val: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Long benign and buried-attack documents from val rows."""
    rng = random.Random(seed)
    benign = val.text[val.label == 0].astype(str).tolist()
    attacks = [t for t in val.text[val.label == 1].astype(str) if len(t) <= MAX_ATTACK_CHARS]

    def paragraphs(target: int) -> list[str]:
        parts, size = [], 0
        while size < target:
            parts.append(rng.choice(benign).strip())
            size += len(parts[-1]) + 2
        return parts

    rows = []
    for length in LENGTHS:
        for _ in range(BENIGN_PER_LENGTH):
            rows.append({"text": "\n\n".join(paragraphs(length)), "label": 0, "length": length})
        for _ in range(BURIED_PER_LENGTH):
            parts = paragraphs(length)
            position = rng.randrange(len(parts) + 1)
            parts.insert(position, rng.choice(attacks).strip())
            rows.append(
                {
                    "text": "\n\n".join(parts),
                    "label": 1,
                    "length": length,
                    "relative_position": position / len(parts),
                }
            )
    return pd.DataFrame(rows)


def guard_scores(clf, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(highest window score after serving preprocessing/bias, window count)."""
    from unittest.mock import patch

    from app.config import Settings
    from app.guardrails.input_guard import _extract_segments, run_input_guardrail
    from app.models.schemas import ChatCompletionRequest, ChatMessage

    # Threshold above 1 so the guard reports the max score without deciding.
    settings = Settings(confidence_threshold=1.0, max_scan_segments=10_000)

    async def run():
        scores, windows = [], []
        with patch("app.guardrails.input_guard.get_settings", return_value=settings):
            for text in texts:
                request = ChatCompletionRequest(
                    model="long-docs", messages=[ChatMessage(role="user", content=text)]
                )
                result = await run_input_guardrail(request, clf)
                if "Classifier error" in result.reason:
                    raise RuntimeError(result.reason)
                scores.append(result.confidence)
                windows.append(len(_extract_segments(request)))
        return np.asarray(scores), np.asarray(windows)

    return asyncio.run(run())


def summarize(short: pd.DataFrame, docs: pd.DataFrame, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    benign, buried = docs[docs.label == 0], docs[docs.label == 1]
    out = {
        "threshold": threshold,
        "doc_auc_buried_vs_long_benign": float(roc_auc_score(docs.label, docs.score)),
        "short_val_benign_fpr": rate(short.score >= threshold),
        "long_benign_fpr": rate(benign.score >= threshold),
        "buried_recall": rate(buried.score >= threshold),
        "by_length": {},
    }
    for length, g in docs.groupby("length"):
        out["by_length"][str(length)] = {
            "windows_median": float(g.windows.median()),
            "doc_auc": float(roc_auc_score(g.label, g.score)),
            "benign_fpr": rate(g[g.label == 0].score >= threshold),
            "buried_recall": rate(g[g.label == 1].score >= threshold),
        }
    sweep = sorted({threshold, *np.round(np.linspace(threshold, 0.99, 9), 4)})
    out["sweep_descriptive_only"] = [
        {
            "threshold": float(t),
            "short_val_benign_fpr": float((short.score >= t).mean()),
            "long_benign_fpr": float((benign.score >= t).mean()),
            "buried_recall": float((buried.score >= t).mean()),
        }
        for t in sweep
    ]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Long-document multiplicity check (val only)")
    parser.add_argument(
        "--model", required=True, choices=["sklearn", "hf2", "onnx2", "cascade2", "cascade2_hf2"]
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--segment-chars", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()
    from app.guardrails import input_guard

    if args.segment_chars:
        input_guard.SEGMENT_CHARS = args.segment_chars
    if args.overlap:
        if args.overlap < MAX_ATTACK_CHARS:
            raise ValueError(f"overlap must be >= {MAX_ATTACK_CHARS}")
        input_guard.SEGMENT_OVERLAP = args.overlap
    cap_threads()
    require_sha(ORIGINAL_DATASET, ORIGINAL_SHA16)
    val = pd.read_csv(ORIGINAL_DATASET).query("split == 'val'").reset_index(drop=True)
    docs = compose(val)
    short = val[val.label == 0].reset_index(drop=True)

    from app.classifiers import resolve_threshold
    from training.eval_suite import load_classifier

    clf = load_classifier(args.model)
    threshold = resolve_threshold(None, clf)
    short["score"], short["windows"] = guard_scores(clf, short.text.astype(str).tolist())
    docs["score"], docs["windows"] = guard_scores(clf, docs.text.tolist())
    report = {
        "model": args.model,
        "segment_chars": input_guard.SEGMENT_CHARS,
        "segment_overlap": input_guard.SEGMENT_OVERLAP,
        "evidence_boundary": "val split of the released dataset only",
        "source_dataset_sha16": ORIGINAL_SHA16,
        "composition": {
            "lengths": LENGTHS,
            "benign_per_length": BENIGN_PER_LENGTH,
            "buried_per_length": BURIED_PER_LENGTH,
            "seed": SEED,
            "val_benign_rows": int((val.label == 0).sum()),
            "val_attack_rows_eligible": int(((val.label == 1) & (val.text.str.len() <= 400)).sum()),
        },
        **summarize(short, docs, threshold),
    }
    write_json(REPORTS / f"long_docs_{args.name}.json", report, overwrite=False)
    logger.info(
        "short FPR %.3f  long FPR %.3f  buried recall %.3f",
        report["short_val_benign_fpr"]["rate"],
        report["long_benign_fpr"]["rate"],
        report["buried_recall"]["rate"],
    )


if __name__ == "__main__":
    main()
