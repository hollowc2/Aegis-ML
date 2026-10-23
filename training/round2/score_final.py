"""Score a serving classifier on the frozen round-2 false-positive set.

Two views, both at the classifier's unchanged serving threshold:
  whole_string  the scanned message content as one string
  input_guard   the full chat request through app.guardrails.input_guard,
                one prediction at a time (every message, windowed as served)
Fail-secure classifier errors and size-limit blocks are counted separately;
neither is a correct benign decision nor a detection. Size-limit rows are
reported on their own and excluded from the FPR aggregates.

Only aggregates are written (reports/round2/<name>.json); no per-row output.

Run (one model at a time; INT8 is CPU-only):
    taskset -c 0-3 uv run python -m training.round2.score_final --model sklearn --name sklearn_fp_baseline
    ONNX2_USE_INT8=true taskset -c 0-3 uv run python -m training.round2.score_final --model onnx2 --name onnx2_int8_fp_baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import numpy as np
import pandas as pd

from training.round2.common import REPORTS, cap_threads, rate, sha256, write_json
from training.round2.final_fp_eval import MANIFEST, OUTPUT

logger = logging.getLogger(__name__)


def _group(df: pd.DataFrame, col: str, flag: str) -> dict:
    return {str(k): rate(g[flag]) for k, g in df.groupby(col)}


def score(clf, threshold: float) -> dict:
    from unittest.mock import patch

    from app.config import Settings
    from app.guardrails.input_guard import run_input_guardrail
    from app.models.schemas import ChatCompletionRequest
    from training.eval_suite import preprocess

    df = pd.read_csv(OUTPUT, keep_default_na=False)
    settings = Settings(confidence_threshold=threshold)

    async def run():
        whole, guard, errors, size_blocks = [], [], [], []
        with patch("app.guardrails.input_guard.get_settings", return_value=settings):
            for text, messages in zip(preprocess(df.text.tolist()), df.messages_json):
                whole.append(float((await clf.predict(text))["malicious_prob"]))
                request = ChatCompletionRequest(model="round2", messages=json.loads(messages))
                result = await run_input_guardrail(request, clf)
                errors.append("Classifier error" in result.reason)
                size_blocks.append("too large to scan" in result.reason)
                guard.append(result.verdict == "block")
        return whole, guard, errors, size_blocks

    whole, guard, errors, size_blocks = asyncio.run(run())
    df = df.assign(
        whole_flag=np.asarray(whole) >= threshold,
        guard_flag=guard,
        error=errors,
        size_block=size_blocks,
    )
    if df.error.any():
        raise RuntimeError(f"{int(df.error.sum())} classifier errors; results invalid")
    scored = df[df.category != "size_limit_benign"]
    out = {
        "threshold": threshold,
        "n": int(len(df)),
        "classifier_errors": int(df.error.sum()),
        "size_limit_rows": {
            "n": int((df.category == "size_limit_benign").sum()),
            "size_limit_blocks": int(df[df.category == "size_limit_benign"].size_block.sum()),
        },
        "size_limit_blocks_outside_size_limit_rows": int(scored.size_block.sum()),
    }
    for view, flag in (("whole_string", "whole_flag"), ("input_guard", "guard_flag")):
        out[view] = {
            "fpr_all": rate(scored[flag]),
            "fpr_by_category": _group(scored, "category", flag),
            "fpr_by_subcategory": _group(scored, "subcategory", flag),
            "fpr_by_source": _group(scored, "source", flag),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Score the frozen round-2 FP set")
    parser.add_argument(
        "--model", required=True, choices=["sklearn", "hf2", "onnx2", "cascade2", "cascade2_hf2"]
    )
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    cap_threads()
    manifest = json.loads(MANIFEST.read_text())
    if sha256(OUTPUT) != manifest["sha256"]:
        raise ValueError("Final FP set does not match its frozen manifest")

    from app.classifiers import resolve_threshold
    from training.eval_suite import load_classifier

    clf = load_classifier(args.model)
    threshold = resolve_threshold(None, clf)
    report = {
        "model": args.model,
        "set": str(OUTPUT),
        "sha256": manifest["sha256"],
        **score(clf, threshold),
    }
    write_json(REPORTS / f"{args.name}.json", report, overwrite=False)
    logger.info(
        "whole-string FPR %.3f, guard FPR %.3f",
        report["whole_string"]["fpr_all"]["rate"],
        report["input_guard"]["fpr_all"]["rate"],
    )


if __name__ == "__main__":
    main()
