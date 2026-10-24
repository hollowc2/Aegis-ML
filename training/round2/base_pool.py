"""Arm B0: the controlled base pool and its sklearn model (train/val only).

The pool is the original released-model dataset (sha256 prefix 0c6e12…) after
removing every row that overlaps a registered evaluation set: exact normalised
texts of external/holdout/targeted/adversarial/round-2 FP rows, every serving
scan window of the targeted and round-2 requests, and the targeted set's
reserved payload substrings. Retained rows keep their original split; removals
are recorded by source/label/split.

B1–B3 (new hard negatives, indirect/exfiltration additions) were not built:
authoring that content was stopped by a safety classifier, and the work was
narrowed (see reports/round2/protocol.json). B0 is therefore not compared
against another arm.

Run:
    taskset -c 0-3 uv run python -m training.round2.base_pool
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from training.round2.common import (
    ARTIFACTS,
    BASE_POOL,
    ORIGINAL_DATASET,
    ORIGINAL_SHA16,
    REPORTS,
    cap_threads,
    fingerprint,
    require_sha,
    safe_output,
    write_json,
)

logger = logging.getLogger(__name__)
GRID = {"C": [4.0, 16.0], "char_max_features": [100_000, 300_000]}
MAX_FPR = 0.02
MODEL_DIR = ARTIFACTS / "models" / "b0_sklearn"


def forbidden_keys() -> tuple[set[str], set[str]]:
    """(exact normalised keys, reserved payload keys) from every evaluation set."""
    from app.guardrails.input_guard import _extract_segments
    from app.models.schemas import ChatCompletionRequest, ChatMessage
    from training.data.prepare_dataset import normalise_key
    from training.eval_suite import EVAL_SETS

    keys: set[str] = set()
    for path, _ in EVAL_SETS.values():
        df = pd.read_csv(path, keep_default_na=False)
        keys.update(df.text.astype(str).map(normalise_key))
        if "messages_json" in df:
            requests = [
                ChatCompletionRequest(model="overlap", messages=json.loads(m))
                for m in df.messages_json
            ]
        else:
            requests = [
                ChatCompletionRequest(
                    model="overlap", messages=[ChatMessage(role="user", content=t)]
                )
                for t in df.text.astype(str)
            ]
        for request in requests:
            keys.update(normalise_key(w) for w in _extract_segments(request))
            keys.update(normalise_key(m.content) for m in request.messages if m.content.strip())
    manifest = json.loads(open("data/targeted_eval.manifest.json").read())
    payloads = {normalise_key(p) for p in manifest["reserved_payloads"] if normalise_key(p)}
    keys.discard("")
    return keys, payloads


def build_pool() -> tuple[pd.DataFrame, dict]:
    from training.data.prepare_dataset import normalise_key

    require_sha(ORIGINAL_DATASET, ORIGINAL_SHA16)
    df = pd.read_csv(ORIGINAL_DATASET)
    keys, payloads = forbidden_keys()
    norm = df.text.astype(str).map(normalise_key)
    exact = norm.isin(keys)
    substring = norm.map(lambda k: any(p in k for p in payloads))
    removed = exact | substring
    record = {
        "removed_exact_or_window": int(exact.sum()),
        "removed_reserved_payload_substring": int((substring & ~exact).sum()),
        "removed_by_source_label_split": {
            f"{s}/{lab}/{sp}": int(n)
            for (s, lab, sp), n in df[removed].groupby(["source", "label", "split"]).size().items()
        },
        "retained_split_label": {
            f"{sp}/{lab}": int(n)
            for (sp, lab), n in df[~removed].groupby(["split", "label"]).size().items()
        },
        "split_preserved": "retained rows keep the original split column unchanged",
    }
    return df[~removed].reset_index(drop=True), record


def train_b0(pool: pd.DataFrame) -> dict:
    import itertools

    import joblib
    from sklearn.metrics import roc_auc_score

    from training.eval_suite import preprocess
    from training.phase1_sklearn.train import build_pipeline
    from training.round2.diagnose import global_metrics, slice_metrics
    from training.round2.sk_common import pick_threshold_strict, proba

    pool = pool.assign(
        text=preprocess(pool.text.astype(str).tolist()), template_id="", payload="", carrier_id=""
    )
    train, val = pool[pool.split == "train"], pool[pool.split == "val"].reset_index(drop=True)
    grid_results, best = [], None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        pipe = build_pipeline("lr", **params).fit(train.text.tolist(), train.label)
        auc = float(roc_auc_score(val.label, proba(pipe, val.text.tolist())))
        grid_results.append({**params, "val_auc": auc})
        logger.info("%s → val AUC %.4f", params, auc)
        if best is None or auc > best[0]:
            best = (auc, pipe, params)
    _, pipe, params = best
    p_val = proba(pipe, val.text.tolist())
    thr = pick_threshold_strict(val.label.to_numpy(), p_val, MAX_FPR)
    joblib.dump(pipe, safe_output(MODEL_DIR / "sklearn_classifier.joblib"), compress=3)
    write_json(
        MODEL_DIR / "sklearn_classifier.json",
        {"threshold": thr["threshold"], "params": params, "dataset": fingerprint(BASE_POOL)},
    )
    return {
        "grid": grid_results,
        "selected_params": params,
        "threshold": thr,
        "val": {
            **global_metrics(val.label, p_val, thr["threshold"]),
            **slice_metrics(val, p_val, thr["threshold"]),
        },
        "model_path": str(MODEL_DIR),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cap_threads()
    pool, record = build_pool()
    pool.to_csv(safe_output(BASE_POOL), index=False)
    report = {
        "pool": fingerprint(BASE_POOL),
        "source_dataset_sha16": ORIGINAL_SHA16,
        "hygiene": record,
        "evidence_boundary": "train and val only",
    }
    report["b0_sklearn"] = train_b0(pool)
    write_json(REPORTS / "b0_base_pool.json", report)
    logger.info("B0 val: %s", {k: report["b0_sklearn"]["val"][k] for k in ("auc", "threshold")})


if __name__ == "__main__":
    main()
