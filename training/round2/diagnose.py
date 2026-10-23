"""Diagnose the rejected targeted experiment using train and validation data only.

Never reads external, test, holdout, targeted or round-2 final evaluation rows.
Writes ``reports/round2/diagnosis.json``.

Sections:
  dataset   committed original (0c6e12…) vs rejected (a68869…) CSVs: label/source
            mix per split, lengths, length-bucket benign composition, groups, carriers.
  stages    approximate stage-by-stage rebuild (``stages.py``; same row count as
            the original but not byte-identical, so counts are indicative only).
  model     the rejected sklearn candidate retrained from its recorded recipe
            (verified by reproducing its recorded val threshold), coefficients,
            false-alarm feature attributions, counterfactuals.
  slices    val recall/FPR by source and by provenance-defined semantic slice,
            with Wilson 95% intervals, for the released baseline (on original val)
            and the candidate (on rejected val), plus both on rows that are val in
            both datasets.
  long      whole-string vs actual input-guard scoring of long val documents,
            with token visibility, windows, errors and size-limit blocks.

Run:
    taskset -c 0-3 uv run python -m training.round2.diagnose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re

import numpy as np
import pandas as pd

from training.round2.common import (
    ARTIFACTS,
    ORIGINAL_DATASET,
    ORIGINAL_SHA16,
    REJECTED_DATASET,
    REJECTED_SHA16,
    REPORTS,
    cap_threads,
    rate,
    require_sha,
    safe_output,
    write_json,
)

logger = logging.getLogger(__name__)

CANDIDATE_PATH = ARTIFACTS / "models" / "rejected_candidate_repro.joblib"
RECORDED_CANDIDATE_THRESHOLD = 0.48513361631912105
CANDIDATE_PARAMS = {"C": 16.0, "char_max_features": 300_000}
HARD_NEGATIVE_SOURCES = ("oasst2_corrections", "synthetic_hardneg", "targeted_carrier_benign",
                         "bipia_email_benign")
# Explicit takeover / secrecy vocabulary (H2). Counted, never used for labels.
TAKEOVER = re.compile(
    r"\b(?:ignore|disregard|override|overrides|overrid|takeover|abandon|supersede|outranks"
    r"|secretly|silently|conceal|suppress|hide|without (?:notifying|telling|approval|seeking)"
    r"|do not (?:tell|disclose|ask)|system patch|system override|top-priority|void)\b",
    re.IGNORECASE,
)
DESTINATION = re.compile(r"https?://|\b[\w.-]+@[\w.-]+\.\w+|\.example\b", re.IGNORECASE)


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path) -> pd.DataFrame:
    from training.eval_suite import preprocess

    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = preprocess(df["text"].astype(str).tolist())
    df["label"] = df["label"].astype(int)
    for col in ("template_id", "payload", "carrier_id"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df


def semantic_slice(row) -> str:
    """Provenance/annotation-defined slice; never derived from model scores."""
    src, tid = row["source"], row["template_id"]
    if row["label"] == 1:
        if src in ("targeted_indirect", "bipia_email_train"):
            return "indirect_injection"
        if src == "targeted_exfil" or row["threat_category"] == "data_exfiltration":
            return "exfiltration"
        return "other_attack"
    if tid.startswith("security:"):
        return "security_discussion"
    if tid in ("authorized:0", "authorized:1", "authorized:2"):
        return "authorized_sharing"
    if tid in ("authorized:3", "authorized:4") or src == "oasst2_corrections":
        return "corrections_overrides"
    if tid == "authorized:5":
        return "quoted_attack_discussion"
    if src in ("targeted_carrier_benign", "bipia_email_benign"):
        return "imperative_benign_documents"
    if src == "synthetic_hardneg":
        return "legacy_templated_hard_negatives"
    return "ordinary_benign"


def slice_metrics(df: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    df = df.assign(p=p, flag=p >= threshold, slice=df.apply(semantic_slice, axis=1),
                   long=df.text.str.len() > 2000)
    out: dict = {"by_slice": {}, "by_source": {}, "long_messages": {}}
    for key, col in (("by_slice", "slice"), ("by_source", "source")):
        for name, g in df.groupby(col):
            out[key][str(name)] = _rates(g)
    for name, g in df[df.long].groupby("label"):
        out["long_messages"]["malicious" if name else "benign"] = _rates(g)
    return out


def _rates(g: pd.DataFrame) -> dict:
    row = {"n": int(len(g))}
    if (g.label == 1).any():
        row["recall"] = rate(g.flag[g.label == 1])
    if (g.label == 0).any():
        row["fpr"] = rate(g.flag[g.label == 0])
    return row


def global_metrics(y, p, threshold) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    return {"auc": float(roc_auc_score(y, p)), "recall": rate(p[y == 1] >= threshold),
            "fpr": rate(p[y == 0] >= threshold), "threshold": threshold}


# ── dataset comparison ───────────────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame) -> dict:
    benign = df[df.label == 0]
    hard = benign.source.isin(HARD_NEGATIVE_SOURCES)
    bucket = np.clip(np.floor(np.log2(df.text.str.len())), 4, 13).astype(int)
    groups = df["split_group"] if "split_group" in df else df.text.map(
        lambda t: " ".join(re.sub(r"[\W_]+", " ", t.lower()).split()[:8]))
    out = {
        "rows": int(len(df)),
        "split_label": df.groupby(["split", "label"]).size().unstack().to_dict(),
        "source_label": df.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(),
        "length_chars_quantiles": {
            str(lab): df[df.label == lab].text.str.len().quantile([.1, .5, .9, .99]).round().to_dict()
            for lab in (0, 1)},
        "benign_ordinary_vs_hard": {"ordinary": int((~hard).sum()), "hard_negative": int(hard.sum())},
        "benign_ordinary_by_split": benign[~hard].split.value_counts().to_dict(),
        "benign_by_length_bucket": {
            str(b): {"ordinary": int(((bucket == b) & (df.label == 0) & ~df.source.isin(HARD_NEGATIVE_SOURCES)).sum()),
                     "hard_negative": int(((bucket == b) & (df.label == 0) & df.source.isin(HARD_NEGATIVE_SOURCES)).sum()),
                     "malicious": int(((bucket == b) & (df.label == 1)).sum())}
            for b in sorted(bucket.unique())},
        "split_groups": int(groups.nunique()),
        "largest_group_rows": int(groups.value_counts().iloc[0]),
        "takeover_vocab_rate": {str(k): float(v) for k, v in
                                df.text.str.contains(TAKEOVER).groupby(df.label).mean().items()},
        "destination_rate": {str(k): float(v) for k, v in
                             df.text.str.contains(DESTINATION).groupby(df.label).mean().items()},
    }
    if "template_id" in df:
        carriers = df.text.map(_carrier_marker)
        tgt = df.source.str.startswith("targeted_") | df.source.str.startswith("bipia")
        out["targeted_carrier_coverage"] = (
            pd.crosstab(carriers[tgt], df.label[tgt]).to_dict())
    return out


_MARKERS = {"email": "From: operations@work.example", "web": "<article>", "tool": '"tool":"retrieve_records"',
            "table": "Quarter,Team,Notes", "code": "def report_total", "summary": "quarterly digest",
            "recipe": "Method:", "howto": "Equipment setup:", "bipia_email": "Subject:"}


def _carrier_marker(text: str) -> str:
    for name, marker in _MARKERS.items():
        if marker in text:
            return name
    return "window_without_header"


def compare_datasets(original: pd.DataFrame, rejected: pd.DataFrame) -> dict:
    o_texts, r_texts = set(original.text), set(rejected.text)
    o_split = dict(zip(original.text, original.split))
    common = rejected[rejected.text.isin(o_texts)]
    moved = common[common.split != common.text.map(o_split)]
    ordinary_o = set(original.text[(original.label == 0) & ~original.source.isin(HARD_NEGATIVE_SOURCES)])
    ordinary_r = set(rejected.text[(rejected.label == 0) & ~rejected.source.isin(HARD_NEGATIVE_SOURCES)])
    return {
        "original": describe_dataset(original),
        "rejected": describe_dataset(rejected),
        "shared_rows": int(len(common)),
        "original_rows_dropped": int(len(o_texts - r_texts)),
        "shared_rows_changed_split": int(len(moved)),
        "changed_split_matrix": pd.crosstab(moved.text.map(o_split), moved.split).to_dict()
        if len(moved) else {},
        "ordinary_benign_original": len(ordinary_o),
        "ordinary_benign_rejected": len(ordinary_r),
        "ordinary_benign_original_retained": len(ordinary_o & ordinary_r),
    }


def stage_summary() -> dict:
    from training.round2 import stages

    raw = stages.load_raw_sources()
    out = {"caveat": ("Rebuild matches the original row count but only ~75% of rows; "
                      "upstream/cache order drift changes sampling. Indicative only.")}
    for recipe in ("original", "rejected"):
        built = stages.build(recipe, raw)
        out[recipe] = {
            "sha256_16": built["sha256"][:16],
            "stages": {name: {"rows": int(len(f)), "label": f.label.value_counts().to_dict(),
                              "hard_negative_benign": int(((f.label == 0) & f.source.isin(
                                  HARD_NEGATIVE_SOURCES)).sum())}
                       for name, f in built["stages"].items()},
            "length_balance_log": built["length_log"],
        }
    return out


# ── candidate model ──────────────────────────────────────────────────────────

def train_candidate(rejected: pd.DataFrame):
    import joblib

    from training.phase1_sklearn.train import build_pipeline
    from training.round2.sk_common import pick_threshold_strict, proba

    if CANDIDATE_PATH.exists():
        pipe = joblib.load(CANDIDATE_PATH)
    else:
        train = rejected[rejected.split == "train"]
        pipe = build_pipeline("lr", **CANDIDATE_PARAMS).fit(train.text.tolist(), train.label)
        joblib.dump(pipe, safe_output(CANDIDATE_PATH), compress=3)
    val = rejected[rejected.split == "val"]
    p_val = proba(pipe, val.text.tolist())
    threshold = pick_threshold_strict(val.label.to_numpy(), p_val, 0.02)["threshold"]
    return pipe, p_val, threshold


def coefficients(pipe, top: int = 30) -> dict:
    feats = pipe.named_steps["features"]
    names = np.concatenate([[f"word:{n}" for n in feats.transformer_list[0][1].get_feature_names_out()],
                            [f"char:{n}" for n in feats.transformer_list[1][1].get_feature_names_out()]])
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return {"names": names, "coef": coef,
            "top_malicious": [[names[i], float(coef[i])] for i in order[::-1][:top]],
            "top_benign": [[names[i], float(coef[i])] for i in order[:top]]}


def false_alarm_attribution(pipe, coefs, val: pd.DataFrame, p, threshold, top=25) -> dict:
    """Sum of positive coef×tfidf contributions over flagged benign val rows, per slice."""
    x = pipe.named_steps["features"].transform(val.text.tolist())
    slices = val.apply(semantic_slice, axis=1).to_numpy()
    out = {}
    for name in sorted(set(slices[val.label.to_numpy() == 0])):
        mask = (val.label.to_numpy() == 0) & (slices == name) & (p >= threshold)
        if not mask.any():
            continue
        contrib = np.asarray(x[mask].multiply(coefs["coef"]).sum(axis=0)).ravel() / mask.sum()
        order = np.argsort(contrib)[::-1][:top]
        out[name] = {"false_alarms": int(mask.sum()),
                     "mean_positive_contributions": [[coefs["names"][i], float(contrib[i])]
                                                     for i in order if contrib[i] > 0]}
    return out


NEUTRAL_SENTENCE = "Please confirm the delivery date with the supplier and update the shared calendar."
TOPIC_SWAP = [("prompt injection", "spelling mistakes"), ("prompt extraction", "spreadsheet formulas"),
              ("jailbreak", "garden design"), ("exfiltration", "proofreading"),
              ("system prompt", "style guide"), ("untrusted", "seasonal"),
              ("malicious", "unusual"), ("red-team", "editorial"), ("attack", "review"),
              ("injection", "formatting"), ("secret", "draft"), ("token", "label")]


def counterfactuals(pipe, val: pd.DataFrame, threshold: float) -> dict:
    """Validation-only edits that remove or replace the labelled evidence."""
    from training.round2.sk_common import proba

    out = {}
    attacks = val[(val.label == 1) & (val.payload != "") & val.apply(
        lambda r: r.payload in r.text, axis=1)]
    if len(attacks):
        before = proba(pipe, attacks.text.tolist())
        removed = proba(pipe, [t.replace(pl, "") for t, pl in zip(attacks.text, attacks.payload)])
        neutral = proba(pipe, [t.replace(pl, NEUTRAL_SENTENCE) for t, pl in zip(attacks.text, attacks.payload)])
        out["attack_windows_payload_removed"] = {
            "n": int(len(attacks)),
            "flagged_before": rate(before >= threshold),
            "flagged_payload_removed": rate(removed >= threshold),
            "flagged_payload_replaced_by_benign_imperative": rate(neutral >= threshold),
            "median_score_before": float(np.median(before)),
            "median_score_removed": float(np.median(removed)),
        }
    sec = val[(val.label == 0) & val.template_id.str.match(r"security:|authorized:5")]
    if len(sec):
        def swap(t):
            for a, b in TOPIC_SWAP:
                t = re.sub(a, b, t, flags=re.IGNORECASE)
            return t
        before = proba(pipe, sec.text.tolist())
        after = proba(pipe, [swap(t) for t in sec.text])
        out["security_benign_topic_words_swapped"] = {
            "n": int(len(sec)), "flagged_before": rate(before >= threshold),
            "flagged_after_swap": rate(after >= threshold),
            "median_before": float(np.median(before)), "median_after": float(np.median(after))}
    auth = val[(val.label == 0) & val.template_id.isin(["authorized:0", "authorized:1", "authorized:2"])]
    if len(auth):
        before = proba(pipe, auth.text.tolist())
        no_dest = proba(pipe, [DESTINATION.sub("my team", t) for t in auth.text])
        out["authorized_sharing_destination_removed"] = {
            "n": int(len(auth)), "flagged_before": rate(before >= threshold),
            "flagged_destination_removed": rate(no_dest >= threshold)}
    return out


def shortcut_probes(rejected: pd.DataFrame) -> dict:
    """How far do cheap surface features separate the targeted training rows? (H2)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def feats(texts):
        return np.column_stack([texts.str.contains(TAKEOVER).astype(float),
                                texts.str.contains(DESTINATION).astype(float),
                                np.log1p(texts.str.len())])

    train, val = rejected[rejected.split == "train"], rejected[rejected.split == "val"]
    clf = LogisticRegression().fit(feats(train.text), train.label)
    out = {"surface_model_val_auc_all": float(roc_auc_score(val.label, clf.predict_proba(feats(val.text))[:, 1]))}
    tgt = val.source.str.startswith("targeted") | val.source.str.startswith("bipia")
    if val[tgt].label.nunique() == 2:
        out["surface_model_val_auc_targeted_rows"] = float(
            roc_auc_score(val[tgt].label, clf.predict_proba(feats(val[tgt].text))[:, 1]))
    sub = rejected[rejected.source.str.startswith("targeted")]
    out["targeted_rows_takeover_vocab_by_label"] = {
        str(k): float(v) for k, v in sub.text.str.contains(TAKEOVER).groupby(sub.label).mean().items()}
    out["targeted_rows_destination_by_label"] = {
        str(k): float(v) for k, v in sub.text.str.contains(DESTINATION).groupby(sub.label).mean().items()}
    out["targeted_unique_payloads"] = int(sub.payload[sub.payload != ""].nunique())
    out["targeted_unique_template_ids"] = int(sub.template_id[sub.template_id != ""].nunique())
    return out


def opposite_label_neighbours(pipe, rejected: pd.DataFrame) -> dict:
    """Val rows whose nearest opposite-label train row is a near-copy (H3)."""
    from sklearn.preprocessing import normalize

    feats = pipe.named_steps["features"]
    train, val = rejected[rejected.split == "train"], rejected[rejected.split == "val"]
    x_train = normalize(feats.transform(train.text.tolist()))
    x_val = normalize(feats.transform(val.text.tolist()))
    yt, yv = train.label.to_numpy(), val.label.to_numpy()
    slices = val.apply(semantic_slice, axis=1).to_numpy()
    best = np.zeros(len(val))
    for lab in (0, 1):
        rows = yv == lab
        sims = x_val[rows] @ x_train[yt != lab].T
        best[rows] = sims.max(axis=1).toarray().ravel()
    out = {}
    for name in sorted(set(slices)):
        m = slices == name
        out[name] = {"n": int(m.sum()), "median_max_cosine": float(np.median(best[m])),
                     "share_cosine_ge_0.8": rate(best[m] >= 0.8)}
    return out


# ── long documents ───────────────────────────────────────────────────────────

def long_documents(pipe_path, val: pd.DataFrame, threshold: float, tokenizer) -> dict:
    from unittest.mock import patch

    from app.classifiers.sklearn_classifier import SklearnClassifier
    from app.config import Settings
    from app.guardrails.input_guard import _extract_segments, run_input_guardrail
    from app.models.schemas import ChatCompletionRequest, ChatMessage

    docs = val[val.text.str.len() > 2000]
    clf = SklearnClassifier(str(pipe_path))
    clf.load()
    settings = Settings(confidence_threshold=threshold)
    whole, guard, n_windows, visible_frac, errors, size_blocks = [], [], [], [], 0, 0

    async def run():
        nonlocal errors, size_blocks
        with patch("app.guardrails.input_guard.get_settings", return_value=settings):
            for text in docs.text:
                req = ChatCompletionRequest(model="diag", messages=[ChatMessage(role="user", content=text)])
                segs = _extract_segments(req)
                n_windows.append(len(segs))
                for seg in segs:
                    offsets = tokenizer(seg, truncation=True, max_length=512,
                                        return_offsets_mapping=True)["offset_mapping"]
                    visible_frac.append(max(b for _, b in offsets) / len(seg))
                result = await run_input_guardrail(req, clf)
                errors += "Classifier error" in result.reason
                size_blocks += "too large to scan" in result.reason
                guard.append(result.confidence)
                whole.append((await clf.predict(text))["malicious_prob"])

    asyncio.run(run())
    whole, guard = np.array(whole), np.array(guard)
    y = docs.label.to_numpy()
    out = {"n": int(len(docs)), "n_malicious": int(y.sum()), "n_benign": int((y == 0).sum()),
           "windows_quantiles": pd.Series(n_windows).quantile([.5, .9, 1.0]).to_dict(),
           "window_char_visibility_512_tokens": pd.Series(visible_frac).quantile([.1, .5, .9]).to_dict(),
           "windows_with_truncated_tail": rate(np.array(visible_frac) < 0.999),
           "classifier_errors": errors, "size_limit_blocks": size_blocks,
           "whole_string": {"recall": rate(whole[y == 1] >= threshold), "fpr": rate(whole[y == 0] >= threshold)},
           "input_guard": {"recall": rate(guard[y == 1] >= threshold), "fpr": rate(guard[y == 0] >= threshold)}}
    # Positive windows in the training data must carry the payload in the first 512 tokens.
    pos = docs[(docs.label == 1) & (docs.payload != "")]
    out["labelled_payload_visible_in_512_tokens"] = rate(
        [_payload_visible(t, pl, tokenizer) for t, pl in zip(pos.text, pos.payload)])
    return out


def _payload_visible(text: str, payload: str, tokenizer) -> bool:
    if payload not in text:
        return False
    end = text.index(payload) + len(payload)
    offsets = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)["offset_mapping"]
    return max(b for _, b in offsets) >= end


def token_overview(df: pd.DataFrame, tokenizer) -> dict:
    lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in df.text]
    s = pd.Series(lengths)
    return {"over_512_tokens": rate(s > 512),
            "over_512_by_label": {str(k): rate(g > 512) for k, g in s.groupby(df.label.to_numpy())},
            "token_quantiles": s.quantile([.5, .9, .99]).to_dict()}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(REPORTS / "diagnosis.json"))
    parser.add_argument("--skip-stages", action="store_true")
    args = parser.parse_args()
    cap_threads()
    require_sha(ORIGINAL_DATASET, ORIGINAL_SHA16)
    require_sha(REJECTED_DATASET, REJECTED_SHA16)
    import joblib
    from transformers import AutoTokenizer

    from training.round2.sk_common import proba

    original, rejected = load(ORIGINAL_DATASET), load(REJECTED_DATASET)
    report: dict = {"inputs": {"original_sha16": ORIGINAL_SHA16, "rejected_sha16": REJECTED_SHA16},
                    "evidence_boundary": "train and val splits only"}
    report["dataset"] = compare_datasets(original, rejected)
    if not args.skip_stages:
        report["stages"] = stage_summary()

    pipe, p_val, threshold = train_candidate(rejected)
    report["candidate_reproduction"] = {
        "params": CANDIDATE_PARAMS, "val_threshold": threshold,
        "recorded_threshold": RECORDED_CANDIDATE_THRESHOLD,
        "reproduced": abs(threshold - RECORDED_CANDIDATE_THRESHOLD) < 1e-9}
    val_r = rejected[rejected.split == "val"].reset_index(drop=True)
    coefs = coefficients(pipe)
    report["candidate_coefficients"] = {"top_malicious": coefs["top_malicious"],
                                        "top_benign": coefs["top_benign"]}
    report["false_alarm_attribution"] = false_alarm_attribution(pipe, coefs, val_r, p_val, threshold)
    report["counterfactuals"] = counterfactuals(pipe, val_r, threshold)
    report["shortcut_probes"] = shortcut_probes(rejected)
    report["opposite_label_neighbours"] = opposite_label_neighbours(pipe, rejected)

    baseline = joblib.load(ARTIFACTS / "baseline_models" / "sklearn_classifier.joblib")
    base_thr = json.loads((ARTIFACTS / "baseline_models" / "sklearn_classifier.json").read_text())["threshold"]
    val_o = original[original.split == "val"].reset_index(drop=True)
    p_base = proba(baseline, val_o.text.tolist())
    common = val_r[val_r.text.isin(set(val_o.text))].reset_index(drop=True)
    report["validation"] = {
        "baseline_on_original_val": {**global_metrics(val_o.label, p_base, base_thr),
                                     **slice_metrics(val_o, p_base, base_thr)},
        "candidate_on_rejected_val": {**global_metrics(val_r.label, p_val, threshold),
                                      **slice_metrics(val_r, p_val, threshold)},
        "rows_val_in_both": int(len(common)),
        "baseline_on_common_val": global_metrics(common.label, proba(baseline, common.text.tolist()), base_thr),
        "candidate_on_common_val": global_metrics(common.label, proba(pipe, common.text.tolist()), threshold),
        "caveat": "Each model is scored on its own dataset's val split (the splits differ).",
    }
    tokenizer = AutoTokenizer.from_pretrained("models/hf2_classifier", local_files_only=True)
    report["tokens"] = {"rejected_val": token_overview(val_r, tokenizer),
                        "original_val": token_overview(val_o, tokenizer)}
    report["long_documents"] = {
        "candidate_on_rejected_val": long_documents(CANDIDATE_PATH, val_r, threshold, tokenizer),
        "baseline_on_original_val": long_documents(
            ARTIFACTS / "baseline_models" / "sklearn_classifier.joblib", val_o, base_thr, tokenizer)}
    write_json(args.output, report)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
