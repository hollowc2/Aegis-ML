"""Rebuild the original and rejected training pools while recording each stage.

``build("original")`` replays the recipe of ``70c2e1e^`` (16,754 rows) and
``build("rejected")`` the recipe of ``70c2e1e`` (22,550 rows). Each returns
the frame after every hygiene/sampling step so the diagnosis can compare them;
the final frames are checked against the recorded SHA256 prefixes, so stage
statistics come from a verified reproduction rather than a reimplementation.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from training.round2 import common  # noqa: F401  (thread caps, source cache)

ORIGINAL_HARD_NEGATIVES = ("oasst2_corrections", "synthetic_hardneg")


def _original_hard_negatives(n: int = 1500, seed: int = 43) -> list[dict]:
    from training.data.synthetic_gen import _gen_hard_negative

    random.seed(seed)
    return [{"text": _gen_hard_negative(), "label": 0, "threat_category": "none"} for _ in range(n)]


def _length_balance(df: pd.DataFrame, hard_sources) -> tuple[pd.DataFrame, list[dict]]:
    """prepare_dataset.length_balance with a configurable priority list and a log."""
    bucket = np.clip(np.floor(np.log2(df["text"].str.len())), 4, 13).astype(int)
    parts, log = [], []
    for b, g in df.groupby(bucket):
        n = int(g["label"].value_counts().min()) if g["label"].nunique() == 2 else 0
        entry = {
            "bucket": int(b),
            "benign_in": int((g.label == 0).sum()),
            "malicious_in": int((g.label == 1).sum()),
            "kept_each": n,
        }
        for label, gl in g.groupby("label"):
            shuffled = gl.sample(frac=1, random_state=42)
            priority = shuffled["source"].isin(hard_sources)
            kept = pd.concat([shuffled[priority], shuffled[~priority]]).head(n)
            parts.append(kept)
            if label == 0:
                entry["benign_priority_in"] = int(priority.sum())
                entry["benign_priority_kept"] = int(kept["source"].isin(hard_sources).sum())
                entry["benign_ordinary_kept"] = int((~kept["source"].isin(hard_sources)).sum())
        log.append(entry)
    return pd.concat(parts), log


def _clean_original(df: pd.DataFrame) -> pd.DataFrame:
    """prepare_dataset._clean as of 70c2e1e^ (no serving preprocessing of stored text)."""
    from training.data.prepare_dataset import MAX_CHARS, MIN_CHARS, normalise_key

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, MAX_CHARS)
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["key"] = df["text"].map(normalise_key)
    df = df[df["key"].str.len() > 0]
    df = df[~(df.groupby("key")["label"].transform("nunique") > 1)]
    return df.drop_duplicates("key")


def load_raw_sources() -> list[pd.DataFrame]:
    from training.data.prepare_dataset import SOURCES, load_source

    return [load_source(spec) for spec in SOURCES if not spec.external]


def build(recipe: str, raw_sources: list[pd.DataFrame] | None = None) -> dict:
    """Return {"stages": {name: frame}, "length_log": [...], "final": frame}."""
    from training.data import prepare_dataset as pdx
    from training.data.seeds import SEED_BENIGN, SEED_MALICIOUS
    from training.data.synthetic_gen import generate_hard_negatives, generate_synthetic_examples

    if recipe not in {"original", "rejected"}:
        raise ValueError(recipe)
    frames = [f.copy() for f in (raw_sources if raw_sources is not None else load_raw_sources())]
    seed_rows = [{"text": t, "label": 0, "threat_category": "none"} for t in SEED_BENIGN] + [
        {"text": t, "label": 1, "threat_category": pdx.assign_threat_category(t, 1)}
        for t in SEED_MALICIOUS
    ]
    frames.append(pd.DataFrame(seed_rows).assign(source="seed"))
    frames.append(pd.DataFrame(generate_synthetic_examples(n=3000)).assign(source="synthetic"))
    if recipe == "original":
        frames.append(pd.DataFrame(_original_hard_negatives()).assign(source="synthetic_hardneg"))
    else:
        frames.append(pd.DataFrame(generate_hard_negatives()).assign(source="synthetic_hardneg"))

    external = pd.read_csv(pdx.EXTERNAL_CSV)
    external["key"] = external.text.map(pdx.normalise_key)
    stages: dict[str, pd.DataFrame] = {}

    if recipe == "original":
        raw = pd.concat(frames, ignore_index=True)
        stages["raw"] = raw
        pool = _clean_original(raw)
        stages["clean_dedup"] = pool
        forbidden = set(external["key"])
        for path in (pdx.HOLDOUT_CSV, pdx.ADVERSARIAL_CSV):
            forbidden.update(pd.read_csv(path)["text"].astype(str).map(pdx.normalise_key))
        pool = pool[~pool["key"].isin(forbidden)]
        stages["eval_excluded"] = pool
        caps = {s.source: s.cap for s in pdx.SOURCES if s.cap and not s.external}
        pool = pdx._cap_sources(pool, caps)
        stages["capped"] = pool
        pool, length_log = _length_balance(pool, ORIGINAL_HARD_NEGATIVES)
        stages["length_balanced"] = pool
        pool["split"] = pool["key"].map(pdx._split_for)
        pool["split_group"] = pool["key"].map(
            lambda k: "legacy:" + " ".join(k.split()[: pdx.SPLIT_GROUP_WORDS])
        )
        final = pool.drop(columns="key").sample(frac=1, random_state=42).reset_index(drop=True)
        csv_columns = ["text", "label", "threat_category", "source", "split"]
    else:
        from transformers import AutoTokenizer

        from app.guardrails.input_guard import _extract_segments
        from app.models.schemas import ChatCompletionRequest, ChatMessage
        from training.data.targeted_gen import generate_targeted_examples, window_rows
        from training.data.targeted_sources import training_rows

        tokenizer = AutoTokenizer.from_pretrained("models/hf2_classifier", local_files_only=True)
        frames.append(pd.DataFrame(generate_targeted_examples(tokenizer)))
        bipia = []
        for row in training_rows():
            bipia.extend(
                window_rows(
                    row["text"],
                    row.get("payload", ""),
                    group="bipia:train",
                    source=row["source"],
                    category=row["threat_category"],
                    tokenizer=tokenizer,
                )
            )
        frames.append(pd.DataFrame(bipia))
        raw = pd.concat(frames, ignore_index=True)
        raw["split_group"] = pdx.split_groups(raw)
        stages["raw"] = raw
        pool = pdx._clean(raw)
        stages["clean_dedup"] = pool
        forbidden = set(external["key"])
        for name in ("holdout", "targeted", "adversarial"):
            path = Path(f"data/{name}_eval.csv")
            forbidden.update(pd.read_csv(path)["text"].astype(str).map(pdx.normalise_key))
        manifest = json.loads(pdx.TARGETED_CSV.with_suffix(".manifest.json").read_text())
        reserved = {pdx.normalise_key(p) for p in manifest["reserved_payloads"]}
        for text in pd.read_csv(pdx.TARGETED_CSV).text:
            req = ChatCompletionRequest(
                model="overlap-check", messages=[ChatMessage(role="user", content=text)]
            )
            forbidden.update(pdx.normalise_key(w) for w in _extract_segments(req))
        leaked = pool["key"].isin(forbidden) | pool["key"].map(
            lambda key: any(p in key for p in reserved if p)
        )
        pool = pool[~leaked]
        stages["eval_excluded"] = pool
        caps = {s.source: s.cap for s in pdx.SOURCES if s.cap and not s.external}
        caps.update(
            targeted_indirect=1800,
            targeted_exfil=1800,
            targeted_carrier_benign=2200,
            synthetic_hardneg=2200,
            bipia_email_train=900,
            bipia_email_benign=300,
        )
        pool = pdx._cap_sources(pool, caps)
        stages["capped"] = pool
        pool, length_log = _length_balance(pool, pdx.HARD_NEGATIVE_SOURCES)
        stages["length_balanced"] = pool
        pool["split"] = pool["split_group"].map(pdx._split_group_for)
        final = pool.drop(columns="key").sample(frac=1, random_state=42).reset_index(drop=True)
        csv_columns = [
            "text",
            "label",
            "threat_category",
            "source",
            "template_id",
            "payload",
            "carrier_id",
            "split_group",
            "split",
        ]

    buf = io.StringIO()
    final[csv_columns].to_csv(buf, index=False)
    import hashlib

    digest = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    return {"stages": stages, "length_log": length_log, "final": final, "sha256": digest}
