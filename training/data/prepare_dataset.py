"""
training/data/prepare_dataset.py
=================================
Builds the training dataset and the external benchmark from public HuggingFace
datasets, hand-written seeds, and the local synthetic generator.

Outputs:
  data/combined_dataset.csv  — training pool with a fixed split column
      Columns: text, label, threat_category, source, split
        - label: 0=benign, 1=malicious
        - threat_category: "none", "prompt_injection", "jailbreak", "data_exfiltration"
        - split: "train" | "val" | "test" — shared by all three training phases
  data/external_eval.csv     — third-party test splits never used for training
      Columns: text, label, source

Data hygiene, in order:
  1. Normalise (NFKC, strip invisible chars) exactly as the serving path does.
  2. Drop exact and normalised duplicates; drop texts that appear with both labels.
  3. Remove anything that also appears in the external, holdout, or adversarial
     evaluation sets, so no evaluation text leaks into training.
  4. Cap each source so no single dataset dominates.
  5. Length-balance: within each log2(length) bucket keep equal benign/malicious
     counts, so text length alone cannot predict the label.
  6. Split by near-duplicate group (first words of the normalised text) rather
     than by row, so template families never straddle train and test.

Run:
    python -m training.data.prepare_dataset
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from app.classifiers.text_preprocessor import TextPreprocessor
from training.data.seeds import SEED_BENIGN, SEED_MALICIOUS
from training.data.synthetic_gen import generate_hard_negatives, generate_synthetic_examples

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "combined_dataset.csv"
EXTERNAL_CSV = DATA_DIR / "external_eval.csv"
HOLDOUT_CSV = DATA_DIR / "holdout_eval.csv"
ADVERSARIAL_CSV = DATA_DIR / "adversarial_eval.csv"

SEED = 42
MIN_CHARS = 8
MAX_CHARS = 12_000          # longer texts are truncated; transformers only see ~512 tokens anyway
SPLIT_GROUP_WORDS = 8       # near-duplicate grouping key: first N normalised words
VAL_PCT, TEST_PCT = 10, 10

# Prompts in "regular" (non-jailbreak) in-the-wild collections that are really
# overrides or jailbreaks. They are dropped from the benign side rather than
# relabelled, because their intent is ambiguous.
_SUSPICIOUS_BENIGN = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier)|disregard (?:all |any )?(?:previous|prior|your)"
    r"|jailbr|\bDAN\b|developer mode|do anything now|no (?:ethical|moral)|unfiltered|uncensored|amoral"
    r"|without (?:any )?(?:restrictions|limits|limitations|censorship|filters)|no restrictions"
    r"|exception to (?:the )?(?:usual )?ai|nsfw|lewd|explicit content|content polic",
    re.IGNORECASE,
)
# Narrower filter for conversational follow-ups, where "forget that" or
# "ignore my last message" is normal: only drop outright jailbreak vocabulary.
_JAILBREAK_TERMS = re.compile(
    r"jailbr|\bDAN\b|developer mode|do anything now|unfiltered|uncensored|amoral"
    r"|no (?:ethical|moral)|without (?:any )?(?:restrictions|censorship|filters)|nsfw|lewd",
    re.IGNORECASE,
)
_OVERRIDE_LANGUAGE = re.compile(
    r"\b(?:ignore|forget|disregard|instead|pretend|act as|you are now|from now on"
    r"|start over|never ?mind|scratch that|role ?play)\b",
    re.IGNORECASE,
)
# Benign sources that teach the model attack-like words are not attacks on their
# own. Length balancing keeps these before sampling other benign rows.
HARD_NEGATIVE_SOURCES = ("oasst2_corrections", "synthetic_hardneg")


# ── Source specification ──────────────────────────────────────────────────────

@dataclass
class Source:
    source: str                                   # short id written to the CSV
    name: str                                     # HF dataset id
    split: str
    text: str | Callable[[pd.DataFrame], pd.Series]
    label: int | str                              # fixed label, or label column name
    label_map: dict = field(default_factory=lambda: {0: 0, 1: 1})
    config: str | None = None
    cap: int | None = None                        # max rows per label after dedup
    category: str | None = None                   # default threat_category for malicious rows
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None
    suspicious_filter: re.Pattern | None = None   # benign rows matching this are dropped
    external: bool = False                        # True → external benchmark, never trained on


def _dolly_text(df: pd.DataFrame) -> pd.Series:
    ctx = df["context"].fillna("").str.strip()
    return np.where(ctx != "", df["instruction"] + "\n\n" + ctx, df["instruction"])


def _alpaca_text(df: pd.DataFrame) -> pd.Series:
    inp = df["input"].fillna("").str.strip()
    return np.where(inp != "", df["instruction"] + "\n\n" + inp, df["instruction"])


SOURCES: list[Source] = [
    # ── Mixed-label prompt-injection datasets ────────────────────────────────
    Source("deepset", "deepset/prompt-injections", "train", "text", "label"),
    Source("spml", "reshabhs/SPML_Chatbot_Prompt_Injection", "train",
           "User Prompt", "Prompt injection", cap=2000),
    Source("xtram1", "xTRam1/safe-guard-prompt-injection", "train", "text", "label"),
    Source("jackhhao", "jackhhao/jailbreak-classification", "train", "prompt", "type",
           label_map={"benign": 0, "jailbreak": 1}, category="jailbreak"),
    Source("toxic_chat", "lmsys/toxic-chat", "train", "user_input", "jailbreaking",
           config="toxicchat0124", category="jailbreak"),
    # ── Malicious-only sources ────────────────────────────────────────────────
    Source("itw_jailbreak", "TrustAIRLab/in-the-wild-jailbreak-prompts", "train", "prompt", 1,
           config="jailbreak_2023_12_25", category="jailbreak"),
    Source("rubend18", "rubend18/ChatGPT-Jailbreak-Prompts", "train", "Prompt", 1,
           category="jailbreak"),
    Source("gandalf", "Lakera/gandalf_ignore_instructions", "train", "text", 1, cap=800),
    Source("multilingual_pi", "yanismiraoui/prompt_injections", "train", "prompt_injections", 1),
    # ── Benign-only sources (incl. long and multilingual prompts) ────────────
    Source("itw_regular", "TrustAIRLab/in-the-wild-jailbreak-prompts", "train", "prompt", 0,
           config="regular_2023_12_25", cap=3000, suspicious_filter=_SUSPICIOUS_BENIGN),
    Source("awesome_prompts", "fka/awesome-chatgpt-prompts", "train", "prompt", 0, cap=1500),
    Source("dolly", "databricks/databricks-dolly-15k", "train", _dolly_text, 0, cap=3000),
    Source("alpaca", "tatsu-lab/alpaca", "train", _alpaca_text, 0, cap=2000),
    Source("oasst2", "OpenAssistant/oasst2", "train", "text", 0, cap=4000,
           row_filter=lambda df: (df["role"] == "prompter") & df["parent_id"].isna(),
           suspicious_filter=_SUSPICIOUS_BENIGN),
    # Follow-up turns: corrections and persona requests are normal in real chats.
    Source("oasst2_corrections", "OpenAssistant/oasst2", "train", "text", 0,
           row_filter=lambda df: (df["role"] == "prompter") & df["parent_id"].notna()
           & df["text"].str.contains(_OVERRIDE_LANGUAGE),
           suspicious_filter=_JAILBREAK_TERMS),
    Source("oasst2_followups", "OpenAssistant/oasst2", "train", "text", 0, cap=2000,
           row_filter=lambda df: (df["role"] == "prompter") & df["parent_id"].notna(),
           suspicious_filter=_SUSPICIOUS_BENIGN),
    # ── External benchmark: official test splits, never trained on ───────────
    Source("deepset", "deepset/prompt-injections", "test", "text", "label", external=True),
    Source("xtram1", "xTRam1/safe-guard-prompt-injection", "test", "text", "label", external=True),
    Source("jackhhao", "jackhhao/jailbreak-classification", "test", "prompt", "type",
           label_map={"benign": 0, "jailbreak": 1}, external=True),
    Source("gandalf", "Lakera/gandalf_ignore_instructions", "test", "text", 1, external=True),
    Source("toxic_chat", "lmsys/toxic-chat", "test", "user_input", "jailbreaking",
           config="toxicchat0124", external=True),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

_preprocessor = TextPreprocessor()


def normalise_key(text: str) -> str:
    """Dedup key: serving-path normalisation, lowercased, punctuation collapsed."""
    cleaned, _ = _preprocessor.preprocess(text)
    return re.sub(r"[\W_]+", " ", cleaned.lower()).strip()


def assign_threat_category(text: str, label: int, default: str | None = None) -> str:
    """Heuristic threat category; source-level defaults win over keyword guesses."""
    if label == 0:
        return "none"
    lower = text.lower()
    exfil_patterns = [
        "repeat your system prompt", "reveal your", "print your system",
        "show me your instructions", "what were your instructions",
        "what is in your context", "expose your", "tell me your secret",
        "system prompt", "initial instructions", "your instructions verbatim",
    ]
    jailbreak_patterns = [
        "dan ", " dan,", "jailbreak", "you are now", "pretend you are",
        "act as", "roleplay as", "developer mode", "do anything now",
        "no restrictions", "without restrictions", "no ethical", "unrestricted",
    ]
    if any(p in lower for p in exfil_patterns):
        return "data_exfiltration"
    if default:
        return default
    if any(p in lower for p in jailbreak_patterns):
        return "jailbreak"
    return "prompt_injection"


def _split_for(key: str) -> str:
    group = " ".join(key.split()[:SPLIT_GROUP_WORDS])
    bucket = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    if bucket < TEST_PCT:
        return "test"
    if bucket < TEST_PCT + VAL_PCT:
        return "val"
    return "train"


def load_source(spec: Source) -> pd.DataFrame:
    """Load one HF dataset split and return columns text, label, threat_category, source."""
    empty = pd.DataFrame(columns=["text", "label", "threat_category", "source"])
    logger.info("Loading %s [%s%s]", spec.name, spec.split, f", {spec.config}" if spec.config else "")
    try:
        raw = load_dataset(spec.name, spec.config, split=spec.split).to_pandas()
    except Exception as exc:
        logger.warning("  failed to load %s: %s — skipping", spec.name, exc)
        return empty

    if spec.row_filter is not None:
        raw = raw[spec.row_filter(raw)]
    texts = spec.text(raw) if callable(spec.text) else raw[spec.text]
    df = pd.DataFrame({"text": np.asarray(texts, dtype=object)}, index=raw.index)

    if isinstance(spec.label, int):
        df["label"] = spec.label
    else:
        df["label"] = raw[spec.label].map(spec.label_map)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    if spec.suspicious_filter is not None:
        suspicious = (df["label"] == 0) & df["text"].str.contains(spec.suspicious_filter)
        if suspicious.any():
            logger.info("  dropped %d suspicious 'benign' prompts", int(suspicious.sum()))
        df = df[~suspicious]

    df["threat_category"] = [
        assign_threat_category(t, lab, spec.category) for t, lab in zip(df["text"], df["label"])
    ]
    df["source"] = spec.source
    logger.info("  → %d rows (benign=%d malicious=%d)",
                len(df), int((df["label"] == 0).sum()), int((df["label"] == 1).sum()))
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, MAX_CHARS)
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["key"] = df["text"].map(normalise_key)
    df = df[df["key"].str.len() > 0]
    # Drop texts that occur with both labels, then keep one copy of each.
    conflicted = df.groupby("key")["label"].transform("nunique") > 1
    if conflicted.any():
        logger.info("Dropping %d rows whose text appears with both labels", int(conflicted.sum()))
    df = df[~conflicted]
    before = len(df)
    df = df.drop_duplicates("key")
    logger.info("Dropped %d duplicate rows", before - len(df))
    return df


def _cap_sources(df: pd.DataFrame, caps: dict[str, int]) -> pd.DataFrame:
    parts = []
    for (source, _label), g in df.groupby(["source", "label"]):
        cap = caps.get(source)
        parts.append(g.sample(cap, random_state=SEED) if cap and len(g) > cap else g)
    return pd.concat(parts)


def length_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Equalise benign/malicious counts inside each log2(length) bucket.

    The majority class is downsampled, keeping HARD_NEGATIVE_SOURCES rows first.
    """
    bucket = np.clip(np.floor(np.log2(df["text"].str.len())), 4, 13).astype(int)
    parts = []
    for b, g in df.groupby(bucket):
        n = int(g["label"].value_counts().min()) if g["label"].nunique() == 2 else 0
        for _label, gl in g.groupby("label"):
            # Hard negatives first, then a random sample of the rest.
            shuffled = gl.sample(frac=1, random_state=SEED)
            priority = shuffled["source"].isin(HARD_NEGATIVE_SOURCES)
            parts.append(pd.concat([shuffled[priority], shuffled[~priority]]).head(n))
        logger.info("  len bucket 2^%d: benign=%d malicious=%d → keep %d each",
                    b, int((g["label"] == 0).sum()), int((g["label"] == 1).sum()), n)
    return pd.concat(parts)


def _eval_keys() -> set[str]:
    keys: set[str] = set()
    for path in (HOLDOUT_CSV, ADVERSARIAL_CSV):
        if path.exists():
            keys.update(pd.read_csv(path)["text"].astype(str).map(normalise_key))
        else:
            logger.warning("%s not found — cannot guarantee it is excluded from training", path)
    return keys


# ── Build ─────────────────────────────────────────────────────────────────────

def build_dataset(n_synthetic: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training pool with split column, external benchmark)."""
    train_frames, external_frames = [], []
    for spec in SOURCES:
        (external_frames if spec.external else train_frames).append(load_source(spec))

    seed_rows = (
        [{"text": t, "label": 0, "threat_category": "none"} for t in SEED_BENIGN]
        + [{"text": t, "label": 1, "threat_category": assign_threat_category(t, 1)}
           for t in SEED_MALICIOUS]
    )
    train_frames.append(pd.DataFrame(seed_rows).assign(source="seed"))
    logger.info("Generating %d synthetic examples...", n_synthetic)
    train_frames.append(pd.DataFrame(generate_synthetic_examples(n=n_synthetic)).assign(source="synthetic"))
    train_frames.append(pd.DataFrame(generate_hard_negatives()).assign(source="synthetic_hardneg"))

    external = _clean(pd.concat(external_frames, ignore_index=True))
    pool = _clean(pd.concat(train_frames, ignore_index=True))

    # Nothing that is evaluated on may appear in training.
    forbidden = _eval_keys() | set(external["key"])
    leaked = pool["key"].isin(forbidden)
    logger.info("Removing %d training rows that overlap evaluation sets", int(leaked.sum()))
    pool = pool[~leaked]

    caps = {s.source: s.cap for s in SOURCES if s.cap and not s.external}
    pool = _cap_sources(pool, caps)
    logger.info("Length-balancing %d rows...", len(pool))
    pool = length_balance(pool)

    pool["split"] = pool["key"].map(_split_for)
    pool = pool.drop(columns="key").sample(frac=1, random_state=SEED).reset_index(drop=True)
    external = external[["text", "label", "source"]].reset_index(drop=True)
    return pool, external


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not HOLDOUT_CSV.exists():
        from training.data.holdout_eval import main as holdout_main
        holdout_main()

    df, external = build_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    external.to_csv(EXTERNAL_CSV, index=False)
    logger.info("Saved %d training rows to %s", len(df), OUTPUT_CSV)
    logger.info("Saved %d external benchmark rows to %s", len(external), EXTERNAL_CSV)

    print("\nRows by split and label:")
    print(df.groupby(["split", "label"]).size().unstack().to_string())
    print("\nRows by source and label:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    print("\nThreat categories:")
    print(df["threat_category"].value_counts().to_string())
    print("\nExternal benchmark:")
    print(external.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
