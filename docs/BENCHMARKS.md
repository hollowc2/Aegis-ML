# Benchmarks

This document records how the classifiers were evaluated and what they
scored. Every number below comes from `reports/<name>.json`, which the shared
harness (`training/eval_suite.py`) writes alongside the model. Where a number
is optimistic or a set is small, the caveat is stated next to it.

## Summary

Operating point: each model's threshold was picked on the **val** split to
keep val FPR ≤ 2%, then applied unchanged to every other set. Cells are
**recall / FPR** (%).

| Model | Threshold | External (n=7,435) | toxic-chat only (n=4,902) | Holdout (n=190) | In-dist. test (n=1,605) |
|---|---:|---:|---:|---:|---:|
| v1 sklearn (previous release) | 0.70 | 70.5 / 6.3 | — | 11.8 / 3.4 | 73.6 / 8.1 † |
| sklearn (TF-IDF word+char, LR) | 0.554 | 91.9 / 1.8 | 79.3 / 2.3 | 36.3 / 11.4 | 93.9 / 1.2 |
| mDeBERTa-v3 multi-task (PyTorch) | 0.802 | 93.4 / 1.3 | 86.2 / 1.6 | 62.7 / 9.1 | 95.1 / 0.7 |
| mDeBERTa ONNX INT8 (CPU) | 0.269 | 88.5 / 1.0 | 75.9 / 1.3 | 47.1 / 5.7 | 90.3 / 0.9 |
| cascade2: sklearn → ONNX INT8 | 0.269 | 89.8 / 0.9 | 79.3 / 1.1 | 44.1 / 5.7 | 91.3 / 0.4 |
| cascade2: sklearn → mDeBERTa FP32 ‡ | 0.802 | 93.5 / 1.2 | 85.1 / 1.5 | 53.9 / 6.8 | 95.4 / 0.4 |

ROC-AUC on the external set: v1 0.928 · sklearn 0.985 · mDeBERTa 0.987 · ONNX INT8 0.978 · cascade2 INT8 0.983 · cascade2 FP32 0.985.

‡ Measured with the PyTorch model as the slow path (`eval_suite --model cascade2_hf2`). The FP32 ONNX graph is checked against PyTorch at export (identical test AUC, 0.9956 on 500 rows), so FP32 ONNX serving behaves like the mDeBERTa rows.

**INT8 trade-off.** Dynamic INT8 cuts the graph from 1.1 GB to 323 MB and runs on any CPU, but costs about 5 points of external recall at the same FPR budget. Use FP32 ONNX (`ONNX2_USE_INT8=false`) when accuracy matters more than memory. The cascade sends only uncertain prompts to the transformer and keeps the transformer's accuracy.

INT8 scores also vary a little between machines. The same INT8 graph on the
Hugging Face Space (different CPU) matched local scores to a median of 0.003
on 240 prompts, but 5% of prompts moved by more than 0.1, and 3% flipped
verdict near the 0.27 threshold (in both directions). The sklearn and FP32
paths matched exactly. Re-check the INT8 threshold on the hardware you deploy
to if you serve INT8.

† The v1 test row is on the new grouped test split, which the v1 model may
have partially trained on under the old random split; it is not a clean
number. The v1 model's previously published 0.9982 AUC / 0.31% FPR came from a
random row split of a dataset with heavy near-duplication and should be
treated as superseded.

### Single-request latency

One `predict()` at a time through the serving classes, after warm-up, on 200
prompts sampled from the external set (median 97 characters;
`training/latency_bench.py`). Hardware: AMD Ryzen 7 1700, process pinned to
4 CPU threads; `hf2` on an AMD RX 7700 XT (ROCm). Longer prompts cost more for
the transformer, roughly linearly up to the 512-token limit.

| Classifier | p50 | p95 | mean |
|---|---:|---:|---:|
| sklearn | 1.4 ms | 2.6 ms | 1.6 ms |
| hf2 (PyTorch, GPU) | 18.2 ms | 19.2 ms | 18.6 ms |
| onnx2 INT8 (CPU) | 39 ms | 266 ms | 76 ms |
| cascade2 INT8 (CPU; 24% escalated) | 2.0 ms | 67 ms | 15 ms |

The classifier is called once per message window, so a request's
classification cost scales with its number of messages.

## Evaluation sets

| Set | Source | Size | Role |
|---|---|---:|---|
| `test` | grouped split of `data/combined_dataset.csv` | 1,605 | in-distribution sanity check |
| `external` | test splits of deepset/prompt-injections, xTRam1/safe-guard-prompt-injection, jackhhao/jailbreak-classification, Lakera/gandalf_ignore_instructions, lmsys/toxic-chat | 7,435 | **primary benchmark** — never trained on, never tuned on |
| `holdout` | `training/data/holdout_eval.py`, hand-written | 190 | hard out-of-distribution attacks and hard negatives |
| `adversarial` | `data/adversarial_eval.csv` | 320 | regression suite; shares vocabulary with the synthetic generator, so optimistic |

### Caveats

- **toxic-chat labels.** toxic-chat's `jailbreaking` label is the attack label;
  everything else counts as benign. A manual audit of the flagged "benign"
  rows found many that are genuine prompt-extraction or role-play jailbreak
  attempts the dataset doesn't label as jailbreaking, so the reported toxic-chat FPR
  overstates real false positives. Its 87 positives also make recall on it
  noisy (±4 points).
- **Holdout has been looked at.** The holdout set was used once, after the
  first retraining round, to diagnose false positives — which led to the
  hard-negative data in round 2. It is no longer a pristine test set; the
  external set is. It is also small: one prompt is 0.5–1.3 points.
- **Adversarial set is optimistic.** It was generated from the same templates
  as `synthetic_gen.py`, so high scores there mostly show the model learned
  those templates.
- **deepset** has only 59 test attacks, many of them in German, and short
  imperative prompts that look like ordinary instructions; recall there is
  the weakest of the external sources for every model.
- **Licences.** Source datasets carry their own licences — toxic-chat, for
  example, is CC-BY-NC-4.0 and is used here both for training and evaluation.
  Check each source's licence before reusing the dataset or trained models
  commercially.

## Where the models still fail

Holdout recall by attack type (sklearn / mDeBERTa, %):

| Attack type | n | sklearn | mDeBERTa |
|---|---:|---:|---:|
| direct paraphrase | 15 | 73 | 100 |
| multilingual (12 languages) | 16 | 38 | 94 |
| obfuscation (rot13, hex, reversed…) | 12 | 33 | 92 |
| prompt extraction | 12 | 75 | 58 |
| role-play jailbreak | 12 | 25 | 50 |
| exfiltration (send data to URL…) | 12 | 33 | 33 |
| indirect injection (in documents/tool output) | 15 | 0 | 33 |
| attack buried in long benign text | 8 | 0 | 13 |

Holdout FPR on benign prompts that *talk about* overrides, security, code or
prompt injection itself is 10–17% for mDeBERTa (sklearn is worse on
meta-discussion, 38%). These are the clear next targets: indirect injection,
exfiltration, and long-context attacks need training data of that shape, not
threshold changes.

The serving path mitigates long inputs: every message is scored separately
and long messages are split into overlapping ~2,000-character windows, with
the request judged on its highest-scoring window (`MAX_SCAN_SEGMENTS` caps the
work; beyond it the request is blocked). The holdout numbers above score each
prompt as a single string, so they understate what the proxy catches for
long multi-message requests.

## Dataset

`training/data/prepare_dataset.py` builds `data/combined_dataset.csv`
(16,754 rows; sha256 prefix `0c6e12ed82b86e36`) from 18 public/synthetic sources
plus a few hand-written seeds:

- **Attacks:** deepset, SPML, xTRam1, jackhhao, toxic-chat (`jailbreaking`),
  TrustAIRLab in-the-wild jailbreaks, rubend18, Gandalf, a multilingual
  prompt-injection set, and synthetic attacks (`synthetic_gen.py`).
- **Benign:** in-the-wild regular prompts (filtered for jailbreak language),
  awesome-chatgpt-prompts, Dolly, Alpaca, OpenAssistant root prompts and
  follow-ups, and hard negatives — OpenAssistant follow-ups that use override
  language ("ignore what I said and…") plus templated hard negatives
  (`generate_hard_negatives`).

Hygiene steps, in order:

1. NFKC/preprocessing identical to serving (`TextPreprocessor`).
2. Exact dedup on a normalised key; texts that appear with both labels are dropped.
3. Any text that also appears in an evaluation set is removed.
4. Per-source caps so no single source dominates.
5. **Length balancing** per log2-length bucket. The previous dataset
   (12,930 rows, 938 exact duplicate texts) could be classified at 0.81 AUC
   from text length alone; after balancing, a length-only model scores 0.50.
6. Split 80/10/10 by a hash of the first eight normalised words, so
   near-duplicate templates land in the same split.

Leave-one-source-out (sklearn trained without a source, scored on it,
recall at 0.5) shows how much the model depends on having seen a source's
style: Gandalf 99%, xTRam1 75%, synthetic 73%, in-the-wild jailbreaks 53%,
SPML 37%, multilingual 27%, deepset 23%. Expect weaker results on attack
styles unlike any training source.

## Procedure

- Train on `split == "train"`. Select hyperparameters (sklearn: C and char
  feature count by val AUC) and checkpoints (mDeBERTa: best val ROC-AUC over
  4 epochs) on `val`.
- mDeBERTa probabilities are temperature-scaled (T = 0.79, fitted on val).
- The threshold is the lowest value keeping val FPR ≤ 2%. The ONNX INT8 graph
  gets its own threshold by the same rule, because quantisation shifts its
  scores (0.27 vs 0.80 for FP32).
- `CONFIDENCE_THRESHOLD` overrides all of this; leave it unset to use the
  model's tuned threshold.

### Reproduce

```bash
uv run python -m training.data.holdout_eval
uv run python -m training.data.prepare_dataset
uv run python -m training.phase1_sklearn.train --loso          # reports/sklearn.json
uv sync --extra hf2
uv run python -m training.phase3_hf2.train --model mdeberta    # reports/hf2_mdeberta.json
uv run python -m training.phase3_hf2.export_onnx --validate
ONNX2_USE_INT8=true uv run python -m training.eval_suite --model cascade2 --name cascade2_int8
uv run python -m training.latency_bench --model sklearn onnx2 cascade2
```

mDeBERTa (microsoft/mdeberta-v3-base, 278M parameters) was trained for 4 epochs
on an AMD RX 7700 XT (ROCm 7.2, PyTorch 2.11) in about an hour; batch 16,
gradient accumulation 2, lr 2e-5. Dataset downloads are from the Hugging Face
Hub, so upstream changes can alter the rebuilt dataset — compare the sha256
prefix above.
