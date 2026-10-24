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


## Targeted experiment: rejected (2026-10-22)

**The candidate did not meet the acceptance criteria.** The sklearn external
FPR rose from 1.85% to **2.91%**, exceeding the predeclared 2% budget.
External recall rose from 91.93% to 92.60% (+0.67 percentage points), satisfying
the recall condition alone. The validation-only selected threshold was
0.4851336163 (C=16, 300,000 char features); the baseline's stored serving
threshold is 0.5544. No thresholds were adjusted after evaluation.

The experiment stopped at this failure as instructed. mDeBERTa was not
retrained; ONNX export/recalibration and after-evaluation of hf2, the FP32
cascade and INT8 were not run. Their after values are **not measured**, not
passes. The original sklearn weights and sidecar were restored byte-for-byte;
the transformer and ONNX weights were never replaced. The release summary
above still describes those baseline models. The rejected sklearn weights
are preserved locally in the session scratchpad's
`weak_categories/rejected_candidate/`, and the original inference artifacts
in `weak_categories/models/`. Neither models nor code were pushed.

Evidence: [acceptance report](../reports/targeted_acceptance.json),
[candidate evaluation](../reports/sklearn_targeted_candidate.json),
[prior sklearn evaluation](../reports/sklearn_before_targeted.json), and
[dataset summary](../reports/dataset_targeted.json). Original external reports
for every model are also preserved as `reports/*_before_targeted.json`.

### Frozen targeted evaluation

`data/targeted_eval.csv` has 2,118 rows, SHA256 prefix
`e049ed098724e003`. It was created before training-data changes and scored
with all four original models. Only aggregate metrics were examined; no
individual misses were inspected or used for tuning. Its manifest stores
reserved payloads, source revisions and file hashes, and the builder refuses
to overwrite it. Registration in `training.eval_suite.EVAL_SETS` automatically
includes it in the dataset's evaluation-overlap exclusion.

Composition: 835 indirect cases (225 BIPIA email-test constructions, 510
InjecAgent direct-harm tool outputs, 100 authored instructions); 644
exfiltration cases (544 InjecAgent data-stealing tool outputs, 100 authored
instructions); 100 authored buried attacks; 100 authored security/legitimate
correction prompts; 100 matching long benign carriers; and 339 NotInject
benign cases. Injected tool-output exfiltration maps to `data_exfiltration`.
These are detector scores, not end-to-end agent attack-success rates.

The following baseline values are percentages. Each model has an independent
saved report `reports/<model>_targeted_baseline.json`. All transformer/cascade/
INT8 **after measurements were stopped**, so these are baseline evidence only.

| Baseline model | Indirect recall | Exfiltration recall | Buried recall | Security benign FPR | NotInject FPR |
|---|---:|---:|---:|---:|---:|
| sklearn | 1.80 | 2.02 | 0.00 | 17.00 | 30.38 |
| hf2 | 11.50 | 8.23 | 0.00 | 22.00 | 28.32 |
| cascade2_hf2 | 10.78 | 7.92 | 0.00 | 20.00 | 27.73 |
| onnx2_int8 | 16.05 | 4.81 | 0.00 | 10.00 | 17.40 |

Sklearn before/after on the **same frozen rows**, percentages:

| Category and metric | n | Before | Rejected candidate |
|---|---:|---:|---:|
| indirect_injection recall | 835 | 1.80 | 14.01 |
| data_exfiltration recall | 644 | 2.02 | 21.12 |
| buried_attack recall | 100 | 0.00 | 0.00 |
| security_benign fpr | 100 | 17.00 | 23.00 |
| notinject_benign fpr | 339 | 30.38 | 43.07 |
| long_benign fpr | 100 | 0.00 | 0.00 |

The new target therefore fails as a whole: indirect and exfiltration improve,
buried attacks do not, and the security-topic FPR worsens. The mDeBERTa
security-FPR target (<5%) has no after measurement; its baseline was 22%.
NotInject is a broader trigger-word benchmark and is reported separately
from security-specific benign prompts.

### Long-text serving path and already-seen holdout

Whole-string buried recall is 0% for all four baselines. Actual requests
through `app.guardrails.input_guard.run_input_guardrail` also scored **0/100
buried attacks** and **0/100 long benign false positives** for all four
baselines, with zero classifier errors or size-limit blocks. The request
settings were `MAX_SCAN_SEGMENTS=64` and invisible-character bias 0.15.
These baseline guard-path results are stored under `serving_long` in each
baseline report. Candidate whole-string buried recall is also 0%; candidate
guard-path scoring was not run after the stop condition. Windowing alone
cannot identify an instruction the window classifier does not recognize.

The old holdout is **already-seen diagnostic evidence**: sklearn recall
36.27% → 50.98%, FPR 11.36% → 11.36%. It is not a clean generalization claim.
All per-category values remain in the candidate report. The unchanged
mDeBERTa holdout values are the baseline values in the release summary.

Toxic-chat only: sklearn recall 79.31% → 81.61%, FPR 2.35% → 3.53%; its 87
positive examples remain a small and noisy recall sample. Overall external
FPR, rather than toxic-chat FPR alone, was the declared acceptance gate.
The primary external CSV remained byte-identical (SHA256 prefix
`580e5a966bf72cd0`, 7,435 rows).

### Candidate training data and hygiene

The released weights were trained on 16,754 rows (SHA256 prefix
`0c6e12ed82b86e36`), which is what `data/combined_dataset.csv` holds. The
rejected candidate's pool, 22,550 rows with SHA256 prefix `a68869d92a3e6dd1`,
was briefly committed in its place and has since been moved out of `data/`.
Its recipe is `training/data/prepare_dataset_targeted.py`, which writes only
to the round-2 artifact root (`~/.cache/aegis-ml-round2`), so the default
`prepare_dataset` and training commands still rebuild the release recipe.

New training sources: BIPIA train email contexts/train attacks, plus original
window-sized indirect/exfiltration examples, benign tool/document carriers,
security discussions, corrections, and user-authorized sends/uploads.
Carriers include email, web, tool output, tables, code, summaries, recipes,
and how-to instructions. Long carriers supply both benign windows and windows
with attacks at the beginning, middle and end. A positive row must contain the
entire payload within both the scan window and the first 512 tokenizer tokens;
partial-span and truncated-away positives are discarded.

Hygiene preserves serving-path NFKC cleaning, conflict removal, normalized
dedup, per-source/per-label caps, and log2-length balancing. New rows use
connected payload/template/carrier groups, merged before dedup so a discarded
duplicate cannot lose split lineage; legacy rows keep their first-eight-word
families. New matched benign windows share the attack family's split.
683 evaluation overlaps were removed, including reserved payload substring
matches and exact normalized serving windows from the targeted evaluation.
The generated CSV was audited for duplicate texts, cross-split components,
invalid labels and exact evaluation overlap; none remained.

Length-only logistic regression fitted on train and scored once on grouped
test produced **0.4647 AUC** (about 0.535 if its ranking is reversed), rather
than a strong length shortcut. No dataset revisions followed that test score.
Split counts (benign/malicious): train 9,198/8,996; val 1,119/1,110;
test 958/1,169. Caps apply per source **and label**, so a source containing
benign neighboring windows can exceed its malicious-row cap in total.

| Candidate source | Rows |
|---|---:|
| xtram1 | 3,805 |
| spml | 2,289 |
| targeted_exfil | 2,229 |
| targeted_indirect | 2,173 |
| synthetic_hardneg | 1,527 |
| synthetic | 1,287 |
| itw_regular | 1,136 |
| multilingual_pi | 974 |
| toxic_chat | 960 |
| targeted_carrier_benign | 938 |
| gandalf | 776 |
| oasst2 | 762 |
| dolly | 747 |
| itw_jailbreak | 652 |
| awesome_prompts | 578 |
| alpaca | 378 |
| oasst2_followups | 361 |
| oasst2_corrections | 308 |
| deepset | 250 |
| bipia_email_train | 227 |
| jackhhao | 136 |
| bipia_email_benign | 39 |
| seed | 16 |
| rubend18 | 2 |

### Source licences and limits

See [source revisions, formats, splits and licences](TARGETED_DATA_LICENSES.md)
and `docs/data_licenses/` for the source notices. BIPIA's MIT text attacks and
MIT email contexts are used; its separately licensed table/code and restricted
news contexts are excluded. InjecAgent has no official train split, so all
1,054 base cases are evaluation-only. NotInject's train/validation/test aliases
are its one/two/three-trigger difficulty groups; all 339 cases are reserved
rather than treating those groups as independent training/test families.
This intentionally substitutes independent synthetic training examples for
unavailable clean InjecAgent/NotInject training splits. No new NC data was
added; existing toxic-chat restrictions remain.

The authored indirect and buried cases reuse the same 100 instructions;
they are paired, not 200 independent attack intents. Long carriers repeat
original archive prose, so these numbers do not measure diverse long-document
generalization. Authoring the evaluation and training scenarios in the same
agent session also limits independence even though their templates and wording
are separate. The frozen set has now been evaluated and reported; future
changes must not tune on its failures.

### Acceptance disposition

- **FAIL — sklearn external:** recall condition passes (+0.67 points), FPR
  fails (2.91% >2%). Candidate rejected and baseline restored.
- **NOT RUN — mDeBERTa external:** stopped before transformer training.
- **FAIL / INCOMPLETE — blind targeted improvement:** sklearn indirect
  +12.22 points and exfiltration +19.10 points, buried +0; other after scores
  not measured.
- **NOT MET — benign security target:** sklearn worsens 17% →23%; mDeBERTa
  baseline 22%, after not measured, so <5% cannot be claimed.
- **REPORTED — old holdout:** already-seen; numbers above and in JSON.
- **PASS — verification:** `uv run pytest tests/`: 75 passed, 18 skipped;
  `ruff check app demo tests` and `ruff format --check app demo tests` pass.
  Skipped tests are not represented as passes.

All substantial CPU work was pinned to cores 0–3 with four-thread limits.
GPU baseline inference used the RX 7700 XT; CPU INT8 scoring was serial.
Training thread caps and sequential ONNX-validation sessions are now explicit.

### Reproduce the experiment protocol

Use a local venv (`UV_PROJECT_ENVIRONMENT=~/.cache/aegis-ml-venv`) with dev/hf2
extras, `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`,
`TOKENIZERS_PARALLELISM=false`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, and prefix
compute commands with `taskset -c 0-3`. Set `TARGETED_SOURCE_CACHE` to the
session scratchpad's `weak_categories` directory to reuse the pinned files.
Preserve baseline model artifacts before any retraining. In a fresh copy
without an already frozen targeted set:

```bash
uv run python -m training.data.targeted_eval
# Sequentially, for sklearn, hf2, cascade2_hf2, onnx2 (ONNX2_USE_INT8=true):
uv run python -m training.eval_suite --model sklearn --sets targeted --serving-long --name sklearn_targeted_baseline
# Corresponding names: hf2_targeted_baseline, cascade2_hf2_targeted_baseline,
# and onnx2_int8_targeted_baseline. Wait for each process to finish.
uv run python -m training.data.prepare_dataset_targeted
# Writes ~/.cache/aegis-ml-round2/rejected_combined_dataset.csv; the trainers
# read data/combined_dataset.csv, so train on a copy of the repo, never in place.
uv run python -m training.phase1_sklearn.train --loso
# Stop if the declared external gate fails; do not retune against its results.
uv run python -m training.targeted_acceptance
```

The retained candidate aggregate report lets the acceptance summary be
regenerated without loading a rejected model. Transformer retraining/export
commands in the earlier reproduction section are not evidence of execution
for this failed round.


## Round 2: narrowed diagnosis and fresh false-positive set (2026-10-23)

**Outcome: no candidate model was trained; released models are unchanged**
(SHA256 re-verified in [`reports/round2/summary.json`](../reports/round2/summary.json)).
The planned round needed new attack-side data: fresh indirect, exfiltration
and buried evaluation cases, attack additions for training, and benign
security-topic training negatives. Authoring or assembling that data was
stopped by a safety classifier. With the maintainer's approval, the round
was narrowed to three parts: a train/val diagnosis of the rejected
experiment, a benign-only fresh final set, and the base-pool arm. **Not run:**
arms B1–B3, the context-aware variant, mDeBERTa retraining, ONNX export, and
every after-measurement. These are not passes.

### Diagnosis of the rejected experiment (train/val only)

[`reports/round2/diagnosis.json`](../reports/round2/diagnosis.json),
built by `training/round2/diagnose.py`. The rejected sklearn candidate was recovered by
retraining from its recorded recipe, and its val threshold was reproduced
exactly (0.48513361631912105). A stage-by-stage rebuild of both datasets
matches the original row count but only ~75% of rows, because of upstream
sampling-order drift. Its stage counts are therefore indicative only. Every
comparison below uses the committed CSVs.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Hard-negative priority displaced ordinary benign rows | Not supported as a count; composition shifted | Ordinary benign rows rose 7,328 → 8,463, but only 3,226 original rows were retained; hard negatives rose 1,049 → 2,812 |
| Repetitive carriers / explicit takeover phrases created shortcuts | Supported for attack vocabulary; carrier shortcut not supported | 67% of new attack rows contain takeover words (0% of their benign controls); 57% vs 4% contain destinations; a 3-feature surface model reaches 0.97 val AUC on those rows; removing the payload drops flags 100% → 0% |
| Labels need authorization/provenance context | Unresolved | No context-paired development data could be built |
| 512-token truncation hides evidence | Partly supported (transformers only) | 44% of long-document val scan windows exceed 512 tokens |
| Overall val AUC hid weak slices | Supported | Candidate own-val FPR 1.97%, but 3.3% vs 1.3% for the baseline on the 1,265 rows that are val in both datasets; legacy templated hard-negative FPR 0% → 9.5%; long-document FPR through the input guard 4.7% → 13.6% |

The candidate's own validation split was padded with easy new slices:
security discussion 0.6% FPR, imperative documents 0% FPR, generated
exfiltration 100% recall. That is how it passed val and then failed external.
Future arms need shared validation slices and a common-row comparison
before any external run.

### Base pool (B0)

The base pool is the original training data minus every row that overlaps an
evaluation set: exact text, served scan windows, and reserved payload
substrings. **Zero rows were removed**, so it equals the released training data
(`0c6e12ed82b86e36`). The fixed grid reselected the release parameters
(C=16, 100k char features; val threshold 0.55440). The selection rule and
gates were frozen beforehand ([`data/round2/protocol.json`](../data/round2/protocol.json)).
GPU training was not warranted: B0 matches the release, and the
security-benign val gate cannot be evaluated (24 annotated rows, below the
50-row minimum).

### Fresh false-positive final set

`data/round2/final_fp_eval.csv`: 467 benign rows, SHA256 prefix
`495339890e79916f`, frozen before scoring
([manifest](../data/round2/final_fp_eval.manifest.json)). Composition:

- 266 authored rows: 110 security-topic requests (including 20 authorized-sharing
  requests and 10 quoted controls), 118 documents across 16 carriers,
  36 long documents of 5k–37.5k characters, and 2 size-limit documents.
- 160 LLMail-Inject false-positive-test emails.
- 41 AgentDojo records with default, uninjected values.

The authored rows were written by a separate agent instance that was given
only category definitions and serving constraints. That is the same model
family as the developer agent, so independence is weaker than human
authorship. Most long documents are 5–6.5k characters, so longer lengths are
thinly covered. Every row carries the full chat request, and is scored both
as a single string and through the real input guard at unchanged serving
thresholds. It contains no attacks, so it measures false positives only.

Released baselines, input-guard FPR % (k/n, 95% Wilson interval). There were no
classifier errors. Both size-limit rows were blocked as too large to scan;
they are counted separately and excluded from these rows.

| Model | All (n=465) | Security-topic (n=110) | Documents (n=319) | Long documents (n=36) |
|---|---:|---:|---:|---:|
| sklearn | 2.6 (12) [1.5–4.5] | 2.7 (3) [0.9–7.7] | 1.9 (6) [0.9–4.0] | 8.3 (3) [2.9–21.8] |
| mDeBERTa (hf2) | 0.9 (4) [0.3–2.2] | 1.8 (2) [0.5–6.4] | 0.3 (1) [0.1–1.8] | 2.8 (1) [0.5–14.2] |
| cascade2 → hf2 | 0.9 (4) [0.3–2.2] | 1.8 (2) [0.5–6.4] | 0.3 (1) [0.1–1.8] | 2.8 (1) [0.5–14.2] |
| ONNX INT8 | 1.7 (8) [0.9–3.4] | 1.8 (2) [0.5–6.4] | 0.9 (3) [0.3–2.7] | 8.3 (3) [2.9–21.8] |

Single-string FPR is lower: 1.9 / 0.4 / 0.4 / 1.1%. Long documents are where
windowing adds false positives. On this independently authored set,
mDeBERTa's security-topic FPR is 1.8%, not the 22% measured on the earlier
targeted set. The earlier figure appears specific to that set's style. The
interval's upper bound (6.4%) still exceeds 5%, and these are baseline
measurements, not a round-2 improvement. Per-model reports:
`reports/round2/*_fp_baseline.json`.

### Long documents: windowing multiplies false alarms (val only)

The guard blocks a message when any 2,000-character window reaches the
threshold, so each extra window is another chance of a false alarm. To size
this without touching evaluation sets, `training/round2/long_docs.py` builds
documents from the released dataset's `val` split: 1,000 benign documents
(200 each at about 2.5k, 5k, 10k, 20k and 40k characters) made by joining
`val` benign rows, and 500 buried-attack documents, which are the same kind of
document with one `val` attack row of at most 400 characters inserted at a
paragraph boundary (so a window always holds it whole). No attack text was
written; every document goes through the real input guard.

| Model, window | Short `val` benign FPR | Long benign FPR | Buried-attack recall | Doc AUC |
|---|---:|---:|---:|---:|
| sklearn, 2,000 (served) | 1.9% | 11.7% (28% at 40k) | 15.0% | 0.55 |
| sklearn, 1,000 | 3.1% | 30.8% | 40.2% | 0.58 |
| sklearn, 600 | 3.6% | 43.2% | 62.6% | 0.61 |
| mDeBERTa, 2,000 (served) | 2.2% | 13.0% (31% at 40k) | 23.8% | not recorded |

Standalone, sklearn catches 94.7% of the same 695 attack rows. Inside a
window of unrelated benign text it barely separates buried-attack documents
from benign ones (AUC 0.55), so the threshold sweeps in the reports cut false
alarms and detections together. Smaller windows raise both rates and exceed
`MAX_SCAN_SEGMENTS=64` on long inputs (202 windows at 40k characters for 600).
**No serving change was made:** neither a threshold for multi-window messages
nor a smaller window improves the trade-off. Fixing it needs training data in
which short attacks and benign text share a window (the missing attack data).

Caveats: joined `val` rows are less coherent than real documents, and the
fresh set's real long documents gave lower rates (sklearn 3/36, mDeBERTa 1/36,
too few to compare). Reports: `reports/round2/long_docs_*.json`.

### Acceptance disposition

- **NOT RUN:** external sklearn/mDeBERTa (no candidate), fresh target-category
  recall (no fresh attack cases), existing targeted/holdout after-scores.
- **BASELINE ONLY:** fresh security-topic mDeBERTa FPR 1.8% (2/110).
- **PASS:** `pytest tests/` 88 passed, 18 skipped (opt-in adversarial
  regression tests, not counted); `ruff check` and `ruff format --check` on
  `app demo tests training/round2`.

### Reproduce

The local venv and CPU pinning are as above. Artifacts go to
`~/.cache/aegis-ml-round2` (`AEGIS_ROUND2_ROOT`); `/tmp` is tmpfs and was lost
on reboot last round. Round-2 runners refuse to write under `models/`, the
release reports or the frozen sets.

```bash
taskset -c 0-3 uv run python -m training.round2.diagnose
taskset -c 0-3 uv run python -m training.round2.final_fp_eval     # refuses to overwrite
taskset -c 0-3 uv run python -m training.round2.score_final --model sklearn --name sklearn_fp_baseline
# hf2, cascade2_hf2, and onnx2 with ONNX2_USE_INT8=true, one at a time
taskset -c 0-3 uv run python -m training.round2.base_pool
taskset -c 0-3 uv run python -m training.round2.long_docs --model sklearn --name sklearn
# --segment-chars 1000|600 --overlap 400 for the window comparison; hf2 on GPU
```
