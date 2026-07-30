# Benchmark notes

This document separates reproduced measurements from design targets. It exists
to make the portfolio claims auditable and to prevent placeholder notebook
values from being presented as model results.

## Reproduced baseline

The Phase 1 scikit-learn evaluation was reproduced from the committed model and
dataset during the portfolio-polish pass:

| Metric | Recorded value |
|---|---:|
| Test examples | 2,586 |
| Threshold | 0.70 |
| ROC-AUC | 0.9982 |
| F1 | 0.9735 |
| False-positive rate | 0.31% |
| False-negative rate | 4.87% |
| Mean inference latency | 0.185 ms/example |

The split is recreated with `train_test_split(..., test_size=0.2,
stratify=labels, random_state=42)`. Latency was measured as one batched
`predict_proba` call divided by the number of examples, so it is not equivalent
to end-to-end single-request API latency.

The older committed comparison notebook contains an earlier 2,377-row
evaluation snapshot. The values above come from rerunning the current
`data/combined_dataset.csv` and bundled model with scikit-learn 1.8.0.

Reproduce it with:

```bash
uv sync --locked
uv run python -m training.phase1_sklearn.evaluate --threshold 0.70
```

## Important caveats

- The combined dataset is curated from public and synthetic sources. Results
  may be inflated by source overlap or near-duplicate prompts.
- The recorded notebook contains placeholder Phase 2 values when its model
  artifact is absent. Those values are not published as benchmark results.
- Phase 2, Phase 3, ONNX, and cascade performance should be published only with
  the exact model artifact, commit, hardware, dataset checksum, and command.
- Prompt-injection performance changes substantially by domain and operating
  threshold. These results do not imply universal protection.

## Publication checklist for new results

For each model, record:

1. Git commit and model artifact identifier.
2. Dataset path, row count, and checksum.
3. Split procedure and random seed.
4. Classification threshold.
5. Precision, recall, F1, ROC-AUC, FPR, and FNR.
6. Per-category adversarial results.
7. Hardware, runtime, batch size, warm-up, and latency percentiles.

The opt-in adversarial suite can be run with:

```bash
RUN_ADVERSARIAL_TESTS=1 CLASSIFIER_TYPE=sklearn uv run pytest \
  tests/test_adversarial_regression.py -v
```
