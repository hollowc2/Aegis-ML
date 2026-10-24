"""Report the stopped targeted experiment without changing thresholds or data.

The candidate's external FPR failed the predeclared budget. This command reads
aggregate reports only; it never trains, scores samples, or changes a model.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def build_report() -> dict:
    before = json.loads((REPORTS / "sklearn_before_targeted.json").read_text())
    candidate = json.loads((REPORTS / "sklearn_targeted_candidate.json").read_text())
    blind_before = json.loads((REPORTS / "sklearn_targeted_baseline.json").read_text())
    external_before, external_after = before["sets"]["external"], candidate["sets"]["external"]
    checks = {
        "sklearn_external": {
            "status": "pass"
            if external_after["recall"] >= external_before["recall"] - 0.01
            and external_after["fpr"] <= 0.02
            else "fail",
            "before": {k: external_before[k] for k in ("recall", "fpr")},
            "after": {k: external_after[k] for k in ("recall", "fpr")},
            "toxic_chat_before": external_before["breakdown"]["toxic_chat"],
            "toxic_chat_after": external_after["breakdown"]["toxic_chat"],
        },
        "mdeberta_external": {
            "status": "not_run",
            "reason": "Stopped on sklearn external FPR failure before GPU training.",
        },
        "blind_targeted": {
            "status": "fail",
            "reason": "Indirect and exfiltration recall rose, but buried recall stayed zero; no transformer candidate was trained.",
            "sklearn_before": blind_before["sets"]["targeted"]["breakdown"],
            "sklearn_after": candidate["sets"]["targeted"]["breakdown"],
        },
        "mdeberta_security_benign_fpr": {
            "status": "not_run",
            "baseline_fpr": json.loads((REPORTS / "hf2_targeted_baseline.json").read_text())[
                "sets"
            ]["targeted"]["breakdown"]["security_benign"]["fpr"],
            "required_after_fpr": "<0.05",
            "reason": "Stopped before transformer training; no after score or success claim.",
        },
        "old_holdout": {
            "status": "reported",
            "caveat": "Already-seen diagnostic set.",
            "sklearn_before": before["sets"]["holdout"],
            "sklearn_after": candidate["sets"]["holdout"],
        },
        "serving_long_after": {
            "status": "not_run",
            "reason": "Stopped on external failure. Baseline guard-path results are saved for all four models.",
        },
    }
    return {
        "status": "rejected",
        "stop_reason": "External sklearn FPR exceeds the predeclared 2% budget. No threshold adjustment, retraining or eval-set changes followed.",
        "released_models": "Original baseline weights restored; rejected sklearn candidate kept in session scratchpad.",
        "candidate_report": "reports/sklearn_targeted_candidate.json",
        "checks": checks,
        "skipped_work": [
            "mDeBERTa retraining",
            "ONNX export and INT8 recalibration",
            "after-evaluation of hf2/cascade2_hf2/onnx2",
            "candidate serving-path evaluation",
        ],
    }


def main() -> None:
    report = build_report()
    (REPORTS / "targeted_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["stop_reason"])


if __name__ == "__main__":
    main()
