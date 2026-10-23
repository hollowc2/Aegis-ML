"""Round-2 guards: release-artifact protection, evaluation isolation and serving behaviour."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.guardrails.input_guard import _extract_segments
from app.models.schemas import ChatCompletionRequest, ChatMessage
from training.round2 import common
from training.round2.final_fp_eval import build_messages, windows
from training.round2.sk_common import pick_threshold_strict


@pytest.mark.parametrize(
    "path",
    [
        "models/sklearn_classifier.joblib",
        "models/hf2_classifier/config.json",
        "data/combined_dataset.csv",
        "data/external_eval.csv",
        "data/targeted_eval.csv",
        "reports/sklearn.json",
    ],
)
def test_round2_outputs_cannot_overwrite_release_artifacts(path):
    with pytest.raises(PermissionError):
        common.safe_output(path)


def test_round2_outputs_allowed_only_under_isolated_roots():
    assert common.safe_output(common.REPORTS / "x.json").name == "x.json"
    assert common.safe_output(common.ARTIFACTS / "models" / "m.joblib").name == "m.joblib"


def test_frozen_json_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ALLOWED_ROOTS", (tmp_path.resolve(),))
    common.write_json(tmp_path / "r.json", {"a": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        common.write_json(tmp_path / "r.json", {"a": 2}, overwrite=False)
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}


def test_strict_threshold_keeps_fpr_budget_above_release_cap():
    # Benign scores all above 0.95: the release helper's 0.95 cap would flag them all.
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0.96, 0.99, 100), np.full(10, 0.999)])
    result = pick_threshold_strict(y, p, 0.02)
    assert result["threshold"] > 0.95
    assert result["val_fpr"] <= 0.02 and result["budget_met"]


def test_window_count_matches_serving_segmentation():
    for n in (10, 2000, 2001, 3600, 9999, 120_000):
        # Non-repeating text: serving de-duplicates identical windows.
        text = " ".join(f"w{i}" for i in range(n))[:n]
        request = ChatCompletionRequest(
            model="t", messages=[ChatMessage(role="user", content=text)]
        )
        assert windows(text) == len(_extract_segments(request))


def test_scanned_text_is_exactly_one_message_with_its_role():
    msgs, scanned = build_messages("Quarterly report body.", "user", "Summarize this.", "sys")
    assert [m["content"] for m in msgs if m["role"] == "user"] == [scanned]
    assert scanned.startswith("Summarize this.") and scanned.endswith("Quarterly report body.")
    msgs, scanned = build_messages("Tool payload.", "tool", "Check my files.", "sys")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert scanned == "Tool payload."


def test_base_pool_excludes_serving_windows_of_evaluation_requests(tmp_path, monkeypatch):
    from training import eval_suite
    from training.data.prepare_dataset import normalise_key
    from training.round2.base_pool import forbidden_keys

    long_doc = " ".join(f"word{i}" for i in range(900))
    messages = [{"role": "system", "content": "s"}, {"role": "tool", "content": long_doc}]
    path = tmp_path / "fp.csv"
    pd.DataFrame(
        {"text": [long_doc], "label": [0], "messages_json": [json.dumps(messages)]}
    ).to_csv(path, index=False)
    monkeypatch.setattr(eval_suite, "EVAL_SETS", {"round2_fp": (path, "category")})
    keys, _ = forbidden_keys()
    request = ChatCompletionRequest(model="t", messages=messages)
    middle_window = _extract_segments(request)[2]
    assert normalise_key(middle_window) in keys


def test_fp_scoring_separates_size_limit_blocks_from_false_positives(tmp_path, monkeypatch):
    from training.round2 import score_final

    rows = []
    for case, text, category in [
        ("a", "Lunch menu for Friday.", "benign_document"),
        ("b", " ".join(f"w{i}" for i in range(30_000)), "size_limit_benign"),
    ]:
        msgs = [{"role": "user", "content": text}]
        rows.append(
            dict(
                case_id=case,
                text=text,
                label=0,
                category=category,
                subcategory=category,
                source="test",
                messages_json=json.dumps(msgs),
            )
        )
    path = tmp_path / "fp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(score_final, "OUTPUT", path)

    class Benign:
        threshold = 0.5

        async def predict(self, text):
            return {"malicious_prob": 0.1, "benign_prob": 0.9}

    out = score_final.score(Benign(), 0.5)
    assert out["size_limit_rows"] == {"n": 1, "size_limit_blocks": 1}
    assert out["input_guard"]["fpr_all"]["n"] == 1
    assert out["input_guard"]["fpr_all"]["k"] == 0
