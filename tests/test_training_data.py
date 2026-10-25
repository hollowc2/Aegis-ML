"""Regression checks for evaluation isolation in the training data."""

from __future__ import annotations

import pandas as pd

from training.data.prepare_dataset import _eval_keys, normalise_key


def test_registered_evaluation_text_is_removed_after_unicode_normalization(tmp_path, monkeypatch):
    from training import eval_suite

    path = tmp_path / "targeted.csv"
    pd.DataFrame({"text": ["ＩＧＮＯＲＥ the test fixture."], "label": [1]}).to_csv(
        path, index=False
    )
    monkeypatch.setattr(eval_suite, "EVAL_SETS", {"targeted": (path, "category")})
    assert normalise_key("IGNORE the test fixture!") in _eval_keys()
