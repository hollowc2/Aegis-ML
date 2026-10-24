"""Round-2 shared paths, resource limits and write guards.

Every round-2 runner writes only under ``data/round2``, ``reports/round2`` or
the local-disk artifact root (``AEGIS_ROUND2_ROOT``, default
``~/.cache/aegis-ml-round2``). Released models, release reports and the frozen
evaluation sets are refused as output paths, so an experiment cannot overwrite
them by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Thread caps must be in place before numpy/torch/onnxruntime start pools.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

MAX_THREADS = 4
DATA = Path("data/round2")
REPORTS = Path("reports/round2")
ARTIFACTS = Path(os.environ.get("AEGIS_ROUND2_ROOT", "~/.cache/aegis-ml-round2")).expanduser()
os.environ.setdefault("TARGETED_SOURCE_CACHE", str(ARTIFACTS / "sources"))

ORIGINAL_DATASET = ARTIFACTS / "original_combined_dataset.csv"
ORIGINAL_SHA16 = "0c6e12ed82b86e36"
# Rejected candidate pool (commit 70c2e1e); copied out of data/ when the release
# dataset was restored.
REJECTED_DATASET = ARTIFACTS / "rejected_combined_dataset.csv"
REJECTED_SHA16 = "a68869d92a3e6dd1"
FINAL_EVAL = DATA / "final_eval.csv"
FINAL_EVAL_MANIFEST = DATA / "final_eval.manifest.json"
BASE_POOL = DATA / "base_pool.csv"
PROTOCOL = DATA / "protocol.json"

# Outputs that must never be written by round-2 code.
_PROTECTED_DIRS = (Path("models").resolve(),)
_PROTECTED_FILES = {
    Path(p).resolve()
    for p in (
        "data/combined_dataset.csv",
        "data/external_eval.csv",
        "data/holdout_eval.csv",
        "data/adversarial_eval.csv",
        "data/targeted_eval.csv",
        "data/targeted_eval.manifest.json",
    )
}
_ALLOWED_ROOTS = (DATA.resolve(), REPORTS.resolve(), ARTIFACTS.resolve())


def safe_output(path: str | Path) -> Path:
    """Return *path* if it is an allowed round-2 output, else raise."""
    resolved = Path(path).expanduser().resolve()
    if resolved in _PROTECTED_FILES or any(resolved.is_relative_to(d) for d in _PROTECTED_DIRS):
        raise PermissionError(f"Refusing to write protected release artifact: {path}")
    if not any(resolved.is_relative_to(root) for root in _ALLOWED_ROOTS):
        raise PermissionError(f"Round-2 outputs must live under {_ALLOWED_ROOTS}: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(path: str | Path) -> dict:
    import pandas as pd

    return {"path": str(path), "sha256": sha256(path), "rows": int(len(pd.read_csv(path)))}


def write_json(path: str | Path, obj, *, overwrite: bool = True) -> Path:
    out = safe_output(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{path} is frozen; refusing to overwrite")
    out.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    return out


def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def cap_threads() -> None:
    """Explicit caps: environment variables alone can be overridden by libraries."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(MAX_THREADS)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(min(MAX_THREADS, len(os.sched_getaffinity(0))))
    except ImportError:
        pass


def require_sha(path: str | Path, sha16: str) -> None:
    actual = sha256(path)[:16]
    if actual != sha16:
        raise ValueError(f"{path} has sha256 prefix {actual}, expected {sha16}")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(flags) -> dict:
    """Rate with count and Wilson interval for a boolean sequence."""
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}
