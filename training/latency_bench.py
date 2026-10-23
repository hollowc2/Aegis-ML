"""
training/latency_bench.py
==========================
Single-request classifier latency, measured through the serving classes the
API uses (one predict() call at a time, after warm-up). Texts are a fixed
random sample of the external benchmark, so prompt lengths are realistic.

Run:
    python -m training.latency_bench --model sklearn onnx2 cascade2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import time

import numpy as np
import pandas as pd

from training.eval_suite import EVAL_SETS, load_classifier, preprocess


async def _bench(clf, texts: list[str], warmup: int) -> dict:
    for text in texts[:warmup]:
        await clf.predict(text)
    latencies, stages = [], []
    for text in texts:
        start = time.perf_counter()
        result = await clf.predict(text)
        latencies.append((time.perf_counter() - start) * 1000)
        stages.append(result.get("stage", ""))
    lat = np.asarray(latencies)
    out = {
        "n": len(texts),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
    }
    if any(stages):
        out["escalated"] = float(np.mean([s != "sklearn" for s in stages]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-request classifier latency")
    parser.add_argument("--model", nargs="+", default=["sklearn"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    path, _ = EVAL_SETS["external"]
    sample = pd.read_csv(path).sample(n=args.n, random_state=0)
    texts = preprocess(sample["text"].astype(str).tolist())
    print(f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"n={args.n}, median prompt {int(np.median([len(t) for t in texts]))} chars")
    for model in args.model:
        stats = asyncio.run(_bench(load_classifier(model), texts, args.warmup))
        extra = f"  escalated {stats['escalated'] * 100:.0f}%" if "escalated" in stats else ""
        print(f"{model:<10} p50 {stats['p50_ms']:7.2f} ms  p95 {stats['p95_ms']:7.2f} ms  "
              f"mean {stats['mean_ms']:7.2f} ms{extra}")


if __name__ == "__main__":
    main()
