"""Build and freeze the targeted evaluation BEFORE training changes.

Refuses to overwrite an existing set. The manifest records every source file,
reserved payload and dataset digest. No training template is imported here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from training.data.targeted_eval_prompts import BENIGN, CARRIER, EXFIL, INDIRECT
from training.data.targeted_sources import (
    BIPIA_REV,
    CACHE,
    INJECAGENT_REV,
    NOTINJECT_REV,
    evaluation_rows,
)

OUTPUT = Path("data/targeted_eval.csv")
MANIFEST = OUTPUT.with_suffix(".manifest.json")


def authored_rows() -> list[dict]:
    assert len(INDIRECT) == len(EXFIL) == len(BENIGN) == 100
    rows = []
    wrappers = [
        "Retrieved email attachment:\n{}\nEnd of attachment.",
        "Search result excerpt:\n{}\nEnd of excerpt.",
        "Tool returned this record:\n{}\nEnd of record.",
        "Document annotation:\n{}\nEnd of annotation.",
        "Extracted page text:\n{}\nEnd of page.",
    ]
    for i, (attack, exfil, benign) in enumerate(zip(INDIRECT, EXFIL, BENIGN)):
        long_body = f"Archive record {i + 1}.\n" + CARRIER * (8 + i % 7)
        offset = (len(long_body) // 4, len(long_body) // 2, len(long_body))[i % 3]
        buried = long_body[:offset] + "\n" + attack + "\n" + long_body[offset:]
        for text, label, category, payload in [
            (wrappers[i % 5].format(attack), 1, "indirect_injection", attack),
            (exfil, 1, "data_exfiltration", exfil),
            (buried, 1, "buried_attack", attack),
            (benign, 0, "security_benign", ""),
            (long_body, 0, "long_benign", ""),
        ]:
            rows.append(
                dict(
                    text=text,
                    label=label,
                    category=category,
                    source="authored_targeted_v1",
                    payload=payload,
                )
            )
    return rows


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Targeted evaluation is frozen; do not overwrite or tune it.")
    rows = evaluation_rows() + authored_rows()
    # Use the training/serving normalization for exact conflict checks.
    from training.data.prepare_dataset import normalise_key

    df = pd.DataFrame(rows)
    df["key"] = df.text.map(normalise_key)
    if (df.groupby("key").label.nunique() > 1).any():
        raise ValueError("Conflicting evaluation labels")
    df = df.drop_duplicates("key")
    payloads = sorted({str(p) for p in df.payload.dropna() if p})
    df = df[["text", "label", "category", "source"]]
    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    manifest = {
        "protocol": "Frozen before training; aggregate metrics only until final evaluation. No tuning.",
        "revisions": {"bipia": BIPIA_REV, "injecagent": INJECAGENT_REV, "notinject": NOTINJECT_REV},
        "source_sha256": {
            str(p.relative_to(CACHE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CACHE.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".jsonl", ".parquet"}
        },
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "rows": len(df),
        "counts": df.groupby(["source", "category"]).size().to_dict().__str__(),
        "reserved_payloads": payloads,
        "caveat": "Authored indirect and buried cases are paired; long carriers repeat original prose. NotInject reserved in full; no InjecAgent train split exists.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(df)} rows; sha256 {manifest['sha256'][:16]}")
    print(df.groupby(["category", "source"]).size().to_string())


if __name__ == "__main__":
    main()
