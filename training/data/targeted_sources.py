"""Pinned, licence-reviewed sources. Test data is never returned by training_rows.

BIPIA: use MIT email contexts/attacks only (not news, tables or Stack Exchange).
InjecAgent and NotInject are reserved in full for evaluation. NotInject's
train/validation/test files alias its one/two/three-trigger benchmark groups.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

BIPIA_REV = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"
INJECAGENT_REV = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
NOTINJECT_REV = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
CACHE = Path(os.environ.get("TARGETED_SOURCE_CACHE", ".cache/targeted_sources"))


def source_file(source: str, path: str) -> Path:
    if source == "bipia":
        url = f"https://raw.githubusercontent.com/microsoft/BIPIA/{BIPIA_REV}/{path}"
    elif source == "injecagent":
        url = f"https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/{INJECAGENT_REV}/{path}"
    elif source == "notinject":
        url = f"https://huggingface.co/datasets/leolee99/NotInject/resolve/{NOTINJECT_REV}/{path}"
    else:
        raise ValueError(source)
    out = CACHE / source / path
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(out)
    return out


def bipia_rows(split: str) -> list[dict]:
    if split not in {"train", "test"}:
        raise ValueError(split)
    attacks = json.loads(source_file("bipia", f"benchmark/text_attack_{split}.json").read_text())
    contexts = [
        json.loads(line)
        for line in source_file("bipia", f"benchmark/email/{split}.jsonl").read_text().splitlines()
    ]
    rows = []
    for family, payloads in attacks.items():
        for j, payload in enumerate(payloads):
            # Same payload in three positions, in distinct carrier documents.
            for position in range(3):
                carrier = contexts[(j * 3 + position) % len(contexts)]
                body = carrier["context"]
                offset = (0, len(body) // 2, len(body))[position]
                text = body[:offset] + "\n" + payload + "\n" + body[offset:]
                rows.append(
                    dict(
                        text=text,
                        label=1,
                        threat_category="prompt_injection",
                        source=f"bipia_email_{split}",
                        payload=payload,
                        template_id=f"bipia:{family}",
                        carrier_id=hashlib.sha256(body.encode()).hexdigest(),
                        category="indirect_injection",
                    )
                )
    if split == "train":
        for carrier in contexts:
            body = carrier["context"]
            rows.append(
                dict(
                    text=body,
                    label=0,
                    threat_category="none",
                    source="bipia_email_benign",
                    carrier_id=hashlib.sha256(body.encode()).hexdigest(),
                )
            )
    return rows


def evaluation_rows() -> list[dict]:
    rows = bipia_rows("test")
    for kind, category in [("dh", "indirect_injection"), ("ds", "data_exfiltration")]:
        cases = json.loads(
            source_file("injecagent", f"data/test_cases_{kind}_base.json").read_text()
        )
        for case in cases:
            rows.append(
                dict(
                    text=case["Tool Response"],
                    label=1,
                    category=category,
                    source=f"injecagent_{kind}_test",
                    payload=case["Attacker Instruction"],
                )
            )
    for difficulty in ("one", "two", "three"):
        df = pd.read_parquet(
            source_file("notinject", f"data/NotInject_{difficulty}-00000-of-00001.parquet")
        )
        rows.extend(
            dict(text=t, label=0, category="notinject_benign", source=f"notinject_{difficulty}")
            for t in df.prompt
        )
    return rows


def training_rows() -> list[dict]:
    return bipia_rows("train")
