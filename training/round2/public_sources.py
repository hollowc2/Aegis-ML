"""Pinned public BENIGN sources for round 2: LLMail-Inject and AgentDojo.

Licences: both MIT (notices in docs/data_licenses/). Downloads are pinned to
the revisions below and cached under ``ARTIFACTS/sources``.

Only benign material is used: LLMail-Inject's false-positive-test emails and
scenario emails, and AgentDojo environment records rendered with each
injection vector's *default* (uninjected) value. Attack submissions and
injection goals are not used in round 2 (see docs/BENCHMARKS.md).
YAML is read with ``yaml.safe_load``/plain text; downloaded code is never run.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from training.round2.common import ARTIFACTS

LLMAIL_REPO = "microsoft/llmail-inject-challenge"
LLMAIL_REV = "1063bdf01ec8762b812d5e06ee768a06faa5a6f7"
AGENTDOJO_REV = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
SOURCES = ARTIFACTS / "sources"
LLMAIL_DIR = SOURCES / "llmail"
AGENTDOJO_DIR = SOURCES / "agentdojo"
SUITES = ("workspace", "travel", "banking", "slack")

LLMAIL_FILES = [
    "data/emails_for_fp_tests.json",
    "data/scenarios.json",
]
AGENTDOJO_FILES = [
    "LICENSE",
    *(f"src/agentdojo/data/suites/{s}/injection_vectors.yaml" for s in SUITES),
    "src/agentdojo/data/suites/banking/environment.yaml",
    "src/agentdojo/data/suites/slack/environment.yaml",
    "src/agentdojo/data/suites/travel/environment.yaml",
    "src/agentdojo/data/suites/workspace/include/calendar.yaml",
    "src/agentdojo/data/suites/workspace/include/cloud_drive.yaml",
    "src/agentdojo/data/suites/workspace/include/inbox.yaml",
]


def fetch_all() -> dict[str, str]:
    """Download any missing pinned file; return {relative path: sha256}."""
    from huggingface_hub import hf_hub_download

    digests = {}
    for path in LLMAIL_FILES:
        out = LLMAIL_DIR / path
        if not out.exists():
            hf_hub_download(
                LLMAIL_REPO, path, repo_type="dataset", revision=LLMAIL_REV, local_dir=LLMAIL_DIR
            )
        digests[f"llmail/{path}"] = hashlib.sha256(out.read_bytes()).hexdigest()
    for path in AGENTDOJO_FILES:
        out = AGENTDOJO_DIR / path
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/{AGENTDOJO_REV}/{path}"
            with urllib.request.urlopen(url, timeout=60) as response:
                out.write_bytes(response.read())
        digests[f"agentdojo/{path}"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return digests


# ── LLMail-Inject ────────────────────────────────────────────────────────────


def llmail_benign_emails() -> dict[str, list[str]]:
    """Benign emails shipped with the challenge: FP-test emails and scenario emails."""
    fp = json.loads((LLMAIL_DIR / "data/emails_for_fp_tests.json").read_text())
    scenarios = json.loads((LLMAIL_DIR / "data/scenarios.json").read_text())
    scenario_emails = list(dict.fromkeys(e for s in scenarios.values() for e in s["emails"]))
    fp_set = set(fp)
    return {
        "fp_tests": list(dict.fromkeys(fp)),
        "scenario": [e for e in scenario_emails if e not in fp_set],
    }


# ── AgentDojo ────────────────────────────────────────────────────────────────
def _placeholder(names) -> re.Pattern:
    return re.compile(
        r"\{(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\}"
    )


def _vector_defaults(suite: str) -> dict[str, str]:
    import yaml

    path = AGENTDOJO_DIR / f"src/agentdojo/data/suites/{suite}/injection_vectors.yaml"
    return {k: str(v.get("default", "")) for k, v in yaml.safe_load(path.read_text()).items()}


def _environment_files(suite: str) -> list[Path]:
    base = AGENTDOJO_DIR / f"src/agentdojo/data/suites/{suite}"
    if suite == "workspace":
        return sorted((base / "include").glob("*.yaml"))
    return [base / "environment.yaml"]


def agentdojo_records() -> list[dict]:
    """One environment record (YAML list item or file line) per injection vector."""
    records = []
    for suite in SUITES:
        defaults = _vector_defaults(suite)
        pattern = _placeholder(defaults)
        for path in _environment_files(suite):
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                for match in pattern.finditer(line):
                    vector = match.group(1)
                    indent = len(line) - len(line.lstrip())
                    start = i
                    while start > 0 and i - start < 30:
                        cand = lines[start]
                        if (
                            cand.lstrip().startswith("- ")
                            and len(cand) - len(cand.lstrip()) < indent
                        ):
                            break
                        start -= 1
                    item_indent = len(lines[start]) - len(lines[start].lstrip())
                    end = i + 1
                    while end < len(lines) and end - i < 40:
                        cand = lines[end]
                        cand_indent = len(cand) - len(cand.lstrip())
                        if cand.strip() and cand_indent <= item_indent:
                            break
                        end += 1
                    snippet = "\n".join(lines[start:end])
                    snippet = _dedent(snippet)
                    records.append(
                        {
                            "suite": suite,
                            "file": path.name,
                            "vector": vector,
                            "template": snippet,
                            "defaults": defaults,
                        }
                    )
    return records


def _dedent(text: str) -> str:
    lines = text.splitlines()
    indent = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    return "\n".join(x[indent:] for x in lines)


def render_record(record: dict) -> str:
    """The record with every injection vector set to its benign default value."""
    return _placeholder(record["defaults"]).sub(
        lambda match: record["defaults"].get(match.group(1), ""), record["template"]
    )
