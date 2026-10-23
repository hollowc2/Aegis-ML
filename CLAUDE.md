# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Aegis-ML is a **drop-in OpenAI-compatible reverse proxy / LLM firewall** that protects locally-hosted language models (llama.cpp, etc.) against prompt injection, jailbreaks, PII leaks, and data exfiltration. Clients point at Aegis instead of the LLM with no code changes required.

## Commands

All commands use `uv run` — the project uses `uv` for dependency management.

### Setup
```bash
uv sync                          # Base install (sklearn only)
uv sync --extra hf               # Add HuggingFace/PyTorch support
uv sync --extra dev              # Add dev tools (pytest, ruff, mypy)
uv sync --all-extras             # Everything
cp .env.example .env             # Configure environment
```

### Running the Service
```bash
uv run python -m app.main        # Development server
uv run aegis-serve               # CLI entry point
uv run aegis-demo                # Launch Gradio demo UI
docker compose up --build        # Docker (Phase 1, sklearn only)
```

### Testing
```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=app --cov-report=html
uv run pytest tests/test_guardrails.py -v    # Single test file
uv run pytest tests/ -k "canary"             # Run tests matching a keyword
```

### Linting & Type Checking
```bash
uv run ruff check .
uv run mypy app/
```

### Training the Classifier
```bash
# Data (writes data/combined_dataset.csv with a fixed split column + data/external_eval.csv)
uv run python -m training.data.holdout_eval        # hand-written holdout set (run first)
uv run python -m training.data.prepare_dataset

# Phase 1 (sklearn — bundled; retrain after any dataset change)
uv run python -m training.phase1_sklearn.train --loso

# Phase 3 (multi-task transformer — requires `uv sync --extra hf2`, GPU recommended)
uv run python -m training.phase3_hf2.train --model mdeberta
uv run python -m training.phase3_hf2.export_onnx --validate

# Shared evaluation suite: any serving classifier, writes reports/<name>.json
uv run python -m training.eval_suite --model onnx2
```

Evaluation rules (see docs/BENCHMARKS.md):
- All phases train on `split == "train"`, pick checkpoints/hyperparameters on
  `val`, and pick the threshold on `val` under an FPR budget. Never tune on
  `test`, `external`, or `holdout`.
- `prepare_dataset.py` removes any training row that also appears in an
  evaluation set. Splits are grouped by near-duplicate family, not by row.
- `export_onnx --validate` fails if either ONNX graph scores below 0.9 AUC on
  the test split, and writes a separate `aegis_threshold_int8` tuned on val
  (dynamic INT8 shifts DeBERTa scores a lot; signed per-tensor INT8 breaks it).
- `adversarial_eval.csv` shares vocabulary with `synthetic_gen.py`, so its
  numbers are optimistic; `external_eval.csv` is the primary benchmark.
- The repo lives on NFS; for GPU training a venv on local disk
  (`UV_PROJECT_ENVIRONMENT=~/.cache/aegis-ml-venv uv sync --extra dev --extra hf2`)
  installs in minutes instead of copying 14 GB of ROCm torch over the network.

## Architecture

### Request Pipeline (6 Stages)

Every `POST /v1/chat/completions` request flows through:

1. **Rate Limiting** — slowapi per-IP throttle (`RATE_LIMIT_PER_MINUTE`)
2. **Input Guardrail** — ML classifier scores every message separately (long messages in overlapping ~2000-char windows, capped by `MAX_SCAN_SEGMENTS`); blocks with HTTP 403 if the highest `malicious_prob` reaches the threshold (`resolve_threshold` in `app/classifiers/__init__.py`)
3. **Canary Token Injection** — random UUID-grade token embedded as `[SYS_REF:TOKEN]` in the system prompt
4. **Reverse Proxy** — httpx async client forwards to `BACKEND_URL` (llama.cpp / OpenAI-compatible endpoint)
5. **Output Guardrail** — checks for canary leak (→ block), applies PII redaction (regex: SSN, CC, email, phone, IPv4, AWS keys), and harmful keyword filter
6. **Audit Log** — every request written to SQLite with verdict, threat type, latency

**Fail-secure**: any exception in guardrail code blocks the request rather than allowing it through.

### Key Module Responsibilities

| Path | Role |
|------|------|
| `app/main.py` | App factory, lifespan startup/shutdown, classifier loading via `_load_classifier()` |
| `app/config.py` | All configuration (Pydantic v2 Settings, loaded from `.env`) |
| `app/models/schemas.py` | OpenAI-compatible request/response schemas + internal types (`ThreatCategory`, `AuditEntry`) |
| `app/guardrails/input_guard.py` | Orchestrates classification; extracts text from all messages including role labels |
| `app/guardrails/canary.py` | Token generation, injection, one-time consumption, 5-min TTL store |
| `app/guardrails/output_guard.py` | Canary leak detection, PII redaction, harmful content filter |
| `app/classifiers/` | Phase 1/2/3 classifiers plus ONNX and cascade variants |
| `app/proxy/llm_proxy.py` | Canary injection into forwarded request, response content extraction/patching |
| `app/api/routes.py` | FastAPI routes: `/v1/chat/completions`, `/health`, `/metrics`, `/audit/logs` |
| `app/models/database.py` | Async SQLite via aiosqlite; single `audit_log` table |

### Classifier Types (set via `CLASSIFIER_TYPE` in `.env`)

- `sklearn` — TF-IDF (word + char n-grams) + Logistic Regression, ~1.5 ms (Phase 1, default)
- `hf` — Fine-tuned DistilBERT/DeBERTa-v3-small, 5–15 ms, ~400 MB RAM (Phase 2)
- `onnx` — ONNX Runtime inference of the Phase 2 model
- `cascade` — sklearn fast-path with Phase 2 ONNX escalation
- `hf2` — Phase 3 multi-task mDeBERTa-v3 model with threat categories
- `onnx2` — INT8-capable ONNX inference for Phase 3
- `cascade2` — sklearn fast-path with Phase 3 ONNX escalation; sklearn scores inside `CASCADE_SK_LOW/HIGH_THRESHOLD` are decided by the ONNX model alone, using its threshold

All classifiers expose the same async interface and are loaded once at startup. The sklearn classifier wraps synchronous inference in `asyncio.to_thread()` to avoid blocking the event loop.

### Canary Token Limitations

The in-memory canary store is **single-process only**. Multi-process or distributed deployments need a shared store (e.g., Redis). The current 5-minute TTL is hardcoded in `app/guardrails/canary.py`.

### Trained Model Artifacts

Trained models are stored under `models/`. Large transformer artifacts are
gitignored; the small sklearn classifier is bundled so the default service and
demo can start without training. The Phase 1 training pipeline downloads
datasets from HuggingFace Hub.

## Configuration

All settings live in `.env` (copy from `.env.example`). Key settings:

| Variable | Default | Effect |
|----------|---------|--------|
| `CLASSIFIER_TYPE` | `sklearn` | Which classifier to load |
| `CONFIDENCE_THRESHOLD` | unset | Block threshold (0–1); unset = the loaded model's tuned threshold (`models/sklearn_classifier.json`, or `aegis_threshold` / `aegis_threshold_int8` in the ONNX/HF2 `config.json`), else 0.70 |
| `BACKEND_URL` | `http://localhost:8080` | llama.cpp / LLM endpoint |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate cap |
| `DATABASE_URL` | `sqlite+aiosqlite:///./logs/audit.db` | Audit log location |
| `REDACT_PROMPTS_IN_LOGS` | `false` | Omit input text from audit log |

## Prometheus Metrics

Available at `GET /metrics`:
- `aegis_requests_total` (labels: verdict)
- `aegis_request_latency_seconds` (histogram)
- `aegis_classifier_latency_seconds` (histogram)
- `aegis_canary_leaks_total`, `aegis_input_blocks_total`, `aegis_output_blocks_total`
