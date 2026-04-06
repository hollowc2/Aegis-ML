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
# Phase 1 (sklearn — required before running the service)
uv run python -m training.data.prepare_dataset
uv run python -m training.phase1_sklearn.train
uv run python -m training.phase1_sklearn.evaluate

# Phase 2 (HuggingFace — requires `uv sync --extra hf`)
uv run python -m training.phase2_hf.train --model distilbert --qlora --epochs 5
uv run python -m training.phase2_hf.evaluate
uv run python -m training.phase2_hf.export_onnx
```

## Architecture

### Request Pipeline (6 Stages)

Every `POST /v1/chat/completions` request flows through:

1. **Rate Limiting** — slowapi per-IP throttle (`RATE_LIMIT_PER_MINUTE`)
2. **Input Guardrail** — ML classifier scores all message text; blocks if `malicious_prob ≥ CONFIDENCE_THRESHOLD` with HTTP 403
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
| `app/classifiers/` | Four classifier implementations: sklearn, hf, onnx, cascade |
| `app/proxy/llm_proxy.py` | Canary injection into forwarded request, response content extraction/patching |
| `app/api/routes.py` | FastAPI routes: `/v1/chat/completions`, `/health`, `/metrics`, `/audit/logs` |
| `app/models/database.py` | Async SQLite via aiosqlite; single `audit_log` table |

### Classifier Types (set via `CLASSIFIER_TYPE` in `.env`)

- `sklearn` — TF-IDF + Logistic Regression, <1 ms, ~50 MB RAM (Phase 1, default)
- `hf` — Fine-tuned DistilBERT/DeBERTa-v3-small, 5–15 ms, ~400 MB RAM (Phase 2)
- `onnx` — ONNX Runtime inference of the HF model, faster than PyTorch in production
- `cascade` — sklearn fast-path for high-confidence cases, ONNX slow-path for uncertainty

All classifiers expose the same async interface and are loaded once at startup. The sklearn classifier wraps synchronous inference in `asyncio.to_thread()` to avoid blocking the event loop.

### Canary Token Limitations

The in-memory canary store is **single-process only**. Multi-process or distributed deployments need a shared store (e.g., Redis). The current 5-minute TTL is hardcoded in `app/guardrails/canary.py`.

### Trained Model Artifacts

Trained models are stored under `models/` (gitignored). The sklearn classifier must be trained before the service can start with `CLASSIFIER_TYPE=sklearn`. The Phase 1 training pipeline downloads datasets from HuggingFace Hub.

## Configuration

All settings live in `.env` (copy from `.env.example`). Key settings:

| Variable | Default | Effect |
|----------|---------|--------|
| `CLASSIFIER_TYPE` | `sklearn` | Which classifier to load |
| `CONFIDENCE_THRESHOLD` | `0.70` | Block threshold (0–1); lower = more aggressive |
| `BACKEND_URL` | `http://localhost:8080` | llama.cpp / LLM endpoint |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate cap |
| `DATABASE_URL` | `sqlite+aiosqlite:///./logs/audit.db` | Audit log location |
| `REDACT_PROMPTS` | `false` | Omit input text from audit log (GDPR) |

## Prometheus Metrics

Available at `GET /metrics`:
- `aegis_requests_total` (labels: verdict)
- `aegis_request_latency_seconds` (histogram)
- `aegis_classifier_latency_seconds` (histogram)
- `aegis_canary_leaks_total`, `aegis_input_blocks_total`, `aegis_output_blocks_total`
