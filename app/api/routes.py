"""
app/api/routes.py
=================
All FastAPI route handlers:

  POST /v1/chat/completions  — main reverse-proxy endpoint
  GET  /health               — liveness / readiness check
  GET  /metrics              — Prometheus text metrics
  GET  /audit/logs           — recent audit log entries (debug)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.guardrails.input_guard import run_input_guardrail
from app.guardrails.output_guard import run_output_guardrail
from app.models.database import get_recent_logs, log_audit_entry
from app.models.schemas import (
    AuditEntry,
    BlockedResponse,
    ChatCompletionRequest,
    GuardrailVerdict,
    HealthResponse,
)
from app.proxy.llm_proxy import (
    extract_assistant_content,
    forward_to_backend,
    patch_response_content,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Prometheus Metrics ────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "aegis_requests_total",
    "Total requests processed",
    ["verdict"],
)
REQUEST_LATENCY = Histogram(
    "aegis_request_latency_seconds",
    "End-to-end request latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
CLASSIFIER_LATENCY = Histogram(
    "aegis_classifier_latency_seconds",
    "Classifier inference latency",
)
CANARY_LEAKS = Counter(
    "aegis_canary_leaks_total",
    "Number of canary token leaks detected",
)
INPUT_BLOCKS = Counter(
    "aegis_input_blocks_total",
    "Requests blocked at input guardrail",
)
OUTPUT_BLOCKS = Counter(
    "aegis_output_blocks_total",
    "Responses blocked at output guardrail",
)


# ── Dependency helpers ────────────────────────────────────────────────────────


def get_classifier(request: Request):
    """Dependency: return the loaded classifier from app state."""
    return request.app.state.classifier


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency: return the shared httpx client from app state."""
    return request.app.state.http_client


def get_limiter(request: Request) -> Limiter:
    return request.app.state.limiter


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    classifier=Depends(get_classifier),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """
    Main reverse-proxy endpoint.

    Flow:
      1. Input guardrail  → block if malicious
      2. Inject canary token into system prompt
      3. Forward to backend LLM
      4. Output guardrail → block / redact if needed
      5. Return cleaned response + audit log
    """
    settings = get_settings()
    start = time.perf_counter()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    client_ip = get_remote_address(request)

    # ── Apply rate limit ───────────────────────────────────────────────────────
    limiter: Limiter = request.app.state.limiter
    rate_limit_str = f"{settings.rate_limit_per_minute}/minute"
    limiter._check_request_limit(request, None, rate_limit_str)  # type: ignore[attr-defined]

    # ── 1. Input Guardrail ─────────────────────────────────────────────────────
    clf_start = time.perf_counter()
    input_result = await run_input_guardrail(body, classifier)
    clf_elapsed = time.perf_counter() - clf_start
    CLASSIFIER_LATENCY.observe(clf_elapsed)

    if input_result.verdict == GuardrailVerdict.block:
        INPUT_BLOCKS.inc()
        REQUESTS_TOTAL.labels(verdict="blocked_input").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000

        await log_audit_entry(
            request_id=request_id,
            client_ip=client_ip,
            input_text=_safe_text(body),
            input_verdict="block",
            input_confidence=input_result.confidence,
            input_threat=input_result.threat_category.value,
            output_verdict="n/a",
            output_threat="none",
            latency_ms=elapsed_ms,
        )

        logger.warning(
            "BLOCKED request %s | confidence=%.2f | threat=%s | ip=%s",
            request_id,
            input_result.confidence,
            input_result.threat_category.value,
            client_ip,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=BlockedResponse().model_dump(),
        )

    # ── 2 & 3. Forward to backend (canary injected inside forward_to_backend) ──
    try:
        backend_response = await forward_to_backend(
            request_payload=body.model_dump(exclude_none=True),
            request_id=request_id,
            client=http_client,
        )
    except httpx.TimeoutException:
        logger.error("Backend timeout for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Backend LLM timed out.",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("Backend HTTP error %d for request %s", exc.response.status_code, request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Backend returned {exc.response.status_code}.",
        )
    except Exception as exc:
        # Fail-secure: unknown backend errors are surfaced as 502
        logger.exception("Unexpected backend error for request %s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Backend error.",
        )

    # ── 4. Output Guardrail ────────────────────────────────────────────────────
    assistant_content = extract_assistant_content(backend_response)
    output_result = await run_output_guardrail(assistant_content, request_id)

    final_response = backend_response
    if output_result.verdict == GuardrailVerdict.block:
        OUTPUT_BLOCKS.inc()
        if output_result.threat_category.value == "canary_leak":
            CANARY_LEAKS.inc()

        REQUESTS_TOTAL.labels(verdict="blocked_output").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000

        await log_audit_entry(
            request_id=request_id,
            client_ip=client_ip,
            input_text=_safe_text(body),
            input_verdict="allow",
            input_confidence=input_result.confidence,
            input_threat=input_result.threat_category.value,
            output_verdict="block",
            output_threat=output_result.threat_category.value,
            latency_ms=elapsed_ms,
        )

        logger.warning(
            "OUTPUT BLOCKED request %s | threat=%s",
            request_id,
            output_result.threat_category.value,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=BlockedResponse(
                error={
                    "message": output_result.reason,
                    "type": "guardrail_violation",
                    "code": output_result.threat_category.value,
                }
            ).model_dump(),
        )

    if output_result.verdict == GuardrailVerdict.redact:
        final_response = patch_response_content(backend_response, output_result.content)

    # ── 5. Audit + return ─────────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start) * 1000
    REQUEST_LATENCY.observe(elapsed_ms / 1000)
    REQUESTS_TOTAL.labels(verdict="allowed").inc()

    await log_audit_entry(
        request_id=request_id,
        client_ip=client_ip,
        input_text=_safe_text(body),
        input_verdict="allow",
        input_confidence=input_result.confidence,
        input_threat=input_result.threat_category.value,
        output_verdict=output_result.verdict.value,
        output_threat=output_result.threat_category.value,
        latency_ms=elapsed_ms,
    )

    return JSONResponse(content=final_response)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Liveness + readiness check — returns classifier status."""
    settings = get_settings()
    classifier = getattr(request.app.state, "classifier", None)
    loaded = classifier.is_loaded() if classifier else False

    return HealthResponse(
        status="ok" if loaded else "degraded",
        classifier=settings.classifier_type,
        classifier_loaded=loaded,
    )


@router.get("/metrics")
async def metrics():
    """Prometheus text metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/audit/logs")
async def audit_logs(limit: int = 50):
    """Return the most recent audit log entries (for debugging / monitoring)."""
    if limit > 500:
        limit = 500
    rows = await get_recent_logs(limit=limit)
    return {"count": len(rows), "logs": rows}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_text(body: ChatCompletionRequest) -> str:
    """
    Extract a single string from the request for audit logging.
    Truncated to 1000 chars to keep the DB tidy.
    """
    parts = [f"[{m.role.value}]: {m.content}" for m in body.messages]
    full = "\n".join(parts)
    return full[:1000]
