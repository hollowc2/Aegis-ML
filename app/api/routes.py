"""
app/api/routes.py
=================
All FastAPI route handlers:

  POST /v1/chat/completions  — main reverse-proxy endpoint
  GET  /v1/models            — passthrough to the backend's model list
  GET  /health               — liveness / readiness check
  GET  /aegis/readiness      — versioned enforcement readiness (aegis.readiness/1)
  GET  /metrics              — Prometheus text metrics (ADMIN_TOKEN if set)
  GET  /audit/logs           — recent audit log entries (requires ADMIN_TOKEN)

ENFORCEMENT_MODE=monitor records input-classifier blocks as "would_block" and
forwards the request anyway, so false positives can be measured on real
traffic. The output guard (canary leak, PII redaction) is enforced in both
modes: it has almost no false positives, and what it catches is the damage.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.guardrails.canary import consume_token as discard_token
from app.guardrails.input_guard import run_input_guardrail
from app.guardrails.output_guard import run_output_guardrail
from app.models.database import get_recent_logs, log_audit_entry
from app.models.schemas import (
    BlockedResponse,
    ChatCompletionRequest,
    GuardrailVerdict,
    HealthResponse,
    ReadinessResponse,
)
from app.proxy.llm_proxy import (
    extract_assistant_content,
    fetch_backend_models,
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
MONITOR_FLAGS = Counter(
    "aegis_monitor_flags_total",
    "Verdicts monitor mode recorded but did not enforce",
    ["stage"],
)


# A caller sends this header to require enforcement: monitor mode or an
# unloaded classifier then refuse the request (503) instead of forwarding it.
REQUIRE_ENFORCEMENT_HEADER = "x-aegis-require-enforcement"
# Set on every proxied reply: the enforcement mode its input was checked under.
ENFORCEMENT_HEADER = "x-aegis-enforcement"
# How often a pending backend request checks whether its caller is still there.
DISCONNECT_POLL_SECONDS = 0.25


class ClientDisconnectedError(Exception):
    """The caller went away while its backend request was running."""


def readiness_reasons(request: Request) -> list[str]:
    """Fixed reason codes why input blocking is not enforced right now."""
    reasons = []
    if get_settings().enforcement_mode != "enforce":
        reasons.append("monitor_mode")
    classifier = getattr(request.app.state, "classifier", None)
    if not (classifier and classifier.is_loaded()):
        reasons.append("classifier_not_loaded")
    return reasons


async def backend_call(request: Request, start_work: Callable[[], Awaitable[Any]]) -> Any:
    """Run one backend request, bounded, counted and tied to its caller.

    The work is counted in ``backend_inflight`` until it has actually finished,
    including the cancellation that closes its upstream connection, so
    /aegis/readiness can show when abandoned work is gone. If the caller
    disconnects the work is cancelled (closing the connection; with
    BACKEND_STREAM llama-server then stops generating) and
    ClientDisconnectedError is raised. The whole
    request, not each read, is bounded by BACKEND_TIMEOUT.
    """
    state = request.app.state

    async def counted() -> Any:
        state.backend_inflight = getattr(state, "backend_inflight", 0) + 1
        try:
            return await start_work()
        finally:
            state.backend_inflight -= 1

    # A caller that left during classification never starts backend work.
    if await request.is_disconnected():
        raise ClientDisconnectedError()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + get_settings().backend_timeout
    task = asyncio.ensure_future(counted())
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise httpx.TimeoutException("backend deadline exceeded")
            done, _ = await asyncio.wait({task}, timeout=min(DISCONNECT_POLL_SECONDS, remaining))
            if task in done:
                return task.result()
            if await request.is_disconnected():
                raise ClientDisconnectedError()
    finally:
        if not task.done():
            task.cancel()
            # Wait for the cancellation to close the connection before returning.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _error(message: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "guardrail_unavailable", "code": code}}


# ── Dependency helpers ────────────────────────────────────────────────────────


async def get_classifier(request: Request):
    """Dependency: return the loaded classifier from app state."""
    return request.app.state.classifier


async def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency: return the shared httpx client from app state."""
    return request.app.state.http_client


async def get_limiter(request: Request) -> Limiter:
    return request.app.state.limiter


def _require_admin(request: Request, *, required: bool) -> None:
    """Check the ADMIN_TOKEN bearer token.

    With no token configured, endpoints that need one (*required*) are
    disabled and the rest stay open.
    """
    token = get_settings().admin_token
    if not token:
        if required:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Disabled: set ADMIN_TOKEN to enable this endpoint.",
            )
        return
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
      0. Enforcement requirement → 503 if the caller requires enforcement and it is off
      1. Input guardrail  → block if malicious, or if the classifier fails or times out
      2. Inject canary token into system prompt
      3. Forward to backend LLM (cancelled if the caller disconnects)
      4. Output guardrail → block / redact if needed
      5. Return cleaned response + audit log
    """
    start = time.perf_counter()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    client_ip = get_remote_address(request)
    settings = get_settings()

    async def audit(input_verdict: str, output_verdict: str, **fields) -> None:
        await log_audit_entry(
            request_id=request_id,
            client_ip=client_ip,
            input_text=_safe_text(body),
            input_verdict=input_verdict,
            input_confidence=fields.get("input_confidence", 0.0),
            input_threat=fields.get("input_threat", "none"),
            output_verdict=output_verdict,
            output_threat=fields.get("output_threat", "none"),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # ── Apply rate limit ───────────────────────────────────────────────────────
    limiter: Limiter = request.app.state.limiter
    limiter._check_request_limit(request, None)  # type: ignore[attr-defined]

    # ── 0. Caller-required enforcement ────────────────────────────────────────
    # A caller that must never run unguarded (cobblebob's public Q&A) sends the
    # header; monitor mode or an unloaded classifier then refuses before any
    # classification or backend work instead of forwarding.
    if request.headers.get(REQUIRE_ENFORCEMENT_HEADER) is not None:
        reasons = readiness_reasons(request)
        if reasons:
            REQUESTS_TOTAL.labels(verdict="enforcement_unavailable").inc()
            await audit("enforcement_unavailable", "n/a")
            logger.warning(
                "Enforcement required but unavailable for request %s: %s",
                request_id,
                ",".join(reasons),
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=_error("Enforcement required but not active.", "enforcement_unavailable"),
            )

    # ── 1. Input Guardrail ─────────────────────────────────────────────────────
    clf_start = time.perf_counter()
    input_result = await run_input_guardrail(body, classifier)
    clf_elapsed = time.perf_counter() - clf_start
    CLASSIFIER_LATENCY.observe(clf_elapsed)

    input_verdict = "allow"
    if input_result.verdict == GuardrailVerdict.block:
        if input_result.guard_error:
            # Fail-secure in every mode: an unscanned request is never forwarded.
            INPUT_BLOCKS.inc()
            REQUESTS_TOTAL.labels(verdict="guard_unavailable").inc()
            await audit("guard_error", "n/a")
            logger.warning("GUARD UNAVAILABLE request %s | blocked fail-secure", request_id)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=_error(
                    "Input guardrail unavailable; request blocked.", "guard_unavailable"
                ),
            )
        if settings.enforcement_mode == "monitor":
            input_verdict = "would_block"
            MONITOR_FLAGS.labels(stage="input").inc()
            logger.warning(
                "MONITOR would block request %s | confidence=%.2f | threat=%s | ip=%s",
                request_id,
                input_result.confidence,
                input_result.threat_category.value,
                client_ip,
            )
        else:
            INPUT_BLOCKS.inc()
            REQUESTS_TOTAL.labels(verdict="blocked_input").inc()
            await audit(
                "block",
                "n/a",
                input_confidence=input_result.confidence,
                input_threat=input_result.threat_category.value,
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

    scored = dict(
        input_confidence=input_result.confidence, input_threat=input_result.threat_category.value
    )

    # ── 2 & 3. Forward to backend (canary injected inside forward_to_backend) ──
    payload = body.model_dump(exclude_none=True)
    try:
        backend_response = await backend_call(
            request,
            lambda: forward_to_backend(
                request_payload=payload, request_id=request_id, client=http_client
            ),
        )
    except ClientDisconnectedError:
        REQUESTS_TOTAL.labels(verdict="cancelled").inc()
        discard_token(request_id)
        await audit(input_verdict, "cancelled", **scored)
        logger.info("Client disconnected; backend request %s cancelled", request_id)
        return Response(status_code=499)
    except httpx.TimeoutException:
        discard_token(request_id)
        await audit(input_verdict, "backend_timeout", **scored)
        logger.error("Backend timeout for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Backend LLM timed out.",
        )
    except httpx.HTTPStatusError as exc:
        discard_token(request_id)
        await audit(input_verdict, "backend_error", **scored)
        logger.error("Backend HTTP error %d for request %s", exc.response.status_code, request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Backend returned {exc.response.status_code}.",
        )
    except Exception as exc:
        # Fail-secure: unknown backend errors are surfaced as 502. Type only:
        # exception messages can quote the request or the reply.
        discard_token(request_id)
        await audit(input_verdict, "backend_error", **scored)
        logger.error("Unexpected backend error for request %s: %s", request_id, type(exc).__name__)
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
        await audit(
            input_verdict, "block", output_threat=output_result.threat_category.value, **scored
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
                    "code": "guard_unavailable"
                    if output_result.guard_error
                    else output_result.threat_category.value,
                }
            ).model_dump(),
        )

    if output_result.verdict == GuardrailVerdict.redact:
        final_response = patch_response_content(backend_response, output_result.content)

    # ── 5. Audit + return ─────────────────────────────────────────────────────
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    REQUESTS_TOTAL.labels(verdict=input_verdict if input_verdict != "allow" else "allowed").inc()
    await audit(
        input_verdict,
        output_result.verdict.value,
        output_threat=output_result.threat_category.value,
        **scored,
    )

    # The caller can check that this reply passed an enforcing input guard.
    return JSONResponse(
        content=final_response,
        headers={ENFORCEMENT_HEADER: settings.enforcement_mode},
    )


@router.get("/aegis/readiness")
async def readiness(request: Request):
    """Versioned readiness for callers that must not run unguarded.

    Unlike /health, it says whether input blocking is actually enforced, and
    how many backend requests are still open (so a caller can tell when work
    it abandoned has been cancelled). It exposes no URLs, tokens or prompts.
    """
    settings = get_settings()
    state = request.app.state
    classifier = getattr(state, "classifier", None)
    reasons = readiness_reasons(request)
    return JSONResponse(
        content=ReadinessResponse(
            revision=getattr(state, "revision", "unknown"),
            instance=getattr(state, "instance", ""),
            ready=not reasons,
            reasons=reasons,
            enforcement=settings.enforcement_mode,
            classifier=settings.classifier_type,
            classifier_loaded=bool(classifier and classifier.is_loaded()),
            scan_roles=list(settings.scan_roles),
            input_guard_timeout_ms=round(settings.input_guard_timeout * 1000),
            require_enforcement_header=REQUIRE_ENFORCEMENT_HEADER,
            backend_redirects=getattr(
                getattr(state, "http_client", None), "follow_redirects", False
            )
            is True,
            backend_stream=settings.backend_stream,
            # Closing a non-streamed request often leaves llama-server generating.
            cancel_on_disconnect=settings.backend_stream,
            audit_prompt_text=not settings.redact_prompts_in_logs,
            audit_retention_days=settings.audit_retention_days,
            backend_inflight=getattr(state, "backend_inflight", 0),
        ).model_dump(by_alias=True),
        headers={"cache-control": "no-store"},
    )


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


@router.get("/v1/models")
async def list_models(request: Request, http_client: httpx.AsyncClient = Depends(get_http_client)):
    """Pass the backend's model list through, so OpenAI clients can probe Aegis."""
    try:
        return JSONResponse(
            content=await backend_call(request, lambda: fetch_backend_models(http_client))
        )
    except ClientDisconnectedError:
        return Response(status_code=499)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Backend LLM timed out."
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Backend returned {exc.response.status_code}.",
        )
    except Exception as exc:
        logger.error("Unexpected backend error listing models: %s", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Backend error.")


@router.get("/metrics")
async def metrics(request: Request):
    """Prometheus text metrics endpoint (requires ADMIN_TOKEN when one is set)."""
    _require_admin(request, required=False)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/audit/logs")
async def audit_logs(request: Request, limit: int = 50):
    """Return the most recent audit log entries. Requires ADMIN_TOKEN."""
    _require_admin(request, required=True)
    if limit > 500:
        limit = 500
    rows = await get_recent_logs(limit=limit)
    return {"count": len(rows), "logs": rows}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_text(body: ChatCompletionRequest) -> str:
    """
    Extract a single string from the request for audit logging: the messages
    the classifier scanned (SCAN_ROLES), so a long trusted system prompt can't
    push the untrusted text out. Truncated to 1000 chars to keep the DB tidy.
    """
    roles = get_settings().scan_roles
    parts = [f"[{m.role.value}]: {m.content}" for m in body.messages if m.role.value in roles]
    full = "\n".join(parts)
    return full[:1000]
