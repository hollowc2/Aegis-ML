"""
app/api/middleware.py
=====================
Custom middleware and rate-limiting setup for the Aegis-ML FastAPI app.

- RequestLoggingMiddleware: structured request/response logging
- setup_rate_limiter: slowapi limiter factory
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────────────────────


def create_limiter(rate_limit_per_minute: int | None = None) -> Limiter:
    """Create a slowapi Limiter keyed by client IP address."""
    rate = rate_limit_per_minute or get_settings().rate_limit_per_minute
    return Limiter(
        key_func=get_remote_address,
        default_limits=[f"{rate}/minute"],
        key_style="url",
    )


def setup_rate_limiter(app: FastAPI, limiter: Limiter) -> None:
    """Attach the limiter and its error handler to the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request Logging Middleware ────────────────────────────────────────────────


class RequestLoggingMiddleware:
    """
    Structured request/response logging.
    Adds an X-Request-ID header to every response for traceability.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        method = scope.get("method", "")
        path = scope.get("path", "")
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        # Starlette's Request.state reads from this ASGI scope dictionary.
        scope.setdefault("state", {})["request_id"] = request_id

        logger.info(
            "→ %s %s  [id=%s  ip=%s]",
            method,
            path,
            request_id,
            client_ip,
        )

        status_code = 500

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
                headers.append("X-Powered-By", "Aegis-ML")
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "← %d  %s %s  %.1f ms  [id=%s]",
                status_code,
                method,
                path,
                elapsed_ms,
                request_id,
            )
