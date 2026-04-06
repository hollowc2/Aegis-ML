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

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────────────────────


def create_limiter() -> Limiter:
    """Create a slowapi Limiter keyed by client IP address."""
    return Limiter(key_func=get_remote_address)


def setup_rate_limiter(app: FastAPI, limiter: Limiter) -> None:
    """Attach the limiter and its error handler to the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request Logging Middleware ────────────────────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request/response logging.
    Adds an X-Request-ID header to every response for traceability.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request_id to the request state so route handlers can read it
        request.state.request_id = request_id

        logger.info(
            "→ %s %s  [id=%s  ip=%s]",
            request.method,
            request.url.path,
            request_id,
            request.client.host if request.client else "unknown",
        )

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "← %d  %s %s  %.1f ms  [id=%s]",
            response.status_code,
            request.method,
            request.url.path,
            elapsed_ms,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Powered-By"] = "Aegis-ML"
        return response
