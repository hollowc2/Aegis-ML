"""
app/main.py
===========
FastAPI application entry point.

Manages the app lifespan (startup / shutdown):
  - Initialises the audit database
  - Loads the classifier (sklearn or HF, via CLASSIFIER_TYPE env var)
  - Creates the shared httpx.AsyncClient
  - Mounts all routes and middleware
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import RequestLoggingMiddleware, create_limiter, setup_rate_limiter
from app.api.routes import router
from app.config import get_settings
from app.models.database import init_db

# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic using the modern FastAPI lifespan API."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("═══════════════════════════════════════════════")
    logger.info("  Aegis-ML LLM Firewall  v1.0.0  starting up  ")
    logger.info("═══════════════════════════════════════════════")
    logger.info("Classifier type : %s", settings.classifier_type)
    logger.info("Backend URL     : %s", settings.backend_url)
    logger.info("Threshold       : %.2f", settings.confidence_threshold)
    if settings.classifier_type == "cascade":
        logger.info(
            "Cascade band    : sk_low=%.2f  sk_high=%.2f",
            settings.cascade_sk_low_threshold,
            settings.cascade_sk_high_threshold,
        )

    # ── Init DB ────────────────────────────────────────────────────────────────
    await init_db()

    # ── Load classifier ────────────────────────────────────────────────────────
    classifier = _load_classifier(settings)
    app.state.classifier = classifier

    # ── Shared HTTP client ─────────────────────────────────────────────────────
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    logger.info("Aegis-ML is armed and operational.")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down Aegis-ML...")
    await app.state.http_client.aclose()
    logger.info("GG.")


def _load_classifier(settings):
    """Load the appropriate classifier based on CLASSIFIER_TYPE."""
    if settings.classifier_type == "sklearn":
        from app.classifiers.sklearn_classifier import SklearnClassifier
        clf = SklearnClassifier(model_path=settings.sklearn_model_path)
        try:
            clf.load()
        except FileNotFoundError as exc:
            logger.warning(
                "scikit-learn model not found (%s). "
                "Run training first: python -m training.phase1_sklearn.train",
                exc,
            )
            # Don't crash — the health endpoint will report degraded state
            # and all requests will be blocked (fail-secure)
        return clf

    elif settings.classifier_type == "hf":
        from app.classifiers.hf_classifier import HFClassifier
        clf = HFClassifier(model_path=settings.hf_model_path)
        try:
            clf.load()
        except FileNotFoundError as exc:
            logger.warning(
                "HuggingFace model not found (%s). "
                "Run training first: python -m training.phase2_hf.train",
                exc,
            )
        return clf

    elif settings.classifier_type == "onnx":
        from app.classifiers.onnx_classifier import ONNXClassifier
        clf = ONNXClassifier(model_path=settings.onnx_model_path)
        try:
            clf.load()
        except (FileNotFoundError, ImportError) as exc:
            logger.warning(
                "ONNX model not loaded (%s). "
                "Run: python -m training.phase2_hf.export_onnx",
                exc,
            )
        return clf

    elif settings.classifier_type == "cascade":
        from app.classifiers.sklearn_classifier import SklearnClassifier
        from app.classifiers.onnx_classifier import ONNXClassifier
        from app.classifiers.cascade_classifier import CascadeClassifier

        sklearn_clf = SklearnClassifier(model_path=settings.sklearn_model_path)
        onnx_clf = ONNXClassifier(model_path=settings.onnx_model_path)

        try:
            sklearn_clf.load()
        except FileNotFoundError as exc:
            logger.warning(
                "Cascade: scikit-learn model not found (%s). "
                "Run: python -m training.phase1_sklearn.train",
                exc,
            )
        try:
            onnx_clf.load()
        except (FileNotFoundError, ImportError) as exc:
            logger.warning(
                "Cascade: ONNX model not loaded (%s). "
                "Run: python -m training.phase2_hf.export_onnx",
                exc,
            )

        clf = CascadeClassifier(
            sklearn_clf=sklearn_clf,
            slow_clf=onnx_clf,
            low_threshold=settings.cascade_sk_low_threshold,
            high_threshold=settings.cascade_sk_high_threshold,
        )
        logger.info(
            "Cascade thresholds: sk_low=%.2f  sk_high=%.2f",
            settings.cascade_sk_low_threshold,
            settings.cascade_sk_high_threshold,
        )
        return clf

    else:
        raise ValueError(
            f"Unknown CLASSIFIER_TYPE: {settings.classifier_type!r}. "
            "Must be 'sklearn', 'hf', 'onnx', or 'cascade'."
        )


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Aegis-ML — LLM Firewall",
        description=(
            "Real-time adversarial prompt injection detector and reverse proxy. "
            "Protects LLMs against prompt injections, jailbreaks, and data-exfiltration attacks."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (permissive for demo; tighten in production) ─────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ────────────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Rate limiting ──────────────────────────────────────────────────────────
    limiter = create_limiter()
    setup_rate_limiter(app, limiter)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    return app


# ── Singleton app instance (imported by uvicorn) ──────────────────────────────
app = create_app()


# ── CLI entry point ───────────────────────────────────────────────────────────

def run() -> None:
    """Called by the 'aegis-serve' CLI script defined in pyproject.toml."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()
