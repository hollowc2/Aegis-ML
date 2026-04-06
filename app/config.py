"""
app/config.py
=============
Centralised, type-safe configuration using Pydantic v2 BaseSettings.
All values can be set via environment variables or a .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service ──────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    redact_prompts_in_logs: bool = Field(
        default=False,
        description="Mask prompt text in audit logs (GDPR-friendly)",
    )

    # ── Backend LLM ──────────────────────────────────────────────────────────
    backend_url: str = Field(
        default="http://localhost:8080/v1/chat/completions",
        description="OpenAI-compatible chat completions URL of the backend LLM",
    )
    backend_api_key: str = Field(
        default="",
        description="API key for the backend (empty = no auth, for local llama.cpp)",
    )
    # Timeout in seconds for forwarding requests to the backend
    backend_timeout: float = Field(default=120.0, ge=1.0)

    # ── Classifier ───────────────────────────────────────────────────────────
    classifier_type: Literal["sklearn", "hf", "onnx", "cascade"] = Field(
        default="sklearn",
        description=(
            "Which classifier to use: 'sklearn' (Phase 1), 'hf' (Phase 2), "
            "'onnx' (Phase 2 via ONNX Runtime), or 'cascade' (sklearn fast path + ONNX slow path)"
        ),
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum injection probability to block a request",
    )

    # ── Model Paths ───────────────────────────────────────────────────────────
    sklearn_model_path: str = Field(
        default="models/sklearn_classifier.joblib",
        description="Path to the trained scikit-learn model artifact",
    )
    hf_model_path: str = Field(
        default="models/hf_classifier",
        description="Path to the fine-tuned HuggingFace model directory",
    )
    onnx_model_path: str = Field(
        default="models/hf_classifier_onnx",
        description="Path to the ONNX-exported model directory (produced by export_onnx.py)",
    )

    # ── Cascade Classifier ────────────────────────────────────────────────────
    cascade_sk_low_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Cascade: sklearn malicious_prob ≤ this is treated as benign without "
            "calling the ONNX model (fast path)"
        ),
    )
    cascade_sk_high_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Cascade: sklearn malicious_prob ≥ this is treated as malicious without "
            "calling the ONNX model (fast path)"
        ),
    )

    # ── Canary Tokens ─────────────────────────────────────────────────────────
    canary_token_length: int = Field(
        default=32,
        ge=16,
        le=128,
        description="Length of per-request canary tokens injected into system prompts",
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute per IP",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./logs/aegis_audit.db",
        description="SQLite database URL for audit logs",
    )

    # ── Demo UI ───────────────────────────────────────────────────────────────
    demo_port: int = Field(default=7860, ge=1, le=65535)
    aegis_api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the running Aegis-ML service (used by the Gradio demo)",
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
