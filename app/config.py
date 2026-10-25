"""
app/config.py
=============
Centralised, type-safe configuration using Pydantic v2 BaseSettings.
All values can be set via environment variables or a .env file.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Role = Literal["system", "user", "assistant", "tool"]


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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
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
    classifier_type: Literal["sklearn", "onnx2", "cascade2"] = Field(
        default="sklearn",
        description=(
            "Which classifier to use: 'sklearn' (fast, bundled), 'onnx2' (mDeBERTa "
            "via ONNX Runtime) or 'cascade2' (sklearn fast path + onnx2 slow path)"
        ),
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum injection probability to block a request. Unset = use the "
            "threshold tuned for the loaded model at training time (see "
            "docs/BENCHMARKS.md), falling back to 0.70 for models without one."
        ),
    )

    # ── Model Paths ───────────────────────────────────────────────────────────
    sklearn_model_path: str = Field(
        default="models/sklearn_classifier.joblib",
        description="Path to the trained scikit-learn model artifact",
    )
    onnx2_model_path: str = Field(
        default="models/hf2_classifier_onnx",
        description="Path to the ONNX-exported mDeBERTa model directory",
    )
    onnx2_use_int8: bool = Field(
        default=False,
        description=(
            "Serve the INT8-quantised ONNX graph for onnx2/cascade2 (smaller, ~5 points "
            "lower recall). Either precision falls back to the other if its file is absent."
        ),
    )

    # ── Text Preprocessing ────────────────────────────────────────────────────
    enable_text_preprocessing: bool = Field(
        default=True,
        description=(
            "Enable Unicode normalization + invisible character detection "
            "before classification (applies to all classifier types)"
        ),
    )
    preprocess_invisible_char_bias: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Extra malicious_prob added when invisible/RTL-override chars are detected. "
            "Set to 0.0 to disable the bias (e.g. for multilingual environments)."
        ),
    )
    max_scan_segments: int = Field(
        default=64,
        ge=1,
        description=(
            "Maximum number of message windows classified per request. Each message is "
            "scored separately and long messages are split into overlapping windows; "
            "requests needing more windows than this are blocked (fail-secure)."
        ),
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

    # ── Enforcement ───────────────────────────────────────────────────────────
    enforcement_mode: Literal["enforce", "monitor"] = Field(
        default="enforce",
        description=(
            "'enforce' blocks and redacts. 'monitor' only records what would have "
            "been blocked or redacted and returns the backend response unchanged, "
            "so false positives can be measured on real traffic before enforcing."
        ),
    )
    scan_roles: Annotated[list[Role], NoDecode] = Field(
        default=["system", "user", "assistant", "tool"],
        description=(
            "Message roles the input classifier scores. Narrow it (e.g. to 'user') "
            "when the caller writes the system prompt and history itself and only "
            "user messages carry untrusted text. Comma-separated."
        ),
    )

    # ── Access control ────────────────────────────────────────────────────────
    admin_token: str = Field(
        default="",
        description=(
            "Bearer token for /audit/logs and /metrics. Unset: /audit/logs is "
            "disabled and /metrics is open."
        ),
    )
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default=[],
        description="Browser origins allowed to call the API (CORS), comma-separated. Empty = none.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./logs/aegis_audit.db",
        description="SQLite database URL for audit logs",
    )
    audit_retention_days: float = Field(
        default=0,
        ge=0,
        description="Delete audit entries older than this many days (0 = keep forever)",
    )

    # ── Demo UI ───────────────────────────────────────────────────────────────
    demo_port: int = Field(default=7860, ge=1, le=65535)
    aegis_api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the running Aegis-ML service (used by the Gradio demo)",
    )

    @field_validator("scan_roles", "cors_allow_origins", mode="before")
    @classmethod
    def split_comma_list(cls, v):
        """Accept "user,system" or a JSON list from the environment."""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json

                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
