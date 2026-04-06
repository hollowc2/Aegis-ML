"""
app/models/schemas.py
=====================
Pydantic v2 schemas for:
  - OpenAI-compatible chat completions API (request & response)
  - Internal guardrail results
  - Audit log entries
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Compatible Schemas
# ─────────────────────────────────────────────────────────────────────────────


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Role
    content: str = Field(default="")
    name: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def coerce_content(cls, v: Any) -> str:
        """
        Normalize OpenAI multimodal content to a plain string.
        Gradio 5.x and some OpenAI clients send content as a list of parts:
          [{"type": "text", "text": "..."}, ...]
        We extract all text parts and join them so downstream code always
        sees a plain string.  None is coerced to "".
        """
        if isinstance(v, list):
            return " ".join(
                item.get("text", "")
                for item in v
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return "" if v is None else str(v)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request body."""

    model: str = Field(default="local-model")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    stop: list[str] | str | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Passthrough any extra fields the backend may need
    model_config = {"extra": "allow"}


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completions response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "local-model"
    choices: list[ChatChoice]
    usage: UsageStats = Field(default_factory=UsageStats)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Guardrail Schemas
# ─────────────────────────────────────────────────────────────────────────────


class ThreatCategory(str, Enum):
    """Categories of detected threats."""

    prompt_injection = "prompt_injection"
    jailbreak = "jailbreak"
    data_exfiltration = "data_exfiltration"
    canary_leak = "canary_leak"
    pii_leak = "pii_leak"
    harmful_content = "harmful_content"
    none = "none"


class GuardrailVerdict(str, Enum):
    allow = "allow"
    block = "block"
    redact = "redact"


class InputGuardrailResult(BaseModel):
    """Result from the input guardrail classifier."""

    verdict: GuardrailVerdict
    is_malicious: bool
    confidence: float = Field(ge=0.0, le=1.0)
    threat_category: ThreatCategory = ThreatCategory.none
    reason: str = ""
    # Raw per-class probabilities (for logging / debugging)
    probabilities: dict[str, float] = Field(default_factory=dict)


class OutputGuardrailResult(BaseModel):
    """Result from the output guardrail filters."""

    verdict: GuardrailVerdict
    is_safe: bool
    threat_category: ThreatCategory = ThreatCategory.none
    reason: str = ""
    # Cleaned / redacted content (same as original when nothing was redacted)
    content: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Audit Log Schema
# ─────────────────────────────────────────────────────────────────────────────


class AuditEntry(BaseModel):
    """One row in the audit log database."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    client_ip: str = ""
    # Input side
    input_text: str = ""          # may be redacted per config
    input_verdict: str = "allow"
    input_confidence: float = 0.0
    input_threat: str = "none"
    # Output side
    output_verdict: str = "allow"
    output_threat: str = "none"
    # Timing
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# API Response Helpers
# ─────────────────────────────────────────────────────────────────────────────


class BlockedResponse(BaseModel):
    """Returned to the client when a request is blocked."""

    error: dict[str, Any] = Field(
        default_factory=lambda: {
            "message": "Request blocked by Aegis-ML guardrails.",
            "type": "guardrail_violation",
            "code": "prompt_injection_detected",
        }
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"] = "ok"
    classifier: str = "unknown"
    classifier_loaded: bool = False
    version: str = "1.0.0"
