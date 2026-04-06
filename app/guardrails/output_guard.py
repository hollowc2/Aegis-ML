"""
app/guardrails/output_guard.py
==============================
Output guardrail — applied to every LLM response before it reaches the client.

Checks (in order):
  1. Canary token leak detection (injection success indicator)
  2. PII pattern detection (SSN, credit card, email, phone, etc.)
  3. Basic harmful content keyword filter
  4. Redaction of any detected PII (optional soft action)

Fail-secure: any exception → block the response.
"""

from __future__ import annotations

import logging
import re

from app.guardrails.canary import consume_token
from app.models.schemas import (
    GuardrailVerdict,
    OutputGuardrailResult,
    ThreatCategory,
)

logger = logging.getLogger(__name__)

# ── PII Patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # (name, pattern, replacement)
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (
        "Credit Card",
        re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
        "[CARD REDACTED]",
    ),
    (
        "Email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL REDACTED]",
    ),
    (
        "Phone (US)",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE REDACTED]",
    ),
    (
        "IPv4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP REDACTED]",
    ),
    (
        "AWS Key",
        re.compile(r"\b(AKIA|AIPA|AIKA|ASIA)[A-Z0-9]{16}\b"),
        "[AWS_KEY REDACTED]",
    ),
    (
        "Private Key Header",
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
        "[PRIVATE_KEY REDACTED]",
    ),
]

# ── Harmful Content Keywords ──────────────────────────────────────────────────
# Keep this list minimal — its job is to catch obvious output that the model
# should never produce, not to be an aggressive content filter.

_HARMFUL_KEYWORDS: list[str] = [
    "how to make a bomb",
    "step-by-step guide to poison",
    "synthesize methamphetamine",
    "child sexual",
    "csam",
    "detailed instructions for murder",
]


# ── Main function ─────────────────────────────────────────────────────────────


async def run_output_guardrail(
    content: str,
    request_id: str,
) -> OutputGuardrailResult:
    """
    Apply all output guardrails to *content*.

    Returns:
        OutputGuardrailResult — content may be redacted even when verdict=allow.
    """
    try:
        return await _check(content, request_id)
    except Exception as exc:
        logger.error("Output guardrail error (fail-secure block): %s", exc)
        return OutputGuardrailResult(
            verdict=GuardrailVerdict.block,
            is_safe=False,
            threat_category=ThreatCategory.prompt_injection,
            reason=f"Output guardrail error — fail-secure block: {exc}",
            content="[RESPONSE BLOCKED]",
        )


async def _check(content: str, request_id: str) -> OutputGuardrailResult:
    """Run all checks and return the combined result."""

    # 1. Canary token leak check
    canary_result = _check_canary(content, request_id)
    if canary_result is not None:
        return canary_result

    # 2. PII detection + redaction
    redacted_content, pii_found = _redact_pii(content)
    if pii_found:
        logger.warning(
            "PII detected in LLM output for request %s — redacting", request_id
        )
        return OutputGuardrailResult(
            verdict=GuardrailVerdict.redact,
            is_safe=True,  # Allow through but with redacted content
            threat_category=ThreatCategory.pii_leak,
            reason="PII detected and redacted from model output.",
            content=redacted_content,
        )

    # 3. Harmful content check
    harm_reason = _check_harmful(content)
    if harm_reason:
        logger.warning(
            "Harmful content detected in LLM output for request %s: %s",
            request_id,
            harm_reason,
        )
        return OutputGuardrailResult(
            verdict=GuardrailVerdict.block,
            is_safe=False,
            threat_category=ThreatCategory.harmful_content,
            reason=harm_reason,
            content="[RESPONSE BLOCKED — HARMFUL CONTENT DETECTED]",
        )

    # All checks passed
    return OutputGuardrailResult(
        verdict=GuardrailVerdict.allow,
        is_safe=True,
        content=content,
    )


def _check_canary(content: str, request_id: str) -> OutputGuardrailResult | None:
    """
    Check whether the canary token for this request appears in the model output.
    Returns a block result if leaked, None if clean.
    """
    token = consume_token(request_id)
    if token is None:
        # No canary token registered (shouldn't happen in normal flow)
        return None

    if token in content:
        logger.warning(
            "CANARY TOKEN LEAK detected in request %s — injection success, blocking.",
            request_id,
        )
        return OutputGuardrailResult(
            verdict=GuardrailVerdict.block,
            is_safe=False,
            threat_category=ThreatCategory.canary_leak,
            reason=(
                "Canary token appeared in model output — "
                "possible successful prompt injection. Response blocked."
            ),
            content="[RESPONSE BLOCKED — INJECTION DETECTED]",
        )

    return None


def _redact_pii(content: str) -> tuple[str, bool]:
    """Apply all PII patterns to *content*, returning (redacted_text, was_pii_found)."""
    found = False
    result = content
    for name, pattern, replacement in _PII_PATTERNS:
        new_result, n_subs = pattern.subn(replacement, result)
        if n_subs:
            found = True
            logger.debug("Redacted %d %s pattern(s) from output", n_subs, name)
            result = new_result
    return result, found


def _check_harmful(content: str) -> str | None:
    """
    Return a reason string if harmful keywords are found, else None.
    Case-insensitive match.
    """
    lower = content.lower()
    for kw in _HARMFUL_KEYWORDS:
        if kw in lower:
            return f"Harmful keyword detected: '{kw}'"
    return None
