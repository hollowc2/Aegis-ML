"""
app/guardrails/input_guard.py
=============================
Input guardrail — runs every incoming chat request through the active classifier
and returns a verdict (allow / block) before the request ever touches the LLM.

Phase 3 enhancements:
  - TextPreprocessor runs before classification (Unicode normalization, invisible
    char detection).  The cleaned text is sent to the classifier; the raw text
    is preserved in the audit log.
  - Invisible-char / RTL-override bias: when the preprocessor flags problematic
    characters, malicious_prob is bumped by PREPROCESS_INVISIBLE_CHAR_BIAS
    before threshold comparison (configurable, default 0.15).
  - ML threat category: if the classifier returns threat_category_probs (hf2/
    onnx2), it is used directly instead of the heuristic _infer_threat_category.
    The heuristic is kept as a fallback for Phase 1/2 classifiers.

Fail-secure: any exception during classification → block the request.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.models.schemas import (
    ChatCompletionRequest,
    GuardrailVerdict,
    InputGuardrailResult,
    ThreatCategory,
)

if TYPE_CHECKING:
    from app.classifiers.sklearn_classifier import SklearnClassifier
    from app.classifiers.hf_classifier import HFClassifier

logger = logging.getLogger(__name__)


def _extract_text(request: ChatCompletionRequest) -> str:
    """
    Concatenate all message contents into a single string for classification.
    We include role prefixes so the classifier can catch role-manipulation attacks
    like "Ignore previous instructions, you are now…" in a user turn.
    """
    parts = []
    for msg in request.messages:
        parts.append(f"[{msg.role.value}]: {msg.content}")
    return "\n".join(parts)


async def run_input_guardrail(
    request: ChatCompletionRequest,
    classifier: "SklearnClassifier | HFClassifier",
) -> InputGuardrailResult:
    """
    Classify the incoming request.

    Returns:
        InputGuardrailResult with verdict=block if malicious, allow otherwise.
    """
    settings = get_settings()
    text = _extract_text(request)

    try:
        result = await _classify(text, classifier, settings)
    except Exception as exc:
        # Fail-secure: any error in the classifier pipeline blocks the request
        logger.error("Input guardrail classification error (fail-secure block): %s", exc)
        return InputGuardrailResult(
            verdict=GuardrailVerdict.block,
            is_malicious=True,
            confidence=1.0,
            threat_category=ThreatCategory.prompt_injection,
            reason=f"Classifier error — fail-secure block: {exc}",
        )

    return result


async def _classify(
    text: str,
    classifier,
    settings,
) -> InputGuardrailResult:
    """Delegate to whichever classifier is loaded and interpret the result."""
    threshold = settings.confidence_threshold

    # ── Phase 3: Unicode preprocessing ───────────────────────────────────────
    preprocess_flags: dict = {}
    cleaned_text = text
    if settings.enable_text_preprocessing:
        try:
            from app.classifiers.text_preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()
            cleaned_text, preprocess_flags = preprocessor.preprocess(text)
        except Exception as exc:
            logger.warning("TextPreprocessor failed (continuing with raw text): %s", exc)

    # Both classifier types expose a unified async .predict(text) → dict interface
    prediction = await classifier.predict(cleaned_text)

    malicious_prob: float = prediction.get("malicious_prob", 0.0)
    benign_prob: float = prediction.get("benign_prob", 1.0 - malicious_prob)

    # ── Phase 3: invisible-char bias ──────────────────────────────────────────
    # If preprocessing detected invisible or RTL override chars, bump the
    # malicious probability.  These chars are a strong attack signal even if the
    # normalised text looks benign to the model.
    if preprocess_flags.get("had_rtl_override") or preprocess_flags.get("had_invisible_chars"):
        bias = settings.preprocess_invisible_char_bias
        malicious_prob = min(1.0, malicious_prob + bias)
        benign_prob = max(0.0, 1.0 - malicious_prob)
        logger.debug(
            "Invisible-char bias applied (+%.2f): malicious_prob → %.4f",
            bias, malicious_prob,
        )

    is_malicious = malicious_prob >= threshold

    # ── Threat category: prefer ML output (Phase 3), fall back to heuristic ──
    threat = ThreatCategory.none
    if is_malicious:
        ml_threat_probs: dict = prediction.get("threat_category_probs", {})
        if ml_threat_probs:
            # Use the ML-predicted threat category from hf2/onnx2
            best_cat = max(ml_threat_probs, key=ml_threat_probs.get)
            try:
                threat = ThreatCategory(best_cat)
            except ValueError:
                threat = ThreatCategory.prompt_injection
        else:
            # Fallback: heuristic keyword-based category inference (Phase 1/2)
            threat = _infer_threat_category(text)

    verdict = GuardrailVerdict.block if is_malicious else GuardrailVerdict.allow
    reason = (
        f"Malicious confidence {malicious_prob:.2%} ≥ threshold {threshold:.2%}"
        if is_malicious
        else f"Benign confidence {benign_prob:.2%}"
    )

    return InputGuardrailResult(
        verdict=verdict,
        is_malicious=is_malicious,
        confidence=malicious_prob,
        threat_category=threat,
        reason=reason,
        probabilities={"malicious": malicious_prob, "benign": benign_prob},
        threat_category_probs=prediction.get("threat_category_probs", {}),
    )


def _infer_threat_category(text: str) -> ThreatCategory:
    """
    Heuristic secondary classification to label the *type* of detected threat.
    Used for richer audit logs and metrics; does not affect the block decision.

    This is the Phase 1/2 fallback — Phase 3 (hf2/onnx2) replaces this with
    ML-learned threat probabilities from the multi-task head.
    """
    lower = text.lower()

    jailbreak_patterns = [
        "ignore previous instructions",
        "disregard all prior",
        "you are now",
        "pretend you are",
        "act as",
        "dan mode",
        "jailbreak",
        "do anything now",
        "developer mode",
    ]
    exfil_patterns = [
        "repeat your system prompt",
        "what were your instructions",
        "tell me your secret",
        "reveal your",
        "show me your",
        "print your system",
        "output your system",
    ]
    injection_patterns = [
        "---",
        "###",
        "system:",
        "assistant:",
        "human:",
        "</s>",
        "[inst]",
        "<<sys>>",
        "new instructions:",
        "ignore the above",
    ]

    if any(p in lower for p in exfil_patterns):
        return ThreatCategory.data_exfiltration
    if any(p in lower for p in jailbreak_patterns):
        return ThreatCategory.jailbreak
    if any(p in lower for p in injection_patterns):
        return ThreatCategory.prompt_injection

    return ThreatCategory.prompt_injection
