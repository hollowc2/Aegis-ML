"""
app/proxy/llm_proxy.py
======================
Async reverse proxy that forwards an OpenAI-compatible chat request
to the local llama.cpp backend (or any OpenAI-compatible endpoint).

Canary tokens are injected here — after the input guardrail approves
the request but before it is forwarded.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import get_settings
from app.guardrails.canary import generate_token, inject_into_system_prompt

logger = logging.getLogger(__name__)


async def forward_to_backend(
    request_payload: dict[str, Any],
    request_id: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Inject a canary token, forward the request to the backend LLM,
    and return the raw response JSON.

    Args:
        request_payload: The validated chat completion request as a dict.
        request_id:      Unique ID for this request (used for canary tracking).
        client:          Shared httpx.AsyncClient (managed by the app lifespan).

    Returns:
        Backend response as a Python dict (parsed from JSON).

    Raises:
        httpx.HTTPStatusError: On non-2xx backend responses.
        httpx.TimeoutException: If backend takes too long.
    """
    settings = get_settings()

    # ── Step 1: Inject canary token ───────────────────────────────────────────
    canary_token = generate_token(request_id)
    messages = request_payload.get("messages", [])
    # inject_into_system_prompt works with list-of-dicts (after .model_dump())
    patched_messages = inject_into_system_prompt(messages, canary_token)
    patched_payload = {**request_payload, "messages": patched_messages}

    logger.debug(
        "Forwarding request %s to backend (canary injected, %d messages)",
        request_id,
        len(patched_messages),
    )

    # ── Step 2: Build headers ──────────────────────────────────────────────────
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.backend_api_key:
        headers["Authorization"] = f"Bearer {settings.backend_api_key}"

    # ── Step 3: POST to backend ────────────────────────────────────────────────
    if settings.backend_stream and not patched_payload.get("stream"):
        return await _forward_streamed(client, patched_payload, headers, request_id)

    response = await client.post(
        settings.backend_url,
        json=patched_payload,
        headers=headers,
        timeout=settings.backend_timeout,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    logger.debug(
        "Backend responded with status %d for request %s", response.status_code, request_id
    )
    return data


async def _forward_streamed(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
) -> dict[str, Any]:
    """Stream the reply from the backend and assemble a chat.completion dict.

    The caller still gets one non-streaming response, but the backend writes
    every token to the open connection, so closing it (the caller left)
    makes llama-server cancel the task at its next write.
    """
    settings = get_settings()
    parts: list[str] = []
    finish_reason: str | None = None
    meta: dict[str, Any] = {}
    usage: dict[str, Any] | None = None
    received = 0
    async with client.stream(
        "POST",
        settings.backend_url,
        json={**payload, "stream": True},
        headers=headers,
        timeout=settings.backend_timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            received += len(line) + 1
            if received > settings.backend_max_response_bytes:
                raise ValueError("backend reply too large")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or "error" in chunk:
                raise ValueError("backend stream error")
            for key in ("id", "created", "model", "system_fingerprint"):
                if key in chunk and key not in meta:
                    meta[key] = chunk[key]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
    logger.debug("Backend stream complete for request %s", request_id)
    return {
        **meta,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(parts)},
                "finish_reason": finish_reason,
            }
        ],
        **({"usage": usage} if usage is not None else {}),
    }


def backend_models_url(backend_url: str) -> str:
    """The backend's /v1/models URL, derived from its chat completions URL."""
    url = backend_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url.removesuffix("/chat/completions") + "/models"
    return url + "/v1/models"


async def fetch_backend_models(client: httpx.AsyncClient) -> dict[str, Any]:
    """GET the backend's model list (OpenAI clients probe it before chatting)."""
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.backend_api_key:
        headers["Authorization"] = f"Bearer {settings.backend_api_key}"
    response = await client.get(
        backend_models_url(settings.backend_url),
        headers=headers,
        timeout=settings.backend_timeout,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def extract_assistant_content(response_data: dict[str, Any]) -> str:
    """
    Pull the assistant's reply text out of an OpenAI-style response dict.
    Returns an empty string if the structure is unexpected.
    """
    try:
        choices = response_data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def patch_response_content(response_data: dict[str, Any], new_content: str) -> dict[str, Any]:
    """
    Return a copy of *response_data* with the assistant reply replaced by *new_content*.
    Used by the output guardrail to swap in redacted content.
    """
    import copy

    patched = copy.deepcopy(response_data)
    try:
        patched["choices"][0]["message"]["content"] = new_content
    except (KeyError, IndexError, TypeError):
        pass
    return patched
