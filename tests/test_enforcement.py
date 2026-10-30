"""
tests/test_enforcement.py
=========================
Guarantees cobblebob's public Q&A relies on before it trusts Aegis:

- a caller that requires enforcement is refused (503) in monitor mode or with
  no classifier, before any classification or backend work;
- a failing or slow classifier blocks (guard_unavailable) in every mode;
- /aegis/readiness tells enforcement apart from monitor mode and shows open
  backend work, without exposing the backend URL or tokens;
- backend redirects are errors, never followed;
- a caller disconnect cancels the backend request and closes its connection
  (real sockets, real uvicorn);
- with REDACT_PROMPTS_IN_LOGS, synthetic visitor/answer canaries reach neither
  the logs nor the audit database on any path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes
from app.main import create_app

REQUIRE = {"x-aegis-require-enforcement": "enforce"}
QUESTION_CANARY = "QCANARY-7d1f0c"
ANSWER_CANARY = "ACANARY-93be42"
SKLEARN_MODEL = Path("models/sklearn_classifier.joblib")


def _settings(**overrides):
    from app.config import Settings

    return Settings(_env_file=None, **overrides)


def _classifier(prob=0.01, loaded=True, error=None, delay=0.0):
    clf = MagicMock()
    clf.is_loaded.return_value = loaded

    async def predict(text):
        if delay:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        return {"label": "benign", "malicious_prob": prob, "benign_prob": 1 - prob}

    clf.predict = AsyncMock(side_effect=predict)
    return clf


def _backend(reply="I'm mining.", status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}]
    }
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    client.follow_redirects = False
    return client


def _body(question="What are you doing?"):
    return {
        "model": "local",
        "messages": [
            {"role": "system", "content": "You are CobbleBob. Facts: {}"},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "max_tokens": 64,
    }


async def _post(settings, classifier, backend, body=None, headers=None, audit=None):
    app = create_app()
    targets = ("app.api.routes", "app.guardrails.input_guard", "app.proxy.llm_proxy")
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(f"{target}.get_settings", return_value=settings))
        stack.enter_context(patch("app.api.routes.log_audit_entry", new=audit or AsyncMock()))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            app.state.classifier = classifier
            app.state.http_client = backend
            app.state.limiter = MagicMock()
            return await c.post("/v1/chat/completions", json=body or _body(), headers=headers)


class TestRequiredEnforcement:
    @pytest.mark.asyncio
    async def test_monitor_mode_refuses_before_classifying_or_forwarding(self):
        clf, backend = _classifier(prob=0.99), _backend()
        resp = await _post(_settings(enforcement_mode="monitor"), clf, backend, headers=REQUIRE)

        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "enforcement_unavailable"
        clf.predict.assert_not_awaited()
        backend.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unloaded_classifier_refuses(self):
        backend = _backend()
        resp = await _post(_settings(), _classifier(loaded=False), backend, headers=REQUIRE)

        assert resp.status_code == 503
        backend.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enforced_benign_question_is_answered_and_marked(self):
        backend = _backend()
        resp = await _post(_settings(scan_roles="user"), _classifier(), backend, headers=REQUIRE)

        assert resp.status_code == 200
        assert resp.headers["x-aegis-enforcement"] == "enforce"
        backend.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flagged_input_never_reaches_the_backend(self):
        backend = _backend()
        resp = await _post(
            _settings(scan_roles="user"), _classifier(prob=0.99), backend, headers=REQUIRE
        )

        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "prompt_injection_detected"
        backend.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_monitor_reply_is_marked_monitor_without_the_header(self):
        resp = await _post(_settings(enforcement_mode="monitor"), _classifier(0.99), _backend())

        assert resp.status_code == 200
        assert resp.headers["x-aegis-enforcement"] == "monitor"

    @pytest.mark.asyncio
    async def test_only_the_user_message_is_scanned(self):
        clf = _classifier()
        await _post(_settings(scan_roles="user"), clf, _backend(), headers=REQUIRE)

        scanned = [call.args[0] for call in clf.predict.await_args_list]
        assert scanned == ["What are you doing?"]


class TestGuardFailure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["enforce", "monitor"])
    async def test_classifier_error_blocks_in_every_mode(self, mode, caplog):
        backend = _backend()
        clf = _classifier(error=RuntimeError(f"tokenizer choked on {QUESTION_CANARY}"))
        with caplog.at_level(logging.DEBUG):
            resp = await _post(_settings(enforcement_mode=mode), clf, backend)

        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "guard_unavailable"
        backend.post.assert_not_awaited()
        assert QUESTION_CANARY not in caplog.text
        assert QUESTION_CANARY not in resp.text

    @pytest.mark.asyncio
    async def test_slow_classifier_times_out_and_blocks(self):
        backend = _backend()
        loop = asyncio.get_running_loop()
        started = loop.time()
        resp = await _post(
            _settings(input_guard_timeout=0.2), _classifier(delay=5), backend, headers=REQUIRE
        )

        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "guard_unavailable"
        assert loop.time() - started < 2
        backend.post.assert_not_awaited()


class TestReadiness:
    async def _get(self, settings, classifier, client=None):
        app = create_app()
        with patch("app.api.routes.get_settings", return_value=settings):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                app.state.classifier = classifier
                app.state.http_client = client or _backend()
                app.state.revision = "a" * 40
                app.state.instance = "i1"
                app.state.backend_inflight = 0
                return await c.get("/aegis/readiness")

    @pytest.mark.asyncio
    async def test_enforced_and_loaded_is_ready(self):
        resp = await self._get(
            _settings(scan_roles="user", redact_prompts_in_logs=True, backend_stream=True),
            _classifier(),
        )
        data = resp.json()

        assert resp.headers["cache-control"] == "no-store"
        assert data["schema"] == "aegis.readiness/1"
        assert data["service"] == "aegis-ml"
        assert data["ready"] is True and data["reasons"] == []
        assert data["enforcement"] == "enforce"
        assert data["scan_roles"] == ["user"]
        assert data["backend_redirects"] is False
        assert data["cancel_on_disconnect"] is True
        assert data["audit_prompt_text"] is False
        assert data["backend_inflight"] == 0
        assert data["require_enforcement_header"] == "x-aegis-require-enforcement"

    @pytest.mark.asyncio
    async def test_monitor_and_unloaded_are_not_ready(self):
        resp = await self._get(_settings(enforcement_mode="monitor"), _classifier(loaded=False))
        data = resp.json()

        assert data["ready"] is False
        assert data["reasons"] == ["monitor_mode", "classifier_not_loaded"]
        assert data["audit_prompt_text"] is True
        assert data["cancel_on_disconnect"] is False, "non-streamed forwarding can't stop llama"

    @pytest.mark.asyncio
    async def test_exposes_no_backend_url_or_token(self):
        settings = _settings(
            backend_url="http://secret-backend.internal:9999/v1/chat/completions",
            admin_token="tok-should-not-leak",
            backend_api_key="key-should-not-leak",
        )
        text = (await self._get(settings, _classifier())).text

        for value in ("secret-backend", "9999", "tok-should-not-leak", "key-should-not-leak"):
            assert value not in text


class TestRedirects:
    @pytest.mark.asyncio
    async def test_lifespan_client_does_not_follow_redirects(self, tmp_path, monkeypatch):
        from app.config import get_settings

        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/a.db")
        monkeypatch.setenv("CLASSIFIER_TYPE", "sklearn")
        get_settings.cache_clear()
        try:
            app = create_app()
            async with app.router.lifespan_context(app):
                assert app.state.http_client.follow_redirects is False
        finally:
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_backend_redirect_is_a_502_and_the_target_is_never_contacted(self):
        contacted = []

        def handler(request: httpx.Request) -> httpx.Response:
            contacted.append(str(request.url))
            if request.url.host == "llm":
                return httpx.Response(302, headers={"location": "http://elsewhere/v1/x"})
            return httpx.Response(200, json={"choices": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        settings = _settings(backend_url="http://llm/v1/chat/completions")
        resp = await _post(settings, _classifier(), client, headers=REQUIRE)
        await client.aclose()

        assert resp.status_code == 502
        assert contacted == ["http://llm/v1/chat/completions"]


class TestBackendCall:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_work_and_settles_the_count(self):
        request = MagicMock()
        request.app.state.backend_inflight = 0
        polls = 0

        async def is_disconnected():
            nonlocal polls
            polls += 1
            return polls >= 2

        request.is_disconnected = is_disconnected
        cancelled = asyncio.Event()

        async def forever():
            assert request.app.state.backend_inflight == 1
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch("app.api.routes.get_settings", return_value=_settings()):
            with pytest.raises(routes.ClientDisconnectedError):
                await routes.backend_call(request, forever)

        assert cancelled.is_set()
        assert request.app.state.backend_inflight == 0

    @pytest.mark.asyncio
    async def test_caller_gone_before_forwarding_starts_no_backend_work(self):
        request = MagicMock()
        request.app.state.backend_inflight = 0
        request.is_disconnected = AsyncMock(return_value=True)
        started = AsyncMock()

        with patch("app.api.routes.get_settings", return_value=_settings()):
            with pytest.raises(routes.ClientDisconnectedError):
                await routes.backend_call(request, started)
        started.assert_not_called()

    @pytest.mark.asyncio
    async def test_total_deadline_bounds_the_work(self):
        request = MagicMock()
        request.app.state.backend_inflight = 0
        request.is_disconnected = AsyncMock(return_value=False)

        with patch("app.api.routes.get_settings", return_value=_settings(backend_timeout=1.0)):
            with pytest.raises(httpx.TimeoutException):
                await routes.backend_call(request, lambda: asyncio.sleep(30))
        assert request.app.state.backend_inflight == 0


# ── Real sockets: uvicorn in front of a fake backend that never answers ───────


class _HangingBackend:
    """Accepts HTTP requests, never answers, and records when callers hang up."""

    def __init__(self) -> None:
        self.requests = 0
        self.closed = asyncio.Event()
        self.closed_at: float | None = None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        with contextlib.suppress(Exception):
            await reader.readuntil(b"\r\n\r\n")
            self.requests += 1
            while await reader.read(65536):
                pass
        self.closed_at = asyncio.get_running_loop().time()
        self.closed.set()
        writer.close()


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    from app.config import get_settings

    db = tmp_path / "audit.db"
    values = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{db}",
        "CLASSIFIER_TYPE": "sklearn",
        "SCAN_ROLES": "user",
        "ENFORCEMENT_MODE": "enforce",
        "REDACT_PROMPTS_IN_LOGS": "true",
        "BACKEND_TIMEOUT": "30",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield db
    get_settings.cache_clear()


async def _serve(app):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.ensure_future(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


@pytest.mark.skipif(not SKLEARN_MODEL.exists(), reason="bundled sklearn model missing")
class TestLiveDisconnect:
    @pytest.mark.asyncio
    async def test_caller_disconnect_closes_the_backend_connection(self, live_env, monkeypatch):
        backend = _HangingBackend()
        fake = await asyncio.start_server(backend.handle, "127.0.0.1", 0)
        backend_port = fake.sockets[0].getsockname()[1]
        monkeypatch.setenv("BACKEND_URL", f"http://127.0.0.1:{backend_port}/v1/chat/completions")
        from app.config import get_settings

        get_settings.cache_clear()
        server, task, port = await _serve(create_app())
        try:
            body = json.dumps(_body(f"What are you doing? {QUESTION_CANARY}")).encode()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/chat/completions HTTP/1.1\r\nhost: t\r\n"
                b"content-type: application/json\r\nx-aegis-require-enforcement: enforce\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            for _ in range(200):
                if backend.requests:
                    break
                await asyncio.sleep(0.02)
            assert backend.requests == 1

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
                assert (await c.get("/aegis/readiness")).json()["backend_inflight"] == 1
                loop = asyncio.get_running_loop()
                hung_up = loop.time()
                writer.close()
                await asyncio.wait_for(backend.closed.wait(), timeout=3)
                assert backend.closed_at - hung_up < 1.5
                for _ in range(100):
                    if (await c.get("/aegis/readiness")).json()["backend_inflight"] == 0:
                        break
                    await asyncio.sleep(0.02)
                assert (await c.get("/aegis/readiness")).json()["backend_inflight"] == 0
        finally:
            server.should_exit = True
            await task
            fake.close()

        import sqlite3

        rows = (
            sqlite3.connect(live_env)
            .execute("select input_text, output_verdict from audit_log")
            .fetchall()
        )
        assert rows == [("[REDACTED]", "cancelled")]


class TestPrivacy:
    """Synthetic canaries never reach logs or the audit DB with redaction on."""

    @pytest.mark.asyncio
    async def test_canaries_absent_from_logs_and_audit_on_every_path(self, tmp_path, caplog):
        import sqlite3

        from app.models import database

        db = tmp_path / "audit.db"
        settings = _settings(
            database_url=f"sqlite+aiosqlite:///{db}",
            redact_prompts_in_logs=True,
            scan_roles="user",
            input_guard_timeout=0.2,
        )
        question = f"what are you doing {QUESTION_CANARY} me@{QUESTION_CANARY}.com"
        email_reply = f"Write to bob@{ANSWER_CANARY}.com, I'm mining. {ANSWER_CANARY}"
        failing = MagicMock()
        failing.post = AsyncMock(side_effect=ValueError(f"bad json {ANSWER_CANARY}"))
        cases = [
            (_classifier(), _backend(email_reply)),  # allowed, PII redacted
            (_classifier(prob=0.99), _backend()),  # input block
            (_classifier(error=RuntimeError(QUESTION_CANARY)), _backend()),  # guard error
            (_classifier(delay=2), _backend()),  # guard timeout
            (_classifier(), failing),  # backend error
            (_classifier(), _backend("how to make a bomb " + ANSWER_CANARY)),  # output block
        ]
        statuses = []
        with patch("app.models.database.get_settings", return_value=settings):
            await database.init_db()
            with caplog.at_level(logging.DEBUG):
                for clf, backend in cases:
                    app = create_app()
                    with contextlib.ExitStack() as stack:
                        for target in ("app.api.routes", "app.guardrails.input_guard"):
                            stack.enter_context(
                                patch(f"{target}.get_settings", return_value=settings)
                            )
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://t") as c:
                            app.state.classifier = clf
                            app.state.http_client = backend
                            app.state.limiter = MagicMock()
                            resp = await c.post(
                                "/v1/chat/completions", json=_body(question), headers=REQUIRE
                            )
                            statuses.append(resp.status_code)
                            if resp.status_code == 200:
                                assert f"bob@{ANSWER_CANARY}.com" not in resp.text
                            else:
                                assert QUESTION_CANARY not in resp.text
                                assert ANSWER_CANARY not in resp.text

        assert statuses == [200, 403, 403, 403, 502, 403]
        assert QUESTION_CANARY not in caplog.text
        assert ANSWER_CANARY not in caplog.text
        raw = db.read_bytes()
        assert QUESTION_CANARY.encode() not in raw and ANSWER_CANARY.encode() not in raw
        rows = sqlite3.connect(db).execute("select input_text from audit_log").fetchall()
        assert len(rows) == 6 and {r[0] for r in rows} == {"[REDACTED]"}


class TestOutputGuardFailure:
    @pytest.mark.asyncio
    async def test_output_guard_error_is_guard_unavailable_not_a_verdict(self, caplog):
        crash = AsyncMock(side_effect=RuntimeError(f"regex blew up on {ANSWER_CANARY}"))
        with patch("app.guardrails.output_guard._check", new=crash):
            with caplog.at_level(logging.DEBUG):
                resp = await _post(
                    _settings(), _classifier(), _backend(ANSWER_CANARY), headers=REQUIRE
                )

        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "guard_unavailable"
        assert ANSWER_CANARY not in resp.text and ANSWER_CANARY not in caplog.text


def _sse(*chunks, done=True):
    lines = [f"data: {json.dumps(c)}" for c in chunks] + (["data: [DONE]"] if done else [])
    return ("\n\n".join(lines) + "\n\n").encode()


class TestStreamedForwarding:
    async def _forward(self, handler, **overrides):
        from app.proxy.llm_proxy import forward_to_backend

        settings = _settings(
            backend_url="http://llm/v1/chat/completions", backend_stream=True, **overrides
        )
        seen = []

        def record(request):
            seen.append(json.loads(request.content))
            return handler(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(record), follow_redirects=False)
        try:
            with patch("app.proxy.llm_proxy.get_settings", return_value=settings):
                return await forward_to_backend(_body(), "rid", client), seen
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_stream_is_assembled_into_one_completion(self):
        body = _sse(
            {"id": "c1", "created": 1, "model": "m", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "I'm "}}]},
            {"choices": [{"delta": {"content": "mining."}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 3}},
        )
        data, seen = await self._forward(lambda r: httpx.Response(200, content=body))

        assert seen[0]["stream"] is True and seen[0]["max_tokens"] == 64
        assert data["object"] == "chat.completion" and data["id"] == "c1"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "I'm mining."}
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"] == {"completion_tokens": 3}

    @pytest.mark.asyncio
    async def test_oversized_error_and_redirect_streams_fail(self):
        big = _sse(*[{"choices": [{"delta": {"content": "x" * 1000}}]}] * 3)
        with pytest.raises(ValueError, match="too large"):
            await self._forward(
                lambda r: httpx.Response(200, content=big), backend_max_response_bytes=2048
            )
        with pytest.raises(ValueError, match="stream error"):
            await self._forward(lambda r: httpx.Response(200, content=_sse({"error": "boom"})))
        with pytest.raises(httpx.HTTPStatusError):
            await self._forward(lambda r: httpx.Response(302, headers={"location": "http://x/"}))

    @pytest.mark.asyncio
    async def test_route_still_answers_non_streaming_and_guards_output(self):
        body = _sse({"choices": [{"delta": {"content": "mail bob@example.com"}}]})
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        )
        settings = _settings(backend_url="http://llm/v1/chat/completions", backend_stream=True)
        resp = await _post(settings, _classifier(), client, headers=REQUIRE)
        await client.aclose()

        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert content == "mail [EMAIL REDACTED]"


class _StreamingBackend:
    """Streams one SSE token every 50 ms until a write fails (the caller hung up)."""

    def __init__(self) -> None:
        self.tokens = 0
        self.stopped = asyncio.Event()
        self.stopped_at: float | None = None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        with contextlib.suppress(Exception):
            head = await reader.readuntil(b"\r\n\r\n")
            length = int(head.lower().split(b"content-length:")[1].split(b"\r\n")[0])
            await reader.readexactly(length)
            writer.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
            )
            while self.tokens < 2000:
                event = _sse({"choices": [{"delta": {"content": "t"}}]}, done=False)
                writer.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
                await writer.drain()
                self.tokens += 1
                await asyncio.sleep(0.05)
        self.stopped_at = asyncio.get_running_loop().time()
        self.stopped.set()
        writer.close()


@pytest.mark.skipif(not SKLEARN_MODEL.exists(), reason="bundled sklearn model missing")
class TestLiveStreamedCancel:
    @pytest.mark.asyncio
    async def test_disconnect_stops_a_streaming_backend_at_its_next_write(
        self, live_env, monkeypatch
    ):
        backend = _StreamingBackend()
        fake = await asyncio.start_server(backend.handle, "127.0.0.1", 0)
        port_b = fake.sockets[0].getsockname()[1]
        monkeypatch.setenv("BACKEND_URL", f"http://127.0.0.1:{port_b}/v1/chat/completions")
        monkeypatch.setenv("BACKEND_STREAM", "true")
        from app.config import get_settings

        get_settings.cache_clear()
        server, task, port = await _serve(create_app())
        try:
            body = json.dumps(_body()).encode()
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/chat/completions HTTP/1.1\r\nhost: t\r\n"
                b"content-type: application/json\r\nx-aegis-require-enforcement: enforce\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            for _ in range(200):
                if backend.tokens >= 5:
                    break
                await asyncio.sleep(0.02)
            loop = asyncio.get_running_loop()
            hung_up = loop.time()
            writer.close()
            await asyncio.wait_for(backend.stopped.wait(), timeout=3)
            assert backend.stopped_at - hung_up < 1.5
            assert backend.tokens < 100
        finally:
            server.should_exit = True
            await task
            fake.close()
