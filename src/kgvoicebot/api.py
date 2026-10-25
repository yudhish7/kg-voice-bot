"""HTTP API (Starlette).

Endpoints
---------
``GET  /health``  liveness + graph size (never requires auth)
``POST /ask``     ``{"question": "..."}``  -> JSON answer with facts and sources
``POST /voice``   raw audio body           -> audio reply; transcript/answer in headers
``POST /ingest``  ``{"doc_id": "...", "text": "..."}``  -> add or replace a document

If ``KGV_API_KEY`` is set, every endpoint except ``/health`` requires
``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .bot import VoiceBot
from .config import Settings
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .models import Answer

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1_000
MAX_DOCUMENT_CHARS = 1_000_000


def _answer_payload(answer: Answer) -> dict[str, Any]:
    ev = answer.evidence
    return {
        "answer": answer.text,
        "entities": ev.entities,
        "facts": [
            {"subject": t.subject, "relation": t.relation, "object": t.object} for t in ev.triples
        ],
        "sources": answer.sources,
    }


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Request body must be valid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    return data


def _string_field(data: dict[str, Any], name: str, max_chars: int) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(422, f"'{name}' must be a non-empty string")
    if len(value) > max_chars:
        raise HTTPException(413, f"'{name}' exceeds {max_chars} characters")
    return value


def create_app(
    bot: VoiceBot,
    ingestor: Ingestor,
    kg: KnowledgeGraph,
    settings: Settings,
    persist: bool = True,
) -> Starlette:
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", **kg.stats()})

    async def ask(request: Request) -> Response:
        data = await _json_body(request)
        question = _string_field(data, "question", MAX_QUESTION_CHARS)
        answer = await run_in_threadpool(bot.ask_text, question)
        return JSONResponse(_answer_payload(answer))

    async def voice(request: Request) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_audio_bytes:
            raise HTTPException(413, "Audio too large")
        audio = await request.body()
        if not audio:
            raise HTTPException(400, "Request body must contain audio")
        if len(audio) > settings.max_audio_bytes:
            raise HTTPException(413, "Audio too large")
        reply = await run_in_threadpool(bot.ask_voice, audio)
        return Response(
            reply.audio,
            media_type=reply.content_type,
            headers={
                "X-Transcript": quote(reply.transcript),
                "X-Answer": quote(reply.answer.text),
            },
        )

    async def ingest(request: Request) -> Response:
        data = await _json_body(request)
        doc_id = _string_field(data, "doc_id", 200)
        text = _string_field(data, "text", MAX_DOCUMENT_CHARS)
        result = await run_in_threadpool(ingestor.ingest_text, doc_id, text)
        if persist:
            await run_in_threadpool(_save, kg, settings)
        return JSONResponse({"doc_id": doc_id, **result, **kg.stats()})

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    async def server_error(request: Request, exc: Exception) -> Response:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"error": "Internal server error"}, status_code=500)

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/ask", ask, methods=["POST"]),
            Route("/voice", voice, methods=["POST"]),
            Route("/ingest", ingest, methods=["POST"]),
        ],
        exception_handlers={HTTPException: http_error, Exception: server_error},
    )

    if settings.api_key:
        expected = f"Bearer {settings.api_key}".encode()

        async def require_key(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            if request.url.path != "/health":
                supplied = request.headers.get("authorization", "").encode()
                if not hmac.compare_digest(supplied, expected):
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=require_key)

    return app


def _save(kg: KnowledgeGraph, settings: Settings) -> None:
    with kg.lock:
        kg.save(settings.graph_path)
