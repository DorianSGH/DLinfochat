"""HTTP API.

    fastapi dev app/main.py     (auto-reload, http://127.0.0.1:8000/docs)
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, StringConstraints

from app.config import get_settings
from app.db import engine
from app.providers import make_chat_model, make_embedder, make_reranker, ollama_client
from app.providers.base import ChatModel
from app.rag.prompt import build_messages
from app.rag.retrieve import Retriever, Source, build_retriever

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Built once at startup and shared by all requests: one pooled HTTP client to
    # Ollama, one DB engine (already module-level in app.db).
    settings = get_settings()
    # Loading a reranker takes a few seconds; do it off the event loop.
    reranker = await asyncio.to_thread(make_reranker, settings.reranker_model)
    async with ollama_client(settings) as ollama:
        app.state.retriever = build_retriever(settings, make_embedder(settings, ollama), reranker)
        app.state.chat_model = make_chat_model(settings, ollama)
        yield
    await engine.dispose()


app = FastAPI(title="DLinfochat", lifespan=lifespan)


# --- dependencies (tests override these with fakes) --------------------------------------

def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_chat_model(request: Request) -> ChatModel:
    return request.app.state.chat_model


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
ChatModelDep = Annotated[ChatModel, Depends(get_chat_model)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


# --- schemas ------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: Question


class SourceOut(BaseModel):
    n: int  # the number the answer cites, e.g. [1]
    title: str
    section: str
    url: str


class SearchHit(SourceOut):
    content: str
    score: float
    semantic_rank: int | None
    keyword_rank: int | None


def to_source_out(n: int, source: Source) -> SourceOut:
    return SourceOut(n=n, title=source.title, section=source.section, url=source.url)


# --- routes -------------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/search")
async def search(
    q: Annotated[str, Query(min_length=1, max_length=500)], retriever: RetrieverDep
) -> list[SearchHit]:
    """What retrieval returns for a question, with each method's rank. For debugging and evals."""
    sources = await retriever.search(q)
    return [
        SearchHit(
            **to_source_out(n, s).model_dump(),
            content=s.content,
            score=s.score,
            semantic_rank=s.semantic_rank,
            keyword_rank=s.keyword_rank,
        )
        for n, s in enumerate(sources, 1)
    ]


async def retrieve_sources(body: ChatRequest, retriever: RetrieverDep) -> list[Source]:
    # Runs as a dependency, i.e. before the stream starts: if retrieval fails the client
    # gets a normal HTTP 500, not a 200 followed by a broken stream.
    return await retriever.search(body.question)


@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    body: ChatRequest,
    sources: Annotated[list[Source], Depends(retrieve_sources)],
    chat_model: ChatModelDep,
) -> AsyncIterator[ServerSentEvent]:
    """Stream an answer as Server-Sent Events.

    Events, in order: `sources` (once, the numbered list the answer cites), `token`
    (many, answer text pieces), then `done`, or `error` if generation fails midway.
    """
    yield ServerSentEvent(
        event="sources",
        data=[to_source_out(n, s).model_dump() for n, s in enumerate(sources, 1)],
    )
    try:
        async for token in chat_model.stream(build_messages(body.question, sources)):
            # data is JSON-encoded, so newlines inside a token can't break SSE framing.
            yield ServerSentEvent(event="token", data={"text": token})
    except Exception:
        # Headers (200) are already sent, so the only way to report this is in-stream.
        log.exception("Answer generation failed")
        yield ServerSentEvent(event="error", data={"message": "The model failed while answering."})
        return
    yield ServerSentEvent(event="done", data={})
