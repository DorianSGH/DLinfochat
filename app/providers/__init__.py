"""The one place that decides which concrete providers the app uses.

Swapping to another provider later means changing these functions (e.g. switching on a
setting); everything else only sees the Embedder / ChatModel protocols.
"""

import httpx

from app.config import Settings
from app.providers.base import ChatModel, Embedder
from app.providers.ollama import OllamaChat, OllamaEmbedder


def ollama_client(settings: Settings) -> httpx.AsyncClient:
    # Generous read timeout: the first request after idling loads the model into VRAM.
    return httpx.AsyncClient(base_url=settings.ollama_url, timeout=httpx.Timeout(300, connect=5))


def make_embedder(settings: Settings, client: httpx.AsyncClient) -> Embedder:
    return OllamaEmbedder(
        client,
        settings.embed_model,
        settings.embed_dimensions,
        query_prefix=settings.embed_query_prefix,
    )


def make_chat_model(settings: Settings, client: httpx.AsyncClient) -> ChatModel:
    return OllamaChat(
        client,
        settings.chat_model,
        settings.chat_num_ctx,
        think=settings.chat_think,
        temperature=settings.chat_temperature,
    )
