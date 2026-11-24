"""Provider interfaces. The rest of the app depends only on these, never on a concrete provider.

To swap models later (e.g. OpenAI embeddings, Claude for chat), write a class that satisfies
the protocol and construct it instead of the Ollama one. Nothing else changes.
"""

from collections.abc import AsyncIterator
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Embedder(Protocol):
    dimensions: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a user question for search.

        Separate from embed_documents because some models are asymmetric: Qwen3-Embedding
        expects an instruction prefix on queries but not on documents. Symmetric models
        (e.g. OpenAI's) just treat both the same.
        """
        ...


class ChatModel(Protocol):
    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield the answer text as it is generated."""
        ...
