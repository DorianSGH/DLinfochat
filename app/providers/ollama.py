"""Ollama implementations, talking to its HTTP API directly with httpx (no SDK)."""

import json
from collections.abc import AsyncIterator

import httpx

from app.providers.base import Message


class OllamaEmbedder:
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        dimensions: int,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed(texts[i : i + self.batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        (vector,) = await self._embed([self.query_prefix + text])
        return vector

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = await self.client.post(
            "/api/embed", json={"model": self.model, "input": inputs}
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        # Catch a model swap that changes vector size before it corrupts the database.
        if len(vectors[0]) != self.dimensions:
            raise ValueError(
                f"{self.model} returned {len(vectors[0])}-dim vectors, expected {self.dimensions}"
            )
        return vectors


class OllamaChat:
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        num_ctx: int,
        think: bool = False,
        temperature: float | None = None,
    ) -> None:
        self.client = client
        self.model = model
        # Ollama silently drops the start of prompts longer than num_ctx, so set it explicitly
        # big enough for system prompt + retrieved chunks + question.
        self.num_ctx = num_ctx
        self.think = think
        self.temperature = temperature  # None keeps the model's default

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {"num_ctx": self.num_ctx},
        }
        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature
        # Ollama streams newline-delimited JSON, one object per token batch.
        async with self.client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                if token := chunk.get("message", {}).get("content"):
                    yield token
