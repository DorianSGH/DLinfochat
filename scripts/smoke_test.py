"""Check that the local models work end to end.

Run from the project root:  python -m scripts.smoke_test
"""

import asyncio
import math
import time

from app.config import get_settings
from app.providers import make_chat_model, make_embedder, ollama_client


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def main() -> None:
    settings = get_settings()
    async with ollama_client(settings) as client:
        embedder = make_embedder(settings, client)
        chat = make_chat_model(settings, client)

        print(f"== Embeddings: {settings.embed_model}")
        docs = [
            "Souls are the in-game currency; players earn them by killing troopers "
            "and spend them on items.",
            "Items are split into three categories: Weapon, Vitality and Spirit.",
            "Preheat the oven to 200 degrees and bake the bread for 25 minutes.",
        ]
        query = "How do I get money to buy stuff?"
        start = time.perf_counter()
        doc_vectors = await embedder.embed_documents(docs)
        query_vector = await embedder.embed_query(query)
        print(f"dims={len(query_vector)}  time={time.perf_counter() - start:.2f}s")
        print(f"query: {query!r}")
        ranked = sorted(
            zip(docs, doc_vectors), key=lambda d: cosine(query_vector, d[1]), reverse=True
        )
        for doc, vector in ranked:
            print(f"  {cosine(query_vector, vector):.3f}  {doc[:70]}")

        print(f"\n== Chat: {settings.chat_model}")
        start = time.perf_counter()
        first_token_at = None
        async for token in chat.stream(
            [
                {"role": "system", "content": "Answer in one or two sentences."},
                {"role": "user", "content": "What is retrieval-augmented generation?"},
            ]
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            print(token, end="", flush=True)
        print(f"\n(first token {first_token_at:.2f}s, total {time.perf_counter() - start:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
