"""Vector search over the ingested chunks, to eyeball retrieval quality.

    python -m scripts.search "what does Abrams' ultimate do?"
"""

import asyncio
import sys

from sqlalchemy import select

from app.config import get_settings
from app.db import Chunk, Page, SessionLocal, engine
from app.providers import make_embedder, ollama_client


async def main(question: str, k: int = 5) -> None:
    settings = get_settings()
    async with ollama_client(settings) as ollama:
        vector = await make_embedder(settings, ollama).embed_query(question)

    # <=> is pgvector's cosine distance; ordering by it uses the HNSW index.
    distance = Chunk.embedding.cosine_distance(vector)
    stmt = (
        select(Page.url, Chunk.content, distance.label("distance"))
        .join(Chunk.page)
        .order_by(distance)
        .limit(k)
    )
    async with SessionLocal() as session:
        rows = (await session.execute(stmt)).all()
    await engine.dispose()

    for rank, row in enumerate(rows, 1):
        print(f"\n#{rank}  similarity {1 - row.distance:.3f}  {row.url}")
        print(row.content[:700])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m scripts.search "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))
