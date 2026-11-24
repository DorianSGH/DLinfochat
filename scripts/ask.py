"""Ask a question from the terminal: same pipeline as the API, no server needed.

    python -m scripts.ask "how much health does Abrams have?"
"""

import asyncio
import sys
import time

from app.config import get_settings
from app.db import engine
from app.providers import make_chat_model, make_embedder, make_reranker, ollama_client
from app.rag.prompt import build_messages
from app.rag.retrieve import build_retriever


async def main(question: str) -> None:
    settings = get_settings()
    try:
        reranker = await asyncio.to_thread(make_reranker, settings.reranker_model)
        async with ollama_client(settings) as ollama:
            retriever = build_retriever(settings, make_embedder(settings, ollama), reranker)
            start = time.perf_counter()
            sources = await retriever.search(question)
            retrieval_s = time.perf_counter() - start

            print(f"Sources ({retrieval_s:.2f}s):")
            for n, s in enumerate(sources, 1):
                ranks = f"semantic #{s.semantic_rank or '-'}, keyword #{s.keyword_rank or '-'}"
                if s.rerank_score is not None:
                    ranks += f", rerank {s.rerank_score:.3f}"
                print(f"  [{n}] {s.title} > {s.section or '(intro)'}   ({ranks})")
            print()

            start = time.perf_counter()
            chat = make_chat_model(settings, ollama)
            async for token in chat.stream(build_messages(question, sources)):
                print(token, end="", flush=True)
            print(f"\n\n(answer {time.perf_counter() - start:.2f}s)")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m scripts.ask "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))
