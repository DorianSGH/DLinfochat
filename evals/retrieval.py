"""Retrieval evals: how often each search method puts the answer's wiki section in its top k.

    python -m evals.retrieval                                    # vector, keyword, hybrid
    python -m evals.retrieval --rerank BAAI/bge-reranker-v2-m3   # + hybrid reranked
    python -m evals.retrieval --misses                           # list what hybrid misses

Needs Postgres (with ingested wiki) and Ollama running. Only retrieval is scored here, not
the generated answers: if the right chunk never reaches the LLM, the answer can't be right.
"""

import argparse
import asyncio
import statistics
import sys
import time

from sqlalchemy import func, select

from app.config import get_settings
from app.db import Chunk, Page, SessionLocal, engine
from app.providers import make_embedder, make_reranker, ollama_client
from app.providers.base import Reranker
from app.rag.retrieve import build_retriever, reciprocal_rank_fusion, rerank_order
from evals.dataset import EvalQuestion, load_all
from evals.metrics import Target, first_hit_rank, hit_rate, mrr

BASE_MODES = ("semantic", "keyword", "hybrid")
RERANK_DEPTHS = (10, 20, 30)
KS = (1, 3, 6, 10, 30)
MRR_K = 10


async def rank_questions(
    questions: list[EvalQuestion], reranker: Reranker | None
) -> tuple[dict[str, list[list[tuple[str, str]]]], dict[str, list[float]]]:
    """Per mode, the retrieved (page, section) list for each question; plus timings."""
    settings = get_settings()
    modes = list(BASE_MODES)
    if reranker:
        modes += [f"hybrid+rerank@{d}" for d in RERANK_DEPTHS]
    id_rankings: dict[str, list[list[int]]] = {mode: [] for mode in modes}
    timings: dict[str, list[float]] = {"retrieval": [], "rerank": []}

    async with ollama_client(settings) as ollama:
        retriever = build_retriever(settings, make_embedder(settings, ollama), reranker)
        for q in questions:
            # One retrieval per question; every mode is scored on the same candidates.
            start = time.perf_counter()
            semantic, keyword = await retriever.candidates(q.question)
            timings["retrieval"].append(time.perf_counter() - start)
            hybrid = [cid for cid, _ in reciprocal_rank_fusion([semantic, keyword], settings.rrf_k)]
            hybrid = hybrid[: settings.retrieval_candidates]
            id_rankings["semantic"].append(semantic)
            id_rankings["keyword"].append(keyword)
            id_rankings["hybrid"].append(hybrid)

            if reranker:
                # Score the deepest shortlist once; shallower depths reuse those scores
                # (a reranker scores each pair independently, so the result is identical).
                rows = await retriever.load(hybrid[: max(RERANK_DEPTHS)])
                start = time.perf_counter()
                scores = await retriever.rerank_scores(q.question, rows)
                timings["rerank"].append(time.perf_counter() - start)
                for depth in RERANK_DEPTHS:
                    id_rankings[f"hybrid+rerank@{depth}"].append(rerank_order(hybrid, scores, depth))

    all_ids = {cid for rankings in id_rankings.values() for ids in rankings for cid in ids}
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Chunk.id, Page.title, Chunk.section).join(Chunk.page).where(Chunk.id.in_(all_ids))
        )
        labels = {row.id: (row.title, row.section) for row in rows}
    retrieved = {
        mode: [[labels[cid] for cid in ids] for ids in rankings]
        for mode, rankings in id_rankings.items()
    }
    return retrieved, timings


async def warn_unknown_targets(questions: list[EvalQuestion]) -> None:
    """Catch typos in labels: a target that matches no chunk can never be found."""
    async with SessionLocal() as session:
        known = (await session.execute(select(Page.title, Chunk.section).join(Chunk.page))).all()
    for q in questions:
        for t in q.targets:
            if not any(t.matches(page, section) for page, section in known):
                print(f"WARNING {q.id}: target {t.page!r} / {t.section!r} matches no chunk")


def _describe(target: Target) -> str:
    if target.section is None:
        return f"{target.page} (any section)"
    return f"{target.page} > {target.section or '(intro)'}"


def _ms(seconds: list[float]) -> str:
    p95 = statistics.quantiles(seconds, n=20)[-1] if len(seconds) >= 20 else max(seconds)
    return f"mean {statistics.mean(seconds) * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms"


def table(title: str, ranks: dict[str, list[int | None]]) -> str:
    n = len(next(iter(ranks.values())))
    head = "| method | " + " | ".join(f"hit@{k}" for k in KS) + f" | MRR@{MRR_K} |"
    rule = "|---" * (len(KS) + 2) + "|"
    rows = [
        f"| {mode} | "
        + " | ".join(f"{hit_rate(r, k):.0%}" for k in KS)
        + f" | {mrr(r, MRR_K):.2f} |"
        for mode, r in ranks.items()
    ]
    return "\n".join([f"\n{title} ({n} questions)", head, rule, *rows])


async def main(reranker_model: str, show_misses: bool) -> None:
    settings = get_settings()
    all_questions = load_all()
    # Questions the wiki doesn't answer have nothing to retrieve; they're for answer evals.
    questions = [q for q in all_questions if q.targets]
    if not questions:
        raise SystemExit("No eval questions found in evals/*.jsonl")

    reranker = await asyncio.to_thread(make_reranker, reranker_model)
    try:
        await warn_unknown_targets(questions)
        retrieved, timings = await rank_questions(questions, reranker)
        async with SessionLocal() as session:
            n_chunks = await session.scalar(select(func.count()).select_from(Chunk))
    finally:
        await engine.dispose()

    ranks = {
        mode: [first_hit_rank(r, q.targets) for r, q in zip(found, questions)]
        for mode, found in retrieved.items()
    }
    print(
        f"embed={settings.embed_model}  chunks={n_chunks}  candidates={settings.retrieval_candidates}"
        f"  rrf_k={settings.rrf_k}  reranker={reranker_model or 'none'}"
        f"  (the LLM sees the top {settings.retrieval_top_k})"
    )
    if skipped := len(all_questions) - len(questions):
        print(f"{skipped} questions with no answer on the wiki skipped (kept for answer evals)")
    print(table("All", ranks))
    for source in sorted({q.source for q in questions}):
        idx = [i for i, q in enumerate(questions) if q.source == source]
        if len(idx) < len(questions):
            print(table(source.capitalize(), {m: [r[i] for i in idx] for m, r in ranks.items()}))

    print(f"\nLatency per question: retrieval (embed + 2 SQL queries) {_ms(timings['retrieval'])}")
    if timings["rerank"]:
        print(f"                      rerank of top {max(RERANK_DEPTHS)}: {_ms(timings['rerank'])}")

    if show_misses:
        mode = f"hybrid+rerank@{settings.rerank_depth}" if reranker else "hybrid"
        mode = mode if mode in ranks else "hybrid"
        k = settings.retrieval_top_k
        print(f"\nQuestions where {mode} misses the top {k}:")
        for i, q in enumerate(questions):
            rank = ranks[mode][i]
            if rank is not None and rank <= k:
                continue
            print(f"\n[{q.id}] {q.question}")
            print(f"  expected: {'; '.join(_describe(t) for t in q.targets)}")
            print("  ranks: " + ", ".join(f"{m} {ranks[m][i]}" for m in ranks))
            for page, section in retrieved[mode][i][:3]:
                print(f"  got: {page} > {section or '(intro)'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m evals.retrieval")
    parser.add_argument(
        "--rerank",
        default=None,
        help="cross-encoder to evaluate (default: RERANKER_MODEL setting; '' to disable)",
    )
    parser.add_argument("--misses", action="store_true", help="list questions that are missed")
    args = parser.parse_args()
    model = get_settings().reranker_model if args.rerank is None else args.rerank
    asyncio.run(main(model, args.misses))
