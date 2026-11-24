"""Retrieval evals: how often each search method puts the answer's wiki section in its top k.

    python -m evals.retrieval            # summary tables
    python -m evals.retrieval --misses   # also list questions hybrid misses in what the LLM sees

Needs Postgres (with ingested wiki) and Ollama running. Only retrieval is scored here, not
the generated answers: if the right chunk never reaches the LLM, the answer can't be right.
"""

import argparse
import asyncio
import sys

from sqlalchemy import func, select

from app.config import get_settings
from app.db import Chunk, Page, SessionLocal, engine
from app.providers import make_embedder, ollama_client
from app.rag.retrieve import build_retriever, reciprocal_rank_fusion
from evals.dataset import EvalQuestion, load_all
from evals.metrics import Target, first_hit_rank, hit_rate, mrr

MODES = ("semantic", "keyword", "hybrid")
KS = (1, 3, 6, 10, 30)
MRR_K = 10


async def rank_questions(questions: list[EvalQuestion]) -> dict[str, list[list[tuple[str, str]]]]:
    """For every question and mode, the retrieved (page, section) list, best first."""
    settings = get_settings()
    id_rankings: dict[str, list[list[int]]] = {mode: [] for mode in MODES}
    async with ollama_client(settings) as ollama:
        retriever = build_retriever(settings, make_embedder(settings, ollama))
        for q in questions:
            # One retrieval per question; all three modes are scored on the same results.
            semantic, keyword = await retriever.candidates(q.question)
            fused = reciprocal_rank_fusion([semantic, keyword], settings.rrf_k)
            id_rankings["semantic"].append(semantic)
            id_rankings["keyword"].append(keyword)
            id_rankings["hybrid"].append([cid for cid, _ in fused][: settings.retrieval_candidates])

    all_ids = {cid for rankings in id_rankings.values() for ids in rankings for cid in ids}
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Chunk.id, Page.title, Chunk.section).join(Chunk.page).where(Chunk.id.in_(all_ids))
        )
        labels = {row.id: (row.title, row.section) for row in rows}
    return {
        mode: [[labels[cid] for cid in ids] for ids in rankings]
        for mode, rankings in id_rankings.items()
    }


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


async def main(show_misses: bool) -> None:
    settings = get_settings()
    questions = load_all()
    if not questions:
        raise SystemExit("No eval questions found in evals/*.jsonl")

    try:
        await warn_unknown_targets(questions)
        retrieved = await rank_questions(questions)
        async with SessionLocal() as session:
            n_chunks = await session.scalar(select(func.count()).select_from(Chunk))
    finally:
        await engine.dispose()

    ranks = {
        mode: [first_hit_rank(r, q.targets) for r, q in zip(retrieved[mode], questions)]
        for mode in MODES
    }
    print(
        f"embed={settings.embed_model}  chunks={n_chunks}  candidates={settings.retrieval_candidates}"
        f"  rrf_k={settings.rrf_k}  (the LLM sees the top {settings.retrieval_top_k})"
    )
    print(table("All", ranks))
    for source in sorted({q.source for q in questions}):
        idx = [i for i, q in enumerate(questions) if q.source == source]
        if len(idx) < len(questions):
            print(table(source.capitalize(), {m: [r[i] for i in idx] for m, r in ranks.items()}))

    if show_misses:
        k = settings.retrieval_top_k
        print(f"\nQuestions where hybrid misses the top {k}:")
        for i, q in enumerate(questions):
            rank = ranks["hybrid"][i]
            if rank is not None and rank <= k:
                continue
            expected = "; ".join(_describe(t) for t in q.targets)
            print(f"\n[{q.id}] {q.question}")
            print(f"  expected: {expected}")
            print(f"  ranks: semantic {ranks['semantic'][i]}, keyword {ranks['keyword'][i]}, hybrid {rank}")
            for page, section in retrieved["hybrid"][i][:3]:
                print(f"  got: {page} > {section or '(intro)'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m evals.retrieval")
    parser.add_argument("--misses", action="store_true", help="list questions hybrid misses")
    asyncio.run(main(parser.parse_args().misses))
