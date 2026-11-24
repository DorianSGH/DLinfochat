"""Hybrid retrieval: semantic (vector) + keyword (full-text) search, fused with RRF.

Why both: vectors match meaning ("money" finds "souls"), keywords match exact names that
embeddings blur ("Siphon Life", "Cold Front"). Each catches what the other misses.
"""

from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import quote

from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from app.config import Settings
from app.db import Chunk, Page, SessionLocal
from app.providers.base import Embedder


@dataclass
class Source:
    chunk_id: int
    title: str
    section: str
    url: str  # links straight to the section, e.g. .../Abrams#Siphon_Life
    content: str
    score: float  # fused RRF score
    semantic_rank: int | None  # 1 = best vector match; None = not in vector results
    keyword_rank: int | None  # 1 = best keyword match; None = not in keyword results


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists. Each item scores the sum of 1 / (k + rank) over the lists it's in.

    Only positions are used, never raw scores, so cosine distances and ts_rank values
    (unrelated scales) are never compared. An item near the top of both lists beats one
    at the very top of only one. k=60 comes from the original RRF paper; a larger k
    shrinks the advantage of rank 1 over rank 2.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def any_word_query(question: str) -> ColumnElement:
    """A tsquery matching chunks that contain ANY of the question's words.

    plainto_tsquery ANDs every word, so "what is the cooldown of Abrams' Siphon Life?"
    would only match chunks containing all of cooldown, abram, siphon and life. Swapping
    & for | turns it into an OR; ts_rank still ranks chunks matching more words higher.
    SQL: replace(plainto_tsquery('english', :q)::text, '&', '|')::tsquery
    """
    all_words = func.plainto_tsquery("english", question)
    return cast(func.replace(cast(all_words, Text), "&", "|"), TSQUERY)


def section_url(page_url: str, section: str) -> str:
    # MediaWiki heading ids are the heading text with spaces as underscores.
    if not section:
        return page_url
    return f"{page_url}#{quote(section.split(' > ')[-1].replace(' ', '_'))}"


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        candidates: int = 30,
        top_k: int = 6,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.sessionmaker = sessionmaker
        self.candidate_limit = candidates
        self.top_k = top_k
        self.rrf_k = rrf_k

    async def candidates(self, question: str) -> tuple[list[int], list[int]]:
        """Chunk ids from each method separately, best first: (semantic, keyword).

        Exposed on its own so evals can score each method and the fusion on the same query.
        """
        vector = await self.embedder.embed_query(question)
        tsquery = any_word_query(question)
        # Sessions here live only for a few quick queries, so a chat request doesn't
        # hold a DB connection during the seconds it spends streaming the answer.
        async with self.sessionmaker() as session:
            semantic_ids = list(
                await session.scalars(
                    select(Chunk.id)
                    .order_by(Chunk.embedding.cosine_distance(vector))  # <=> operator
                    .limit(self.candidate_limit)
                )
            )
            keyword_ids = list(
                await session.scalars(
                    select(Chunk.id)
                    .where(Chunk.search_vector.op("@@")(tsquery))  # uses the GIN index
                    .order_by(func.ts_rank(Chunk.search_vector, tsquery).desc())
                    .limit(self.candidate_limit)
                )
            )
        return semantic_ids, keyword_ids

    async def search(self, question: str) -> list[Source]:
        semantic_ids, keyword_ids = await self.candidates(question)
        fused = reciprocal_rank_fusion([semantic_ids, keyword_ids], self.rrf_k)[: self.top_k]
        async with self.sessionmaker() as session:
            rows = await session.execute(
                select(Chunk.id, Chunk.section, Chunk.content, Page.title, Page.url)
                .join(Chunk.page)
                .where(Chunk.id.in_([chunk_id for chunk_id, _ in fused]))
            )
            by_id = {row.id: row for row in rows}

        return [
            Source(
                chunk_id=chunk_id,
                title=row.title,
                section=row.section,
                url=section_url(row.url, row.section),
                content=row.content,
                score=score,
                semantic_rank=_rank_of(chunk_id, semantic_ids),
                keyword_rank=_rank_of(chunk_id, keyword_ids),
            )
            for chunk_id, score in fused
            if (row := by_id.get(chunk_id))
        ]


def _rank_of(item: int, ranking: list[int]) -> int | None:
    return ranking.index(item) + 1 if item in ranking else None


def build_retriever(settings: Settings, embedder: Embedder) -> Retriever:
    return Retriever(
        embedder,
        SessionLocal,
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
    )
