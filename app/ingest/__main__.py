"""Ingest the Deadlock wiki into Postgres.

    python -m app.ingest                          # new and changed pages only
    python -m app.ingest --full                   # re-ingest everything
    python -m app.ingest --limit 20               # first 20 pending pages
    python -m app.ingest --title Abrams --dry-run # print chunks; no embedding, no DB
"""

import argparse
import asyncio
import logging
import sys
import time

import httpx
from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import Settings, get_settings
from app.db import Chunk, Page, SessionLocal, engine, init_db
from app.ingest.chunk import ChunkData, chunk_sections
from app.ingest.clean import html_to_sections
from app.ingest.wiki import PageInfo, WikiClient
from app.providers import make_embedder, ollama_client
from app.providers.base import Embedder

log = logging.getLogger("ingest")


async def build_chunks(wiki: WikiClient, info: PageInfo, settings: Settings) -> tuple[list[ChunkData], int]:
    html, revision_id = await wiki.parse(info.id)
    return chunk_sections(info.title, html_to_sections(html), settings.chunk_max_chars), revision_id


async def ingest_page(wiki: WikiClient, embedder: Embedder, info: PageInfo, settings: Settings) -> int:
    chunks, revision_id = await build_chunks(wiki, info, settings)
    vectors = await embedder.embed_documents([c.content for c in chunks]) if chunks else []

    # One transaction per page: a crash mid-run never leaves a page half-written.
    async with SessionLocal() as session, session.begin():
        # Replace the page's chunks wholesale; simpler and safer than diffing them.
        await session.execute(delete(Chunk).where(Chunk.page_id == info.id))
        values = {
            "title": info.title,
            "url": info.url,
            "revision_id": revision_id,
            "embed_model": settings.embed_model,
            "ingested_at": func.now(),
        }
        await session.execute(
            pg_insert(Page)
            .values(id=info.id, **values)
            .on_conflict_do_update(index_elements=[Page.id], set_=values)
        )
        if chunks:
            await session.execute(
                insert(Chunk),
                [
                    {
                        "page_id": info.id,
                        "position": i,
                        "section": c.section,
                        "content": c.content,
                        "embedding": v,
                    }
                    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
    return len(chunks)


async def dry_run(wiki: WikiClient, pages: list[PageInfo], settings: Settings) -> None:
    for info in pages:
        chunks, _ = await build_chunks(wiki, info, settings)
        print(f"\n######## {info.title}: {len(chunks)} chunks  ({info.url})")
        for i, c in enumerate(chunks):
            print(f"\n--- chunk {i}  [{len(c.content)} chars]\n{c.content}")


async def run(args: argparse.Namespace) -> None:
    try:
        await _run(args)
    finally:
        await engine.dispose()  # close pooled DB connections inside this event loop


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    headers = {"User-Agent": settings.wiki_user_agent}
    async with (
        httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True) as http,
        ollama_client(settings) as ollama,
    ):
        wiki = WikiClient(http, settings.wiki_api_url)
        all_pages = await wiki.list_pages()
        if not all_pages:
            raise SystemExit("Wiki returned no pages; refusing to continue.")
        log.info("Wiki lists %d pages after filtering", len(all_pages))

        pages = all_pages
        if args.title:
            pages = [p for p in all_pages if p.title in set(args.title)]
            if missing := set(args.title) - {p.title for p in pages}:
                log.warning("Not found or filtered out: %s", ", ".join(sorted(missing)))

        if args.dry_run:
            await dry_run(wiki, pages[: args.limit or None], settings)
            return

        await init_db()
        async with SessionLocal() as session:
            rows = await session.execute(select(Page.id, Page.revision_id, Page.embed_model))
            stored = {pid: (rev, model) for pid, rev, model in rows}

        # A page is pending if it's new, edited since we stored it, or embedded with
        # another model.
        pending = [
            p for p in pages
            if args.full or stored.get(p.id) != (p.revision_id, settings.embed_model)
        ]
        todo = pending[: args.limit] if args.limit else pending

        # Pages deleted (or newly filtered out) on the wiki. Chunks go with them via
        # ON DELETE CASCADE.
        stale = stored.keys() - {p.id for p in all_pages}
        if stale:
            async with SessionLocal() as session, session.begin():
                await session.execute(delete(Page).where(Page.id.in_(stale)))
            log.info("Removed %d pages no longer on the wiki", len(stale))

        log.info(
            "%d pages pending (%d up to date); ingesting %d",
            len(pending), len(pages) - len(pending), len(todo),
        )
        embedder = make_embedder(settings, ollama)
        semaphore = asyncio.Semaphore(settings.ingest_concurrency)
        done = 0
        total_chunks = 0
        failures: list[tuple[str, str]] = []
        start = time.perf_counter()

        async def worker(info: PageInfo) -> None:
            nonlocal done, total_chunks
            async with semaphore:
                try:
                    n = await ingest_page(wiki, embedder, info, settings)
                    total_chunks += n
                    done += 1
                    log.info("[%d/%d] %s: %d chunks", done, len(todo), info.title, n)
                except Exception as exc:  # one bad page shouldn't stop the run
                    failures.append((info.title, repr(exc)))
                    log.error("Failed %s: %r", info.title, exc)

        await asyncio.gather(*(worker(p) for p in todo))

    log.info(
        "Done: %d pages, %d chunks in %.0fs, %d failures",
        done, total_chunks, time.perf_counter() - start, len(failures),
    )
    for title, error in failures:
        log.info("  failed: %s  %s", title, error)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="re-ingest every page")
    parser.add_argument("--limit", type=int, help="only process the first N pending pages")
    parser.add_argument("--title", action="append", help="only this page (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print chunks, write nothing")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # page titles like "Black Château" on Windows
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
