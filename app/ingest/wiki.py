"""Minimal async client for the MediaWiki Action API."""

import asyncio
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# Subpages that are mostly noise for Q&A (audio, images, transcripts) or would surface
# outdated numbers (patch history). Matched against the part after the last "/".
SKIPPED_SUBPAGES = {
    "update history", "voice lines", "sounds", "gallery", "transcription", "test", "zh-hans",
}
# The main page is a navigation portal, not content.
SKIPPED_TITLES = {"Deadlock Wiki"}


@dataclass(frozen=True)
class PageInfo:
    id: int
    title: str
    revision_id: int
    url: str


class WikiError(Exception):
    pass


class WikiClient:
    def __init__(self, client: httpx.AsyncClient, api_url: str) -> None:
        self.client = client
        self.api_url = api_url

    async def _get(self, **params: object) -> dict:
        # maxlag asks the wiki to refuse (rather than slowly serve) requests while its
        # database is under load; we then back off and retry.
        params |= {"format": "json", "formatversion": 2, "maxlag": 5}
        for attempt in range(5):
            response = await self.client.get(self.api_url, params=params)
            if response.status_code == 403 and "Just a moment" in response.text:
                raise WikiError(
                    "Blocked by Cloudflare's bot challenge. Some query parameters trigger it "
                    f"(e.g. gapfilterredir); request was: {response.url}"
                )
            if response.status_code in (429, 500, 502, 503, 504):
                await self._backoff(attempt, response.headers.get("retry-after"))
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                if data["error"].get("code") == "maxlag":
                    await self._backoff(attempt, response.headers.get("retry-after"))
                    continue
                raise WikiError(f"{data['error'].get('code')}: {data['error'].get('info')}")
            return data
        raise WikiError(f"Gave up after retries: {params}")

    @staticmethod
    async def _backoff(attempt: int, retry_after: str | None) -> None:
        delay = float(retry_after) if retry_after else 2**attempt
        log.warning("Wiki busy, retrying in %.0fs", delay)
        await asyncio.sleep(delay)

    async def list_pages(self) -> list[PageInfo]:
        """All main-namespace articles with their latest revision id.

        prop=info returns lastrevid for up to 500 pages per request, which is what makes
        incremental re-ingestion cheap: one pass of ~5 requests tells us what changed.
        """
        pages: list[PageInfo] = []
        cont: dict = {}
        while True:
            data = await self._get(
                action="query",
                generator="allpages",
                gapnamespace=0,
                gaplimit="max",
                prop="info",
                inprop="url",
                **cont,
            )
            for p in data["query"]["pages"]:
                # Filtered here instead of with gapfilterredir=nonredirects, which this
                # wiki's Cloudflare setup answers with a bot challenge.
                if p.get("redirect") or _is_skipped(p["title"]):
                    continue
                pages.append(PageInfo(p["pageid"], p["title"], p["lastrevid"], p["fullurl"]))
            if "continue" not in data:
                return pages
            cont = data["continue"]

    async def parse(self, page_id: int) -> tuple[str, int]:
        """Rendered HTML of a page (templates expanded), plus the revision it came from."""
        data = await self._get(
            action="parse",
            pageid=page_id,
            prop="text|revid",
            disabletoc=1,
            disableeditsection=1,
        )
        return data["parse"]["text"], data["parse"]["revid"]


def _is_skipped(title: str) -> bool:
    if title in SKIPPED_TITLES:
        return True
    return "/" in title and title.rsplit("/", 1)[1].lower() in SKIPPED_SUBPAGES
