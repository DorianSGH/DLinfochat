"""Retrieval metrics. Pure functions, unit-tested in tests/test_eval_metrics.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """Where the answer to an eval question lives on the wiki.

    Labelled by page and section name, not chunk id: chunk ids change on every re-ingest,
    names don't. A long section split into several chunks counts as found if any of its
    chunks is retrieved.
    """

    page: str
    # Heading path such as "Abilities > Siphon Life"; "" is the page intro.
    # None means any chunk of the page counts.
    section: str | None = None

    def matches(self, page: str, section: str) -> bool:
        if page != self.page:
            return False
        if self.section is None:
            return True
        # A chunk from a subsection ("Souls > Haunted Souls > X") also answers a
        # question labelled with its parent section.
        return section == self.section or section.startswith(self.section + " > ")


def first_hit_rank(retrieved: list[tuple[str, str]], targets: list[Target]) -> int | None:
    """1-based rank of the first retrieved (page, section) matching any target, else None."""
    for rank, (page, section) in enumerate(retrieved, start=1):
        if any(target.matches(page, section) for target in targets):
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Share of questions with a relevant chunk in the top k (a.k.a. recall@k when
    each question has one relevant section)."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], k: int) -> float:
    """Mean reciprocal rank, cut off at k: 1.0 if the first result is relevant, 0.5 if
    the second is, ..., 0 if nothing relevant in the top k. Rewards ranking it *first*,
    not just somewhere in the top k."""
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)
