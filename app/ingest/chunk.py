"""Split cleaned sections into chunks that fit the embedding sweet spot."""

import re
from dataclasses import dataclass

from app.ingest.clean import Section

# Fragments shorter than this (after the header) carry too little to be worth a vector.
MIN_BODY_CHARS = 40


@dataclass
class ChunkData:
    section: str
    content: str


def chunk_sections(title: str, sections: list[Section], max_chars: int) -> list[ChunkData]:
    """One chunk per section, split further on line then sentence boundaries if too long.

    Every chunk starts with "Title > Section" so it still makes sense on its own: a chunk
    saying "Cooldown 42s" is useless unless you know it's about Abrams' Siphon Life.
    """
    chunks: list[ChunkData] = []
    for section in sections:
        header = f"{title} > {section.heading}" if section.heading else title
        budget = max(max_chars - len(header) - 2, 200)
        for body in _pack(_units(section.text, budget), budget):
            if len(body) >= MIN_BODY_CHARS:
                chunks.append(ChunkData(section.heading, f"{header}\n\n{body}"))
    return chunks


def _units(text: str, budget: int) -> list[str]:
    """Lines, with any line longer than the budget broken into sentences, then hard-cut."""
    units: list[str] = []
    for line in text.split("\n"):
        if len(line) <= budget:
            units.append(line)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            while len(sentence) > budget:
                units.append(sentence[:budget])
                sentence = sentence[budget:]
            if sentence:
                units.append(sentence)
    return units


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedily join consecutive units into pieces no longer than the budget."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > budget:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(unit)
        size += len(unit) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces
