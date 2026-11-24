"""Draft synthetic eval questions: the local LLM writes one question per random wiki chunk.

    python -m evals.generate --n 60 --seed 1

Overwrites evals/questions_synthetic.jsonl. Each question is labelled with the page and
section of the chunk it was written from. Review the file afterwards and delete questions
that are vague, wrong or not something a player would ask.

Known bias: questions written while looking at a chunk tend to reuse its words, which
flatters keyword search. That's why hand-written questions are reported separately.
"""

import argparse
import asyncio
import random
import sys
from collections import defaultdict

from sqlalchemy import func, select

from app.config import get_settings
from app.db import Chunk, Page, SessionLocal, engine
from app.providers import make_chat_model, ollama_client
from evals.dataset import SYNTHETIC_FILE, EvalQuestion, save
from evals.metrics import Target

PROMPT = """\
Here is an excerpt from the Deadlock wiki:

<excerpt>
{content}
</excerpt>

Write ONE question that a Deadlock player might ask, which this excerpt answers.
- Name the subject (hero, item, ability, mechanic) so the question makes sense on its own.
- Phrase it the way a player would type it. Don't copy sentences or unusual phrases from the excerpt.
- Ask about one specific fact.
- If the excerpt has nothing a player would plausibly ask about (for example it is only a list \
of names), reply SKIP.

Reply with only the question on one line, or SKIP."""


def is_usable(text: str) -> bool:
    return "\n" not in text and text.endswith("?") and 10 <= len(text) <= 200


async def main(n: int, seed: int, min_chars: int) -> None:
    settings = get_settings()
    try:
        async with SessionLocal() as session:
            rows = (
                await session.execute(
                    select(Page.title, Chunk.section, Chunk.content)
                    .join(Chunk.page)
                    .where(func.length(Chunk.content) >= min_chars)
                )
            ).all()
    finally:
        await engine.dispose()

    # One chunk per page, so a few huge pages don't dominate the set.
    by_page = defaultdict(list)
    for row in rows:
        by_page[row.title].append(row)
    rng = random.Random(seed)
    pages = rng.sample(sorted(by_page), k=len(by_page))  # shuffled; extra pages replace SKIPs

    questions: list[EvalQuestion] = []
    skipped = 0
    async with ollama_client(settings) as ollama:
        chat = make_chat_model(settings, ollama)
        for title in pages:
            if len(questions) == n:
                break
            chunk = rng.choice(by_page[title])
            messages = [{"role": "user", "content": PROMPT.format(content=chunk.content)}]
            text = "".join([token async for token in chat.stream(messages)]).strip()
            if not is_usable(text):
                skipped += 1
                continue
            questions.append(
                EvalQuestion(
                    id=f"syn-{len(questions) + 1:03d}",
                    question=text,
                    targets=[Target(chunk.title, chunk.section)],
                    source="synthetic",
                )
            )
            print(f"{len(questions):3d}. [{chunk.title} > {chunk.section or '(intro)'}] {text}")

    save(SYNTHETIC_FILE, questions)
    print(f"\nWrote {len(questions)} questions to {SYNTHETIC_FILE.name} ({skipped} chunks skipped)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m evals.generate")
    parser.add_argument("--n", type=int, default=60, help="number of questions")
    parser.add_argument("--seed", type=int, default=1, help="random seed (same seed = same chunks)")
    parser.add_argument("--min-chars", type=int, default=200, help="skip chunks shorter than this")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.seed, args.min_chars))
