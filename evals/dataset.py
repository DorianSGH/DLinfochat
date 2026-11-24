"""Load and save eval questions (JSON Lines: one question per line, easy to diff and edit)."""

import json
from dataclasses import dataclass
from pathlib import Path

from evals.metrics import Target

EVALS_DIR = Path(__file__).parent
MANUAL_FILE = EVALS_DIR / "questions_manual.jsonl"  # hand-written; edit freely
SYNTHETIC_FILE = EVALS_DIR / "questions_synthetic.jsonl"  # generated; review, then edit


@dataclass
class EvalQuestion:
    id: str
    question: str
    # Empty = the wiki doesn't answer it. Skipped by retrieval evals, kept for checking
    # that the bot says it doesn't know.
    targets: list[Target]
    source: str  # "manual" or "synthetic"
    note: str | None = None  # for humans: partial answers, false premises, etc.


def load(path: Path) -> list[EvalQuestion]:
    if not path.exists():
        return []
    questions = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            questions.append(
                EvalQuestion(
                    id=row["id"],
                    question=row["question"],
                    targets=[Target(t["page"], t.get("section")) for t in row["targets"]],
                    source=row["source"],
                    note=row.get("note"),
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path.name} line {line_no}: {exc!r}") from exc
    return questions


def load_all() -> list[EvalQuestion]:
    return load(MANUAL_FILE) + load(SYNTHETIC_FILE)


def save(path: Path, questions: list[EvalQuestion]) -> None:
    def to_row(q: EvalQuestion) -> dict:
        row = {
            "id": q.id,
            "question": q.question,
            "targets": [{"page": t.page, "section": t.section} for t in q.targets],
            "source": q.source,
        }
        if q.note:
            row["note"] = q.note
        return row

    lines = [json.dumps(to_row(q), ensure_ascii=False) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
