from app.ingest.chunk import chunk_sections
from app.ingest.clean import Section


def test_chunks_carry_header_and_respect_max_chars():
    text = "\n".join(f"Line number {i} has some words in it." for i in range(200))
    chunks = chunk_sections("Abrams", [Section("Abilities > Siphon Life", text)], max_chars=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.startswith("Abrams > Abilities > Siphon Life\n\n")
        assert len(chunk.content) <= 500
        assert chunk.section == "Abilities > Siphon Life"


def test_overlong_line_is_split_on_sentences():
    line = " ".join(["This sentence keeps going on and on."] * 60)
    chunks = chunk_sections("Souls", [Section("", line)], max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.content) <= 500 for c in chunks)
    assert chunks[0].content.startswith("Souls\n\n")
    # No sentence is cut in half when sentences fit the budget.
    assert all(c.content.endswith(".") for c in chunks)


def test_tiny_fragments_are_dropped():
    assert chunk_sections("Abrams", [Section("Trivia", "Short.")], max_chars=500) == []
