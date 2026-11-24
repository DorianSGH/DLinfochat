"""API tests with fake retriever and chat model: no database, Ollama or network needed."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_chat_model, get_retriever
from app.rag.retrieve import Source

SOURCE = Source(
    chunk_id=1,
    title="Abrams",
    section="Abilities > Siphon Life",
    url="https://deadlock.wiki/Abrams#Siphon_Life",
    content="Abrams > Abilities > Siphon Life\n\nCooldown 42s",
    score=0.03,
    semantic_rank=1,
    keyword_rank=2,
)


class FakeRetriever:
    async def search(self, question: str) -> list[Source]:
        return [SOURCE]


class FakeChat:
    def __init__(self, tokens: list[str], fail_after: int | None = None) -> None:
        self.tokens = tokens
        self.fail_after = fail_after
        self.messages: list | None = None

    async def stream(self, messages):
        self.messages = messages
        for i, token in enumerate(self.tokens):
            if i == self.fail_after:
                raise RuntimeError("model crashed")
            yield token


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        if "data" in fields:  # skip keep-alive comment blocks
            events.append((fields.get("event"), json.loads(fields["data"])))
    return events


@pytest.fixture
def use_fakes():
    def install(chat: FakeChat, retriever=None):
        app.dependency_overrides[get_retriever] = lambda: retriever or FakeRetriever()
        app.dependency_overrides[get_chat_model] = lambda: chat

    yield install
    app.dependency_overrides.clear()


def test_chat_streams_sources_then_tokens_then_done(use_fakes):
    chat = FakeChat(["Siphon Life has a ", "42s cooldown.\n", "[1]"])
    use_fakes(chat)

    response = TestClient(app).post("/api/chat", json={"question": "Siphon Life cooldown?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[0] == (
        "sources",
        [{"n": 1, "title": "Abrams", "section": "Abilities > Siphon Life",
          "url": "https://deadlock.wiki/Abrams#Siphon_Life"}],
    )
    tokens = [data["text"] for event, data in events if event == "token"]
    assert "".join(tokens) == "Siphon Life has a 42s cooldown.\n[1]"  # newline survived
    assert events[-1] == ("done", {})
    # The retrieved chunk reached the model, numbered, along with the question.
    prompt = chat.messages[-1]["content"]
    assert "[1] Abrams > Abilities > Siphon Life" in prompt
    assert prompt.endswith("Question: Siphon Life cooldown?")


def test_generation_failure_is_reported_in_stream(use_fakes):
    use_fakes(FakeChat(["partial ", "answer"], fail_after=1))

    response = TestClient(app).post("/api/chat", json={"question": "anything"})

    events = parse_sse(response.text)
    assert [event for event, _ in events] == ["sources", "token", "error"]


def test_retrieval_failure_is_a_real_http_error(use_fakes):
    class BrokenRetriever:
        async def search(self, question):
            raise RuntimeError("database down")

    use_fakes(FakeChat([]), retriever=BrokenRetriever())

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/chat", json={"question": "anything"}
    )
    assert response.status_code == 500


def test_blank_question_is_rejected(use_fakes):
    use_fakes(FakeChat([]))
    response = TestClient(app).post("/api/chat", json={"question": "   "})
    assert response.status_code == 422
