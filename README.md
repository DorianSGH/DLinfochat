# DLinfochat

RAG chatbot that answers Deadlock questions from [deadlock.wiki](https://deadlock.wiki).

## Setup

Requires Python 3.12+, Docker, and [Ollama](https://ollama.com) running locally.

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
docker compose up -d --wait
```

## Commands

| Command | What it does |
|---|---|
| `fastapi dev app/main.py` | Run the API with auto-reload; interactive docs at http://127.0.0.1:8000/docs. |
| `python -m scripts.ask "question"` | Answer a question in the terminal (same pipeline as the API), showing sources and ranks. |
| `python -m app.ingest` | Fetch new/changed wiki pages, chunk, embed, store. Re-run any time. |
| `python -m app.ingest --full` | Re-ingest every page (e.g. after changing the cleaner). |
| `python -m app.ingest --title Abrams --dry-run` | Print the chunks for a page; no embedding, no DB writes. |
| `python -m scripts.search "question"` | Show the 5 chunks closest to a question. |
| `python -m scripts.smoke_test` | Check the local chat and embedding models respond. |
| `python -m evals.retrieval [--misses]` | Score retrieval (hit@k, MRR) for vector, keyword and hybrid search on the eval questions. |
| `python -m evals.generate --n 60` | Regenerate the synthetic eval questions with the local LLM (review them afterwards). |
| `python -m pytest` | Unit tests (cleaner, chunker, fusion, API, eval metrics). |

Run them with the venv's Python (`.venv\Scripts\python`) from the project root.

## API

| Endpoint | Description |
|---|---|
| `POST /api/chat` `{"question": "..."}` | Streams Server-Sent Events: `sources` (numbered list the answer cites), then `token` events (`{"text": ...}`), then `done`, or `error` if generation fails midway. |
| `GET /api/search?q=...` | The fused retrieval results with per-method ranks, for debugging. |
| `GET /health` | Liveness check. |

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "What is the cooldown of Siphon Life?"}'
```

## Evaluation

Retrieval is scored separately from answer quality: if the chunk containing the answer
never reaches the LLM, the answer can't be right.

- **Questions** live in `evals/questions_manual.jsonl` (hand-written) and
  `evals/questions_synthetic.jsonl` (written by the local LLM from random wiki chunks, then
  reviewed). Each is labelled with the wiki page and section that answers it, not chunk ids,
  which change on every re-ingest.
- **Metrics:** hit@k (is an answering chunk in the top k?) and MRR@10 (how high is the
  first one?). The LLM sees the top 6, so hit@6 is the number that matters most; hit@30 is
  the ceiling a reranker could reach.
- **Results are reported per question source**, because synthetic questions reuse the
  chunk's wording and so flatter keyword search.
- **Label errors** are fixed only after reading the retrieved chunk and confirming it
  answers the question.

## Layout

```
app/main.py          FastAPI app: /api/chat (SSE), /api/search, /health
app/rag/             hybrid retrieval (vector + full-text, RRF) and prompt building
app/config.py        settings (override via .env, see .env.example)
app/providers/       Embedder / ChatModel interfaces + Ollama implementations
app/db/              SQLAlchemy models (pages, chunks) and engine
app/ingest/          wiki API client, HTML cleaner, chunker, CLI
evals/               retrieval eval questions, metrics and runner
scripts/             search and smoke-test helpers
tests/               pytest
```
