from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 (::1) first, and the
    # refused attempt costs ~2s per new connection because Docker/Ollama listen on IPv4.
    database_url: str = "postgresql+asyncpg://dlinfochat:dlinfochat@127.0.0.1:5433/dlinfochat"

    ollama_url: str = "http://127.0.0.1:11434"

    chat_model: str = "qwen3.5:9b"
    chat_num_ctx: int = 8192
    # Qwen3.5 can "think" before answering. Off by default: slower, and rarely needed
    # when the answer is already in the retrieved chunks.
    chat_think: bool = False
    # Low temperature: for answering from sources we want faithful, not creative.
    chat_temperature: float = 0.2

    embed_model: str = "qwen3-embedding:0.6b"
    embed_dimensions: int = 1024
    # Qwen3-Embedding's query format from its model card: "Instruct: <task>\nQuery:<question>".
    embed_query_prefix: str = (
        "Instruct: Given a question about the video game Deadlock, "
        "retrieve wiki passages that answer the question\nQuery:"
    )

    wiki_api_url: str = "https://deadlock.wiki/api.php"
    # MediaWiki asks API clients to identify themselves with a descriptive User-Agent.
    wiki_user_agent: str = "DLinfochat/0.1 (personal RAG learning project)"
    # Pages fetched at once. Kept low to be polite to a volunteer-run wiki.
    ingest_concurrency: int = 2
    chunk_max_chars: int = 1500

    # Hybrid search: each method returns this many candidates before fusion...
    retrieval_candidates: int = 30
    # ...and this many fused chunks go to the LLM (6 x <=1500 chars fits in chat_num_ctx).
    retrieval_top_k: int = 6
    rrf_k: int = 60
    # Cross-encoder that re-scores the fused candidates before the top k are picked, e.g.
    # "BAAI/bge-reranker-v2-m3". Empty = no reranking (and no PyTorch needed).
    reranker_model: str = ""
    # How many fused candidates the reranker re-scores. Evals: depth 10 missed answers
    # ranked 11-20 (hit@6 94%), depth 20 and 30 tied (99%), so 20 for less work.
    rerank_depth: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
