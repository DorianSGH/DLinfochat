"""Local cross-encoder reranker via sentence-transformers.

Optional: needs PyTorch + sentence-transformers (requirements-rerank.txt). Only imported
when a reranker model is configured, so the rest of the app runs without them.
"""

import asyncio

import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32) -> None:
        # Downloads the model from Hugging Face on first use, then loads from cache.
        # Uses the GPU automatically when PyTorch sees one. On GPU, half precision uses the
        # tensor cores: measured 3.6x faster (378 -> 106 ms for 30 pairs) with the same
        # scores to 3 decimals. CPUs gain nothing from it, so keep fp32 there.
        model_kwargs = {"dtype": torch.float16} if torch.cuda.is_available() else {}
        self.model = CrossEncoder(model_name, max_length=max_length, model_kwargs=model_kwargs)
        self.batch_size = batch_size
        # The first GPU call sets up CUDA kernels (~0.5s). Pay that at startup, not on
        # the first user's request.
        self.model.predict([("warm-up", "warm-up")], show_progress_bar=False)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        # predict() is synchronous and compute-bound: run it in a worker thread so the
        # event loop keeps serving other requests in the meantime.
        scores = await asyncio.to_thread(
            self.model.predict,
            [(query, doc) for doc in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
