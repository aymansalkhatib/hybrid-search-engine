"""HTTP client the retrieval-service uses for topic-based re-ranking.

Asks the topic-service for the dominant LDA topic of a batch of texts (the query + the
candidate pool's original texts). Used only when a search requests ``topic_rerank``; a
404 (no topic model built) or a connection error propagates so the caller can fall back
to the base ranking instead of failing the search.
"""

from __future__ import annotations

import httpx

from shared.contracts import InferRequest, InferResponse


class TopicClient:
    """Thin wrapper over the topic-service ``/infer`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def dominant_topics(self, dataset: str, texts: list[str]) -> list[int]:
        """Return the dominant topic id for each text (same order). Raises on a non-2xx."""
        req = InferRequest(dataset=dataset, texts=texts)
        resp = self._client.post("/infer", json=req.model_dump())
        resp.raise_for_status()
        body = InferResponse(**resp.json())
        return [r.dominant_topic for r in body.results]

    def close(self) -> None:
        self._client.close()
