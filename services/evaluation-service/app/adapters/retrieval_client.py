"""HTTP client the evaluation-service uses to run queries through the retrieval-service.

Evaluation reproduces the **real online path**: for each test query it calls the
retrieval-service's ``POST /search`` exactly as the UI would, then scores the returned
ranking against the qrels. We always pass ``with_text=False`` — evaluation needs the
ranked **ids and scores**, not the original text — which keeps each call light.

The underlying ``httpx.Client`` is shared across the evaluator's worker threads (httpx
clients are thread-safe), so a full run can issue several searches concurrently.
"""

from __future__ import annotations

from typing import Optional

import httpx

from shared.contracts import EvalRunSpec, SearchRequest, SearchResponse

# A ranked result: (external doc_id, score), higher = more relevant.
Scored = tuple[str, float]


class RetrievalClient:
    """Thin wrapper over the retrieval-service ``/search`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        """True if the service answers /health — used to fail an evaluation fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def search(
        self, *, dataset: str, spec: EvalRunSpec, query: str, top_k: int,
        cluster_rerank: bool = False, topic_rerank: bool = False,
    ) -> tuple[list[Scored], Optional[str], float]:
        """Run one query with the run's model config → ``([(doc_id, score)], mode, took_ms)``.

        ``took_ms`` is the retrieval-service's own measured search time (server-side, so it
        is a fair per-query latency regardless of how many queries we issue concurrently).
        ``cluster_rerank`` enables the extra-feature cluster re-ranking for the with/without
        comparison. Raises ``httpx.HTTPStatusError`` for a non-2xx (e.g. 404 = model not
        built), which the evaluator uses to fail just that run cleanly.
        """
        req = SearchRequest(
            dataset=dataset,
            model=spec.model,
            query=query,
            top_k=top_k,
            k1=spec.k1,
            b=spec.b,
            hybrid=spec.hybrid,
            with_text=False,
            cluster_rerank=cluster_rerank,
            topic_rerank=topic_rerank,
        )
        resp = self._client.post("/search", json=req.model_dump(mode="json"))
        resp.raise_for_status()
        body = SearchResponse(**resp.json())
        return [(h.doc_id, h.score) for h in body.hits], body.mode, body.took_ms

    def close(self) -> None:
        self._client.close()
