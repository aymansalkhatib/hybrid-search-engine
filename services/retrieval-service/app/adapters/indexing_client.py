"""HTTP client for the indexing-service Boolean-match primitive.

Boolean (inverted-index-only) search matches documents by the postings lists alone —
no scoring model. The set algebra (AND/OR) runs where the index lives (the
indexing-service owns it and normalizes the query with its own build options), so this
client just passes the query + operator through and gets back the matched doc ids.
That mirrors how :class:`RepresentationClient` delegates per-model scoring.
"""

from __future__ import annotations

import httpx

from shared.contracts import BooleanMatchRequest, BooleanMatchResponse


class IndexingClient:
    """Thin wrapper over the indexing-service Boolean-match API."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        """True if the service answers /health — used to fail a search fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def match(
        self, *, dataset: str, query: str, operator: str, top_k: int
    ) -> BooleanMatchResponse:
        """Boolean-match the query over the inverted index → matched doc ids."""
        req = BooleanMatchRequest(dataset=dataset, query=query, operator=operator, top_k=top_k)
        resp = self._client.post("/match", json=req.model_dump())
        resp.raise_for_status()
        return BooleanMatchResponse(**resp.json())

    def close(self) -> None:
        self._client.close()
