"""HTTP client the retrieval-service uses for cluster-based re-ranking.

Asks the clustering-service which cluster a batch of texts (the query + the candidate
pool's original texts) falls into. Used only when a search requests ``cluster_rerank``;
a 404 (no clustering built) or a connection error propagates so the caller can fall
back to the base ranking instead of failing the search.
"""

from __future__ import annotations

import httpx

from shared.contracts import AssignRequest, AssignResponse


class ClusteringClient:
    """Thin wrapper over the clustering-service ``/assign`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def assign(self, dataset: str, texts: list[str]) -> list[int]:
        """Return the cluster id for each text (same order). Raises on a non-2xx."""
        req = AssignRequest(dataset=dataset, texts=texts)
        resp = self._client.post("/assign", json=req.model_dump())
        resp.raise_for_status()
        body = AssignResponse(**resp.json())
        return [a.cluster_id for a in body.assignments]

    def close(self) -> None:
        self._client.close()
