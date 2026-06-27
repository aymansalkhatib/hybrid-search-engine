"""HTTP client the retrieval-service uses for cluster-based **pruning**.

Asks the clustering-service for the member doc ids of the query's nearest clusters — the
candidate pool a search is restricted to. Used only when a search requests
``cluster_prune``; a 404 (no clustering built) or a connection error propagates so the
caller can fall back to a full (un-pruned) search instead of failing.
"""

from __future__ import annotations

import httpx

from shared.contracts import ClusterMembersRequest, ClusterMembersResponse


class ClusteringClient:
    """Thin wrapper over the clustering-service ``/members`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def members(self, dataset: str, query: str, top_n: int, max_members: int | None = None) -> list[str]:
        """Doc ids in the query's ``top_n`` nearest clusters (the pruned search space).

        Empty when the query has no in-vocabulary terms. Raises on a non-2xx (e.g. 404 =
        no clustering built) so the caller falls back to a full search."""
        req = ClusterMembersRequest(dataset=dataset, query=query, top_n=top_n, max_members=max_members)
        resp = self._client.post("/members", json=req.model_dump())
        resp.raise_for_status()
        return ClusterMembersResponse(**resp.json()).doc_ids

    def close(self) -> None:
        self._client.close()
