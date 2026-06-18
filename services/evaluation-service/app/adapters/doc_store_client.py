"""HTTP client the evaluation-service uses to read a dataset's ground truth.

The doc-store owns the dataset's persisted facts. Evaluation needs two of them:

* the **qrels** — ``GET /qrels/all`` returns the whole ``{query_id: {doc_id: relevance}}``
  map in one call (the exact shape ranx consumes); and
* the **test queries** — paged via ``GET /queries`` and assembled into ``{query_id: text}``
  so each judged query can be searched.

Reading through the doc-store API (never pymongo directly) keeps the SOA boundary intact:
the doc-store stays the single owner of the database.
"""

from __future__ import annotations

import httpx

from shared.contracts import AllQrelsResponse, DatasetStatus, QueryListResponse


class DocStoreClient:
    """Thin wrapper over the doc-store REST API — qrels + test queries for evaluation."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def dataset_status(self, dataset_id: str) -> DatasetStatus:
        """Readiness counts (docs/queries/qrels) — used to pre-flight an evaluation."""
        resp = self._client.get("/dataset/status", params={"dataset": dataset_id})
        resp.raise_for_status()
        return DatasetStatus(**resp.json())

    def all_qrels(self, dataset_id: str) -> dict[str, dict[str, int]]:
        """The full qrels as ``{query_id: {doc_id: relevance}}`` (the gold judgments)."""
        resp = self._client.get("/qrels/all", params={"dataset": dataset_id})
        resp.raise_for_status()
        return AllQrelsResponse(**resp.json()).qrels

    def all_queries(self, dataset_id: str, page_size: int = 5000) -> dict[str, str]:
        """Every stored test query as ``{query_id: text}`` (paged through ``/queries``)."""
        out: dict[str, str] = {}
        offset = 0
        while True:
            resp = self._client.get(
                "/queries", params={"dataset": dataset_id, "offset": offset, "limit": page_size}
            )
            resp.raise_for_status()
            page = QueryListResponse(**resp.json())
            for q in page.queries:
                out[q.query_id] = q.text
            offset += len(page.queries)
            if not page.queries or offset >= page.total:
                break
        return out

    def close(self) -> None:
        self._client.close()
