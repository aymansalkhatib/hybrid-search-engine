"""HTTP client the retrieval-service uses to read ORIGINAL docs by id for display.

This is the **by-id** read of the *original* documents the assignment grades (§2.11):
after ranking, the top-k results show the raw text, fetched from Mongo through the
doc-store (never pymongo directly — that keeps the SOA boundary intact). Missing ids
are simply absent from the result, so a doc-store blip degrades to "ids only" rather
than failing the whole search.
"""

from __future__ import annotations

import httpx

from shared.contracts import DocsRequest, DocsResponse


class DocStoreClient:
    """Thin wrapper over the doc-store REST API — query-time original-text lookups."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def fetch_originals(self, dataset_id: str, doc_ids: list[str]) -> dict[str, str]:
        """Return ``{doc_id: original_text}`` for the given ids (query-time display)."""
        if not doc_ids:
            return {}
        req = DocsRequest(dataset=dataset_id, doc_ids=doc_ids)
        resp = self._client.post("/docs", json=req.model_dump())
        resp.raise_for_status()
        body = DocsResponse(**resp.json())
        return {d.doc_id: d.text for d in body.docs}

    def close(self) -> None:
        self._client.close()
