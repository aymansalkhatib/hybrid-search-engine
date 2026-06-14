"""HTTP client the builder uses to read the corpus from the doc-store.

DB-centric pipeline: a representation is fitted from the **stored**
documents in MongoDB, fetched a page at a time through the doc-store (never pymongo
directly — that keeps the SOA boundary intact). Same paginated endpoint the indexer
and the UI's database browser use, so every view of the corpus comes from one source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import httpx

from shared.contracts import DocListResponse


@dataclass(frozen=True)
class Doc:
    """A document streamed from the store (what the builder consumes)."""

    doc_id: str
    text: str


class DocStoreClient:
    """Thin wrapper over the doc-store REST API used as the build source."""

    def __init__(self, base_url: str, *, page_size: int = 2000, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._page_size = page_size

    def is_healthy(self) -> bool:
        """True if the doc-store answers /health — used to fail builds fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def ingested_count(self, dataset_id: str) -> int:
        """How many docs are stored for this dataset (0 ⇒ not ingested yet)."""
        resp = self._client.get("/dataset/status", params={"dataset": dataset_id})
        resp.raise_for_status()
        return resp.json()["ingested_count"]

    def iter_docs(
        self, dataset_id: str, start: int | None = None, stop: int | None = None
    ) -> Iterator[Doc]:
        """Yield stored docs as ``Doc(doc_id, text)`` for seq range ``[start:stop)``.

        Pages through the doc-store ``/docs/list`` endpoint so memory stays bounded
        regardless of corpus size. ``start``/``stop`` are seq positions; omit both for
        the whole stored corpus.
        """
        seq = start or 0
        while stop is None or seq < stop:
            page_limit = self._page_size if stop is None else min(self._page_size, stop - seq)
            resp = self._client.get(
                "/docs/list",
                params={"dataset": dataset_id, "offset": seq, "limit": page_limit},
            )
            resp.raise_for_status()
            page = DocListResponse(**resp.json())
            if not page.docs:
                break
            for item in page.docs:
                yield Doc(doc_id=item.doc_id, text=item.text)
            seq += len(page.docs)
            if len(page.docs) < page_limit:  # short page ⇒ reached the end
                break

    def close(self) -> None:
        self._client.close()
