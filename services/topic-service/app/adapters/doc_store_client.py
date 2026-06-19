"""HTTP client the topic modeller uses to read the corpus from the doc-store.

Same SOA boundary the other builders keep: the corpus is fetched a page at a time
through the doc-store REST API (never pymongo directly). Build-time only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import httpx

from shared.contracts import DocListResponse


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str


class DocStoreClient:
    """Thin wrapper over the doc-store REST API used as the build source."""

    def __init__(self, base_url: str, *, page_size: int = 2000, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._page_size = page_size

    def is_healthy(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def ingested_count(self, dataset_id: str) -> int:
        resp = self._client.get("/dataset/status", params={"dataset": dataset_id})
        resp.raise_for_status()
        return resp.json()["ingested_count"]

    def iter_docs(self, dataset_id: str, stop: int | None = None) -> Iterator[Doc]:
        seq = 0
        while stop is None or seq < stop:
            page_limit = self._page_size if stop is None else min(self._page_size, stop - seq)
            resp = self._client.get(
                "/docs/list", params={"dataset": dataset_id, "offset": seq, "limit": page_limit}
            )
            resp.raise_for_status()
            page = DocListResponse(**resp.json())
            if not page.docs:
                break
            for item in page.docs:
                yield Doc(doc_id=item.doc_id, text=item.text)
            seq += len(page.docs)
            if len(page.docs) < page_limit:
                break

    def close(self) -> None:
        self._client.close()
