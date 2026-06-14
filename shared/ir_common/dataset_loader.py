"""Dataset-agnostic loader built on top of `ir_datasets`.

This is the *single* place in the codebase that knows about concrete datasets.
Everything else depends only on the uniform interface below, so switching datasets
is just editing the `DATASETS` list in `.env` — never a code change.

Example
-------
    loader = DatasetLoader("beir/quora/test")
    for doc in loader.iter_docs():
        ...
    qrels = loader.get_qrels()          # {query_id: {doc_id: relevance}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import ir_datasets


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


class DatasetLoader:
    """Uniform wrapper over an `ir_datasets` dataset."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self._ds = ir_datasets.load(dataset_id)

    def iter_docs(self) -> Iterator[Doc]:
        """Yield documents as `Doc(doc_id, text)` in the dataset's native order.

        Used by the **download** (materialize the corpus locally) and **ingest**
        (migrate raw docs to MongoDB) steps. Range selection happens later, at the
        doc-store level (by ``seq``), so the indexer pages a stored range from Mongo
        rather than re-reading ir_datasets.
        """
        for d in self._ds.docs_iter():
            yield Doc(doc_id=d.doc_id, text=_doc_text(d))

    def iter_queries(self) -> Iterator[Query]:
        """Yield queries as `Query(query_id, text)`."""
        for q in self._ds.queries_iter():
            yield Query(query_id=q.query_id, text=_query_text(q))

    def get_qrels(self) -> dict[str, dict[str, int]]:
        """Return relevance judgments as {query_id: {doc_id: relevance}}."""
        qrels: dict[str, dict[str, int]] = {}
        for qrel in self._ds.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
        return qrels

    def doc_count(self) -> int:
        """Number of documents (fast when the dataset exposes a count)."""
        try:
            return self._ds.docs_count()
        except Exception:  # pragma: no cover - fallback for datasets w/o count
            return sum(1 for _ in self._ds.docs_iter())

    def has_qrels(self) -> bool:
        return self._ds.has_qrels()

    def metadata(self) -> dict:
        """ir-datasets' shipped metadata (doc/query/qrel counts) for this dataset.

        Network-free: read from the package's bundled metadata, so it works as a
        **preview before any download**. Shape (keys optional per dataset)::

            {"docs": {"count": N}, "queries": {"count": M}, "qrels": {"count": K}}

        Returns ``{}`` for datasets that ship no metadata.
        """
        try:
            return self._ds.metadata() or {}
        except Exception:  # pragma: no cover - dataset without a metadata provider
            return {}


def _doc_text(doc) -> str:
    """Best-effort extraction of a document's searchable text.

    `ir_datasets` doc tuples vary by dataset; we combine title + body when present.
    """
    title = (getattr(doc, "title", "") or "").strip()
    body = getattr(doc, "text", None)
    if body is None:
        for attr in ("body", "contents", "abstract"):
            body = getattr(doc, attr, None)
            if body:
                break
    body = (body or "").strip()
    return f"{title} {body}".strip()


def _query_text(query) -> str:
    """Best-effort extraction of a query's text."""
    for attr in ("text", "title", "query", "description"):
        val = getattr(query, attr, None)
        if val:
            return str(val).strip()
    return ""
