"""Streaming ingest of raw documents — framework/DB-independent.

Batches a document stream and hands each batch to a ``write_batch`` callback
(implemented by the Mongo adapter). Keeps memory bounded on 200K+ corpora and
knows nothing about MongoDB or HTTP.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

logger = logging.getLogger("doc-store-service")

# Persist a batch of (seq, doc_id, raw_text) rows (seq = position in the corpus).
WriteBatch = Callable[[list[tuple[int, str, str]]], None]


def ingest_documents(
    *,
    docs: Iterable,                 # yields objects exposing .doc_id and .text
    write_batch: WriteBatch,
    batch_size: int = 5000,
    limit: int | None = None,
    progress_every: int = 50000,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Write the (raw) documents in batches; return how many were ingested.

    Each doc is stored with its ``seq`` (0-based position in the corpus) so the store
    has a stable order for browsing and for the indexer to page through. ``on_progress``
    (if given) is called after each flushed batch with the running total.
    """
    batch: list[tuple[int, str, str]] = []
    count = 0

    def flush() -> None:
        nonlocal count
        if not batch:
            return
        write_batch(batch)
        count += len(batch)
        if on_progress is not None:
            on_progress(count)
        if count % progress_every < len(batch):
            logger.info("ingested %d raw docs", count)
        batch.clear()

    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            break
        batch.append((i, doc.doc_id, doc.text))
        if len(batch) >= batch_size:
            flush()
    flush()
    return count
