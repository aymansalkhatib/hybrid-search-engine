"""Streaming ingest of a dataset's facts — framework/DB-independent.

Batches an item stream and hands each batch to a ``write_batch`` callback (implemented
by the Mongo adapter). The same batching core ingests **documents**, **queries** and
**qrels**; only the per-item row shape differs. Keeps memory bounded on 200K+ corpora
and knows nothing about MongoDB or HTTP.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger("doc-store-service")

T = TypeVar("T")

# Persist a batch of rows (the first field is always ``seq`` = position in the stream).
WriteBatch = Callable[[list], None]


def _ingest_batched(
    *,
    items: Iterable[T],
    make_row: Callable[[int, T], tuple],
    write_batch: WriteBatch,
    label: str,
    batch_size: int = 5000,
    progress_every: int = 50000,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Write **all** items in batches; return how many were written.

    Each item gets its 0-based stream position as ``seq`` (via ``make_row``), giving the
    store a stable order for browsing/paging. ``on_progress`` (if given) is called after
    each flushed batch with the running total.
    """
    batch: list[tuple] = []
    count = 0

    def flush() -> None:
        nonlocal count
        if not batch:
            return
        write_batch(list(batch))
        count += len(batch)
        if on_progress is not None:
            on_progress(count)
        if count % progress_every < len(batch):
            logger.info("ingested %d %s", count, label)
        batch.clear()

    for i, item in enumerate(items):
        batch.append(make_row(i, item))
        if len(batch) >= batch_size:
            flush()
    flush()
    return count


def ingest_documents(
    *,
    docs: Iterable,                 # yields objects exposing .doc_id and .text
    write_batch: WriteBatch,
    batch_size: int = 5000,
    progress_every: int = 50000,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Write all raw documents as ``(seq, doc_id, text)`` rows; return how many."""
    return _ingest_batched(
        items=docs,
        make_row=lambda i, d: (i, d.doc_id, d.text),
        write_batch=write_batch,
        label="raw docs",
        batch_size=batch_size,
        progress_every=progress_every,
        on_progress=on_progress,
    )


def ingest_queries(
    *,
    queries: Iterable,              # yields objects exposing .query_id and .text
    write_batch: WriteBatch,
    batch_size: int = 5000,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Write all test queries as ``(seq, query_id, text)`` rows; return how many."""
    return _ingest_batched(
        items=queries,
        make_row=lambda i, q: (i, q.query_id, q.text),
        write_batch=write_batch,
        label="queries",
        batch_size=batch_size,
        on_progress=on_progress,
    )


def ingest_qrels(
    *,
    qrels: Iterable,               # yields (query_id, doc_id, relevance) tuples
    write_batch: WriteBatch,
    batch_size: int = 5000,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Write all judgments as ``(seq, query_id, doc_id, relevance)`` rows; return how many."""
    return _ingest_batched(
        items=qrels,
        make_row=lambda i, r: (i, r[0], r[1], r[2]),
        write_batch=write_batch,
        label="qrels",
        batch_size=batch_size,
        on_progress=on_progress,
    )
