"""Build an :class:`InvertedIndex` from a document stream.

Preprocessing is delegated over HTTP to the preprocessing-service (SOA: no
service imports another service's internals). Documents are streamed and flushed
in fixed-size batches so memory and request sizes stay bounded on 200K+ corpora.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Iterable

from app.domain.inverted_index import INDEX_VERSION, InvertedIndex

logger = logging.getLogger("indexing-service")

# Given a batch of texts, return a parallel list of token lists.
PreprocessBatch = Callable[[list[str]], list[list[str]]]


def build_index(
    *,
    dataset_id: str,
    docs: Iterable,                 # yields objects exposing .doc_id and .text
    preprocess_batch: PreprocessBatch,
    options: dict,                  # PreprocessOptions.model_dump() — recorded in the index
    batch_size: int = 1000,
    limit: int | None = None,
    progress_every: int = 20000,
    on_progress: Callable[[int], None] | None = None,
) -> InvertedIndex:
    """Build the index from ``docs`` (already sliced to the wanted range by the caller).

    ``on_progress`` (if given) is called after each flushed batch with the running
    count — used by the job runner to report a live percentage.
    """
    index = InvertedIndex(dataset_id=dataset_id, version=INDEX_VERSION, options=options)
    ids: list[str] = []
    texts: list[str] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not ids:
            return
        token_lists = preprocess_batch(texts)
        # The batch endpoint returns one token list per input text. A mismatch
        # would silently misalign doc_ids and tokens (zip truncates), so fail loud.
        if len(token_lists) != len(ids):
            raise RuntimeError(
                f"preprocessing returned {len(token_lists)} token lists "
                f"for {len(ids)} documents — corpus/token misalignment"
            )
        for doc_id, tokens in zip(ids, token_lists):
            index.add_document(doc_id, tokens)
        processed += len(ids)
        if on_progress is not None:
            on_progress(processed)
        if processed % progress_every < len(ids):
            logger.info("indexed %d docs (vocab=%d)", processed, index.vocab_size)
        ids.clear()
        texts.clear()

    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            break
        ids.append(doc.doc_id)
        texts.append(doc.text)
        if len(ids) >= batch_size:
            flush()
    flush()

    index.built_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        "index built for %s: %d docs, vocab=%d, postings=%d, avgdl=%.2f",
        dataset_id, index.num_docs, index.vocab_size, index.num_postings, index.avg_doc_length,
    )
    return index
