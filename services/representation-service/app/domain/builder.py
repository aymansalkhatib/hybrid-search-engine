"""Build a representation from a document stream.

Documents are streamed from the doc-store and (for the lexical models) preprocessed
over HTTP in fixed-size batches, so memory and request sizes stay bounded on 200K+
corpora. The embedding model reads **raw** text (it tokenizes internally), so for it
we skip preprocessing entirely. The fitted model object is returned; persisting it is
the caller's job.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator

from app.domain.base import BaseRepresentation, get_model_class

logger = logging.getLogger("representation-service")

# Given a batch of raw texts, return a parallel list of normalized (space-joined) texts.
NormalizeBatch = Callable[[list[str]], list[str]]


def build_representation(
    *,
    model: str,
    dataset_id: str,
    docs: Iterable,                 # yields objects exposing .doc_id and .text
    normalize_batch: NormalizeBatch,
    options: dict,                  # PreprocessOptions.model_dump() — recorded with the model
    params: dict,                   # model-specific params (model_dump of the params model)
    batch_size: int = 1000,
    progress_every: int = 20000,
    on_progress: Callable[[int], None] | None = None,
) -> BaseRepresentation:
    """Stream ``docs`` → (preprocess if lexical) → fit ``model``. Returns the ready model.

    ``doc_ids`` are collected in stream order and stay aligned row-for-row with the
    fitted matrix (the model consumes ``corpus`` in the same order they are appended).
    """
    model_cls = get_model_class(model)
    if model_cls is None:
        raise ValueError(f"unknown representation model '{model}'")
    needs_pp = model_cls.requires_preprocessing

    doc_ids: list[str] = []
    processed = 0

    def _flush(ids: list[str], texts: list[str]) -> list[str]:
        nonlocal processed
        out = normalize_batch(texts) if needs_pp else list(texts)
        # One output text per input, or doc_ids and rows would silently misalign.
        if len(out) != len(ids):
            raise RuntimeError(
                f"preprocessing returned {len(out)} texts for {len(ids)} documents "
                "— corpus/text misalignment"
            )
        doc_ids.extend(ids)
        processed += len(ids)
        if on_progress is not None:
            on_progress(processed)
        if processed % progress_every < len(ids):
            logger.info("prepared %d docs for %s", processed, model)
        return out

    def corpus() -> Iterator[str]:
        ids: list[str] = []
        texts: list[str] = []
        for doc in docs:
            ids.append(doc.doc_id)
            texts.append(doc.text)
            if len(ids) >= batch_size:
                yield from _flush(ids, texts)
                ids, texts = [], []
        if ids:
            yield from _flush(ids, texts)

    rep = model_cls.build(
        dataset_id=dataset_id, corpus=corpus(), doc_ids=doc_ids, options=options, params=params,
    )
    extra = rep.stats_extra()
    logger.info("%s built for %s: %d docs %s", model, dataset_id, rep.num_docs, extra)
    return rep
