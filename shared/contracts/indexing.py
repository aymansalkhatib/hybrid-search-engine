"""Contracts for the indexing-service (inverted index build + queries).

The inverted index stores postings and per-doc lengths — everything BM25/TF-IDF
need (`tf`, `df`, doc length, `avgdl`) — but **not** the raw document text. The
original text lives in the doc-store (MongoDB) and is fetched by ID at query time.
A dataset is referenced by its **id** (a value from the
configured ``DATASETS`` catalog), so callers stay dataset-agnostic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.jobs import JobRef
from shared.contracts.preprocessing import PreprocessOptions


class BuildIndexRequest(BaseModel):
    """Build the index over the whole corpus, or a positional range of it.

    Range selection (``start``/``stop``) indexes only docs ``[start:stop)`` of the
    dataset's native, stable order — handled by lazy slicing so a huge corpus never
    overloads memory. ``build all`` = leave both unset. ``limit`` alone is shorthand
    for ``stop = (start or 0) + limit`` (kept for quick smoke builds).
    """

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    start: Optional[int] = Field(
        default=None, ge=0, description="Range start (inclusive) over the corpus order"
    )
    stop: Optional[int] = Field(
        default=None, ge=1, description="Range stop (exclusive) over the corpus order"
    )
    limit: Optional[int] = Field(
        default=None, ge=1, description="Shorthand cap: stop = (start or 0) + limit"
    )
    force: bool = Field(default=False, description="Rebuild even if a cached index exists")

    @model_validator(mode="after")
    def _check_range(self) -> "BuildIndexRequest":
        if self.start is not None and self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self

    def effective_stop(self) -> Optional[int]:
        """Resolve ``stop`` taking the ``limit`` shorthand into account."""
        if self.stop is not None:
            return self.stop
        if self.limit is not None:
            return (self.start or 0) + self.limit
        return None


class IndexStats(BaseModel):
    """Summary of a built index (returned by /build and /stats)."""

    dataset_id: str         # concrete ir-datasets id
    num_docs: int
    vocab_size: int
    num_postings: int       # total (term, doc) pairs across the index
    total_tokens: int
    avg_doc_length: float   # BM25's avgdl
    version: str
    options: PreprocessOptions
    built_at: str           # ISO-8601 UTC
    cached: bool = False    # True when served from an existing build (not rebuilt)


class Posting(BaseModel):
    doc_id: str
    tf: int                 # term frequency within that document


class PostingsResponse(BaseModel):
    term: str
    df: int                 # document frequency (# docs containing the term)
    cf: int                 # collection frequency (sum of tf over all docs)
    postings: list[Posting]


class TermStats(BaseModel):
    term: str
    df: int
    cf: int


class DocInfo(BaseModel):
    """Per-doc info the index knows. Raw text is NOT here — see the doc-store."""

    doc_id: str
    length: int             # token count after preprocessing


class BuiltIndexes(BaseModel):
    datasets: list[str]     # concrete dataset ids that have a persisted index


class IndexStatusResponse(BaseModel):
    """Live indexing status for a dataset: what's built + any build in flight."""

    dataset_id: str                       # concrete ir-datasets id
    built: bool                           # a persisted index exists
    stats: Optional[IndexStats] = None    # summary of the built index (if any)
    active_job: Optional[JobRef] = None   # in-flight build (with live progress), if any
