"""Contracts for the indexing-service (inverted index build + queries).

The inverted index stores postings and per-doc lengths — everything BM25/TF-IDF
need (`tf`, `df`, doc length, `avgdl`) — but **not** the raw document text. The
original text lives in the doc-store (MongoDB) and is fetched by ID at query time.
A dataset is referenced by its **id** (a value from the
configured ``DATASETS`` catalog), so callers stay dataset-agnostic.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.jobs import JobRef
from shared.contracts.preprocessing import PreprocessOptions

BooleanOperator = Literal["and", "or"]


class BuildIndexRequest(BaseModel):
    """Build the index over the **whole corpus** (the index always covers every
    ingested doc), optionally restricted to a positional range via ``start``/``stop``.

    The preprocessing ``options`` are part of the request: the corpus is tokenized
    with exactly these settings, and queries must later use the same ones to stay
    comparable. ``build all`` = leave ``start``/``stop`` unset.
    """

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    start: Optional[int] = Field(
        default=None, ge=0, description="Range start (inclusive) over the corpus order"
    )
    stop: Optional[int] = Field(
        default=None, ge=1, description="Range stop (exclusive) over the corpus order"
    )
    force: bool = Field(default=False, description="Rebuild even if a cached index exists")

    @model_validator(mode="after")
    def _check_range(self) -> "BuildIndexRequest":
        if self.start is not None and self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self

    def effective_stop(self) -> Optional[int]:
        """The exclusive range stop (``None`` ⇒ index to the end of the corpus)."""
        return self.stop


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


# --------------------------------------------------------------------------- #
#  Boolean match — inverted-index-only retrieval (no scoring model)
# --------------------------------------------------------------------------- #
#
# The oldest, most direct retrieval model: match documents that contain the query
# terms using the postings lists alone — set intersection (AND) or union (OR), with
# **no** relevance scoring (no TF-IDF / BM25). This is an **internal** primitive the
# retrieval-service calls; the index owns its build options, so it normalizes the raw
# query itself (exactly as the representation-service does for /rank).


class MatchedTerm(BaseModel):
    """A normalized query term and how many docs contain it (df, 0 ⇒ absent)."""

    term: str
    df: int


class BooleanMatchRequest(BaseModel):
    """Boolean retrieval over the inverted index — match-only, no ranking model."""

    dataset: str
    query: str = Field(min_length=1)
    operator: BooleanOperator = Field(default="and", description="AND = all terms; OR = any term")
    top_k: int = Field(default=10, ge=1, le=1000)


class BooleanMatchHit(BaseModel):
    doc_id: str
    matched: int            # how many distinct query terms this doc contains (coordination)


class BooleanMatchResponse(BaseModel):
    dataset_id: str
    operator: str
    terms: list[MatchedTerm]    # the normalized query terms (+ df), for transparency
    total_matched: int          # size of the matched set before top_k
    hits: list[BooleanMatchHit]  # up to top_k, ordered by coordination desc then doc_id


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


class IndexDeleteResult(BaseModel):
    """Result of removing a dataset's built index (artifact + in-memory cache)."""

    dataset_id: str
    index_deleted: bool     # True if a persisted artifact was removed
    was_cached: bool        # True if an in-memory copy was evicted
