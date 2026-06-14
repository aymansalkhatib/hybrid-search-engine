"""Contracts for the representation-service (offline build + query encoding).

A *representation* turns the corpus into numbers a retriever can score: TF-IDF
(VSM) today; BM25 stats and dense embeddings are pluggable additions later. The
matrix and fitted model are heavy artifacts built **offline** and loaded ready at
startup (nothing is fitted during a query). The raw text still
lives only in the doc-store (read by ID at query time); a representation references
documents by their **external id**, aligned row-for-row with the matrix.

A dataset is referenced by its **id** (a value from the configured ``DATASETS``
catalog), so callers stay dataset-agnostic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.jobs import JobRef
from shared.contracts.preprocessing import PreprocessOptions


class TfidfParams(BaseModel):
    """Vectorizer knobs for the TF-IDF model — recorded with the artifact so a
    query is later weighted exactly the way the corpus was."""

    min_df: int = Field(
        default=1, ge=1, description="Drop terms appearing in fewer than N documents"
    )
    max_df: float = Field(
        default=1.0, gt=0.0, le=1.0,
        description="Drop terms appearing in more than this fraction of documents",
    )
    sublinear_tf: bool = Field(
        default=True, description="Use 1 + log(tf) term-frequency scaling (recommended)"
    )


class BuildRepresentationRequest(BaseModel):
    """Build a representation over the **whole corpus** (every ingested doc),
    optionally restricted to a positional range via ``start``/``stop``.

    The preprocessing ``options`` are part of the request: the corpus is tokenized
    with exactly these settings, and queries are later normalized the same way to
    stay comparable. ``build all`` = leave ``start``/``stop`` unset.
    """

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    model: str = Field(default="tfidf", description="Representation model to build")
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    params: TfidfParams = Field(default_factory=TfidfParams)
    start: Optional[int] = Field(
        default=None, ge=0, description="Range start (inclusive) over the corpus order"
    )
    stop: Optional[int] = Field(
        default=None, ge=1, description="Range stop (exclusive) over the corpus order"
    )
    force: bool = Field(default=False, description="Rebuild even if a cached model exists")

    @model_validator(mode="after")
    def _check_range(self) -> "BuildRepresentationRequest":
        if self.start is not None and self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self


class RepresentationStats(BaseModel):
    """Summary of a built representation (returned by /build and /stats)."""

    dataset_id: str          # concrete ir-datasets id
    model: str
    num_docs: int            # rows in the doc-term matrix
    vocab_size: int          # columns (unique terms kept)
    nnz: int                 # non-zero weights in the matrix
    density: float           # nnz / (num_docs * vocab_size) — sparsity gauge
    version: str
    options: PreprocessOptions
    params: TfidfParams
    built_at: str            # ISO-8601 UTC
    cached: bool = False     # True when served from an existing build (not rebuilt)


class RepresentationStatusResponse(BaseModel):
    """Live status for a (dataset, model): what's built + any build in flight."""

    dataset_id: str
    model: str
    built: bool
    stats: Optional[RepresentationStats] = None
    active_job: Optional[JobRef] = None


class BuiltRepresentation(BaseModel):
    dataset_id: str
    model: str


class BuiltRepresentations(BaseModel):
    items: list[BuiltRepresentation]


class RepresentationDeleteResult(BaseModel):
    dataset_id: str
    model: str
    deleted: bool            # True if a persisted artifact was removed
    was_cached: bool         # True if an in-memory copy was evicted


class EncodeRequest(BaseModel):
    """Vectorize query text(s) with a built model — the same weighting the corpus
    got, so query and documents live in one space (used by retrieval, and by the
    console to *see* the algorithm)."""

    dataset: str
    model: str = "tfidf"
    texts: list[str] = Field(min_length=1, description="Query texts to vectorize")
    top_terms: int = Field(
        default=30, ge=1, le=500, description="Max weighted terms returned per text"
    )


class WeightedTerm(BaseModel):
    term: str
    weight: float


class EncodedVector(BaseModel):
    nnz: int                       # non-zero terms in this query's vector
    terms: list[WeightedTerm]      # top-weighted terms, descending


class EncodeResponse(BaseModel):
    model: str
    dim: int                       # vocabulary size (vector dimensionality)
    options: PreprocessOptions     # the options the query was normalized with
    vectors: list[EncodedVector]
