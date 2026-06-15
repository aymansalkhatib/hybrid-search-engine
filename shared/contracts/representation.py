"""Contracts for the representation-service (offline build + online query/search).

A *representation* turns the corpus into something a retriever can score. The
assignment (§2) requires four, all hosted here:

* **TF-IDF (VSM)** — sparse weights, cosine similarity.
* **BM25** — probabilistic ranking with per-query tunable ``k1`` / ``b``.
* **Embedding** — dense vectors (sentence-transformers / BERT).
* **Hybrid** — combine the above two ways: **serial** (one model retrieves
  candidates, another re-ranks) and **parallel** (run several models and merge
  their ranked lists with a **fusion method**, e.g. RRF).

Heavy artifacts are built **offline** and loaded ready at startup (no fitting at
query time). Raw text lives only in the doc-store and is fetched
**by id** at query time for display. Datasets/models are referenced by id/name, so
callers stay dataset- and model-agnostic.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.jobs import JobRef
from shared.contracts.preprocessing import PreprocessOptions


# --------------------------------------------------------------------------- #
#  Build — per-model parameters
# --------------------------------------------------------------------------- #

class TfidfParams(BaseModel):
    """Vectorizer knobs for TF-IDF — recorded with the artifact so a query is
    weighted exactly the way the corpus was."""

    min_df: int = Field(default=1, ge=1, description="Drop terms appearing in fewer than N docs")
    max_df: float = Field(default=1.0, gt=0.0, le=1.0,
                          description="Drop terms appearing in more than this fraction of docs")
    sublinear_tf: bool = Field(default=True, description="Use 1 + log(tf) scaling (recommended)")


class Bm25BuildParams(BaseModel):
    """BM25 build knobs. Note: ``k1``/``b`` are **query-time** (see SearchRequest),
    not build-time — only the corpus statistics are precomputed here."""

    min_df: int = Field(default=1, ge=1, description="Drop terms appearing in fewer than N docs")


class EmbeddingBuildParams(BaseModel):
    """Word2Vec build knobs (the dense-embedding model, trained offline on the corpus)."""

    vector_size: int = Field(default=100, ge=8, le=1024, description="Embedding dimensionality")
    window: int = Field(default=5, ge=1, le=20, description="Context window size")
    min_count: int = Field(default=2, ge=1, description="Ignore words below this corpus frequency")
    epochs: int = Field(default=5, ge=1, le=50, description="Training passes over the corpus")
    sg: int = Field(default=1, ge=0, le=1, description="1 = skip-gram, 0 = CBOW")


class BertBuildParams(BaseModel):
    """BERT (sentence-transformers) build knobs."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="sentence-transformers model id (downloaded once, cached on the volume)",
    )
    batch_size: int = Field(default=64, ge=1, le=512, description="Encode batch size")


class BuildRepresentationRequest(BaseModel):
    """Build a representation over the **whole corpus** (every ingested doc),
    optionally restricted to a positional range via ``start``/``stop``.

    Preprocessing ``options`` apply to the lexical models (TF-IDF, BM25); the
    embedding model reads raw text (it has its own tokenizer). Only the params
    block matching ``model`` is used. ``build all`` = leave ``start``/``stop`` unset.
    """

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    model: Literal["tfidf", "bm25", "embedding", "bert"] = Field(default="tfidf")
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    params: TfidfParams = Field(default_factory=TfidfParams, description="TF-IDF params")
    bm25: Bm25BuildParams = Field(default_factory=Bm25BuildParams, description="BM25 build params")
    embedding: EmbeddingBuildParams = Field(default_factory=EmbeddingBuildParams,
                                            description="Word2Vec build params")
    bert: BertBuildParams = Field(default_factory=BertBuildParams, description="BERT build params")
    start: Optional[int] = Field(default=None, ge=0, description="Range start (inclusive)")
    stop: Optional[int] = Field(default=None, ge=1, description="Range stop (exclusive)")
    force: bool = Field(default=False, description="Rebuild even if a cached model exists")

    @model_validator(mode="after")
    def _check_range(self) -> "BuildRepresentationRequest":
        if self.start is not None and self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self


# --------------------------------------------------------------------------- #
#  Build — status / stats
# --------------------------------------------------------------------------- #

class RepresentationStats(BaseModel):
    """Summary of a built representation (returned by /build and /stats).

    ``params`` is model-specific (a free-form record of what it was built with);
    ``vocab_size`` / ``nnz`` apply to the sparse lexical models, ``dim`` to the
    dense embedding model, ``avgdl`` to BM25.
    """

    dataset_id: str
    model: str
    num_docs: int
    vocab_size: Optional[int] = None   # lexical models (terms / columns)
    nnz: Optional[int] = None          # lexical models (non-zero weights)
    density: Optional[float] = None    # lexical models
    dim: Optional[int] = None          # embedding model (vector dimensionality)
    avgdl: Optional[float] = None       # BM25 (average doc length)
    version: str
    options: PreprocessOptions
    params: dict[str, Any] = Field(default_factory=dict)
    built_at: str
    cached: bool = False


class RepresentationStatusResponse(BaseModel):
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
    deleted: bool
    was_cached: bool


# --------------------------------------------------------------------------- #
#  Encode — inspect a query's weighting (lexical models)
# --------------------------------------------------------------------------- #

class EncodeRequest(BaseModel):
    """Vectorize query text(s) with a lexical model and return the top-weighted
    terms — lets the UI *see* what TF-IDF / BM25 considers important."""

    dataset: str
    model: Literal["tfidf", "bm25"] = "tfidf"
    texts: list[str] = Field(min_length=1, description="Query texts to weigh")
    top_terms: int = Field(default=30, ge=1, le=500, description="Max weighted terms per text")


class WeightedTerm(BaseModel):
    term: str
    weight: float


class EncodedVector(BaseModel):
    nnz: int
    terms: list[WeightedTerm]


class EncodeResponse(BaseModel):
    model: str
    dim: int
    options: PreprocessOptions
    vectors: list[EncodedVector]


# --------------------------------------------------------------------------- #
#  Search — the online query path (single model + hybrid)
# --------------------------------------------------------------------------- #

FusionMethod = Literal["rrf", "weighted"]
HybridMode = Literal["serial", "parallel"]


class HybridSpec(BaseModel):
    """How to combine models for a hybrid search.

    * **serial**: ``first`` retrieves ``candidates`` docs, then ``rerank`` re-scores
      just those and reorders them.
    * **parallel**: each model in ``components`` searches independently and the
      ranked lists are merged by a **fusion method** (RRF or weighted score sum).
    """

    mode: HybridMode = "parallel"
    # parallel
    components: list[Literal["tfidf", "bm25", "embedding", "bert"]] = Field(
        default_factory=lambda: ["bm25", "bert"],
        description="Models to run and fuse (parallel mode)",
    )
    fusion: FusionMethod = "rrf"
    weights: Optional[list[float]] = Field(
        default=None, description="Per-component weights (weighted fusion; defaults to equal)"
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF damping constant")
    # serial
    first: Literal["tfidf", "bm25", "embedding", "bert"] = Field(
        default="bm25", description="Fast model that retrieves candidates (serial mode)"
    )
    rerank: Literal["tfidf", "bm25", "embedding", "bert"] = Field(
        default="bert", description="Model that re-ranks the candidates (serial mode)"
    )
    candidates: int = Field(default=100, ge=1, le=2000,
                            description="How many candidates the first stage passes on (serial)")

    @model_validator(mode="after")
    def _check(self) -> "HybridSpec":
        if self.mode == "parallel" and len(self.components) < 2:
            raise ValueError("parallel hybrid needs at least 2 components")
        if self.weights is not None and len(self.weights) != len(self.components):
            raise ValueError("weights must match the number of components")
        if self.mode == "serial" and self.first == self.rerank:
            raise ValueError("serial hybrid needs two different models (first != rerank)")
        return self


class SearchRequest(BaseModel):
    """Run a query and return ranked documents. ``model='hybrid'`` uses ``hybrid``."""

    dataset: str
    model: Literal["tfidf", "bm25", "embedding", "bert", "hybrid"] = "bm25"
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    # BM25 per-query tuning (assignment: must be controllable per query from the UI)
    k1: float = Field(default=1.5, ge=0.0, le=10.0, description="BM25 term-saturation")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization")
    hybrid: Optional[HybridSpec] = Field(default=None, description="Required when model='hybrid'")
    with_text: bool = Field(default=True, description="Fetch the original doc text by id for display")

    @model_validator(mode="after")
    def _check(self) -> "SearchRequest":
        if self.model == "hybrid" and self.hybrid is None:
            self.hybrid = HybridSpec()
        return self


class SearchHit(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: Optional[str] = None          # original text (when with_text and found in the store)


class SearchResponse(BaseModel):
    dataset_id: str
    model: str
    mode: Optional[str] = None          # hybrid mode, if applicable
    query: str
    took_ms: float
    total: int                          # hits returned
    hits: list[SearchHit]
