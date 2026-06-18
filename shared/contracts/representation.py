"""Contracts for the representation-service (offline build + per-model scoring).

A *representation* turns the corpus into something a retriever can score. The
assignment (§2) requires four, all hosted here:

* **TF-IDF (VSM)** — sparse weights, cosine similarity.
* **BM25** — probabilistic ranking with per-query tunable ``k1`` / ``b``.
* **Embedding (Word2Vec)** — dense distributional vectors.
* **BERT** — dense contextual vectors (sentence-transformers).

Heavy artifacts are built **offline** and loaded ready at startup (no fitting at
query time). This service owns the fitted models and exposes the
**scoring primitives** the retrieval-service orchestrates: ``/rank`` (rank docs with
one model) and ``/score`` (score a specific set of docs, for serial re-ranking).
Combining models into a hybrid (serial / parallel + fusion) and showing the original
text are the **retrieval-service**'s job (see ``shared.contracts.retrieval``).
Datasets/models are referenced by id/name, so callers stay dataset- and model-agnostic.
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
#  Scoring primitives — the per-model ranking the retrieval-service orchestrates
# --------------------------------------------------------------------------- #
#
# These are **internal** service-to-service contracts: the retrieval-service calls
# them to obtain one model's scores, then applies the retrieval strategy (single /
# serial / parallel) on top. They operate on a single concrete model (never the
# "hybrid" pseudo-model) and never fetch original text — that is retrieval's job.

ScoringModel = Literal["tfidf", "bm25", "embedding", "bert"]


class RankRequest(BaseModel):
    """Rank the whole corpus for a query with **one** model — the core scoring call."""

    dataset: str
    model: ScoringModel = "bm25"
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=2000)
    # BM25 per-query tuning (ignored by the other models)
    k1: float = Field(default=1.5, ge=0.0, le=10.0, description="BM25 term-saturation")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization")


class ScoredDoc(BaseModel):
    doc_id: str
    score: float


class RankResponse(BaseModel):
    dataset_id: str
    model: str
    hits: list[ScoredDoc]               # ranked desc, up to top_k, positive scores only


class ScoreRequest(BaseModel):
    """Score a **specific** set of doc ids for a query with one model.

    Used by serial hybrid re-ranking: the first stage's candidate ids are re-scored
    by a second model. Ids the model doesn't know are simply absent from the result.
    """

    dataset: str
    model: ScoringModel = "bert"
    query: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=1, description="Candidate doc ids to score")
    k1: float = Field(default=1.5, ge=0.0, le=10.0, description="BM25 term-saturation")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization")


class ScoreResponse(BaseModel):
    dataset_id: str
    model: str
    scores: dict[str, float]            # {doc_id: score} for the ids the model knows
