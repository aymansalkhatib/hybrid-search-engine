"""Contracts for the retrieval-service — the online query path.

Retrieval is the **matching & ranking** layer: it takes a query and
returns ranked documents, either with a single representation or a **hybrid** that
combines several. It does not own the model artifacts — it asks the
representation-service for per-model scores (``/rank`` / ``/score``), then applies the
retrieval *strategy* (single / serial re-rank / parallel fusion) and fetches the top-k
**original** docs by id from the doc-store for display.

* **serial** hybrid — one model retrieves candidates, another re-ranks just those.
* **parallel** hybrid — several models run independently and their ranked lists are
  merged by a **fusion method** (RRF or weighted score sum).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.indexing import BooleanOperator, MatchedTerm

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
    # Extra feature (§11): cluster-based re-ranking. When true, the query is assigned to
    # its document cluster and candidates sharing that cluster are floated to the top —
    # a toggleable stage, so its before/after effect is evaluable. No-op (falls back to
    # the base ranking) if no clustering is built for the dataset.
    cluster_rerank: bool = Field(default=False, description="Re-rank so candidates in the query's cluster come first")
    # Extra feature (§11): topic-based re-ranking — float candidates sharing the query's
    # dominant LDA topic to the top. No-op (base ranking) if no topic model is built.
    topic_rerank: bool = Field(default=False, description="Re-rank so candidates in the query's dominant topic come first")

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


# --------------------------------------------------------------------------- #
#  Boolean search — inverted-index-only retrieval (no scoring model)
# --------------------------------------------------------------------------- #
#
# A separate, classic retrieval mode the UI offers alongside the model-based search:
# match documents purely by the inverted index (AND/OR over postings), with no
# relevance scoring. The retrieval-service delegates the matching to the
# indexing-service's ``/match`` primitive, then attaches the original text by id.


class BooleanSearchRequest(BaseModel):
    """Search the inverted index only — match documents containing the query terms
    (AND = all, OR = any), with **no** ranking model (no TF-IDF / BM25)."""

    dataset: str
    query: str = Field(min_length=1)
    operator: BooleanOperator = Field(default="and", description="AND = all terms; OR = any term")
    top_k: int = Field(default=10, ge=1, le=200)
    with_text: bool = Field(default=True, description="Fetch the original doc text by id for display")


class BooleanSearchHit(BaseModel):
    rank: int
    doc_id: str
    matched: int                        # # of distinct query terms the doc contains
    text: Optional[str] = None          # original text (when with_text and found in the store)


class BooleanSearchResponse(BaseModel):
    dataset_id: str
    operator: str
    query: str
    terms: list[MatchedTerm]            # normalized query terms (+ df) — what actually matched
    took_ms: float
    total: int                          # size of the matched set (before top_k)
    hits: list[BooleanSearchHit]
