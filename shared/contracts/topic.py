"""Contracts for the topic-service — topic modelling (extra feature, §11).

Topic detection is an **offline** batch job: the corpus is streamed from the doc-store,
vectorised with raw-text counts (English stop-words) and modelled with Latent Dirichlet
Allocation (LDA). The fitted artifact makes the online endpoints cheap: list the topics
(top terms + their weights) for the **topic charts** the TA expects, report the corpus's
dominant-topic distribution, and infer the topic mixture of a new query/text. It's
self-contained (builds its own counts) so it works on any ingested dataset and is
toggleable / evaluable on its own.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TopicBuildRequest(BaseModel):
    """Offline build of an LDA topic model for a dataset (idempotent unless ``force``)."""

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    n_topics: int = Field(default=10, ge=2, le=100, description="Number of topics (LDA components)")
    max_features: int = Field(default=20000, ge=500, le=200000, description="Vocabulary cap")
    max_docs: int = Field(
        default=20000, ge=100, le=2_000_000,
        description="Model at most this many docs (a representative sample keeps it fast on 200K+ corpora)",
    )
    max_iter: int = Field(default=10, ge=1, le=100, description="LDA passes over the corpus")
    max_df: float = Field(
        default=0.5, gt=0.0, le=1.0,
        description="Drop terms appearing in more than this fraction of docs — trims ubiquitous, "
                    "non-discriminative words (good/like/best) so topics are sharper. Lower = stricter.",
    )
    force: bool = Field(default=False, description="Rebuild & overwrite even if a model exists")


class TopicInfo(BaseModel):
    """One topic's summary for the table/charts."""

    topic_id: int
    label: str                              # the top two terms, for a human-readable handle
    size: int                               # docs whose dominant topic is this one
    top_terms: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)  # per-term weight, aligned with top_terms


class TopicStats(BaseModel):
    """The persisted model's headline facts (drives the dashboard summary + charts)."""

    dataset_id: str
    n_topics: int
    num_docs: int
    max_features: int
    perplexity: Optional[float] = None      # LDA fit quality (lower is better)
    topics: list[TopicInfo] = Field(default_factory=list)
    built_at: Optional[str] = None


class TopicStatusResponse(BaseModel):
    dataset_id: str
    built: bool
    stats: Optional[TopicStats] = None
    active_job: Optional[dict] = None       # JobRef when a build is in progress


class TopicWeight(BaseModel):
    topic_id: int
    label: str
    weight: float


class InferRequest(BaseModel):
    dataset: str
    texts: list[str] = Field(min_length=1, description="Texts/queries to infer the topic mixture of")


class TopicInference(BaseModel):
    text: str
    dominant_topic: int
    top_terms: list[str] = Field(default_factory=list)   # dominant topic's terms
    distribution: list[TopicWeight] = Field(default_factory=list)  # mixture over all topics


class InferResponse(BaseModel):
    dataset_id: str
    n_topics: int
    results: list[TopicInference] = Field(default_factory=list)


class TopicMembersRequest(BaseModel):
    """Ask which docs to *search within* for a query: the members of its nearest topics.

    Powers retrieval's topic **pruning** (restrict the search space) — distinct from the
    older re-ranking of a full-corpus pool."""

    dataset: str
    query: str = Field(min_length=1, description="The query to locate in topic space")
    top_n: int = Field(default=3, ge=1, le=50,
                       description="Search within this many highest-weight topics (1 = strictest pruning)")
    max_members: Optional[int] = Field(
        default=None, ge=1, description="Cap the returned candidate pool for latency (None = no cap)"
    )


class TopicMembersResponse(BaseModel):
    dataset_id: str
    topic_ids: list[int] = Field(default_factory=list,
                                 description="The nearest topics chosen ([] if the query has no in-vocab terms)")
    doc_ids: list[str] = Field(default_factory=list, description="Member doc ids to restrict the search to")


class TopicDeleteResult(BaseModel):
    dataset_id: str
    deleted: bool
