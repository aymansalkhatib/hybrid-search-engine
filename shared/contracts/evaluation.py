"""Contracts for the evaluation-service — offline IR effectiveness measurement.

The evaluation-service answers the assignment's most-graded question:
*how good is each representation?* For a dataset it runs the stored **test queries**
through the retrieval-service, compares each ranked result list against the **qrels**
(both read from the doc-store), and reports **MAP, nDCG, Recall, P@10** per model — the
**primary** metrics being **MAP** and **nDCG** (TA emphasis).

A full run is **offline** and can be long (thousands of queries × several models), so it
runs as a background **job**: ``POST /evaluate`` returns a ``JobStatus`` and the client
polls ``GET /jobs/{id}``. The finished :class:`EvaluationReport` is persisted (JSON + a
flat CSV) so the Arabic report's **charts** can be drawn from it, and reports are
**labeled** so the required **before/after** comparison of extra features is just two
labels (e.g. ``baseline`` vs ``with-refinement``).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shared.contracts.refinement import RefineOptions
from shared.contracts.retrieval import HybridSpec

# The models retrieval can rank with (mirrors SearchRequest.model).
EvalModel = Literal["tfidf", "bm25", "embedding", "bert", "hybrid"]

# Default metric set. ``map`` & ``ndcg@10`` are the TA's primary metrics; ``recall@100``
# and ``precision@10`` complete the four the assignment requires (§10). Names follow the
# ranx/trec_eval ``metric@cutoff`` convention.
DEFAULT_METRICS: list[str] = ["map", "ndcg@10", "recall@100", "precision@10"]


class EvalRunSpec(BaseModel):
    """One model configuration to evaluate (and compare against the others in a report).

    Mirrors the fields of a retrieval ``SearchRequest`` so a run reproduces exactly what
    the online search would do: a single representation, or a ``hybrid`` (serial re-rank /
    parallel fusion). ``label`` is how the run appears in the report and its charts.
    """

    label: str = Field(min_length=1, description="Label for this run in the report, e.g. 'bm25' or 'hybrid-rrf'")
    model: EvalModel = "bm25"
    k1: float = Field(default=1.5, ge=0.0, le=10.0, description="BM25 term-saturation (used when model='bm25')")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization (used when model='bm25')")
    hybrid: Optional[HybridSpec] = Field(default=None, description="Hybrid strategy; defaulted when model='hybrid'")

    @model_validator(mode="after")
    def _check(self) -> "EvalRunSpec":
        if self.model == "hybrid" and self.hybrid is None:
            self.hybrid = HybridSpec()
        return self


class EvaluateRequest(BaseModel):
    """Evaluate one or more model configurations on a dataset's stored test queries."""

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    label: str = Field(
        default="baseline",
        min_length=1,
        description="Report label — the unit of before/after comparison (e.g. 'baseline' vs 'with-refinement')",
    )
    runs: list[EvalRunSpec] = Field(
        default_factory=list,
        description="Model configs to evaluate; empty = the four base models (tfidf, bm25, embedding, bert)",
    )
    metrics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_METRICS),
        description="ranx-style metric names (metric@cutoff); 'map' and 'ndcg@10' are the primary ones",
    )
    top_k: int = Field(default=100, ge=1, le=1000, description="Ranking depth retrieved per query (the run cutoff)")
    max_queries: Optional[int] = Field(
        default=None, ge=1, description="Evaluate only the first N judged queries (a quick sample); None = all"
    )
    per_query: bool = Field(
        default=True,
        description="Also compute & persist per-query scores (a sidecar) for best/worst & win-loss drill-down",
    )
    refine: Optional[RefineOptions] = Field(
        default=None,
        description=(
            "Query-refinement applied to every test query before retrieval (spell-correct / "
            "expand). None = no refinement (the 'before' baseline). Set it to measure the "
            "'after' effect — run the same models with a different label (e.g. 'with-refinement') "
            "and compare reports. Refinement is applied once per query and shared across all runs."
        ),
    )
    cluster_prune: bool = Field(
        default=False,
        description=(
            "Apply cluster-based **pruning** to every run (extra feature §11): the search is "
            "restricted to the members of the query's nearest clusters. False = the 'before' "
            "baseline; set it and label the report (e.g. 'with-clustering') for the before/after effect."
        ),
    )
    topic_prune: bool = Field(
        default=False,
        description=(
            "Apply topic-based **pruning** to every run (extra feature §11): the search is "
            "restricted to the members of the query's nearest LDA topics. Label the report "
            "(e.g. 'with-topics') and compare to 'baseline' for the before/after effect."
        ),
    )
    prune_top_n: int = Field(
        default=3, ge=1, le=50,
        description="When pruning, search within this many nearest clusters/topics (1 = strictest)",
    )
    force: bool = Field(default=False, description="Re-run and overwrite even if a report with this label exists")


class RunEvaluation(BaseModel):
    """The metric scores for one evaluated model configuration."""

    label: str
    model: str
    mode: Optional[str] = None                                   # hybrid mode (serial/parallel), if applicable
    params: dict = Field(default_factory=dict)                   # what was evaluated (k1/b, hybrid spec)
    num_queries: int = 0                                         # judged queries actually scored
    metrics: dict[str, float] = Field(default_factory=dict)      # {metric_name: mean value over queries}
    took_ms: float = 0.0                                         # wall time for the whole run (all queries)
    avg_query_ms: Optional[float] = None                         # mean per-query search latency (quality-vs-speed)
    error: Optional[str] = None                                  # set if this run failed (the others still report)


class EvaluationReport(BaseModel):
    """A full evaluation — every model's metrics on one dataset, persisted for charts."""

    report_id: str                                              # stable id: <dataset>__<label>
    dataset_id: str
    label: str
    created_at: str
    metrics: list[str]                                          # metric names evaluated (column order)
    top_k: int
    num_queries: int                                           # judged queries in the gold set (after sampling)
    runs: list[RunEvaluation]
    engine: str = "ranx"                                       # metric engine that produced the numbers


class ReportSummary(BaseModel):
    """One row of the saved-reports listing (cheap metadata, no per-query detail)."""

    report_id: str
    dataset_id: str
    label: str
    created_at: str
    models: list[str]
    num_queries: int


class ReportsList(BaseModel):
    items: list[ReportSummary]


class ReportDeleteResult(BaseModel):
    report_id: str
    deleted: bool


class PerQueryReport(BaseModel):
    """Per-query metric scores (a sidecar to :class:`EvaluationReport`), loaded on demand.

    Powers the drill-down: *which* queries each model wins or loses, and head-to-head
    comparisons between two runs. Kept separate from the main report so the table/charts
    stay light — this can be large (queries × runs × metrics) and is fetched only when
    the user opens the analysis. Optionally filtered to a subset of run labels.
    """

    report_id: str
    dataset_id: str
    metrics: list[str]
    queries: dict[str, str]                              # query_id -> original query text (judged set)
    runs: dict[str, dict[str, dict[str, float]]]         # run label -> query_id -> {metric: value}
