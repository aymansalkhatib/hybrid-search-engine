"""Contracts for the clustering-service — unsupervised document clustering (extra feature, §11).

Clustering is an **offline** batch job: the corpus is streamed from the doc-store,
normalized by the preprocessing-service, vectorised with TF-IDF and partitioned with
(MiniBatch) KMeans. The fitted artifact is persisted so the online endpoints are cheap:
list clusters (size + top terms), assign new text to its nearest cluster, and serve a
2-D projection for the scatter plot. It's independent of the trained representations —
it builds its own lexical vectors — so it works on any ingested dataset and is
toggleable / evaluable on its own (cluster plots + silhouette), exactly as the TA asks.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClusterBuildRequest(BaseModel):
    """Offline build of a clustering for a dataset (idempotent unless ``force``)."""

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    n_clusters: int = Field(default=8, ge=2, le=100, description="K — number of clusters")
    max_features: int = Field(default=20000, ge=500, le=200000, description="TF-IDF vocabulary cap")
    max_docs: int = Field(
        default=20000, ge=100, le=2_000_000,
        description="Cluster at most this many docs (a representative sample keeps it fast on 200K+ corpora)",
    )
    force: bool = Field(default=False, description="Rebuild & overwrite even if a clustering exists")


class ClusterInfo(BaseModel):
    """One cluster's summary for the table/charts."""

    cluster_id: int
    size: int                              # docs assigned to this cluster
    top_terms: list[str] = Field(default_factory=list)  # highest-weight terms at the centroid


class ClusterStats(BaseModel):
    """The persisted clustering's headline facts (drives the dashboard summary + charts)."""

    dataset_id: str
    n_clusters: int
    num_docs: int                          # docs actually clustered (after the max_docs sample)
    max_features: int
    silhouette: Optional[float] = None     # cluster-separation quality on a sample (−1..1; higher better)
    inertia: Optional[float] = None        # KMeans within-cluster sum of squares
    clusters: list[ClusterInfo] = Field(default_factory=list)
    built_at: Optional[str] = None


class ClusterStatusResponse(BaseModel):
    dataset_id: str
    built: bool
    stats: Optional[ClusterStats] = None
    active_job: Optional[dict] = None      # JobRef when a build is in progress


class ClusterPoint(BaseModel):
    """A document projected to 2-D (TruncatedSVD) for the scatter plot."""

    x: float
    y: float
    cluster: int


class ClusterPlotResponse(BaseModel):
    dataset_id: str
    n_clusters: int
    points: list[ClusterPoint] = Field(default_factory=list)


class AssignRequest(BaseModel):
    dataset: str
    texts: list[str] = Field(min_length=1, description="Texts/queries to assign to their nearest cluster")


class Assignment(BaseModel):
    text: str
    cluster_id: int
    top_terms: list[str] = Field(default_factory=list)


class AssignResponse(BaseModel):
    dataset_id: str
    assignments: list[Assignment] = Field(default_factory=list)


class ClusterDeleteResult(BaseModel):
    dataset_id: str
    deleted: bool
