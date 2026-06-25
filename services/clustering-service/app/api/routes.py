"""HTTP layer (thin) for the clustering-service — maps requests to the domain.

* ``POST /build`` — offline clustering build as a background job (idempotent; live
  progress via ``GET /jobs/{id}`` or ``GET /status``).
* ``GET /status`` — is a clustering built? + its summary (sizes, top terms, silhouette).
* ``GET /clusters`` — the per-cluster table.
* ``GET /plot`` — 2-D points for the scatter chart.
* ``POST /assign`` — assign text(s)/a query to their nearest cluster.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.clusterer import ClusterModel, build_clustering
from shared.contracts import (
    AssignRequest,
    AssignResponse,
    Assignment,
    ClusterBuildRequest,
    ClusterDeleteResult,
    ClusterInfo,
    ClusterPlotResponse,
    ClusterPoint,
    ClusterStats,
    ClusterStatusResponse,
    JobStatus,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("clustering-service")
router = APIRouter(tags=["clustering"])

JOB_TYPE = "cluster"


def _resolve_or_400(dataset: str) -> str:
    dataset_id = settings.resolve_dataset(dataset)
    if dataset_id is None:
        catalog = settings.datasets
        detail = (
            "no datasets configured — set DATASETS in .env"
            if not catalog
            else f"unknown dataset '{dataset}' — not in the configured catalog ({', '.join(catalog)})"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return dataset_id


def _job_key(dataset_id: str) -> str:
    return dataset_id


def _get_model(request: Request, dataset_id: str) -> Optional[ClusterModel]:
    cache = request.app.state.models
    if dataset_id not in cache:
        loaded = request.app.state.store.load(dataset_id)
        if loaded is not None:
            cache[dataset_id] = loaded
    return cache.get(dataset_id)


def _stats(model: ClusterModel) -> ClusterStats:
    return ClusterStats(
        dataset_id=model.dataset_id,
        n_clusters=model.n_clusters,
        num_docs=model.num_docs,
        max_features=model.max_features,
        silhouette=model.silhouette,
        inertia=model.inertia,
        clusters=[
            ClusterInfo(cluster_id=c, size=model.sizes[c], top_terms=model.top_terms[c])
            for c in range(model.n_clusters)
        ],
        built_at=model.built_at,
    )


def _submit_or_409(request: Request, key: str, fn):
    try:
        return request.app.state.jobs.submit(JOB_TYPE, key, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---- build ---------------------------------------------------------------

@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: ClusterBuildRequest, request: Request) -> JobStatus:
    """Start a background clustering build. The **raw** corpus is read from the doc-store
    (vectorised in-service — no preprocessing-service), so the dataset must be **ingested**
    first. **409** if not ingested or a build is already running; **503** if a dependency is down.
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state

    if not state.doc_store.is_healthy():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"doc-store unavailable at {settings.doc_store_url}; clustering reads the corpus from it")
    try:
        ingested = state.doc_store.ingested_count(dataset_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"doc-store error while checking ingest status: {exc}") from exc
    if ingested <= 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=(f"dataset '{dataset_id}' has no documents in the store; ingest it first "
                                    "via the doc-store POST /dataset/prepare"))

    n_clusters, max_features, max_docs = req.n_clusters, req.max_features, req.max_docs
    force = req.force
    doc_store = state.doc_store

    def fn(progress: Progress) -> dict:
        if not force and state.store.exists(dataset_id):
            existing = _get_model(request, dataset_id)
            if existing is not None:
                return {"skipped": True, "stats": _stats(existing).model_dump()}
        total = min(max_docs, ingested)
        progress.update(total=total, message=f"clustering into {n_clusters} groups")
        model = build_clustering(
            dataset_id=dataset_id,
            n_clusters=n_clusters,
            max_features=max_features,
            docs=doc_store.iter_docs(dataset_id, stop=max_docs),
            batch_size=settings.preprocess_batch_size,
            plot_sample=settings.plot_sample_size,
            on_progress=lambda n: progress.update(processed=n),
        )
        state.store.save(model)
        state.models[dataset_id] = model
        return {"skipped": False, "stats": _stats(model).model_dump()}

    return status_from_job(_submit_or_409(request, _job_key(dataset_id), fn))


# ---- status / read -------------------------------------------------------

@router.get("/status", response_model=ClusterStatusResponse)
def cluster_status(request: Request, dataset: str = Query(..., description="Dataset id")) -> ClusterStatusResponse:
    dataset_id = _resolve_or_400(dataset)
    model = _get_model(request, dataset_id)
    active = request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id))
    return ClusterStatusResponse(
        dataset_id=dataset_id,
        built=model is not None,
        stats=_stats(model) if model is not None else None,
        active_job=ref_from_job(active).model_dump() if active else None,
    )


def _require(request: Request, dataset: str) -> ClusterModel:
    dataset_id = _resolve_or_400(dataset)
    model = _get_model(request, dataset_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"no clustering for dataset '{dataset_id}'; build it first via POST /build")
    return model


@router.get("/clusters", response_model=list[ClusterInfo])
def clusters(request: Request, dataset: str = Query(...)) -> list[ClusterInfo]:
    model = _require(request, dataset)
    return [ClusterInfo(cluster_id=c, size=model.sizes[c], top_terms=model.top_terms[c])
            for c in range(model.n_clusters)]


@router.get("/plot", response_model=ClusterPlotResponse)
def plot(request: Request, dataset: str = Query(...)) -> ClusterPlotResponse:
    model = _require(request, dataset)
    return ClusterPlotResponse(
        dataset_id=model.dataset_id,
        n_clusters=model.n_clusters,
        points=[ClusterPoint(x=x, y=y, cluster=c) for (x, y, c) in model.projection],
    )


@router.post("/assign", response_model=AssignResponse)
def assign(req: AssignRequest, request: Request) -> AssignResponse:
    """Assign each text/query to its nearest cluster (no re-fit — transform + predict)."""
    model = _require(request, req.dataset)
    pairs = model.assign(req.texts)
    return AssignResponse(
        dataset_id=model.dataset_id,
        assignments=[Assignment(text=t, cluster_id=c, top_terms=terms)
                     for t, (c, terms) in zip(req.texts, pairs)],
    )


# ---- jobs / delete -------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JobStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found")
    return status_from_job(job)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(request: Request, dataset: Optional[str] = Query(None)) -> list[JobStatus]:
    jobs = request.app.state.jobs.list(type=JOB_TYPE)
    if dataset is not None:
        dataset_id = _resolve_or_400(dataset)
        jobs = [j for j in jobs if j.key == _job_key(dataset_id)]
    return [status_from_job(j) for j in jobs]


@router.delete("/clustering", response_model=ClusterDeleteResult)
def delete_clustering(request: Request, dataset: str = Query(...)) -> ClusterDeleteResult:
    dataset_id = _resolve_or_400(dataset)
    if request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"a clustering build is in progress for '{dataset_id}'; wait for it to finish")
    removed = request.app.state.store.delete(dataset_id)
    request.app.state.models.pop(dataset_id, None)
    return ClusterDeleteResult(dataset_id=dataset_id, deleted=removed)
