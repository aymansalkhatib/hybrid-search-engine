"""HTTP layer (thin) for the topic-service — maps requests to the domain.

* ``POST /build`` — offline LDA build as a background job (idempotent; live progress).
* ``GET /status`` — is a model built? + its summary (topics, perplexity).
* ``GET /topics`` — the per-topic table (top terms + weights) for the charts.
* ``POST /infer`` — the topic mixture of a query/text (dominant topic + distribution).
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.topic_model import TopicModel, build_topics
from shared.contracts import (
    InferRequest,
    InferResponse,
    JobStatus,
    TopicBuildRequest,
    TopicDeleteResult,
    TopicInfo,
    TopicInference,
    TopicStats,
    TopicStatusResponse,
    TopicWeight,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("topic-service")
router = APIRouter(tags=["topic"])

JOB_TYPE = "topic"


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


def _get_model(request: Request, dataset_id: str) -> Optional[TopicModel]:
    cache = request.app.state.models
    if dataset_id not in cache:
        loaded = request.app.state.store.load(dataset_id)
        if loaded is not None:
            cache[dataset_id] = loaded
    return cache.get(dataset_id)


def _topic_infos(model: TopicModel) -> list[TopicInfo]:
    return [
        TopicInfo(
            topic_id=t, label=model.label(t), size=model.sizes[t],
            top_terms=model.top_terms[t], weights=model.weights[t],
        )
        for t in range(model.n_topics)
    ]


def _stats(model: TopicModel) -> TopicStats:
    return TopicStats(
        dataset_id=model.dataset_id,
        n_topics=model.n_topics,
        num_docs=model.num_docs,
        max_features=model.max_features,
        perplexity=model.perplexity,
        topics=_topic_infos(model),
        built_at=model.built_at,
    )


def _submit_or_409(request: Request, key: str, fn):
    try:
        return request.app.state.jobs.submit(JOB_TYPE, key, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---- build ---------------------------------------------------------------

@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: TopicBuildRequest, request: Request) -> JobStatus:
    """Start a background LDA build. The corpus is read from the doc-store, so the
    dataset must be **ingested** first. **409** if not ingested / a build is running;
    **503** if the doc-store is down."""
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state

    if not state.doc_store.is_healthy():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"doc-store unavailable at {settings.doc_store_url}; topic modelling reads the corpus from it")
    try:
        ingested = state.doc_store.ingested_count(dataset_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"doc-store error while checking ingest status: {exc}") from exc
    if ingested <= 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=(f"dataset '{dataset_id}' has no documents in the store; ingest it first "
                                    "via the doc-store POST /dataset/prepare"))

    n_topics, max_features, max_docs, max_iter = req.n_topics, req.max_features, req.max_docs, req.max_iter
    max_df = req.max_df
    force = req.force
    doc_store = state.doc_store

    def fn(progress: Progress) -> dict:
        if not force and state.store.exists(dataset_id):
            existing = _get_model(request, dataset_id)
            if existing is not None:
                return {"skipped": True, "stats": _stats(existing).model_dump()}
        total = min(max_docs, ingested)
        progress.update(total=total, message=f"modelling {n_topics} topics")
        model = build_topics(
            dataset_id=dataset_id,
            n_topics=n_topics,
            max_features=max_features,
            max_iter=max_iter,
            max_df=max_df,
            docs=doc_store.iter_docs(dataset_id, stop=max_docs),
            batch_size=settings.stream_batch_size,
            on_progress=lambda n: progress.update(processed=n),
        )
        state.store.save(model)
        state.models[dataset_id] = model
        return {"skipped": False, "stats": _stats(model).model_dump()}

    return status_from_job(_submit_or_409(request, _job_key(dataset_id), fn))


# ---- status / read -------------------------------------------------------

@router.get("/status", response_model=TopicStatusResponse)
def topic_status(request: Request, dataset: str = Query(...)) -> TopicStatusResponse:
    dataset_id = _resolve_or_400(dataset)
    model = _get_model(request, dataset_id)
    active = request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id))
    return TopicStatusResponse(
        dataset_id=dataset_id,
        built=model is not None,
        stats=_stats(model) if model is not None else None,
        active_job=ref_from_job(active).model_dump() if active else None,
    )


def _require(request: Request, dataset: str) -> TopicModel:
    dataset_id = _resolve_or_400(dataset)
    model = _get_model(request, dataset_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"no topic model for dataset '{dataset_id}'; build it first via POST /build")
    return model


@router.get("/topics", response_model=list[TopicInfo])
def topics(request: Request, dataset: str = Query(...)) -> list[TopicInfo]:
    return _topic_infos(_require(request, dataset))


@router.post("/infer", response_model=InferResponse)
def infer(req: InferRequest, request: Request) -> InferResponse:
    """Infer the topic mixture of each text (dominant topic + distribution over topics)."""
    model = _require(request, req.dataset)
    inferred = model.infer(req.texts)
    results: list[TopicInference] = []
    for text, (dominant, dist) in zip(req.texts, inferred):
        results.append(TopicInference(
            text=text,
            dominant_topic=dominant,
            top_terms=model.top_terms[dominant],
            distribution=[TopicWeight(topic_id=t, label=model.label(t), weight=w) for t, w in enumerate(dist)],
        ))
    return InferResponse(dataset_id=model.dataset_id, n_topics=model.n_topics, results=results)


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


@router.delete("/topic", response_model=TopicDeleteResult)
def delete_topic(request: Request, dataset: str = Query(...)) -> TopicDeleteResult:
    dataset_id = _resolve_or_400(dataset)
    if request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"a topic build is in progress for '{dataset_id}'; wait for it to finish")
    removed = request.app.state.store.delete(dataset_id)
    request.app.state.models.pop(dataset_id, None)
    return TopicDeleteResult(dataset_id=dataset_id, deleted=removed)
