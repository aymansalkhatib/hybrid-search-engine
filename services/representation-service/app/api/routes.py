"""HTTP layer (thin) for the representation-service — maps requests to the domain.

Building is a long offline step, so ``POST /build`` runs as a **background job**: it
returns a ``JobStatus`` immediately and reports live progress (poll ``GET /jobs/{id}``
or ``GET /status``). ``POST /encode`` is the online path — it vectorizes a query with a
loaded model and is fast (no fitting). Builds for the same dataset+model can't overlap.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.base import (
    REPRESENTATION_VERSION,
    BaseRepresentation,
    available_models,
    get_model_class,
)
from app.domain.builder import build_representation
from shared.contracts import (
    BuildRepresentationRequest,
    BuiltRepresentations,
    EncodeRequest,
    EncodeResponse,
    EncodedVector,
    JobStatus,
    PreprocessOptions,
    RepresentationDeleteResult,
    RepresentationStats,
    RepresentationStatusResponse,
    TfidfParams,
    WeightedTerm,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("representation-service")
router = APIRouter(tags=["representation"])

JOB_TYPE = "represent"


# ---- helpers -------------------------------------------------------------

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


def _validate_model_or_400(model: str) -> str:
    if get_model_class(model) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown model '{model}' — available: {', '.join(available_models())}",
        )
    return model


def _job_key(dataset_id: str, model: str) -> str:
    """One in-flight build per dataset+model (different models may build concurrently)."""
    return f"{dataset_id}::{model}"


def _get_model(request: Request, dataset_id: str, model: str) -> Optional[BaseRepresentation]:
    """Return the model from the in-memory cache, lazily loading it from disk."""
    cache = request.app.state.models
    key = (dataset_id, model)
    if key not in cache:
        loaded = request.app.state.store.load(dataset_id, model, REPRESENTATION_VERSION)
        if loaded is not None:
            cache[key] = loaded
    return cache.get(key)


def _require_model(request: Request, dataset: str, model: str) -> BaseRepresentation:
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    rep = _get_model(request, dataset_id, model)
    if rep is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no '{model}' representation for dataset '{dataset_id}'; build it first via POST /build",
        )
    return rep


def _stats(rep: BaseRepresentation, *, cached: bool) -> RepresentationStats:
    return RepresentationStats(
        dataset_id=rep.dataset_id,
        model=rep.model,
        num_docs=rep.num_docs,
        vocab_size=rep.vocab_size,
        nnz=rep.nnz,
        density=rep.density,
        version=rep.version,
        options=PreprocessOptions(**rep.options) if rep.options else PreprocessOptions(),
        params=TfidfParams(**rep.params) if rep.params else TfidfParams(),
        built_at=rep.built_at,
        cached=cached,
    )


def _submit_or_409(request: Request, key: str, fn):
    """Start a build job, mapping a duplicate (in-flight) build to HTTP 409."""
    try:
        return request.app.state.jobs.submit(JOB_TYPE, key, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---- endpoints -----------------------------------------------------------

@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: BuildRepresentationRequest, request: Request) -> JobStatus:
    """Start a background build of a representation (TF-IDF) for a dataset.

    The corpus is read from the **doc-store** (MongoDB), so the dataset must be
    **ingested** first (pipeline: download → ingest → represent). **409** if not
    ingested; **503** if the doc-store or preprocessing-service is down. Idempotent:
    a cached model with the same options/params ends the job as ``skipped`` unless
    ``force``. A concurrent build of the same dataset+model → **409**.
    """
    dataset_id = _resolve_or_400(req.dataset)
    model = _validate_model_or_400(req.model)
    state = request.app.state
    doc_store = state.doc_store

    # Synchronous preconditions so the caller gets a clear status immediately.
    if not doc_store.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store unavailable at {settings.doc_store_url}; a build reads the corpus from it",
        )
    try:
        ingested = doc_store.ingested_count(dataset_id)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store error while checking ingest status: {exc}",
        ) from exc
    if ingested <= 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"dataset '{dataset_id}' has no documents in the store; ingest it first "
                "via the doc-store POST /dataset/prepare (download → ingest → represent)"
            ),
        )
    if not state.preprocessing.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"preprocessing-service unavailable at {settings.preprocessing_url}; "
                "a build needs it to tokenize the corpus"
            ),
        )

    start = req.start
    stop = req.stop
    options_dump = req.options.model_dump()
    params_dump = req.params.model_dump()
    force = req.force

    def fn(progress: Progress) -> dict:
        # Reuse a cached model with identical options+params unless a rebuild is forced.
        if not force:
            existing = _get_model(request, dataset_id, model)
            if existing is not None and existing.options == options_dump and existing.params == params_dump:
                return {"skipped": True, "stats": _stats(existing, cached=True).model_dump()}
        total = max(stop - (start or 0), 0) if stop is not None else ingested
        progress.update(total=total, message=f"building {model}")
        rep = build_representation(
            model=model,
            dataset_id=dataset_id,
            docs=doc_store.iter_docs(dataset_id, start=start, stop=stop),
            normalize_batch=lambda texts: state.preprocessing.normalize_batch(texts, req.options),
            options=options_dump,
            params=params_dump,
            batch_size=settings.preprocess_batch_size,
            on_progress=lambda n: progress.update(processed=n),
        )
        state.store.save(rep)
        state.models[(dataset_id, model)] = rep
        return {"skipped": False, "stats": _stats(rep, cached=False).model_dump()}

    return status_from_job(_submit_or_409(request, _job_key(dataset_id, model), fn))


@router.get("/status", response_model=RepresentationStatusResponse)
def representation_status(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationStatusResponse:
    """What's built for a (dataset, model) plus any build in flight (live progress)."""
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    rep = _get_model(request, dataset_id, model)
    active = request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id, model))
    return RepresentationStatusResponse(
        dataset_id=dataset_id,
        model=model,
        built=rep is not None,
        stats=_stats(rep, cached=True) if rep is not None else None,
        active_job=ref_from_job(active) if active else None,
    )


@router.get("/built", response_model=BuiltRepresentations)
def built(request: Request) -> BuiltRepresentations:
    """List (dataset, model) pairs that currently have a persisted artifact on disk."""
    return BuiltRepresentations(items=request.app.state.store.list_built(REPRESENTATION_VERSION))


@router.get("/stats", response_model=RepresentationStats)
def stats(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationStats:
    return _stats(_require_model(request, dataset, model), cached=True)


@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest, request: Request) -> EncodeResponse:
    """Vectorize query text(s) with a built model — the online representation path.

    The query is normalized with the **exact options the model was built with** (so it
    lands in the same space), then transformed. Returns the top-weighted terms per
    query so the algorithm is inspectable. **404** if the model isn't built; **503** if
    the preprocessing-service is down.
    """
    rep = _require_model(request, req.dataset, req.model)
    if not request.app.state.preprocessing.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"preprocessing-service unavailable at {settings.preprocessing_url}; "
                "encoding a query needs it to normalize the text the same way as the corpus"
            ),
        )
    options = PreprocessOptions(**rep.options) if rep.options else PreprocessOptions()
    try:
        normalized = request.app.state.preprocessing.normalize_batch(req.texts, options)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"preprocessing error while normalizing the query: {exc}",
        ) from exc

    matrix = rep.encode(normalized)
    names = rep.feature_names()
    vectors: list[EncodedVector] = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        idx, data = row.indices, row.data
        order = data.argsort()[::-1][: req.top_terms]  # heaviest weights first
        terms = [WeightedTerm(term=str(names[idx[j]]), weight=round(float(data[j]), 6)) for j in order]
        vectors.append(EncodedVector(nnz=int(row.nnz), terms=terms))

    return EncodeResponse(
        model=rep.model,
        dim=rep.vocab_size,
        options=options,
        vectors=vectors,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JobStatus:
    """Poll a background build's live progress."""
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found")
    return status_from_job(job)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
    model: Optional[str] = Query(None, description="Filter by model (requires dataset)"),
) -> list[JobStatus]:
    key = None
    if dataset is not None:
        dataset_id = _resolve_or_400(dataset)
        key = _job_key(dataset_id, model) if model else None
        if key is None:
            # Filter to this dataset's jobs across all models.
            jobs = request.app.state.jobs.list(type=JOB_TYPE)
            prefix = f"{dataset_id}::"
            return [status_from_job(j) for j in jobs if j.key.startswith(prefix)]
    return [status_from_job(j) for j in request.app.state.jobs.list(type=JOB_TYPE, key=key)]


@router.delete("/representation", response_model=RepresentationDeleteResult)
def delete_representation(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationDeleteResult:
    """Remove a built representation — the artifact on disk **and** the in-memory
    cache. **409** if a build is in flight for this dataset+model."""
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    if request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id, model)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a build is in progress for '{dataset_id}' / '{model}'; wait for it to finish",
        )
    removed = request.app.state.store.delete(dataset_id, model, REPRESENTATION_VERSION)
    was_cached = request.app.state.models.pop((dataset_id, model), None) is not None
    return RepresentationDeleteResult(
        dataset_id=dataset_id, model=model, deleted=removed, was_cached=was_cached
    )
