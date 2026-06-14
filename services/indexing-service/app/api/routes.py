"""HTTP layer (thin) for the indexing-service — maps requests to the domain.

Building is a long offline step, so ``POST /build`` runs as a **background job**:
it returns a ``JobRef`` immediately and reports live progress (poll ``GET /jobs/{id}``
or ``GET /index/status``). A build can cover the whole corpus or a positional range.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.builder import build_index
from app.domain.inverted_index import INDEX_VERSION, InvertedIndex
from shared.contracts import (
    BuildIndexRequest,
    BuiltIndexes,
    DocInfo,
    IndexDeleteResult,
    IndexStats,
    IndexStatusResponse,
    JobStatus,
    Posting,
    PostingsResponse,
    PreprocessOptions,
    TermStats,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("indexing-service")
router = APIRouter(tags=["indexing"])


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


def _get_index(request: Request, dataset_id: str) -> Optional[InvertedIndex]:
    """Return the index from the in-memory cache, lazily loading it from disk."""
    cache = request.app.state.indexes
    if dataset_id not in cache:
        loaded = request.app.state.store.load(dataset_id, INDEX_VERSION)
        if loaded is not None:
            cache[dataset_id] = loaded
    return cache.get(dataset_id)


def _require_index(request: Request, dataset: str) -> InvertedIndex:
    dataset_id = _resolve_or_400(dataset)
    index = _get_index(request, dataset_id)
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no index for dataset '{dataset_id}'; "
                "build it first via POST /build"
            ),
        )
    return index


def _stats(index: InvertedIndex, *, cached: bool) -> IndexStats:
    options = PreprocessOptions(**index.options) if index.options else PreprocessOptions()
    return IndexStats(
        dataset_id=index.dataset_id,
        num_docs=index.num_docs,
        vocab_size=index.vocab_size,
        num_postings=index.num_postings,
        total_tokens=index.total_tokens,
        avg_doc_length=index.avg_doc_length,
        version=index.version,
        options=options,
        built_at=index.built_at,
        cached=cached,
    )


def _submit_or_409(request: Request, dataset_id: str, fn):
    """Start a build job, mapping a duplicate (in-flight) build to HTTP 409."""
    try:
        return request.app.state.jobs.submit("build", dataset_id, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---- endpoints -----------------------------------------------------------

@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: BuildIndexRequest, request: Request) -> JobStatus:
    """Start a background build of the inverted index for a dataset.

    The corpus is read from the **doc-store** (MongoDB), so the dataset must be
    **ingested** first (pipeline: download → ingest → build). The whole corpus is
    indexed (optionally a positional range via ``start``/``stop`` over the stored
    ``seq`` order). **409** if not ingested; **503** if the doc-store or
    preprocessing-service is down. Idempotent: a cached index with the same options
    ends the job as ``skipped`` unless ``force``. Concurrent build → **409**.
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state
    doc_store = state.doc_store

    # Synchronous preconditions so the caller gets a clear status immediately.
    if not doc_store.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"doc-store unavailable at {settings.doc_store_url}; "
                "a build reads the corpus from it"
            ),
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
                "via the doc-store POST /dataset/prepare (download → ingest → build)"
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
    stop = req.effective_stop()
    options_dump = req.options.model_dump()
    force = req.force

    def fn(progress: Progress) -> dict:
        # Reuse a cached index with identical options unless a rebuild is forced.
        if not force:
            existing = _get_index(request, dataset_id)
            if existing is not None and existing.options == options_dump:
                return {"skipped": True,
                        "stats": _stats(existing, cached=True).model_dump()}
        total = max(stop - (start or 0), 0) if stop is not None else ingested
        progress.update(total=total, message="indexing")
        index = build_index(
            dataset_id=dataset_id,
            docs=doc_store.iter_docs(dataset_id, start=start, stop=stop),
            preprocess_batch=lambda texts: state.preprocessing.preprocess_batch(texts, req.options),
            options=options_dump,
            batch_size=settings.preprocess_batch_size,
            on_progress=lambda n: progress.update(processed=n),
        )
        state.store.save(index)
        state.indexes[dataset_id] = index
        return {"skipped": False,
                "stats": _stats(index, cached=False).model_dump()}

    return status_from_job(_submit_or_409(request, dataset_id, fn))


@router.get("/index/status", response_model=IndexStatusResponse)
def index_status(request: Request, dataset: str = Query(..., description="Dataset id from the catalog")) -> IndexStatusResponse:
    """What's built for a dataset plus any build in flight (live 'docs indexed' progress)."""
    dataset_id = _resolve_or_400(dataset)
    index = _get_index(request, dataset_id)
    active = request.app.state.jobs.active_for("build", dataset_id)
    return IndexStatusResponse(
        dataset_id=dataset_id,
        built=index is not None,
        stats=_stats(index, cached=True) if index is not None else None,
        active_job=ref_from_job(active) if active else None,
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
) -> list[JobStatus]:
    key = _resolve_or_400(dataset) if dataset else None
    return [status_from_job(j) for j in request.app.state.jobs.list(type="build", key=key)]


@router.get("/stats", response_model=IndexStats)
def stats(request: Request, dataset: str = Query(..., description="Dataset id from the catalog")) -> IndexStats:
    return _stats(_require_index(request, dataset), cached=True)


@router.get("/postings", response_model=PostingsResponse)
def postings(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    term: str = Query(..., description="A normalized (preprocessed) term"),
    limit: int = Query(100, ge=1, le=10000, description="Max postings returned"),
) -> PostingsResponse:
    index = _require_index(request, dataset)
    pairs = index.postings_for(term)
    return PostingsResponse(
        term=term,
        df=len(pairs),
        cf=sum(tf for _, tf in pairs),
        postings=[Posting(doc_id=d, tf=tf) for d, tf in pairs[:limit]],
    )


@router.get("/term", response_model=TermStats)
def term_stats(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    term: str = Query(..., description="A normalized (preprocessed) term"),
) -> TermStats:
    index = _require_index(request, dataset)
    return TermStats(term=term, df=index.df(term), cf=index.cf(term))


@router.get("/doc", response_model=DocInfo)
def doc_info(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    doc_id: str = Query(...),
) -> DocInfo:
    index = _require_index(request, dataset)
    length = index.doc_length(doc_id)
    if length is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"doc '{doc_id}' not in index",
        )
    return DocInfo(doc_id=doc_id, length=length)


@router.get("/built", response_model=BuiltIndexes)
def built(request: Request) -> BuiltIndexes:
    """List dataset ids that currently have a persisted index artifact on disk."""
    return BuiltIndexes(datasets=request.app.state.store.list_datasets(INDEX_VERSION))


@router.delete("/index", response_model=IndexDeleteResult)
def delete_index(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> IndexDeleteResult:
    """Remove a dataset's built index — the artifact on disk **and** the in-memory
    cache — so a dataset delete also clears its index. **409** if a build is in flight."""
    dataset_id = _resolve_or_400(dataset)
    if request.app.state.jobs.active_for("build", dataset_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a build is in progress for '{dataset_id}'; wait for it to finish",
        )
    removed = request.app.state.store.delete(dataset_id, INDEX_VERSION)
    was_cached = request.app.state.indexes.pop(dataset_id, None) is not None
    return IndexDeleteResult(dataset_id=dataset_id, index_deleted=removed, was_cached=was_cached)
