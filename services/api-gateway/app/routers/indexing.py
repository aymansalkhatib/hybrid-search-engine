"""Passthrough to the indexing-service (inverted index: postings, df, doc lengths, avgdl).

Mirrors the service's own API under ``/indexing`` and reuses its contracts, so the
gateway's Swagger documents and validates it. ``POST /indexing/build`` is the raw,
single-service build; the orchestrated ``POST /datasets/index`` (lifecycle) is the
recommended product path.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from app.proxy import clean, proxy
from shared.contracts import (
    BuildIndexRequest,
    BuiltIndexes,
    DocInfo,
    IndexDeleteResult,
    IndexStats,
    IndexStatusResponse,
    JobStatus,
    PostingsResponse,
    TermStats,
)

router = APIRouter(prefix="/indexing", tags=["indexing"])


@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: BuildIndexRequest, request: Request) -> JSONResponse:
    """Start a background build of the inverted index for a dataset."""
    return proxy(lambda: request.app.state.indexing.post("/build", json=req.model_dump(mode="json")))


@router.get("/status", response_model=IndexStatusResponse)
def index_status(request: Request, dataset: str = Query(..., description="Dataset id from the catalog")) -> JSONResponse:
    """What's built for a dataset plus any build in flight (live progress)."""
    return proxy(lambda: request.app.state.indexing.get("/index/status", params={"dataset": dataset}))


@router.get("/stats", response_model=IndexStats)
def stats(request: Request, dataset: str = Query(..., description="Dataset id from the catalog")) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get("/stats", params={"dataset": dataset}))


@router.get("/postings", response_model=PostingsResponse)
def postings(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    term: str = Query(..., description="A normalized (preprocessed) term"),
    limit: int = Query(100, ge=1, le=10000, description="Max postings returned"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get("/postings", params={"dataset": dataset, "term": term, "limit": limit}))


@router.get("/term", response_model=TermStats)
def term_stats(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    term: str = Query(..., description="A normalized (preprocessed) term"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get("/term", params={"dataset": dataset, "term": term}))


@router.get("/doc", response_model=DocInfo)
def doc_info(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    doc_id: str = Query(...),
) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get("/doc", params={"dataset": dataset, "doc_id": doc_id}))


@router.get("/built", response_model=BuiltIndexes)
def built(request: Request) -> JSONResponse:
    """List dataset ids that currently have a persisted index artifact."""
    return proxy(lambda: request.app.state.indexing.get("/built"))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get(f"/jobs/{job_id}"))


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.get("/jobs", params=clean({"dataset": dataset})))


@router.delete("/index", response_model=IndexDeleteResult)
def delete_index(request: Request, dataset: str = Query(..., description="Dataset id from the catalog")) -> JSONResponse:
    return proxy(lambda: request.app.state.indexing.request("DELETE", "/index", params={"dataset": dataset}))
