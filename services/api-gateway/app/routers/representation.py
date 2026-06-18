"""Passthrough to the representation-service (TF-IDF · BM25 · Word2Vec · BERT).

Mirrors the service's API under ``/representation``: offline builds (background jobs),
``/encode`` (inspect a lexical query's weighting), and stats/status/delete. The online
query path lives in the **retrieval-service** (``/retrieval/search``); the per-model
scoring primitives (``/rank`` / ``/score``) are internal service-to-service calls and
are intentionally not re-exposed here.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from app.proxy import clean, proxy
from shared.contracts import (
    BuildRepresentationRequest,
    BuiltRepresentations,
    EncodeRequest,
    EncodeResponse,
    JobStatus,
    RepresentationDeleteResult,
    RepresentationStats,
    RepresentationStatusResponse,
)

router = APIRouter(prefix="/representation", tags=["representation"])


@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: BuildRepresentationRequest, request: Request) -> JSONResponse:
    """Start a background build of a representation (tfidf / bm25 / embedding / bert)."""
    return proxy(lambda: request.app.state.representation.post("/build", json=req.model_dump(mode="json")))


@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest, request: Request) -> JSONResponse:
    """Show the top-weighted terms a lexical model assigns to a query (TF-IDF / BM25)."""
    return proxy(lambda: request.app.state.representation.post("/encode", json=req.model_dump(mode="json")))


@router.get("/status", response_model=RepresentationStatusResponse)
def representation_status(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.representation.get("/status", params={"dataset": dataset, "model": model}))


@router.get("/built", response_model=BuiltRepresentations)
def built(request: Request) -> JSONResponse:
    """List every built {dataset, model} representation artifact."""
    return proxy(lambda: request.app.state.representation.get("/built"))


@router.get("/stats", response_model=RepresentationStats)
def stats(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.representation.get("/stats", params={"dataset": dataset, "model": model}))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JSONResponse:
    return proxy(lambda: request.app.state.representation.get(f"/jobs/{job_id}"))


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
    model: Optional[str] = Query(None, description="Filter by model (requires dataset)"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.representation.get("/jobs", params=clean({"dataset": dataset, "model": model})))


@router.delete("", response_model=RepresentationDeleteResult)
def delete_representation(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> JSONResponse:
    return proxy(
        lambda: request.app.state.representation.request(
            "DELETE", "/representation", params={"dataset": dataset, "model": model}
        )
    )
