"""Passthrough to the doc-store read surface (the by-id / browse API).

The graded path: the **original** document text is read **by id** from MongoDB at
query time. This router exposes those reads plus the queries/qrels browse endpoints.
The offline write flow (download / ingest / delete) lives in the ``lifecycle`` router
because it orchestrates the indexing service too.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy
from shared.contracts import (
    AllQrelsResponse,
    DocListResponse,
    DocsRequest,
    DocsResponse,
    QrelsForQueryResponse,
    QrelsListResponse,
    QueryListResponse,
    RawDoc,
    RawQuery,
)

router = APIRouter(prefix="/docstore", tags=["docstore"])


@router.get("/doc", response_model=RawDoc)
def get_doc(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    doc_id: str = Query(...),
) -> JSONResponse:
    """Return the ORIGINAL document text by id (query-time display path — graded)."""
    return proxy(lambda: request.app.state.doc_store.get("/doc", params={"dataset": dataset, "doc_id": doc_id}))


@router.post("/docs", response_model=DocsResponse)
def get_docs(req: DocsRequest, request: Request) -> JSONResponse:
    """Batch fetch originals by id — used to show top-k results."""
    return proxy(lambda: request.app.state.doc_store.post("/docs", json=req.model_dump(mode="json")))


@router.get("/docs/list", response_model=DocListResponse)
def list_docs(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> JSONResponse:
    """Browse stored documents by page, ordered by ingest position (``seq``)."""
    return proxy(lambda: request.app.state.doc_store.get("/docs/list", params={"dataset": dataset, "offset": offset, "limit": limit}))


@router.get("/queries", response_model=QueryListResponse)
def list_queries(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> JSONResponse:
    """Browse the stored test queries by page."""
    return proxy(lambda: request.app.state.doc_store.get("/queries", params={"dataset": dataset, "offset": offset, "limit": limit}))


@router.get("/query", response_model=RawQuery)
def get_query(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    query_id: str = Query(...),
) -> JSONResponse:
    """Return a single test query's text by id."""
    return proxy(lambda: request.app.state.doc_store.get("/query", params={"dataset": dataset, "query_id": query_id}))


@router.get("/qrels", response_model=QrelsForQueryResponse)
def get_qrels_for_query(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    query_id: str = Query(...),
) -> JSONResponse:
    """All judged documents (the gold set) for a single query."""
    return proxy(lambda: request.app.state.doc_store.get("/qrels", params={"dataset": dataset, "query_id": query_id}))


@router.get("/qrels/list", response_model=QrelsListResponse)
def list_qrels(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> JSONResponse:
    """Browse all stored relevance judgments by page."""
    return proxy(lambda: request.app.state.doc_store.get("/qrels/list", params={"dataset": dataset, "offset": offset, "limit": limit}))


@router.get("/qrels/all", response_model=AllQrelsResponse)
def all_qrels(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> JSONResponse:
    """The full qrels as ``{query_id: {doc_id: relevance}}`` — the eval/fusion input shape."""
    return proxy(lambda: request.app.state.doc_store.get("/qrels/all", params={"dataset": dataset}))
