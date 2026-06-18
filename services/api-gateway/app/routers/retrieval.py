"""Passthrough to the retrieval-service — the online query path.

Mirrors the service's API under ``/retrieval``:

* ``POST /retrieval/search`` — model-based ranking (single model or hybrid
  serial/parallel + fusion). BM25 ``k1``/``b`` are per-query fields on the body.
* ``POST /retrieval/boolean`` — inverted-index-only retrieval (AND/OR, no scoring).

Both return the top-k with their **original** text, fetched by id from the doc-store.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy
from shared.contracts import (
    BooleanSearchRequest,
    BooleanSearchResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request) -> JSONResponse:
    """Rank documents for a query (single model or hybrid) and return the top-k with
    their original text. The online query path."""
    return proxy(lambda: request.app.state.retrieval.post("/search", json=req.model_dump(mode="json")))


@router.post("/boolean", response_model=BooleanSearchResponse)
def boolean_search(req: BooleanSearchRequest, request: Request) -> JSONResponse:
    """Boolean search over the inverted index only (AND/OR, no scoring model) — return
    the matched docs with their original text."""
    return proxy(lambda: request.app.state.retrieval.post("/boolean", json=req.model_dump(mode="json")))
