"""Passthrough to the retrieval-service — the online query path.

Mirrors the service's API under ``/retrieval``. ``POST /retrieval/search`` ranks
documents for a query (single model or hybrid serial/parallel + fusion) and returns
the top-k with their **original** text, fetched by id from the doc-store. BM25
``k1``/``b`` are per-query fields on the search body.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy
from shared.contracts import SearchRequest, SearchResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request) -> JSONResponse:
    """Rank documents for a query (single model or hybrid) and return the top-k with
    their original text. The online query path."""
    return proxy(lambda: request.app.state.retrieval.post("/search", json=req.model_dump(mode="json")))
