"""Passthrough to the query-refinement-service (spell-correct / expand a query).

Refinement runs on the raw query upstream of retrieval (assignment §5 req 5); the UI
calls this to offer "did you mean" and an opt-in synonym-expansion toggle, then sends
the refined query to ``/retrieval/search``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy
from shared.contracts import RefineRequest, RefineResponse

router = APIRouter(prefix="/refinement", tags=["query-refinement"])


@router.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest, request: Request) -> JSONResponse:
    """Refine a raw query: spell-correction + optional synonym expansion."""
    return proxy(lambda: request.app.state.query_refinement.post("/refine", json=req.model_dump(mode="json")))
