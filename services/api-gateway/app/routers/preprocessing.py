"""Passthrough to the preprocessing-service (normalize / tokenize text).

The same normalization is applied to documents (offline) and queries (online), so
this is exposed for query-time normalization and for inspecting how text is processed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy
from shared.contracts import (
    PreprocessBatchRequest,
    PreprocessBatchResponse,
    PreprocessRequest,
    PreprocessResult,
)

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


@router.post("", response_model=PreprocessResult)
def preprocess(req: PreprocessRequest, request: Request) -> JSONResponse:
    """Normalize a single text (used for queries and single documents)."""
    return proxy(lambda: request.app.state.preprocessing.post("/preprocess", json=req.model_dump(mode="json")))


@router.post("/batch", response_model=PreprocessBatchResponse)
def preprocess_batch(req: PreprocessBatchRequest, request: Request) -> JSONResponse:
    """Normalize many texts in one call (used when preprocessing a corpus)."""
    return proxy(lambda: request.app.state.preprocessing.post("/preprocess/batch", json=req.model_dump(mode="json")))
