"""HTTP layer (thin) — maps requests to the domain preprocessor."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from shared.contracts import (
    PreprocessBatchRequest,
    PreprocessBatchResponse,
    PreprocessRequest,
    PreprocessResult,
)

router = APIRouter(tags=["preprocessing"])


def _result(preprocessor, text: str, options) -> PreprocessResult:
    tokens = preprocessor.process(text, options)
    return PreprocessResult(tokens=tokens, normalized_text=" ".join(tokens))


@router.post("/preprocess", response_model=PreprocessResult)
def preprocess(req: PreprocessRequest, request: Request) -> PreprocessResult:
    """Normalize a single text (used for queries and single documents)."""
    return _result(request.app.state.preprocessor, req.text, req.options)


@router.post("/preprocess/batch", response_model=PreprocessBatchResponse)
def preprocess_batch(
    req: PreprocessBatchRequest, request: Request
) -> PreprocessBatchResponse:
    """Normalize many texts in one call (used when preprocessing a corpus)."""
    if len(req.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"batch of {len(req.texts)} exceeds max_batch_size="
                f"{settings.max_batch_size}; split into smaller chunks"
            ),
        )
    pp = request.app.state.preprocessor
    items = [_result(pp, text, req.options) for text in req.texts]
    return PreprocessBatchResponse(items=items)
