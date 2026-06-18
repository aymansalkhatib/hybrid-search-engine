"""HTTP layer (thin) for the retrieval-service — the online query path.

``POST /search`` ranks documents for a query and returns the top-k with their
**original** text. It supports a single representation or a **hybrid**:

* ``model`` ∈ {tfidf, bm25, embedding, bert} → one model's ranking (``/rank``).
* ``model='hybrid'`` → ``serial`` re-ranks one model's candidates with another, or
  ``parallel`` fuses several models' lists (RRF / weighted).

Scoring itself happens in the representation-service (single owner of the artifacts);
this service orchestrates the strategy, then fetches the original docs by id from the
doc-store for display (the graded by-id read).
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.domain.hybrid import parallel_search, serial_search
from shared.contracts import SearchHit, SearchRequest, SearchResponse

logger = logging.getLogger("retrieval-service")
router = APIRouter(tags=["retrieval"])


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


def _representation_error(exc: httpx.HTTPStatusError) -> HTTPException:
    """Re-raise a downstream representation error with the SAME status & message.

    The representation-service already returns clean, specific errors (404 = model not
    built, 503 = preprocessing down). Forwarding them verbatim means the UI still sees
    "build the model first" instead of a generic 502.
    """
    try:
        body = exc.response.json()
        message = body.get("error", {}).get("message") or body.get("detail") or exc.response.text
    except ValueError:
        message = exc.response.text
    return HTTPException(status_code=exc.response.status_code, detail=message)


# ---- search (the online query path) --------------------------------------

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request) -> SearchResponse:
    """Rank documents for a query (single model or hybrid) and return the top-k with
    their original text.

    BM25's ``k1``/``b`` are taken from the request (per-query tuning). **400** for an
    unknown dataset; **404** if a needed model isn't built; **503** if the
    representation-service is down (it does the scoring).
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state
    rep = state.representation

    if not rep.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"representation-service unavailable at {settings.representation_url}; "
                "it scores the query for retrieval"
            ),
        )

    t0 = time.perf_counter()
    try:
        if req.model == "hybrid":
            spec = req.hybrid
            if spec.mode == "serial":
                results = serial_search(
                    client=rep, dataset=dataset_id, first=spec.first, rerank=spec.rerank,
                    query=req.query, candidates=spec.candidates, top_k=req.top_k,
                    k1=req.k1, b=req.b,
                )
            else:
                results = parallel_search(
                    client=rep, dataset=dataset_id, components=spec.components,
                    query=req.query, fusion=spec.fusion, weights=spec.weights,
                    rrf_k=spec.rrf_k, top_k=req.top_k, k1=req.k1, b=req.b,
                )
            mode = spec.mode
        else:
            results = rep.rank(
                dataset=dataset_id, model=req.model, query=req.query,
                top_k=req.top_k, k1=req.k1, b=req.b,
            )
            mode = None
    except httpx.HTTPStatusError as exc:
        raise _representation_error(exc) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"representation-service error at {settings.representation_url}: {exc}",
        ) from exc

    # Show the ORIGINAL document text — read by id from the doc-store (graded path).
    texts: dict[str, str] = {}
    if req.with_text and results:
        try:
            texts = state.doc_store.fetch_originals(dataset_id, [d for d, _ in results])
        except httpx.HTTPError:
            texts = {}  # degrade gracefully: still return the ranking if the doc-store blips

    hits = [
        SearchHit(rank=i + 1, doc_id=d, score=round(float(s), 6), text=texts.get(d))
        for i, (d, s) in enumerate(results)
    ]
    return SearchResponse(
        dataset_id=dataset_id, model=req.model, mode=mode, query=req.query,
        took_ms=round((time.perf_counter() - t0) * 1000, 2), total=len(hits), hits=hits,
    )
