"""HTTP layer (thin) for the retrieval-service — the online query path.

Two endpoints, both returning the top-k with their **original** text:

* ``POST /search`` — model-based ranking. ``model`` ∈ {tfidf, bm25, embedding, bert}
  for one model's ranking, or ``model='hybrid'`` for ``serial`` re-rank / ``parallel``
  fusion. Scoring happens in the representation-service.
* ``POST /boolean`` — inverted-index-only retrieval (AND/OR over postings, **no**
  scoring model). Matching happens in the indexing-service's ``/match`` primitive.

Either way this service only orchestrates the strategy, then fetches the original docs
by id from the doc-store for display (the graded by-id read).
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.domain.hybrid import parallel_search, serial_search
from app.domain.prune import prune_candidates
from shared.contracts import (
    BooleanSearchHit,
    BooleanSearchRequest,
    BooleanSearchResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

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


def _downstream_error(exc: httpx.HTTPStatusError) -> HTTPException:
    """Re-raise a downstream error (representation/indexing) with its SAME status & message.

    Those services already return clean, specific errors (404 = not built, 503 =
    preprocessing down). Forwarding them verbatim means the UI still sees "build it
    first" instead of a generic 502.
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

    # Pruning (extra features §11): restrict the search space to the members of the query's
    # nearest clusters/topics, so the model scores that smaller candidate pool instead of
    # the whole corpus. ``candidates`` is None → a normal full search. Falls back to a full
    # search if a service is down/not built or the query has no in-vocab terms — optional,
    # never breaks a search.
    candidates: list[str] | None = None
    prune_label: str | None = None
    if req.cluster_prune or req.topic_prune:
        candidates, labels = prune_candidates(
            query=req.query, dataset=dataset_id, top_n=req.prune_top_n,
            cluster_client=state.clustering if req.cluster_prune else None,
            topic_client=state.topic if req.topic_prune else None,
            max_members=settings.prune_max_members,
        )
        if candidates:
            prune_label = "+".join(labels)

    try:
        if req.model == "hybrid":
            spec = req.hybrid
            if spec.mode == "serial":
                results = serial_search(
                    client=rep, dataset=dataset_id, first=spec.first, rerank=spec.rerank,
                    query=req.query, candidates=spec.candidates, top_k=req.top_k,
                    k1=req.k1, b=req.b, restrict_to=candidates,
                )
            else:
                results = parallel_search(
                    client=rep, dataset=dataset_id, components=spec.components,
                    query=req.query, fusion=spec.fusion, weights=spec.weights,
                    rrf_k=spec.rrf_k, top_k=req.top_k, k1=req.k1, b=req.b, restrict_to=candidates,
                )
            mode = spec.mode
        else:
            results = rep.rank(
                dataset=dataset_id, model=req.model, query=req.query,
                top_k=req.top_k, k1=req.k1, b=req.b, candidates=candidates,
            )
            mode = None
    except httpx.HTTPStatusError as exc:
        raise _downstream_error(exc) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"representation-service error at {settings.representation_url}: {exc}",
        ) from exc

    if prune_label:
        mode = f"{mode}+{prune_label}" if mode else prune_label
    results = results[:req.top_k]

    # Show the ORIGINAL document text — read by id from the doc-store (graded path).
    texts: dict[str, str] = {}
    if req.with_text and results:
        try:
            texts = state.doc_store.fetch_originals(dataset_id, [d for d, _ in results])
        except httpx.HTTPError:
            pass  # degrade gracefully: still return the ranking if the doc-store blips

    hits = [
        SearchHit(rank=i + 1, doc_id=d, score=round(float(s), 6), text=texts.get(d))
        for i, (d, s) in enumerate(results)
    ]
    return SearchResponse(
        dataset_id=dataset_id, model=req.model, mode=mode, query=req.query,
        took_ms=round((time.perf_counter() - t0) * 1000, 2), total=len(hits), hits=hits,
    )


# ---- boolean search (inverted-index-only, no scoring model) ---------------

@router.post("/boolean", response_model=BooleanSearchResponse)
def boolean_search(req: BooleanSearchRequest, request: Request) -> BooleanSearchResponse:
    """Search the inverted index only — match docs containing the query terms (AND =
    all, OR = any), with **no** ranking model.

    The matching is done by the indexing-service's ``/match`` primitive (it owns the
    index); this service attaches the **original** text by id for display. **400** for
    an unknown dataset; **404** if the index isn't built; **503** if indexing is down.
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state
    idx = state.indexing

    if not idx.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"indexing-service unavailable at {settings.indexing_url}; "
                "it owns the inverted index Boolean search matches against"
            ),
        )

    t0 = time.perf_counter()
    try:
        matched = idx.match(
            dataset=dataset_id, query=req.query, operator=req.operator, top_k=req.top_k
        )
    except httpx.HTTPStatusError as exc:
        raise _downstream_error(exc) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"indexing-service error at {settings.indexing_url}: {exc}",
        ) from exc

    texts: dict[str, str] = {}
    if req.with_text and matched.hits:
        try:
            texts = state.doc_store.fetch_originals(dataset_id, [h.doc_id for h in matched.hits])
        except httpx.HTTPError:
            texts = {}

    hits = [
        BooleanSearchHit(rank=i + 1, doc_id=h.doc_id, matched=h.matched, text=texts.get(h.doc_id))
        for i, h in enumerate(matched.hits)
    ]
    return BooleanSearchResponse(
        dataset_id=dataset_id, operator=matched.operator, query=req.query,
        terms=matched.terms, took_ms=round((time.perf_counter() - t0) * 1000, 2),
        total=matched.total_matched, hits=hits,
    )
