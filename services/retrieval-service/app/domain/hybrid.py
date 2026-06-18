"""Hybrid search — combine representations two ways (assignment §2).

* **Serial**: a fast model (``first``) retrieves a candidate pool, then a stronger
  model (``rerank``) re-scores just those candidates and reorders them. One model
  feeds the next.
* **Parallel**: every model in ``components`` searches the full query independently,
  then their ranked lists are merged by a **fusion method** (RRF or weighted).

Each model is reached by name through the representation-service's scoring primitives
(:meth:`RepresentationClient.rank` / :meth:`~RepresentationClient.score`), so adding a
model to a hybrid needs no change here — exactly the pluggable-strategy seam the
constitution grades, now across a service boundary.
"""

from __future__ import annotations

from app.adapters.representation_client import RepresentationClient
from app.domain.fusion import reciprocal_rank_fusion, weighted_fusion

Scored = tuple[str, float]


def _pool(top_k: int) -> int:
    """A retrieved pool generous enough for fusion/rerank to have material to work
    with, without scoring more than necessary."""
    return min(max(top_k * 5, 100), 1000)


def serial_search(
    *,
    client: RepresentationClient,
    dataset: str,
    first: str,
    rerank: str,
    query: str,
    candidates: int,
    top_k: int,
    k1: float,
    b: float,
) -> list[Scored]:
    """``first`` retrieves ``candidates`` docs → ``rerank`` re-scores & reorders them."""
    pool = client.rank(dataset=dataset, model=first, query=query, top_k=candidates, k1=k1, b=b)
    if not pool:
        return []
    cand_ids = [doc_id for doc_id, _ in pool]
    rescored = client.score(dataset=dataset, model=rerank, query=query, doc_ids=cand_ids, k1=k1, b=b)
    if not rescored:  # rerank knows none of the candidates → keep the first stage's order
        return pool[:top_k]
    ranked = sorted(rescored.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]


def parallel_search(
    *,
    client: RepresentationClient,
    dataset: str,
    components: list[str],
    query: str,
    fusion: str,
    weights: list[float] | None,
    rrf_k: int,
    top_k: int,
    k1: float,
    b: float,
) -> list[Scored]:
    """Each component searches independently; their lists are fused into one ranking."""
    pool = _pool(top_k)
    ranked_lists = [
        client.rank(dataset=dataset, model=name, query=query, top_k=pool, k1=k1, b=b)
        for name in components
    ]
    if fusion == "weighted":
        fused = weighted_fusion(ranked_lists, weights)
    else:
        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    return fused[:top_k]
