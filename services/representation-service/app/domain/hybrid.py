"""Hybrid search — combine representations two ways (assignment §2).

* **Serial**: a fast model (``first``) retrieves a candidate pool, then a stronger
  model (``rerank``) re-scores just those candidates and reorders them. One model
  feeds the next.
* **Parallel**: every model in ``components`` searches the full query independently,
  then their ranked lists are merged by a **fusion method** (RRF or weighted).

Models are addressed by name through the same :class:`BaseRepresentation` interface,
so adding a model to a hybrid needs no change here.
"""

from __future__ import annotations

from app.domain.base import BaseRepresentation, Normalizer, Scored
from app.domain.fusion import reciprocal_rank_fusion, weighted_fusion

# A retrieved pool that is generous enough for fusion/rerank to have material to work
# with, without scoring more than necessary.
def _pool(top_k: int) -> int:
    return min(max(top_k * 5, 100), 1000)


def serial_search(
    *,
    first: BaseRepresentation,
    rerank: BaseRepresentation,
    raw_query: str,
    normalize: Normalizer,
    candidates: int,
    top_k: int,
    **params,
) -> list[Scored]:
    """``first`` retrieves ``candidates`` docs → ``rerank`` re-scores & reorders them."""
    pool = first.search(raw_query=raw_query, top_k=candidates, normalize=normalize, **params)
    if not pool:
        return []
    cand_ids = [doc_id for doc_id, _ in pool]
    rescored = rerank.score_docs(doc_ids=cand_ids, raw_query=raw_query, normalize=normalize, **params)
    if not rescored:  # rerank knows none of the candidates → keep the first stage's order
        return pool[:top_k]
    ranked = sorted(rescored.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]


def parallel_search(
    *,
    models: dict[str, BaseRepresentation],
    components: list[str],
    raw_query: str,
    normalize: Normalizer,
    fusion: str,
    weights: list[float] | None,
    rrf_k: int,
    top_k: int,
    **params,
) -> list[Scored]:
    """Each component searches independently; their lists are fused into one ranking."""
    pool = _pool(top_k)
    ranked_lists = [
        models[name].search(raw_query=raw_query, top_k=pool, normalize=normalize, **params)
        for name in components
    ]
    if fusion == "weighted":
        fused = weighted_fusion(ranked_lists, weights)
    else:
        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    return fused[:top_k]
