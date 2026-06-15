"""Fusion methods — merge several ranked lists into one (parallel hybrid).

The assignment requires a **Fusion Method** for parallel hybrid scoring. We provide
the two most common, dependency-free:

* **RRF** (Reciprocal Rank Fusion) — rank-based, scale-free, robust default. Each
  list contributes ``1/(k + rank)`` to a doc's fused score.
* **Weighted** — score-based: min-max normalize each list to [0, 1], then add up
  per-component weights. Lets one model count more than another.
"""

from __future__ import annotations

from typing import Optional

Scored = tuple[str, float]


def reciprocal_rank_fusion(ranked_lists: list[list[Scored]], k: int = 60) -> list[Scored]:
    """Fuse by reciprocal rank. ``k`` damps the influence of top ranks (RRF default 60)."""
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def weighted_fusion(
    scored_lists: list[list[Scored]], weights: Optional[list[float]] = None
) -> list[Scored]:
    """Fuse by weighted sum of **min-max normalized** scores (each list → [0, 1])."""
    weights = weights if weights is not None else [1.0] * len(scored_lists)
    fused: dict[str, float] = {}
    for lst, w in zip(scored_lists, weights):
        if not lst:
            continue
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        for doc_id, s in lst:
            norm = 1.0 if rng == 0 else (s - lo) / rng
            fused[doc_id] = fused.get(doc_id, 0.0) + w * norm
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
