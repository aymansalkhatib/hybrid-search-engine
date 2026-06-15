"""Small ranking helpers shared by the scoring models."""

from __future__ import annotations

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, sorted descending.

    Uses ``argpartition`` (O(n)) to find the top-k, then sorts only those — much
    cheaper than a full sort over a 200K+ score vector.
    """
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(scores[part])[::-1]].tolist()
