"""Cluster-based re-ranking — an extra-feature retrieval stage.

Classic cluster-based retrieval: assign the query to its document cluster, then float
the candidates that share that cluster to the top (preserving each group's base-score
order — a stable partition, not a re-score). This sharpens topical precision when the
query lands in a coherent cluster, and is a no-op-ish reordering otherwise. Because it
only reorders an already-retrieved pool it never loses recall within that pool, and its
before/after effect on MAP/nDCG is directly measurable.

The query and the pool's original texts are assigned in a **single** call to the
clustering-service (item 0 = the query, the rest = the pool, in order).
"""

from __future__ import annotations

from app.adapters.clustering_client import ClusteringClient

Scored = tuple[str, float]


def cluster_rerank(
    *,
    results: list[Scored],
    query: str,
    texts: dict[str, str],
    client: ClusteringClient,
    dataset: str,
    top_k: int,
) -> list[Scored]:
    """Reorder ``results`` so docs in the query's cluster come first; return top_k.

    Raises ``httpx.HTTPError`` if the clustering-service is unavailable / not built —
    the caller falls back to the base ranking in that case.
    """
    if not results:
        return results
    ids = [doc_id for doc_id, _ in results]
    pool_texts = [texts.get(doc_id, "") for doc_id in ids]
    clusters = client.assign(dataset, [query] + pool_texts)
    if not clusters:
        return results[:top_k]

    query_cluster = clusters[0]
    if query_cluster < 0:
        # The query has no in-vocabulary terms → no dominant cluster; re-ranking would just
        # float an arbitrary group, so keep the base ranking untouched.
        return results[:top_k]
    doc_clusters = clusters[1:]
    same: list[Scored] = []
    other: list[Scored] = []
    for scored, cluster in zip(results, doc_clusters):
        (same if cluster == query_cluster else other).append(scored)
    # Any docs beyond the returned cluster labels keep their place at the tail.
    other.extend(results[len(doc_clusters):])
    return (same + other)[:top_k]
