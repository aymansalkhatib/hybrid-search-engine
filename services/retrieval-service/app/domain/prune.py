"""Cluster/topic-based **pruning** — an extra-feature retrieval stage.

Classic cluster-based retrieval done the *pruning* way: locate the query in cluster /
topic space and restrict the search to the members of its ``top_n`` nearest groups,
scoring that smaller candidate pool instead of the whole corpus. This is a genuine
reduction of the search space (not a re-ordering of a full-corpus pool), so its
before/after effect on MAP/nDCG — and on latency — is directly measurable.

Searching only the *nearest* groups (not a single one) protects recall: relevant docs
scattered across a few neighbouring clusters/topics are still reached. Pruning is always
optional and never breaks a search: if a service is down / not built, or the query has no
in-vocabulary terms (no nearest group), it falls back to a full search.
"""

from __future__ import annotations

import httpx

from app.adapters.clustering_client import ClusteringClient
from app.adapters.topic_client import TopicClient


def prune_candidates(
    *,
    query: str,
    dataset: str,
    top_n: int,
    cluster_client: ClusteringClient | None,
    topic_client: TopicClient | None,
    max_members: int | None = None,
) -> tuple[list[str] | None, list[str]]:
    """Gather the candidate doc ids a search should be restricted to.

    Returns ``(candidates, labels)``. ``candidates`` is ``None`` when no pruning applies
    (no client requested, the service is down/not built, or the query has no in-vocabulary
    terms) — the caller then runs a normal full search. ``labels`` names what pruned
    (``["cluster"]`` / ``["topic"]`` / both) for the response ``mode``.

    With both cluster and topic pruning on, the candidate set is their **intersection**
    (most restrictive); if that is empty we fall back to their **union** so a search never
    returns nothing.
    """
    sets: list[set[str]] = []
    labels: list[str] = []

    if cluster_client is not None:
        try:
            ids = cluster_client.members(dataset, query, top_n, max_members)
            if ids:
                sets.append(set(ids))
                labels.append("cluster")
        except httpx.HTTPError:
            pass  # clustering unavailable/not built → skip (fall back to full search)

    if topic_client is not None:
        try:
            ids = topic_client.members(dataset, query, top_n, max_members)
            if ids:
                sets.append(set(ids))
                labels.append("topic")
        except httpx.HTTPError:
            pass  # topic model unavailable/not built → skip

    if not sets:
        return None, []
    if len(sets) == 1:
        return list(sets[0]), labels
    combined = set.intersection(*sets) or set.union(*sets)
    return list(combined), labels
