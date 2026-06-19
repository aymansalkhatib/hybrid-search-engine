"""Topic-based re-ranking — an extra-feature retrieval stage.

The topic analogue of cluster re-ranking: infer the query's **dominant LDA topic**, then
float the pooled candidates that share that topic to the top (preserving each group's
base-score order — a stable partition, not a re-score). Because it only reorders an
already-retrieved pool it never loses recall within that pool, and its before/after
effect on MAP/nDCG is directly measurable.

The query and the pool's original texts are inferred in a **single** call to the
topic-service (item 0 = the query, the rest = the pool, in order).
"""

from __future__ import annotations

from app.adapters.topic_client import TopicClient

Scored = tuple[str, float]


def topic_rerank(
    *,
    results: list[Scored],
    query: str,
    texts: dict[str, str],
    client: TopicClient,
    dataset: str,
    top_k: int,
) -> list[Scored]:
    """Reorder ``results`` so docs in the query's dominant topic come first; return top_k.

    Raises ``httpx.HTTPError`` if the topic-service is unavailable / not built — the
    caller falls back to the base ranking in that case.
    """
    if not results:
        return results
    ids = [doc_id for doc_id, _ in results]
    pool_texts = [texts.get(doc_id, "") for doc_id in ids]
    topics = client.dominant_topics(dataset, [query] + pool_texts)
    if not topics:
        return results[:top_k]

    query_topic = topics[0]
    doc_topics = topics[1:]
    same: list[Scored] = []
    other: list[Scored] = []
    for scored, topic in zip(results, doc_topics):
        (same if topic == query_topic else other).append(scored)
    other.extend(results[len(doc_topics):])
    return (same + other)[:top_k]
