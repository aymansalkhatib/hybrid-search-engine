"""Core clustering logic — framework-independent (no FastAPI here).

Offline pipeline — the sklearn-recommended **LSA + KMeans** for text: stream the **raw**
corpus → TF-IDF (in-service: English stop-words + a min_df noise filter) → TruncatedSVD
(LSA, a compact dense space) → L2-normalise → KMeans. Plain KMeans on raw TF-IDF clusters
poorly (a huge, sparse, Euclidean-unfriendly space → lopsided clusters); LSA + L2 turns it
into a low-dimensional space where KMeans' Euclidean distance behaves like cosine, giving
tighter, better-separated, less lopsided clusters. The whole transform is **self-contained**
(no preprocessing-service): a query or candidate doc is assigned with the exact same
analyser + LSA used at fit time (transform → predict), so there's no normalisation mismatch
and assignment is cheap enough to cluster-rerank a candidate pool on every query.

The fitted :class:`ClusterModel` keeps everything the online endpoints need without a
re-fit: the TF-IDF + LSA + KMeans transformers, per-cluster sizes & top terms (centroids
mapped back to TF-IDF features via ``svd.inverse_transform``), a silhouette score, and a
2-D projection sample (the first two LSA dims) for the scatter plot. A text with no in-
vocabulary terms (empty/stop-words-only query) is reported as ``-1`` — *no cluster* — so
the re-ranker can treat it as a no-op instead of floating an arbitrary group.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger("clustering-service")

# Bump when the persisted shape changes so stale pickles are auto-rebuilt (see ClusteringStore.load).
# v3 adds per-cluster membership (doc ids) so retrieval can *prune* the search space to the
# query's nearest clusters instead of merely re-ranking a pool.
SCHEMA_VERSION = 3

_TOP_TERMS = 10
_SILHOUETTE_SAMPLE = 2000
NO_CLUSTER = -1   # a text with no in-vocabulary terms belongs to no cluster


@dataclass
class ClusterModel:
    dataset_id: str
    schema_version: int
    n_clusters: int
    num_docs: int
    max_features: int
    svd_components: int
    built_at: str
    sizes: list[int]
    top_terms: list[list[str]]
    silhouette: float | None
    inertia: float | None
    projection: list[tuple[float, float, int]]   # (x, y, cluster) sample for the scatter
    members: list[list[str]]                      # cluster_id → doc ids assigned to it (corpus membership)
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: KMeans

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Raw texts → (L2-normalised LSA vectors, nnz per text). nnz==0 ⇒ no in-vocab terms."""
        tfidf = self.vectorizer.transform(texts)
        nnz = tfidf.getnnz(axis=1)
        reduced = normalize(self.svd.transform(tfidf))
        return reduced, nnz

    def assign_ids(self, texts: list[str]) -> list[int]:
        """Cluster id per text (the lean assignment primitive); -1 if no in-vocab terms."""
        reduced, nnz = self._embed(texts)
        preds = self.kmeans.predict(reduced)
        return [int(p) if nnz[i] > 0 else NO_CLUSTER for i, p in enumerate(preds)]

    def assign(self, texts: list[str]) -> list[tuple[int, list[str]]]:
        """Assign each (raw) text to its nearest cluster → ``(cluster_id, top_terms)`` ((-1, []) if none)."""
        return [(c, self.top_terms[c] if c >= 0 else []) for c in self.assign_ids(texts)]

    def nearest_members(
        self, query: str, top_n: int, max_members: int | None = None
    ) -> tuple[list[int], list[str]]:
        """Pruning primitive: pick the query's ``top_n`` **nearest** clusters (by centroid
        distance) and return their members' doc ids — the candidate pool retrieval scores
        *instead of* the whole corpus. Returns ``([], [])`` for a query with no in-vocab
        terms (no nearest cluster → caller falls back to a full search). ``max_members``
        caps the pool for latency."""
        reduced, nnz = self._embed([query])
        if nnz[0] == 0:
            return [], []
        # transform → distance to every centroid; the smallest are the nearest clusters.
        distances = self.kmeans.transform(reduced)[0]
        order = np.argsort(distances)[: max(1, top_n)]
        chosen = [int(c) for c in order]
        ids: list[str] = []
        for c in chosen:
            ids.extend(self.members[c])
            if max_members is not None and len(ids) >= max_members:
                return chosen, ids[:max_members]
        return chosen, ids


def _fit_vectorizer(texts: list[str], *, max_features: int, min_df: int, max_df: float) -> tuple[TfidfVectorizer, "object"]:
    """TF-IDF the corpus, falling back to no pruning if min_df/max_df would empty the vocabulary
    (keeps tiny/edge corpora working; the real 20K+ builds always hit the pruned path)."""
    try:
        vec = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=min_df, max_df=max_df)
        return vec, vec.fit_transform(texts)
    except ValueError:
        logger.warning("min_df=%d/max_df=%.2f pruned the whole vocabulary — retrying without pruning", min_df, max_df)
        vec = TfidfVectorizer(max_features=max_features, stop_words="english")
        return vec, vec.fit_transform(texts)


def _assign_members(
    model_parts: tuple[TfidfVectorizer, TruncatedSVD, KMeans],
    n_clusters: int,
    member_docs: Iterable,
    batch_size: int,
) -> list[list[str]]:
    """Stream the (full) corpus and bucket each doc id under its nearest cluster.

    A second cheap pass (transform → predict, **no re-fit**) so membership covers every
    ingested doc — even though the model was *fit* on a representative sample — which is
    what lets retrieval prune to a cluster without losing the rest of the corpus. Docs with
    no in-vocabulary terms belong to no cluster and are skipped."""
    vectorizer, svd, kmeans = model_parts
    members: list[list[str]] = [[] for _ in range(n_clusters)]
    buf_ids: list[str] = []
    buf_texts: list[str] = []

    def flush() -> None:
        if not buf_texts:
            return
        tfidf = vectorizer.transform(buf_texts)
        nnz = tfidf.getnnz(axis=1)
        preds = kmeans.predict(normalize(svd.transform(tfidf)))
        for i, c in enumerate(preds):
            if nnz[i] > 0:
                members[int(c)].append(buf_ids[i])
        buf_ids.clear()
        buf_texts.clear()

    for doc in member_docs:
        if doc.text and doc.text.strip():
            buf_ids.append(doc.doc_id)
            buf_texts.append(doc.text)
            if len(buf_texts) >= batch_size:
                flush()
    flush()
    return members


def build_clustering(
    *,
    dataset_id: str,
    n_clusters: int,
    max_features: int,
    docs: Iterable,                 # yields objects exposing .doc_id/.text (the fit sample, capped to max_docs)
    member_docs: Iterable | None = None,  # full corpus for membership (defaults to the fit sample)
    batch_size: int = 1000,
    plot_sample: int = 1500,
    svd_components: int = 128,
    min_df: int = 2,
    max_df: float = 0.6,
    on_progress: Callable[[int], None] | None = None,
) -> ClusterModel:
    """Stream raw docs → TF-IDF → LSA → KMeans → summarise. Returns the fitted model.

    The model is *fit* on ``docs`` (a sample). Membership (cluster → doc ids, for retrieval
    pruning) is then assigned over ``member_docs`` — the full corpus — so pruning can reach
    every ingested doc, not just the sample. If ``member_docs`` is None the fit sample's own
    labels are reused as membership.
    """
    texts: list[str] = []
    sample_ids: list[str] = []
    processed = 0

    for doc in docs:
        if doc.text and doc.text.strip():
            texts.append(doc.text)
            sample_ids.append(doc.doc_id)
        processed += 1
        if on_progress is not None and processed % batch_size == 0:
            on_progress(processed)
    if on_progress is not None:
        on_progress(processed)
    if len(texts) < n_clusters:
        raise ValueError(
            f"only {len(texts)} non-empty docs — need at least n_clusters={n_clusters}"
        )

    logger.info("vectorising %d docs (max_features=%d)", len(texts), max_features)
    vectorizer, matrix = _fit_vectorizer(texts, max_features=max_features, min_df=min_df, max_df=max_df)

    # LSA: reduce TF-IDF to a compact dense space, then L2-normalise so KMeans' Euclidean
    # distance behaves like cosine. n_components must stay below both vocabulary and corpus size.
    n_comp = max(2, min(svd_components, matrix.shape[1] - 1, matrix.shape[0] - 1))
    logger.info("reducing to %d LSA dims, then clustering into %d groups", n_comp, n_clusters)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced_raw = svd.fit_transform(matrix)
    reduced = normalize(reduced_raw)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced)

    # Membership (cluster → doc ids) for retrieval pruning. Assign over the full corpus
    # when given (so pruning reaches every doc); otherwise reuse the fit sample's labels.
    if member_docs is not None:
        logger.info("assigning corpus membership for pruning")
        members = _assign_members((vectorizer, svd, kmeans), n_clusters, member_docs, batch_size)
    else:
        members = [[] for _ in range(n_clusters)]
        for doc_id, c in zip(sample_ids, labels):
            members[int(c)].append(doc_id)

    # Map LSA centroids back to TF-IDF feature space for human-readable top terms.
    feature_names = vectorizer.get_feature_names_out()
    original_centroids = svd.inverse_transform(kmeans.cluster_centers_)
    order = original_centroids.argsort(axis=1)[:, ::-1]
    top_terms = [[str(feature_names[i]) for i in order[c, :_TOP_TERMS]] for c in range(n_clusters)]
    sizes = np.bincount(labels, minlength=n_clusters).astype(int).tolist()

    silhouette: float | None = None
    try:
        if reduced.shape[0] > n_clusters:
            silhouette = float(
                silhouette_score(
                    reduced, labels,  # euclidean on L2-normalised ≡ cosine ordering
                    sample_size=min(_SILHOUETTE_SAMPLE, reduced.shape[0]), random_state=42,
                )
            )
    except Exception:  # noqa: BLE001 — silhouette is a nicety, never fail the build for it
        logger.warning("silhouette computation failed — reporting None", exc_info=True)

    # 2-D scatter from the first two LSA dims (most variance), on a random sample to keep the payload small.
    n = reduced_raw.shape[0]
    idx = list(range(n))
    if n > plot_sample:
        idx = random.Random(42).sample(idx, plot_sample)
    projection = [(round(float(reduced_raw[i, 0]), 4), round(float(reduced_raw[i, 1]), 4), int(labels[i])) for i in idx]

    return ClusterModel(
        dataset_id=dataset_id,
        schema_version=SCHEMA_VERSION,
        n_clusters=n_clusters,
        num_docs=len(texts),
        max_features=max_features,
        svd_components=n_comp,
        built_at=datetime.now(timezone.utc).isoformat(),
        sizes=sizes,
        top_terms=top_terms,
        silhouette=silhouette,
        inertia=float(kmeans.inertia_),
        projection=projection,
        members=members,
        vectorizer=vectorizer,
        svd=svd,
        kmeans=kmeans,
    )
