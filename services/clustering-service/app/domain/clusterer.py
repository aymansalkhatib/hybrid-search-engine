"""Core clustering logic — framework-independent (no FastAPI here).

Offline pipeline: stream the **raw** corpus → TF-IDF → MiniBatchKMeans. The TF-IDF
vectoriser tokenises raw text itself (lowercasing + English stop-word removal), so the
clustering is **self-contained** — it doesn't depend on the preprocessing-service, and,
crucially, a new query or candidate document can be *assigned* with the **same**
analyser used at fit time (transform → predict) with no normalisation mismatch. That
keeps assignment cheap enough to cluster-rerank a candidate pool on every query.

The fitted :class:`ClusterModel` keeps everything the online endpoints need without a
re-fit: per-cluster sizes & top terms (from the centroids), a silhouette score for
cluster quality, and a 2-D TruncatedSVD projection sample for the scatter plot.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger("clustering-service")

_TOP_TERMS = 10
_SILHOUETTE_SAMPLE = 2000


@dataclass
class ClusterModel:
    dataset_id: str
    n_clusters: int
    num_docs: int
    max_features: int
    built_at: str
    sizes: list[int]
    top_terms: list[list[str]]
    silhouette: float | None
    inertia: float | None
    projection: list[tuple[float, float, int]]   # (x, y, cluster) sample for the scatter
    vectorizer: TfidfVectorizer
    kmeans: MiniBatchKMeans

    def assign(self, texts: list[str]) -> list[tuple[int, list[str]]]:
        """Assign each (raw) text to its nearest cluster → ``(cluster_id, top_terms)``."""
        matrix = self.vectorizer.transform(texts)
        labels = self.kmeans.predict(matrix)
        return [(int(c), self.top_terms[int(c)]) for c in labels]

    def assign_ids(self, texts: list[str]) -> list[int]:
        """Cluster id per text (the lean primitive the retrieval re-ranker uses)."""
        return [int(c) for c in self.kmeans.predict(self.vectorizer.transform(texts))]


def build_clustering(
    *,
    dataset_id: str,
    n_clusters: int,
    max_features: int,
    docs: Iterable,                 # yields objects exposing .text (capped to max_docs by the caller)
    batch_size: int = 1000,
    plot_sample: int = 1500,
    on_progress: Callable[[int], None] | None = None,
) -> ClusterModel:
    """Stream raw docs → TF-IDF → KMeans → summarise. Returns the fitted model."""
    texts: list[str] = []
    processed = 0

    for doc in docs:
        if doc.text and doc.text.strip():
            texts.append(doc.text)
        processed += 1
        if on_progress is not None and processed % batch_size == 0:
            on_progress(processed)
    if on_progress is not None:
        on_progress(processed)
    if len(texts) < n_clusters:
        raise ValueError(
            f"only {len(texts)} non-empty docs after preprocessing — need at least n_clusters={n_clusters}"
        )

    logger.info("vectorising %d docs (max_features=%d)", len(texts), max_features)
    # Raw-text TF-IDF: the vectoriser lowercases and strips English stop-words itself,
    # so fit and assign share one analyser (no preprocessing-service dependency).
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    logger.info("clustering into %d groups", n_clusters)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=1024)
    labels = kmeans.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_
    top_terms = [
        [str(feature_names[i]) for i in centers[c].argsort()[::-1][:_TOP_TERMS]]
        for c in range(n_clusters)
    ]
    sizes = np.bincount(labels, minlength=n_clusters).astype(int).tolist()

    silhouette: float | None = None
    try:
        if matrix.shape[0] > n_clusters:
            silhouette = float(
                silhouette_score(
                    matrix, labels, metric="cosine",
                    sample_size=min(_SILHOUETTE_SAMPLE, matrix.shape[0]), random_state=42,
                )
            )
    except Exception:  # noqa: BLE001 — silhouette is a nicety, never fail the build for it
        logger.warning("silhouette computation failed — reporting None", exc_info=True)

    # 2-D projection for the scatter plot, on a random sample to keep the payload small.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(matrix)
    n = coords.shape[0]
    idx = list(range(n))
    if n > plot_sample:
        idx = random.Random(42).sample(idx, plot_sample)
    projection = [(round(float(coords[i, 0]), 4), round(float(coords[i, 1]), 4), int(labels[i])) for i in idx]

    return ClusterModel(
        dataset_id=dataset_id,
        n_clusters=n_clusters,
        num_docs=len(texts),
        max_features=max_features,
        built_at=datetime.now(timezone.utc).isoformat(),
        sizes=sizes,
        top_terms=top_terms,
        silhouette=silhouette,
        inertia=float(kmeans.inertia_),
        projection=projection,
        vectorizer=vectorizer,
        kmeans=kmeans,
    )
