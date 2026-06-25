"""Core topic-modelling logic — framework-independent (no FastAPI here).

Offline pipeline: stream the **raw** corpus → CountVectorizer (lowercase + English
stop-words + min_df/max_df noise filter) → Latent Dirichlet Allocation. The vectoriser
tokenises raw text itself, so the model is self-contained (no preprocessing-service) and a
new query is *inferred* with the same analyser used at fit time. LDA uses **batch**
variational Bayes: the corpus is capped (``max_docs``) and loaded in memory, so batch
converges to sharper topics than the online (mini-batch) learner, which is meant for
streaming corpora that don't fit in RAM.

The fitted :class:`TopicModel` keeps what the online endpoints need without a re-fit:
each topic's top terms + weights (the topic charts), the corpus's dominant-topic sizes,
the LDA perplexity (fit quality), and the vectoriser + LDA so a query's topic mixture can
be inferred. A query with no in-vocabulary terms (empty/stop-words-only) has no dominant
topic (reported as ``-1``) so the re-ranker can treat it as a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("topic-service")

# Bump when the persisted shape changes so stale pickles are auto-rebuilt (see TopicStore.load).
SCHEMA_VERSION = 2

_TOP_TERMS = 12
NO_TOPIC = -1   # a text with no in-vocabulary terms has no dominant topic


@dataclass
class TopicModel:
    dataset_id: str
    schema_version: int
    n_topics: int
    num_docs: int
    max_features: int
    built_at: str
    sizes: list[int]                              # docs whose dominant topic is t
    top_terms: list[list[str]]                    # per topic
    weights: list[list[float]]                    # per topic, aligned with top_terms
    perplexity: float | None
    vectorizer: CountVectorizer
    lda: LatentDirichletAllocation

    def label(self, topic_id: int) -> str:
        if topic_id < 0:
            return "—"
        terms = self.top_terms[topic_id]
        return " / ".join(terms[:2]) if terms else f"topic {topic_id}"

    def terms_for(self, topic_id: int) -> list[str]:
        return self.top_terms[topic_id] if topic_id >= 0 else []

    def infer(self, texts: list[str]) -> list[tuple[int, list[float]]]:
        """Topic mixture per text → ``(dominant_topic, [weight per topic])`` (dominant -1 if no in-vocab terms)."""
        counts = self.vectorizer.transform(texts)
        nnz = counts.getnnz(axis=1)
        dist = self.lda.transform(counts)
        out: list[tuple[int, list[float]]] = []
        for i, row in enumerate(dist):
            # Normalise each row to a proper distribution (LDA rows already ~sum to 1).
            total = float(row.sum()) or 1.0
            probs = (row / total).tolist()
            dominant = int(np.argmax(row)) if nnz[i] > 0 else NO_TOPIC
            out.append((dominant, [round(p, 4) for p in probs]))
        return out


def build_topics(
    *,
    dataset_id: str,
    n_topics: int,
    max_features: int,
    max_iter: int,
    max_df: float = 0.5,
    docs: Iterable,                 # yields objects exposing .text (capped to max_docs by the caller)
    batch_size: int = 1000,
    on_progress: Callable[[int], None] | None = None,
) -> TopicModel:
    """Stream raw docs → counts → LDA → summarise. Returns the fitted model."""
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

    if len(texts) < n_topics:
        raise ValueError(f"only {len(texts)} non-empty docs — need at least n_topics={n_topics}")

    logger.info("vectorising %d docs (max_features=%d, max_df=%.2f)", len(texts), max_features, max_df)
    # Raw-text counts (LDA wants term counts, not TF-IDF); drop English stop-words, terms
    # in <2 docs (noise), and terms in >max_df of docs (ubiquitous, non-discriminative —
    # the main source of generic, blurry topics on short Quora text) so topics are crisp.
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english", min_df=2, max_df=max_df)
    matrix = vectorizer.fit_transform(texts)

    logger.info("fitting LDA with %d topics (max_iter=%d, batch)", n_topics, max_iter)
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=max_iter, learning_method="batch",
        random_state=42, n_jobs=1,
    )
    doc_topic = lda.fit_transform(matrix)

    feature_names = vectorizer.get_feature_names_out()
    components = lda.components_
    top_terms: list[list[str]] = []
    weights: list[list[float]] = []
    for t in range(n_topics):
        order = components[t].argsort()[::-1][:_TOP_TERMS]
        total = float(components[t].sum()) or 1.0
        top_terms.append([str(feature_names[i]) for i in order])
        weights.append([round(float(components[t][i] / total), 5) for i in order])
    sizes = np.bincount(doc_topic.argmax(axis=1), minlength=n_topics).astype(int).tolist()

    perplexity: float | None = None
    try:
        perplexity = float(lda.perplexity(matrix))
    except Exception:  # noqa: BLE001 — a nicety; never fail the build for it
        logger.warning("perplexity computation failed — reporting None", exc_info=True)

    return TopicModel(
        dataset_id=dataset_id,
        schema_version=SCHEMA_VERSION,
        n_topics=n_topics,
        num_docs=len(texts),
        max_features=max_features,
        built_at=datetime.now(timezone.utc).isoformat(),
        sizes=sizes,
        top_terms=top_terms,
        weights=weights,
        perplexity=perplexity,
        vectorizer=vectorizer,
        lda=lda,
    )
