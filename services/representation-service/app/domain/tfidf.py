"""TF-IDF (Vector Space Model) representation — sparse weights, cosine ranking.

Built with scikit-learn's ``TfidfVectorizer``. The corpus arrives **already
preprocessed**, so the analyzer is a plain whitespace split (we never re-tokenize),
which keeps document and query weighting identical and the build fast. Rows are
L2-normalized, so a query·document dot product is exactly **cosine similarity** —
search is a single sparse matrix-vector product.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.domain.base import REPRESENTATION_VERSION, BaseRepresentation, ModelKind, Normalizer, Scored, register
from app.domain.ranking import top_k_indices


def _split(text: str) -> list[str]:
    """Analyzer for ``TfidfVectorizer``: text is already normalized, just split on
    whitespace. Module-level so the fitted vectorizer stays picklable."""
    return text.split()


@register
class TfidfRepresentation(BaseRepresentation):
    model = ModelKind.TFIDF
    requires_preprocessing = True

    def __init__(self, *, dataset_id, vectorizer, matrix, doc_ids, options, params, built_at,
                 version=REPRESENTATION_VERSION):
        super().__init__(dataset_id=dataset_id, doc_ids=doc_ids, options=options,
                         params=params, built_at=built_at, version=version)
        self.vectorizer = vectorizer
        self.matrix = matrix  # CSR float32, L2-normalized rows
        self._index: dict[str, int] | None = None  # lazy doc_id -> row

    # ---- build ----
    @classmethod
    def build(cls, *, dataset_id, corpus: Iterable[str], doc_ids, options, params):
        vectorizer = TfidfVectorizer(
            analyzer=_split, lowercase=False, dtype=np.float32,
            min_df=params.get("min_df", 1), max_df=params.get("max_df", 1.0),
            sublinear_tf=params.get("sublinear_tf", True), norm="l2",
        )
        try:
            matrix = vectorizer.fit_transform(corpus).tocsr()
        except ValueError as exc:
            raise RuntimeError(
                f"TF-IDF build produced an empty vocabulary for '{dataset_id}' ({exc}) "
                "— loosen the preprocessing options or min_df"
            ) from exc
        return cls(dataset_id=dataset_id, vectorizer=vectorizer, matrix=matrix, doc_ids=doc_ids,
                   options=options, params=params, built_at=datetime.now(timezone.utc).isoformat())

    # ---- query ----
    def _query_vec(self, raw_query: str, normalize: Normalizer):
        return self.vectorizer.transform([normalize(raw_query, self.options)])  # 1×V, L2-normalized

    def search(self, *, raw_query, top_k, normalize, **_) -> list[Scored]:
        qv = self._query_vec(raw_query, normalize)
        scores = (self.matrix @ qv.T).toarray().ravel()  # cosine (both L2-normalized)
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices(scores, top_k) if scores[i] > 0]

    def score_docs(self, *, doc_ids, raw_query, normalize, **_) -> dict[str, float]:
        qv = self._query_vec(raw_query, normalize)
        idx = self._doc_index()
        rows = [(d, idx[d]) for d in doc_ids if d in idx]
        if not rows:
            return {}
        sub = self.matrix[[r for _, r in rows]]
        scores = (sub @ qv.T).toarray().ravel()
        return {d: float(s) for (d, _), s in zip(rows, scores)}

    # ---- introspection ----
    def encode(self, normalized_texts):
        return self.vectorizer.transform(normalized_texts)

    def feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def stats_extra(self) -> dict:
        vocab = len(self.vectorizer.vocabulary_)
        denom = self.num_docs * vocab
        return {
            "vocab_size": vocab,
            "nnz": int(self.matrix.nnz),
            "density": (self.matrix.nnz / denom) if denom else 0.0,
        }

    # ---- internals ----
    def _doc_index(self) -> dict[str, int]:
        if self._index is None:
            self._index = {d: i for i, d in enumerate(self.doc_ids)}
        return self._index

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_index", None)  # derived; rebuilt on demand
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._index = None
