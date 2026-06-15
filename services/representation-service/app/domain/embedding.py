"""Dense Embedding representation — Word2Vec (gensim), trained offline on the corpus.

Word2Vec learns a dense vector per word; a document's vector is the **L2-normalized
mean** of its word vectors, so cosine similarity is again a dot product. Unlike the
lexical models (which match exact terms), this captures distributional similarity —
two questions can rank close without sharing words.

Word2Vec is light and CPU-only (no torch), which is why it's the default dense model
here. A transformer (BERT / sentence-transformers) can be added later as **another**
registered model — the interface is identical — where a faster network / GPU is
available; nothing else in the system changes.

Training reads the **preprocessed** tokens (same normalization as TF-IDF/BM25), so
queries and documents live in one space.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from app.domain.base import REPRESENTATION_VERSION, BaseRepresentation, ModelKind, Normalizer, Scored, register
from app.domain.ranking import top_k_indices


def _mean_vector(tokens: list[str], kv, dim: int) -> np.ndarray:
    """L2-normalized mean of the in-vocabulary word vectors (zeros if none match)."""
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    v = np.mean(vecs, axis=0).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@register
class EmbeddingRepresentation(BaseRepresentation):
    model = ModelKind.EMBEDDING
    requires_preprocessing = True  # Word2Vec trains on the same normalized tokens

    def __init__(self, *, dataset_id, matrix, doc_ids, kv, dim, options, params, built_at,
                 version=REPRESENTATION_VERSION):
        super().__init__(dataset_id=dataset_id, doc_ids=doc_ids, options=options,
                         params=params, built_at=built_at, version=version)
        self.matrix = matrix      # np.ndarray float32 (N, dim), L2-normalized rows
        self.kv = kv              # gensim KeyedVectors (word vectors) for query encoding
        self.dim = dim
        self._index: dict[str, int] | None = None

    # ---- build ----
    @classmethod
    def build(cls, *, dataset_id, corpus: Iterable[str], doc_ids, options, params):
        from gensim.models import Word2Vec  # lazy: keeps service import light
        sentences = [text.split() for text in corpus]   # materialize (drives build progress)
        if not sentences:
            raise RuntimeError(f"Word2Vec build saw no documents for '{dataset_id}'")
        w2v = Word2Vec(
            sentences=sentences,
            vector_size=int(params.get("vector_size", 100)),
            window=int(params.get("window", 5)),
            min_count=int(params.get("min_count", 2)),
            epochs=int(params.get("epochs", 5)),
            sg=int(params.get("sg", 1)),
            workers=4,
        )
        kv = w2v.wv
        dim = int(kv.vector_size)
        matrix = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, toks in enumerate(sentences):
            matrix[i] = _mean_vector(toks, kv, dim)
        return cls(dataset_id=dataset_id, matrix=matrix, doc_ids=doc_ids, kv=kv, dim=dim,
                   options=options, params=params, built_at=datetime.now(timezone.utc).isoformat())

    # ---- query ----
    def _encode_query(self, raw_query: str, normalize: Normalizer) -> np.ndarray:
        tokens = normalize(raw_query, self.options).split()
        return _mean_vector(tokens, self.kv, self.dim)

    def search(self, *, raw_query, top_k, normalize, **_) -> list[Scored]:
        qv = self._encode_query(raw_query, normalize)
        if not qv.any():            # query had no in-vocabulary words
            return []
        scores = self.matrix @ qv   # cosine (rows + query L2-normalized)
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices(scores, top_k) if scores[i] > 0]

    def score_docs(self, *, doc_ids, raw_query, normalize, **_) -> dict[str, float]:
        qv = self._encode_query(raw_query, normalize)
        if not qv.any():
            return {}
        index = self._doc_index()
        rows = [(d, index[d]) for d in doc_ids if d in index]
        if not rows:
            return {}
        sub = self.matrix[[r for _, r in rows]]
        scores = sub @ qv
        return {d: float(s) for (d, _), s in zip(rows, scores)}

    def stats_extra(self) -> dict:
        return {"dim": self.dim, "vocab_size": len(self.kv)}

    # ---- internals ----
    def _doc_index(self) -> dict[str, int]:
        if self._index is None:
            self._index = {d: i for i, d in enumerate(self.doc_ids)}
        return self._index

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_index", None)   # derived map — rebuilt on demand
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._index = None
