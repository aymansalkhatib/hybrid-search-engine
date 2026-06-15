"""Dense Embedding representation — BERT via sentence-transformers.

A bi-encoder transformer maps each document to a contextual dense vector; vectors are
L2-normalized so cosine similarity is a dot product. Unlike Word2Vec (static word
vectors averaged), BERT embeddings are **contextual** and usually rank notably better
on semantic retrieval — at the cost of a heavier dependency (torch) and slower builds.

Both dense models coexist as separate registered strategies (``embedding`` = Word2Vec,
``bert`` = transformer), so the evaluation can compare them and a parallel hybrid can
even fuse both.

BERT reads the **raw** text (its own tokenizer), so no preprocessing is applied. The
transformer itself is **not** pickled (only the doc-vector matrix + model name are);
it is lazily (re)loaded by name and cached both in-process and on the data volume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from app.domain.base import REPRESENTATION_VERSION, BaseRepresentation, ModelKind, Normalizer, Scored, register
from app.domain.ranking import top_k_indices

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Process-wide cache of loaded transformers (keyed by model name).
_ST_CACHE: dict = {}


def _load_st(model_name: str):
    """Lazily import + load a sentence-transformer (kept out of module import so the
    service starts even before torch is exercised)."""
    if model_name not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_CACHE[model_name]


@register
class BertRepresentation(BaseRepresentation):
    model = ModelKind.BERT
    requires_preprocessing = False  # the transformer tokenizes raw text itself

    def __init__(self, *, dataset_id, matrix, doc_ids, model_name, dim, options, params,
                 built_at, version=REPRESENTATION_VERSION):
        super().__init__(dataset_id=dataset_id, doc_ids=doc_ids, options=options,
                         params=params, built_at=built_at, version=version)
        self.matrix = matrix          # np.ndarray float32 (N, dim), L2-normalized rows
        self.model_name = model_name
        self.dim = dim
        self._index: dict[str, int] | None = None

    # ---- build ----
    @classmethod
    def build(cls, *, dataset_id, corpus: Iterable[str], doc_ids, options, params):
        model_name = params.get("model_name", DEFAULT_MODEL)
        batch_size = int(params.get("batch_size", 64))
        st = _load_st(model_name)
        dim = int(st.get_sentence_embedding_dimension())

        chunks: list[np.ndarray] = []
        buf: list[str] = []

        def flush_buf():
            if not buf:
                return
            vecs = st.encode(buf, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            chunks.append(np.asarray(vecs, dtype=np.float32))
            buf.clear()

        for text in corpus:           # raw text; consuming this drives build progress
            buf.append(text)
            if len(buf) >= batch_size:
                flush_buf()
        flush_buf()

        matrix = np.vstack(chunks) if chunks else np.zeros((0, dim), dtype=np.float32)
        return cls(dataset_id=dataset_id, matrix=matrix, doc_ids=doc_ids, model_name=model_name,
                   dim=dim, options=options, params=params,
                   built_at=datetime.now(timezone.utc).isoformat())

    # ---- query ----
    def _encode_query(self, raw_query: str) -> np.ndarray:
        st = _load_st(self.model_name)
        vec = st.encode([raw_query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def search(self, *, raw_query, top_k, normalize, **_) -> list[Scored]:
        qv = self._encode_query(raw_query)
        scores = self.matrix @ qv  # cosine (rows + query L2-normalized)
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices(scores, top_k) if scores[i] > 0]

    def score_docs(self, *, doc_ids, raw_query, normalize, **_) -> dict[str, float]:
        qv = self._encode_query(raw_query)
        index = self._doc_index()
        rows = [(d, index[d]) for d in doc_ids if d in index]
        if not rows:
            return {}
        sub = self.matrix[[r for _, r in rows]]
        scores = sub @ qv
        return {d: float(s) for (d, _), s in zip(rows, scores)}

    def stats_extra(self) -> dict:
        return {"dim": self.dim}

    # ---- internals ----
    def _doc_index(self) -> dict[str, int]:
        if self._index is None:
            self._index = {d: i for i, d in enumerate(self.doc_ids)}
        return self._index

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_index", None)   # the transformer is never an attribute, so never pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._index = None
