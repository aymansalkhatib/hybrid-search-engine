"""TF-IDF (Vector Space Model) representation — the first, simplest model.

Built with scikit-learn's ``TfidfVectorizer`` (battle-tested, C-accelerated). The
corpus arrives **already preprocessed** (tokenized/normalized by the preprocessing-
service), so the analyzer is a plain whitespace split — we never re-lowercase or
re-tokenize, which keeps document and query weighting identical and the build fast.

Rows are L2-normalized (``norm="l2"``), so a dot product between a query vector and a
document row is exactly **cosine similarity** — the retrieval step (Phase 4) becomes a
single sparse matrix-vector product. Weights are stored as ``float32`` to halve memory
and speed up that product on a 200K+ corpus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.domain.base import REPRESENTATION_VERSION, BaseRepresentation, ModelKind, register


def _split(text: str) -> list[str]:
    """Analyzer for ``TfidfVectorizer``: the text is already normalized, so just
    split on whitespace. Defined at module level so the fitted vectorizer (which
    holds a reference to it) stays picklable."""
    return text.split()


@register
class TfidfRepresentation(BaseRepresentation):
    model = ModelKind.TFIDF

    def __init__(
        self,
        *,
        dataset_id: str,
        vectorizer: TfidfVectorizer,
        matrix,                 # scipy CSR, float32, L2-normalized rows
        doc_ids: list[str],
        options: dict,
        params: dict,
        built_at: str,
        version: str = REPRESENTATION_VERSION,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id, doc_ids=doc_ids, options=options,
            params=params, built_at=built_at, version=version,
        )
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def build(cls, *, dataset_id, corpus: Iterable[str], doc_ids, options, params):
        vectorizer = TfidfVectorizer(
            analyzer=_split,        # corpus is pre-tokenized; just split on spaces
            lowercase=False,        # preprocessing already did any casing
            dtype=np.float32,       # compact + fast cosine
            min_df=params.get("min_df", 1),
            max_df=params.get("max_df", 1.0),
            sublinear_tf=params.get("sublinear_tf", True),
            norm="l2",              # rows L2-normalized ⇒ dot product = cosine
        )
        try:
            matrix = vectorizer.fit_transform(corpus).tocsr()
        except ValueError as exc:
            # Almost always "empty vocabulary" — every doc was filtered away by the
            # preprocessing options (e.g. min length too high). Surface it clearly.
            raise RuntimeError(
                f"TF-IDF build produced an empty vocabulary for '{dataset_id}' "
                f"({exc}) — loosen the preprocessing options or min_df"
            ) from exc
        return cls(
            dataset_id=dataset_id,
            vectorizer=vectorizer,
            matrix=matrix,
            doc_ids=doc_ids,
            options=options,
            params=params,
            built_at=datetime.now(timezone.utc).isoformat(),
        )

    def encode(self, normalized_texts: list[str]):
        """Transform queries with the fitted vocabulary/idf → CSR (L2-normalized)."""
        return self.vectorizer.transform(normalized_texts)

    def feature_names(self):
        return self.vectorizer.get_feature_names_out()

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    @property
    def nnz(self) -> int:
        return int(self.matrix.nnz)
