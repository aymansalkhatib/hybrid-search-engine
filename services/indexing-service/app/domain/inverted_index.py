"""In-memory inverted index — the core data structure of the indexing-service.

Framework-independent and picklable. Stores, per term, a postings list of
``(doc_idx, tf)`` and, per document, its length (token count). That is exactly
what BM25 (`df`, `tf`, doc length, `avgdl`) and TF-IDF need.

Raw document text is **deliberately not stored here** — it lives in the doc-store
(MongoDB) and is fetched by ID at query time. Documents are
referenced internally by a dense integer ``doc_idx`` (0..N-1) to keep postings
compact; ``doc_ids[doc_idx]`` maps back to the external id.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

INDEX_VERSION = "1"


@dataclass
class InvertedIndex:
    dataset_id: str
    version: str = INDEX_VERSION
    options: dict = field(default_factory=dict)  # PreprocessOptions used at build time
    built_at: str = ""

    # internal doc index i (0..N-1) ↔ external doc_id
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    # term -> [(doc_idx, tf), ...], appended in increasing doc_idx order
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Lazy reverse map (doc_id -> idx); built on demand, never pickled.
        self._doc_index: dict[str, int] | None = None

    # ---- build ----
    def add_document(self, doc_id: str, tokens: list[str]) -> None:
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            self.postings.setdefault(term, []).append((idx, tf))
        self._doc_index = None  # invalidate reverse map

    # ---- corpus stats ----
    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    @property
    def total_tokens(self) -> int:
        return sum(self.doc_lengths)

    @property
    def num_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    @property
    def avg_doc_length(self) -> float:
        return self.total_tokens / self.num_docs if self.num_docs else 0.0

    # ---- term/doc queries ----
    def df(self, term: str) -> int:
        """Document frequency: # docs containing the term."""
        return len(self.postings.get(term, ()))

    def cf(self, term: str) -> int:
        """Collection frequency: total occurrences of the term across the corpus."""
        return sum(tf for _, tf in self.postings.get(term, ()))

    def postings_for(self, term: str) -> list[tuple[str, int]]:
        """Postings as external ``(doc_id, tf)`` pairs, ordered by doc index."""
        return [(self.doc_ids[i], tf) for i, tf in self.postings.get(term, ())]

    def doc_length(self, doc_id: str) -> int | None:
        idx = self._doc_index_map().get(doc_id)
        return None if idx is None else self.doc_lengths[idx]

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self._doc_index_map()

    # ---- internals ----
    def _doc_index_map(self) -> dict[str, int]:
        if self._doc_index is None:
            self._doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        return self._doc_index

    def __getstate__(self) -> dict:
        # Drop the derived reverse map from the pickle (rebuilt on demand).
        state = self.__dict__.copy()
        state.pop("_doc_index", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._doc_index = None
