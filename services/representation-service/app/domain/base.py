"""Pluggable representation strategies — the seam that keeps models swappable.

A representation is a framework-independent object that (1) is **built offline**
from the corpus and (2) at query time can **search** (rank docs) and **score** a
given set of docs. TF-IDF, BM25 and dense Embeddings each subclass
:class:`BaseRepresentation` and register themselves with :func:`register`; the
hybrid layer (serial / parallel + fusion) composes these via the same interface —
no caller changes, exactly the "pluggable strategies" the constitution grades.

Raw document text is **never** stored here (it lives in the doc-store, read by id
at query time). A representation only keeps ``doc_ids`` aligned row-for-row with its
matrix, so a hit's matrix row maps back to an external document id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Type

# Bumped only on a breaking change to the on-disk artifact layout.
REPRESENTATION_VERSION = "1"

# Normalize one raw text with a given PreprocessOptions dict → a normalized string.
Normalizer = Callable[[str, dict], str]

# A ranked result: (external doc_id, score), higher score = more relevant.
Scored = tuple[str, float]


class ModelKind:
    """Known representation model names (the ``model`` field across contracts)."""

    TFIDF = "tfidf"
    BM25 = "bm25"
    EMBEDDING = "embedding"   # Word2Vec (gensim)
    BERT = "bert"             # sentence-transformers (dense, contextual)


_REGISTRY: dict[str, Type["BaseRepresentation"]] = {}


def register(model_cls: Type["BaseRepresentation"]) -> Type["BaseRepresentation"]:
    """Class decorator: make a model resolvable by its ``model`` name."""
    _REGISTRY[model_cls.model] = model_cls
    return model_cls


def get_model_class(model: str) -> Type["BaseRepresentation"] | None:
    return _REGISTRY.get(model)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


class BaseRepresentation(ABC):
    """Common interface + introspection shared by every representation model."""

    model: str = ""               # overridden by each concrete model (its registry key)
    requires_preprocessing = True  # lexical models normalize the query; embeddings use raw text

    def __init__(
        self,
        *,
        dataset_id: str,
        doc_ids: list[str],
        options: dict,   # PreprocessOptions.model_dump() used at build time
        params: dict,    # model-specific params (model_dump of the params model)
        built_at: str,
        version: str = REPRESENTATION_VERSION,
    ) -> None:
        self.dataset_id = dataset_id
        self.doc_ids = doc_ids
        self.options = options
        self.params = params
        self.built_at = built_at
        self.version = version

    # ---- build (offline) ----
    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        dataset_id: str,
        corpus: Iterable[str],   # normalized docs (lexical) or raw docs (embedding), in order
        doc_ids: list[str],      # filled, in order, as ``corpus`` is consumed
        options: dict,
        params: dict,
    ) -> "BaseRepresentation":
        """Fit the model over ``corpus`` and return a ready-to-serve instance."""

    # ---- query (online) ----
    @abstractmethod
    def search(self, *, raw_query: str, top_k: int, normalize: Normalizer, **params) -> list[Scored]:
        """Return up to ``top_k`` ``(doc_id, score)`` for the query, sorted desc.

        ``normalize(text, options)`` is provided so lexical models can normalize the
        query with **their own** build-time ``options``; embedding models ignore it.
        """

    @abstractmethod
    def score_docs(
        self, *, doc_ids: list[str], raw_query: str, normalize: Normalizer, **params
    ) -> dict[str, float]:
        """Score a specific set of ``doc_ids`` for the query (used by serial re-ranking)."""

    # ---- introspection (lexical models override; sensible defaults otherwise) ----
    def encode(self, normalized_texts: list[str]):
        raise NotImplementedError(f"{self.model} does not support /encode")

    def feature_names(self):
        raise NotImplementedError(f"{self.model} has no term vocabulary")

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def stats_extra(self) -> dict:
        """Model-specific stat fields (e.g. vocab_size/nnz/density, or dim, or avgdl)."""
        return {}

    def warmup(self) -> None:
        """Load any lazy query-time resource (e.g. a transformer encoder) into memory now,
        so the first online query doesn't pay that cost on the request path and trip the
        gateway timeout. No-op by default; dense transformer models override it. Keeps the
        online query within budget — §10: models are loaded ready at startup."""
        return None
