"""Pluggable representation strategies — the seam that keeps models swappable.

A representation is a framework-independent object that (1) is **built offline**
from the preprocessed corpus and (2) can **encode** a query into the same space at
runtime. TF-IDF is the only concrete model today; BM25 and dense embeddings slot in
by subclassing :class:`BaseRepresentation` and decorating with :func:`register` —
no caller changes, exactly the "pluggable strategies" the constitution grades.

Raw document text is **never** stored here (it lives in the doc-store, read by ID
at query time). A representation only keeps ``doc_ids`` aligned row-for-row with its
matrix, so a retriever can map a matrix row back to an external document id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Type

# Bumped only on a breaking change to the on-disk artifact layout.
REPRESENTATION_VERSION = "1"


class ModelKind:
    """Known representation model names (the ``model`` field across contracts)."""

    TFIDF = "tfidf"


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

    model: str = ""  # overridden by each concrete model (its registry key)

    def __init__(
        self,
        *,
        dataset_id: str,
        doc_ids: list[str],
        options: dict,   # PreprocessOptions.model_dump() used at build time
        params: dict,    # model params (e.g. TfidfParams.model_dump())
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
        corpus: Iterable[str],   # yields one normalized (space-joined tokens) doc per document
        doc_ids: list[str],      # filled, in order, as ``corpus`` is consumed
        options: dict,
        params: dict,
    ) -> "BaseRepresentation":
        """Fit the model over ``corpus`` and return a ready-to-serve instance."""

    # ---- query (online) ----
    @abstractmethod
    def encode(self, normalized_texts: list[str]):
        """Vectorize already-normalized query texts → a scipy CSR matrix in the
        document space (one row per text)."""

    @abstractmethod
    def feature_names(self):
        """Array mapping a matrix column index → its term (for introspection)."""

    # ---- introspection ----
    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def nnz(self) -> int: ...

    @property
    def density(self) -> float:
        denom = self.num_docs * self.vocab_size
        return (self.nnz / denom) if denom else 0.0
