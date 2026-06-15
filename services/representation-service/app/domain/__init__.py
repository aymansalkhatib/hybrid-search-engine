"""Domain layer for the representation-service.

Importing the package registers every pluggable model with the strategy registry
in :mod:`app.domain.base` (so ``get_model_class("tfidf")`` resolves without the
caller importing the concrete class). Add a new model = add one import here.
"""

from app.domain import tfidf as tfidf          # noqa: F401  (registers TfidfRepresentation)
from app.domain import bm25 as bm25            # noqa: F401  (registers Bm25Representation)
from app.domain import embedding as embedding  # noqa: F401  (registers EmbeddingRepresentation — Word2Vec)
from app.domain import bert as bert            # noqa: F401  (registers BertRepresentation — sentence-transformers)
