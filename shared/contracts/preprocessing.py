"""Contracts for the preprocessing-service.

The *same* options object is applied to documents (at indexing time) and to
queries (at search time), which is exactly what the assignment requires: queries
must be processed with the same technique as documents so they stay comparable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreprocessOptions(BaseModel):
    """Per-request toggles for each normalization stage."""

    lowercase: bool = True
    remove_stopwords: bool = True
    # Stemming and lemmatization are both offered; typically pick one. Default to
    # lemmatization (keeps real words) and leave stemming off. Note: stem/lemma
    # output is always lowercase (Porter lowercases by design; WordNet lemma
    # lookup is case-insensitive), even when `lowercase` is False.
    stem: bool = False
    lemmatize: bool = True
    min_token_length: int = Field(default=2, ge=1)


class PreprocessRequest(BaseModel):
    text: str
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)


class PreprocessResult(BaseModel):
    """Output for one piece of text."""

    tokens: list[str]
    # tokens re-joined by space — convenient to feed straight into vectorizers.
    normalized_text: str


class PreprocessBatchRequest(BaseModel):
    texts: list[str]
    options: PreprocessOptions = Field(default_factory=PreprocessOptions)


class PreprocessBatchResponse(BaseModel):
    items: list[PreprocessResult]
