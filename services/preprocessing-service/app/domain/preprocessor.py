"""Core text-normalization logic — framework-independent and reusable.

This module knows nothing about FastAPI/HTTP. It takes text + options and returns
tokens, so the exact same code path serves both documents and queries.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from shared.contracts import PreprocessOptions

# Tokenizer: keep runs of ASCII letters/digits, which also drops punctuation.
# Accents are folded to ASCII first (Müller -> Muller) and apostrophes removed
# (don't -> dont) so contractions/possessives survive as a single token.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_APOSTROPHES = str.maketrans("", "", "'’`")


def _fold_to_ascii(text: str) -> str:
    """NFKD-normalize and strip combining marks so accented Latin folds to ASCII."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


class Stemmer(Protocol):
    def stem(self, token: str) -> str: ...


class Lemmatizer(Protocol):
    def lemmatize(self, token: str, pos: str) -> str: ...


class PosTagger(Protocol):
    """Tags a token sequence, returning a WordNet POS char ('n','v','a','r')
    parallel to the input. Sequence-level so surrounding words inform each tag."""

    def tag(self, tokens: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class TextPreprocessor:
    """Pure normalization pipeline. Dependencies (stopwords, stemmer, lemmatizer,
    pos_tagger) are injected so the logic stays decoupled from NLTK."""

    stopwords: frozenset[str]
    stemmer: Stemmer
    lemmatizer: Lemmatizer
    pos_tagger: PosTagger

    def __post_init__(self) -> None:
        # Project the stopword set into the same normalized space as tokens
        # (fold accents, drop apostrophes, lowercase). Stopword lists keep
        # contractions with the apostrophe ("don't") while our tokens never
        # contain one ("dont") — without this, contraction stopwords leak
        # straight into the index.
        normalized = frozenset(
            _fold_to_ascii(w).translate(_APOSTROPHES).lower() for w in self.stopwords
        )
        object.__setattr__(self, "stopwords", normalized)

    def process(self, text: str, options: PreprocessOptions) -> list[str]:
        text = _fold_to_ascii(text).translate(_APOSTROPHES)
        if options.lowercase:
            text = text.lower()

        raw = _TOKEN_RE.findall(text)
        # POS-tag the whole sequence up front so the lemmatizer gets real context
        # (verb vs. noun → "studying"→"study", not "studying"). Only pay the
        # tagging cost when lemmatization is actually requested.
        tags = self.pos_tagger.tag(raw) if options.lemmatize else None

        tokens: list[str] = []
        for i, tok in enumerate(raw):
            # Stopwords are matched case-insensitively on the surface form, before
            # stem/lemma, so common words are caught regardless of their root and
            # regardless of the `lowercase` toggle.
            if options.remove_stopwords and tok.lower() in self.stopwords:
                continue
            if options.lemmatize:
                tok = self.lemmatizer.lemmatize(tok, tags[i])  # type: ignore[index]
            if options.stem:
                tok = self.stemmer.stem(tok)
            if len(tok) < options.min_token_length:
                continue
            tokens.append(tok)
        return tokens
