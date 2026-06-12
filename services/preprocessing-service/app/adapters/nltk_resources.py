"""NLTK resource loading — the IO/adapter boundary for the domain.

Builds a ready-to-use TextPreprocessor with English stopwords, a Porter stemmer,
a WordNet lemmatizer and a POS tagger. Stem/lemmatize are memoized because the
vocabulary is far smaller than the token stream on a 200K+ document corpus.
Required corpora are pre-baked into the Docker image; locally they are downloaded
on first run if missing.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from app.domain.preprocessor import TextPreprocessor
from shared.contracts import PreprocessOptions

logger = logging.getLogger("preprocessing-service")

# (lookup path, download id) for each NLTK resource the preprocessor needs.
# Note: only the core English `wordnet` is required for lemmatization — the
_REQUIRED = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

# Penn Treebank tag prefix -> WordNet POS char. Anything else (determiners,
# prepositions, ...) falls back to noun, which is WordNetLemmatizer's own default.
_PENN_TO_WORDNET = {"J": "a", "V": "v", "N": "n", "R": "r"}


def _to_wordnet_pos(penn_tag: str) -> str:
    return _PENN_TO_WORDNET.get(penn_tag[:1].upper(), "n")


def ensure_nltk_data() -> None:
    """Download any missing NLTK resources (no-op when already present)."""
    for path, pkg in _REQUIRED:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            if not nltk.download(pkg, quiet=True):
                raise RuntimeError(
                    f"could not download NLTK resource '{pkg}' — check network "
                    "access or pre-install it into NLTK_DATA"
                )


class CachedLemmatizer:
    """WordNet lemmatizer memoized per (token, pos) to avoid repeated lookups.

    Tokens are lowercased for the lookup: WordNet stores lowercase lemmas, so
    "Dogs" would silently come back unchanged while "dogs" -> "dog". Lemma
    output is therefore always lowercase (as is stemmer output, by Porter's
    own default), regardless of the `lowercase` option.
    """

    def __init__(self, maxsize: int = 100_000) -> None:
        self._lemmatize = lru_cache(maxsize=maxsize)(WordNetLemmatizer().lemmatize)

    def lemmatize(self, token: str, pos: str) -> str:
        return self._lemmatize(token.lower(), pos)


class CachedStemmer:
    """Porter stemmer memoized per token."""

    def __init__(self, maxsize: int = 100_000) -> None:
        self._stem = lru_cache(maxsize=maxsize)(PorterStemmer().stem)

    def stem(self, token: str) -> str:
        return self._stem(token)


class NltkPosTagger:
    """Sequence POS tagger mapping Penn Treebank tags to WordNet POS chars."""

    def tag(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        return [_to_wordnet_pos(tag) for _, tag in pos_tag(tokens)]


def build_preprocessor() -> TextPreprocessor:
    ensure_nltk_data()
    preprocessor = TextPreprocessor(
        stopwords=frozenset(nltk_stopwords.words("english")),
        stemmer=CachedStemmer(),
        lemmatizer=CachedLemmatizer(),
        pos_tagger=NltkPosTagger(),
    )
    # Exercise the full pipeline once: this loads the lazy POS-tagger model and
    # WordNet now, so a missing/corrupt resource fails startup instead of the
    # first request (where /health would stay green while requests 500), and the
    # first real request doesn't pay the model-load latency.
    preprocessor.process("startup warmup probe", PreprocessOptions(stem=True))
    return preprocessor
