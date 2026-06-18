"""Loads the offline resources the refiner needs: a SymSpell spell-checker primed
with an English frequency dictionary, plus a WordNet handle and a stopword set.

Everything here is loaded **once at startup** (no network, no per-query fitting), in
keeping with the offline-build / online-query split — refinement must stay well
inside the ≤20 s query budget.
"""

from __future__ import annotations

import importlib.resources
import logging

from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn
from symspellpy import SymSpell

logger = logging.getLogger("query-refinement-service")

# SymSpell is built once with the largest edit distance we'll ever allow (2); each
# per-query lookup may request a smaller budget, never a larger one.
_MAX_EDIT_DISTANCE = 2
_PREFIX_LENGTH = 7
# Bundled with the symspellpy wheel — ~83k words with corpus frequencies. No download.
_FREQ_DICT = "frequency_dictionary_en_82_765.txt"


def build_symspell() -> SymSpell:
    """A SymSpell checker loaded with symspellpy's bundled frequency dictionary."""
    sym = SymSpell(max_dictionary_edit_distance=_MAX_EDIT_DISTANCE, prefix_length=_PREFIX_LENGTH)
    dict_path = importlib.resources.files("symspellpy") / _FREQ_DICT
    if not sym.load_dictionary(str(dict_path), term_index=0, count_index=1):
        raise RuntimeError(f"failed to load SymSpell dictionary at {dict_path}")
    logger.info("SymSpell loaded (%d words, max_edit_distance=%d)", sym.word_count, _MAX_EDIT_DISTANCE)
    return sym


def load_stopwords() -> set[str]:
    """English stopwords — content terms only get synonym expansion."""
    return set(nltk_stopwords.words("english"))


def warm_wordnet() -> None:
    """Force WordNet to load its data now, so the first online query doesn't pay
    the lazy-load cost on the request path."""
    wn.synsets("information")
