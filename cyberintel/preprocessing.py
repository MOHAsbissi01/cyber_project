"""Classical text preprocessing (AA3, course Chap. 1): normalisation, tokenisation, stop words, stemming, lemmatisation, POS.

`make_analyzer(mode)` returns a callable usable as `TfidfVectorizer(analyzer=...)`, so the effect of each preprocessing
step on classification is MEASURED (see classify.preprocessing_ablation) instead of assumed.
"""
from __future__ import annotations

import re
from functools import lru_cache

import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import ROOT
from .textutil import tokenize

nltk.data.path.insert(0, str(ROOT / "nltk_data"))   # offline copy of wordnet / tagger (downloaded once, see README)

_URL = re.compile(r"https?://\S+|www\.\S+")
_CVE = re.compile(r"cve-\d{4}-\d{4,7}", re.I)


def normalize(text: str) -> str:
    """Lower-case, replace URLs by a token, keep CVE ids as one token (they are information, not noise)."""
    text = _URL.sub(" url ", text)
    return _CVE.sub(lambda m: m.group(0).lower().replace("-", "_"), text.lower())


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    lem.lemmatize("tests")   # fail early (clear error) if the wordnet data is missing
    return lem


@lru_cache(maxsize=200_000)
def _stem(w: str) -> str:
    return _stemmer().stem(w)


@lru_cache(maxsize=200_000)
def _lemma(w: str) -> str:
    lem = _lemmatizer()
    return lem.lemmatize(lem.lemmatize(w, "v"), "n")


def make_analyzer(mode: str = "stop"):
    """mode: 'raw' (tokens only) | 'stop' (+ stop-word removal) | 'stem' (+ Porter stemming) | 'lemma' (+ WordNet lemmatisation)."""
    if mode not in ("raw", "stop", "stem", "lemma"):
        raise ValueError(mode)

    def analyzer(doc: str) -> list[str]:
        toks = tokenize(normalize(doc))
        if mode != "raw":
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 1]
        if mode == "stem":
            toks = [_stem(t) for t in toks]
        elif mode == "lemma":
            toks = [_lemma(t) for t in toks]
        return toks

    return analyzer


def pos_profile(text: str, top: int = 8) -> list[tuple[str, int]]:
    """Part-of-speech tag counts of a text (shows the linguistic-analysis step; used in the notebook/report)."""
    toks = tokenize(text)[:300]
    tags = [t for _, t in nltk.pos_tag(toks)]
    from collections import Counter
    return Counter(tags).most_common(top)


def preprocessing_demo(text: str) -> dict:
    """Step-by-step view of the pipeline on one sentence (for the report / UI)."""
    toks = tokenize(normalize(text))
    nostop = [t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return {"tokens": toks, "no_stopwords": nostop, "stems": [_stem(t) for t in nostop], "lemmas": [_lemma(t) for t in nostop],
            "pos": nltk.pos_tag(toks)}
