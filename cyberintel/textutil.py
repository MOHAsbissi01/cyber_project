"""Small text utilities shared by NER, chunking, RAG and preprocessing (sentence splitting, tokenisation)."""
from __future__ import annotations

import re

# a sentence ends with . ! ? followed by a capital/digit/bullet, or a numbered-list marker ("3) ", "4. ") starts a new unit
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\*\-\(\"'`\[])|\s+(?=\d{1,2}[.)]\s+[A-Z])")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\._]*[a-z0-9]|[a-z0-9]", re.I)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (offline, no model download). Returns non-empty stripped sentences."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence, so entity offsets can be mapped to sentences."""
    spans, pos = [], 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return [(a, b) for a, b in spans if b > a]


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens keeping technical tokens such as 'log4j', 'sql-injection', 'cve-2021-44228'."""
    return [t.lower() for t in _TOKEN.findall(text)]
