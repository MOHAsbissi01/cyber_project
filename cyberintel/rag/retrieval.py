"""Retriever (course Chap. 4): chunked knowledge base with lexical (TF-IDF), dense (embeddings) and hybrid (rank fusion) search.

Why hybrid: dense embeddings capture meaning but blur rare identifiers (log4j, CVE ids, tool names); TF-IDF over the FULL vocabulary
(min_df=1) is exact on those. Reciprocal Rank Fusion combines both without having to calibrate their score scales.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import cfg
from ..embeddings import Embedder

_TOKEN = r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\._]*[a-zA-Z0-9]\b"


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    out: dict[int, float] = {}
    for r in rankings:
        for pos, d in enumerate(r):
            out[d] = out.get(d, 0.0) + 1.0 / (k + pos + 1)
    return out


class ChunkIndex:
    def __init__(self, chunks: pd.DataFrame, questions: dict[int, str], dense: str = "st"):
        self.chunks = chunks.reset_index(drop=True)
        self.dense_kind = dense
        self.texts = [f"{questions[i]} | {t}" for i, t in zip(self.chunks.incident_id, self.chunks.text)]
        self.vec = TfidfVectorizer(token_pattern=_TOKEN, ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english",
                                   lowercase=True)
        self.X = self.vec.fit_transform(self.texts)
        self.emb = None
        self.E = None
        if dense != "none":
            self.emb = Embedder(dense).fit(self.texts if dense == "tfidf" else [])
            self.E = self.emb.encode(self.texts, batch_size=64)

    # ---- searches return [(chunk_row, score)] best first
    def lexical(self, q: str, k: int = 30) -> list[tuple[int, float]]:
        s = (self.X @ self.vec.transform([q]).T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]

    def dense(self, q: str, k: int = 30) -> list[tuple[int, float]]:
        if self.E is None:
            return []
        s = self.E @ self.emb.encode([q])[0]
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    def hybrid(self, q: str, k: int = 30) -> list[tuple[int, float, float, float]]:
        """[(chunk_row, rrf_score, lexical_score, dense_score)] best first."""
        lx, ds = self.lexical(q, 60), self.dense(q, 60)
        lmap, dmap = dict(lx), dict(ds)
        fused = rrf([[i for i, _ in lx], [i for i, _ in ds]], cfg("rag", "rrf_k"))
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(i, fused[i], lmap.get(i, 0.0), dmap.get(i, 0.0)) for i in order]

    def incident_of(self, row: int) -> int:
        return int(self.chunks.incident_id.iloc[row])

    def chunk_text(self, row: int) -> str:
        return str(self.chunks.text.iloc[row])

    # ---- persistence (the dense model object is re-created lazily to keep the pickle small)
    def save(self, path: Path):
        emb = self.emb
        self.emb = None
        joblib.dump({"obj": self, "dense_kind": self.dense_kind, "st_name": getattr(emb, "name", None)}, path)
        self.emb = emb

    @staticmethod
    def load(path: Path) -> "ChunkIndex":
        d = joblib.load(path)
        idx: ChunkIndex = d["obj"]
        if idx.dense_kind == "st":
            idx.emb = Embedder("st", name=d["st_name"]).fit([])
        elif idx.dense_kind == "tfidf":
            idx.emb = None
            idx.E = None
            idx.dense_kind = "none"
        return idx
