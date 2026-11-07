"""Text embeddings (course Chap. 1 vectorisation): TF-IDF + SVD (LSA, offline) or sentence-transformers (contextual)."""
from __future__ import annotations

import numpy as np

from .config import cfg


class Embedder:
    """kind='tfidf': TF-IDF (1-2 grams, sub-linear tf) -> TruncatedSVD (LSA) -> L2-normalised dense vectors.
    kind='st'   : sentence-transformers model (contextual embeddings, e.g. all-MiniLM-L6-v2), L2-normalised.
    Always fit on TRAIN texts only when used inside an evaluation (no leakage); the app fits on the whole corpus."""

    def __init__(self, kind: str | None = None, name: str | None = None, dim: int | None = None):
        self.kind = kind or cfg("embedding", "default")
        self.name = name or cfg("embedding", "st_model")
        self.dim = dim or cfg("embedding", "lsa_dim")

    def fit(self, texts) -> "Embedder":
        if self.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
        else:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english", max_features=60000)
            X = self.vec.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        if self.kind == "st":
            return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False),
                              dtype=np.float32)
        return self._norm(self.svd.transform(self.vec.transform(texts))).astype(np.float32)

    def fit_transform(self, texts) -> np.ndarray:
        texts = list(texts)
        return self.fit(texts).encode(texts)

    @staticmethod
    def _norm(M: np.ndarray) -> np.ndarray:
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def cosine_topk(E: np.ndarray, q: np.ndarray, k: int, exclude: int | None = None) -> list[tuple[int, float]]:
    """Top-k cosine neighbours of vector q among the L2-normalised rows of E (course Chap. 1: cosine similarity)."""
    s = E @ q
    if exclude is not None:
        s[exclude] = -1
    idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
    idx = idx[np.argsort(-s[idx])]
    return [(int(i), float(s[i])) for i in idx]
