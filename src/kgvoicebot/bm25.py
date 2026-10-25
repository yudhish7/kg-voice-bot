"""A small, dependency-free Okapi BM25 index."""

from __future__ import annotations

import math
from collections import Counter

from .text import content_tokens


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        df: Counter[str] = Counter()
        for doc_id, text in docs.items():
            tokens = content_tokens(text)
            self._tf[doc_id] = Counter(tokens)
            self._len[doc_id] = len(tokens)
            df.update(set(tokens))
        n = len(docs)
        self._avg = (sum(self._len.values()) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        """BM25 score for every document with a non-zero match."""
        terms = set(content_tokens(query))
        out: dict[str, float] = {}
        for doc_id, tf in self._tf.items():
            norm = 1 - self.b + self.b * (self._len[doc_id] / self._avg if self._avg else 0)
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                out[doc_id] = score
        return out
