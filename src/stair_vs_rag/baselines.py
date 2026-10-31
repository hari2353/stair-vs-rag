"""Baseline retrievers: BM25 over leaf-section text (paper used Elasticsearch;
rank_bm25 keeps this local and dependency-light — documented deviation)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25Retriever:
    def __init__(self, leaf_texts: dict[str, str]):
        ids = sorted(leaf_texts)
        self.ids = ids
        self.bm25 = BM25Okapi([_tok(leaf_texts[i]) for i in ids])

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tok(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: -x[1])
        out = [(leaf, float(s)) for leaf, s in ranked]
        return out[:k] if k else out
