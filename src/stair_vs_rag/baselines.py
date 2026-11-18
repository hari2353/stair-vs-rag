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


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Word chunks with overlap — the fixed-chunk RAG default.

    The trailing chunk is dropped when it adds no word beyond the previous
    chunk (pure overlap), so short documents don't produce a duplicate tail.
    """
    if size < 1:
        msg = f"size must be >= 1, got {size}"
        raise ValueError(msg)
    if overlap < 0 or overlap >= size:
        msg = f"overlap must be in [0, size), got {overlap} (size={size})"
        raise ValueError(msg)
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[str] = []
    prev_end = -1
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        if end <= prev_end:  # pure-overlap tail: adds no word
            continue
        chunks.append(" ".join(words[start:end]))
        prev_end = end
    return chunks


class ChunkRetriever:
    """BM25 over fixed-size chunks; search returns leaves (best chunk wins).

    This is the naive RAG baseline STAIR argues against: chunks ignore the
    section tree, but each leaf still gets one vote so top-k is comparable.
    """

    def __init__(self, leaf_texts: dict[str, str],
                 size: int = 120, overlap: int = 20) -> None:
        self.ids = sorted(leaf_texts)
        self.chunks: list[tuple[str, str]] = [
            (leaf, chunk)
            for leaf in self.ids
            for chunk in chunk_text(leaf_texts[leaf], size, overlap)
        ]
        self.bm25 = (
            BM25Okapi([_tok(chunk) for _, chunk in self.chunks])
            if self.chunks else None
        )

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tok(query))
        best: dict[str, float] = {}
        for (leaf, _chunk), score in zip(self.chunks, scores, strict=True):
            if leaf not in best or score > best[leaf]:
                best[leaf] = float(score)
        ranked = sorted(best.items(), key=lambda x: -x[1])
        return ranked[:k] if k else ranked
