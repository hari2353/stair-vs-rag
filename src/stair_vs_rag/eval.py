"""Retrieval scoring: Recall@k / MRR / nDCG and output-validity flags."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class Score:
    r1: int
    r3: int = 0
    ndcg: float = 0.0
    hallucinated: bool = False
    valid: bool = True   # prediction resolves to some known leaf


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^\.*", "", s)
    s = re.sub(r"^leaf\b\s*", "", s)
    return re.sub(r"[^a-z0-9.]", "", s)


def _matches(pred: str, leaf: str) -> bool:
    """Lenient prefix match in either direction of truncation."""
    return bool(_norm(pred)) and _norm(leaf).startswith(_norm(pred))


def score_prediction(gold_is_leaf: bool, pred: str, gold: str,
                     known_leaves: frozenset[str] | None = None) -> Score:
    """A prediction hits when its normalized text is a prefix of the gold
    leaf (lenient to truncated generations).

    With `known_leaves`, `hallucinated` means the output matches no valid
    leaf — the paper's definition (an invented citation). A wrong-but-real
    leaf is a retrieval miss (`valid=True`, `r1=0`). Without `known_leaves`
    the legacy behavior is kept: any miss is flagged as hallucinated.
    """
    if not gold_is_leaf:
        return Score(r1=0, hallucinated=False)
    hit = _matches(pred, gold)
    if known_leaves is None:
        return Score(r1=1 if hit else 0, hallucinated=not hit)
    valid = any(_matches(pred, leaf) for leaf in known_leaves)
    return Score(r1=1 if hit else 0, hallucinated=not valid, valid=valid)


def _check_k(k: int) -> None:
    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)


def recall_at_k(ranked: list[str], gold: str, k: int) -> float:
    """1.0 when the gold leaf is inside the top k (single-gold recall)."""
    _check_k(k)
    return 1.0 if gold in ranked[:k] else 0.0


def mrr_at_k(ranked: list[str], gold: str, k: int) -> float:
    """Reciprocal rank of the first (only) relevant leaf, 0 when absent."""
    _check_k(k)
    for rank, leaf in enumerate(ranked[:k], start=1):
        if leaf == gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: str, k: int) -> float:
    """nDCG@k with one relevant document: 1/log2(rank+1)."""
    _check_k(k)
    for rank, leaf in enumerate(ranked[:k], start=1):
        if leaf == gold:
            return 1.0 / math.log2(rank + 1)
    return 0.0
