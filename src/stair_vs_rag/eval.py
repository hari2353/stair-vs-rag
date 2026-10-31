"""Retrieval scoring: Recall@1 style hit and hallucination flag."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Score:
    r1: int
    r3: int = 0
    ndcg: float = 0.0
    hallucinated: bool = False


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^\.*", "", s)
    s = re.sub(r"^leaf\b\s*", "", s)
    return re.sub(r"[^a-z0-9.]", "", s)


def score_prediction(gold_is_leaf: bool, pred: str, gold: str) -> Score:
    """A prediction hits when its normalized text is a prefix of the gold
    leaf (lenient to truncated generations). Any non-matching prediction is
    treated as a hallucination for aggregate metrics; corpus-level validity
    checks live in toc.exact_leaf_match."""
    if not gold_is_leaf:
        return Score(r1=0, hallucinated=False)
    hit = bool(_norm(pred)) and _norm(gold).startswith(_norm(pred))
    return Score(r1=1 if hit else 0, hallucinated=not hit)
