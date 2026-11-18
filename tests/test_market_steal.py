"""Market-steal additions (2026-11-18): BEIR-style ranking metrics, honest
hallucination semantics, fixed-chunk BM25 baseline.

Compared products treat these as baseline: retrieval reports R@k/MRR/nDCG,
hallucination means an invalid citation (not a retrieval miss), and naive RAG
chunks documents fixed-size with overlap instead of using the section tree.
"""

from __future__ import annotations

import math

import pytest

from stair_vs_rag.baselines import BM25Retriever, ChunkRetriever, chunk_text
from stair_vs_rag.benchmark import aggregate
from stair_vs_rag.eval import mrr_at_k, ndcg_at_k, recall_at_k, score_prediction

LEAVES = {
    "1.1 Origins": "behaviourism watson skinner observable behaviour",
    "2.3 Design": "budget one hour per week design time per credit hour",
    "3.7 Future": "technology changes faster than teachers can absorb it",
}


# ── BEIR-style ranking metrics ──────────────────────────────────────────────
class TestRankingMetrics:
    RANKED = ["wrong-a", "gold", "wrong-b", "wrong-c"]

    def test_recall_cutoff_is_inclusive(self) -> None:
        assert recall_at_k(self.RANKED, "gold", 2) == 1.0
        assert recall_at_k(self.RANKED, "gold", 1) == 0.0
        assert recall_at_k(self.RANKED, "absent", 10) == 0.0

    def test_mrr_uses_first_hit(self) -> None:
        assert mrr_at_k(self.RANKED, "gold", 10) == 0.5
        assert mrr_at_k(self.RANKED, "gold", 1) == 0.0
        assert mrr_at_k(["gold"], "gold", 10) == 1.0
        assert mrr_at_k([], "gold", 10) == 0.0

    def test_ndcg_discounts_by_rank(self) -> None:
        assert ndcg_at_k(["gold"], "gold", 10) == 1.0
        assert ndcg_at_k(self.RANKED, "gold", 10) == pytest.approx(1 / math.log2(3))
        assert ndcg_at_k(self.RANKED, "gold", 1) == 0.0
        assert 0.0 <= ndcg_at_k(self.RANKED, "absent", 10) <= 1.0

    def test_bad_cutoff_rejected(self) -> None:
        with pytest.raises(ValueError, match="k"):
            recall_at_k(self.RANKED, "gold", 0)


# ── honest hallucination semantics ──────────────────────────────────────────
class TestValidity:
    GOLD = "2.3 Design: budget one hour per week"
    KNOWN = frozenset({"1.1 Origins: behaviourism", "2.3 Design: budget one hour per week"})

    def test_wrong_but_real_leaf_is_a_miss_not_a_hallucination(self) -> None:
        s = score_prediction(True, "1.1 Origins", self.GOLD, known_leaves=self.KNOWN)
        assert s.r1 == 0 and s.valid is True and s.hallucinated is False

    def test_invented_leaf_is_invalid_and_hallucinated(self) -> None:
        s = score_prediction(True, "9.9 Invented section", self.GOLD, known_leaves=self.KNOWN)
        assert s.r1 == 0 and s.valid is False and s.hallucinated is True

    def test_truncated_valid_leaf_prefix_still_valid(self) -> None:
        s = score_prediction(True, "2.3 Des", self.GOLD, known_leaves=self.KNOWN)
        assert s.valid is True and s.r1 == 1

    def test_empty_prediction_is_invalid(self) -> None:
        s = score_prediction(True, "", self.GOLD, known_leaves=self.KNOWN)
        assert s.valid is False and s.hallucinated is True

    def test_legacy_call_without_known_leaves_unchanged(self) -> None:
        s = score_prediction(True, "1.1 Origins", self.GOLD)
        assert s.hallucinated is True  # legacy: any miss counted as hallucination

    def test_non_leaf_gold_is_neither(self) -> None:
        s = score_prediction(False, "anything", self.GOLD, known_leaves=self.KNOWN)
        assert s.r1 == 0 and s.valid is True and s.hallucinated is False


# ── fixed-chunk BM25 baseline ───────────────────────────────────────────────
class TestChunkBaseline:
    def test_chunk_text_respects_size_and_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(50))
        chunks = chunk_text(text, size=20, overlap=5)
        assert [len(c.split()) for c in chunks] == [20, 20, 20]
        assert chunks[1].split()[0] == "w15"  # 20 - 5 overlap

    def test_invalid_chunk_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="size"):
            chunk_text("a b c", size=0)
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("a b c", size=10, overlap=10)

    def test_late_chunk_still_retrieves_leaf(self) -> None:
        filler = " ".join(f"filler{i}" for i in range(200))
        leaf = "2.3 Design: " + filler + " unique keyword xylophone"
        ret = ChunkRetriever({"1.1 Origins": "behaviourism skinner", "2.3 Design": leaf})
        top = ret.search("where is the xylophone mentioned?")
        assert top[0][0] == "2.3 Design"

    def test_one_vote_per_leaf(self) -> None:
        texts = {"2.3 Design": "design " * 40}
        texts.update({f"x{i}": "behaviourism" for i in range(20)})
        ret = ChunkRetriever(texts)
        ranked = ret.search("design")
        ids = [leaf for leaf, _ in ranked]
        assert len(ids) == len(set(ids)) and ids[0] == "2.3 Design"
        assert len(ret.chunks) > 3  # several chunks, still one vote
        # the best chunk's score wins, never a sum over chunks
        assert ranked[0][1] == pytest.approx(ret.search("design", k=1)[0][1])

    def test_empty_text_produces_no_chunks(self) -> None:
        ret = ChunkRetriever({"empty": "   ", "1.1 Origins": "behaviourism"})
        assert [leaf for leaf, _ in ret.search("behaviourism")] == ["1.1 Origins"]


# ── aggregate carries the new metrics ───────────────────────────────────────
class TestAggregateMetrics:
    def test_new_metrics_default_from_r1_for_old_rows(self) -> None:
        out = aggregate([
            {"system": "bm25", "r1": 1, "hallucinated": False},
            {"system": "bm25", "r1": 0, "hallucinated": False},
        ])
        assert out["bm25"] == {
            "queries": 2, "recall_1": 0.5, "recall_5": 0.5, "recall_10": 0.5,
            "mrr_10": 0.5, "ndcg_10": 0.5, "hallucination_rate": 0.0,
        }

    def test_explicit_rank_metrics_are_averaged(self) -> None:
        out = aggregate([
            {"system": "chunks", "r1": 0, "r5": 1, "r10": 1,
             "mrr": 0.5, "ndcg": 1 / math.log2(3), "hallucinated": False},
            {"system": "chunks", "r1": 0, "r5": 0, "r10": 0,
             "mrr": 0.0, "ndcg": 0.0, "hallucinated": False},
        ])
        assert out["chunks"]["recall_1"] == 0.0
        assert out["chunks"]["recall_5"] == 0.5
        assert out["chunks"]["mrr_10"] == 0.25
        assert out["chunks"]["ndcg_10"] == pytest.approx((1 / math.log2(3)) / 2)
