from stair_vs_rag.benchmark import aggregate


def test_aggregate_counts_r1_and_hallucination_rate():
    results = [
        {"system": "bm25", "r1": 1, "hallucinated": False},
        {"system": "bm25", "r1": 0, "hallucinated": False},
        {"system": "stair", "r1": 1, "hallucinated": False},
        {"system": "stair", "r1": 0, "hallucinated": True},
    ]
    out = aggregate(results)
    assert out["bm25"] == {"queries": 2, "recall_1": 0.5, "hallucination_rate": 0.0}
    assert out["stair"] == {"queries": 2, "recall_1": 0.5, "hallucination_rate": 0.5}


def test_aggregate_empty_is_safe():
    assert aggregate([]) == {}
