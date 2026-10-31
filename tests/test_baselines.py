from stair_vs_rag.baselines import BM25Retriever

LEAF_TEXTS = {
    "12.9 Step seven: design course structure and learning":
        "Budget one hour per week of design time per credit hour. Set learning "
        "goals first, then structure the course around those goals.",
    "1.1.1 Origins and controversies":
        "Behaviourism originated with Watson and Skinner, who argued that "
        "learning is a change in observable behaviour.",
    "13.7 Building the future":
        "Technology will keep changing faster than teachers can absorb it.",
}


def test_bm25_ranks_goal_query_first():
    ret = BM25Retriever(LEAF_TEXTS)
    top3 = [leaf for leaf, _ in ret.search("how should I allocate time to set learning goals for a course?")]
    assert "12.9 Step seven: design course structure and learning" == top3[0]


def test_bm25_vectorizes_and_returns_ranked_list():
    ret = BM25Retriever(LEAF_TEXTS)
    scores = ret.search("technology changing fast")
    assert scores[0][0] == "13.7 Building the future"
