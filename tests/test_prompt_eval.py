from stair_vs_rag.prompt import build_stair_prompt
from stair_vs_rag.eval import score_prediction


def test_prompt_includes_book_toc_query_and_says_feafle():
    toc = ["1 Chapter 1: Fundamental Change in Education",
           "leaf 12.9 Step seven: design course structure and learning"]
    p = build_stair_prompt("Teaching in a Digital Age", toc,
                           "What is the suggested time allocation for students?")
    assert p.startswith("### Book: Teaching in a Digital Age")
    assert "Pick the best section" in p
    assert "leaf 12.9 Step seven: design course structure and learning" in p
    assert "### Query: What is the suggested time allocation" in p
    assert p.rstrip().endswith("### Section:")


def test_score_prediction_r1_and_hallucination():
    hits = score_prediction(True, "12.9 Step seven", "12.9 Step seven: design course structure and learning")
    assert hits.r1 == 1 and hits.hallucinated is False
    miss = score_prediction(True, "1 Chapter 1", "12.9 Step seven: design course structure and learning")
    assert miss.r1 == 0
    bogus = score_prediction(True, "1 Chapter 1", "Invented section title")
    assert bogus.hallucinated is True and bogus.r1 == 0
