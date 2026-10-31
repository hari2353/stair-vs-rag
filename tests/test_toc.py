from fixture_toc import FIXTURE_TOC

from stair_vs_rag.toc import ToC, leaf_entries, exact_leaf_match, render_toc, parse_toc


def test_parse_toc_marks_leaves():
    toc = parse_toc(FIXTURE_TOC)
    assert ("1", "Chapter 1: Fundamental Change in Education", False) == (toc[0].num, toc[0].title, toc[0].is_leaf)
    assert toc[4].is_leaf is False  # "12 Chapter 12:..." is a chapter heading
    leaf_titles = [e.title for e in toc if e.is_leaf]
    assert "Step seven: design course structure and learning" in leaf_titles
    assert len(leaf_entries(toc)) == 6


def test_parse_toc_counts():
    toc = parse_toc(FIXTURE_TOC)
    assert len(toc) == 10


def test_exact_leaf_match_normalizes():
    assert exact_leaf_match("12.9  step seven:   design course structure and LEARNING",
                            parse_toc(FIXTURE_TOC)) == "12.9 Step seven: design course structure and learning"


def test_exact_leaf_match_may_end_w_leaf_prefix():
    assert exact_leaf_match("leaf 12.9 Step seven: design course structure and learning",
                            parse_toc(FIXTURE_TOC)) == "12.9 Step seven: design course structure and learning"


def test_hallucination_returns_none():
    assert exact_leaf_match("1 Chapter 1: Fundamental Change in Education",
                            parse_toc(FIXTURE_TOC)) is None
    assert exact_leaf_match("Section 4.2.3 Plurality voting", parse_toc(FIXTURE_TOC)) is None


def test_dotted_line_heights():
    # model may prefix the number with a dotted leader; ensure still matched
    assert exact_leaf_match("... 12.7 Step five: master the technology",
                            parse_toc(FIXTURE_TOC)) == "12.7 Step five: master the technology"


def test_render_toc_restores_text():
    assert render_toc(parse_toc(FIXTURE_TOC)) == FIXTURE_TOC


def test_toc_class_roundtrip():
    toc = ToC.from_text(FIXTURE_TOC)
    assert len(toc) == 10 and render_toc(toc) == FIXTURE_TOC
