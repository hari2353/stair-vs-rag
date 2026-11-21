"""Self-correcting retrieval: accept only known, scored leaves; else abstain."""

from __future__ import annotations

from stair_vs_rag.selfcheck import SelfCorrectingRetriever

KNOWN = frozenset({"2.3 Design: budget one hour per week", "1.1 Origins: behaviourism"})
UNKNOWN = "9.9 Invented section"


class FakeRetriever:
    """Deterministic stand-in: canned results per exact query, call log kept."""

    def __init__(self, responses: dict[str, list[tuple[str, float]]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.calls.append((query, k))
        return list(self.responses.get(query, []))[:k]


def test_direct_valid_hit_is_accepted() -> None:
    ret = FakeRetriever({"course design": [("2.3 Design: budget one hour per week", 0.8)]})
    answer = SelfCorrectingRetriever(ret, KNOWN).answer("course design")
    assert answer.leaf == "2.3 Design: budget one hour per week"
    assert answer.abstained is False
    assert answer.attempts == 1
    assert answer.rewritten_query is None


def test_unknown_top_leaf_then_valid_after_rewrite() -> None:
    ret = FakeRetriever({
        "What is the course design?": [(UNKNOWN, 0.9)],
        "course design": [("2.3 Design: budget one hour per week", 0.7)],
    })
    answer = SelfCorrectingRetriever(ret, KNOWN).answer("What is the course design?")
    assert answer.leaf == "2.3 Design: budget one hour per week"
    assert answer.abstained is False
    assert answer.attempts == 2
    assert answer.rewritten_query == "course design"
    assert [q for q, _ in ret.calls] == ["What is the course design?", "course design"]


def test_empty_results_abstain() -> None:
    ret = FakeRetriever({"photosynthesis": []})
    answer = SelfCorrectingRetriever(ret, KNOWN).answer("photosynthesis")
    assert answer.abstained is True
    assert answer.leaf is None
    assert answer.citations == ()
    assert answer.reason == "no_results"
    assert answer.attempts == 1


def test_below_threshold_abstains() -> None:
    ret = FakeRetriever({"course design": [("2.3 Design: budget one hour per week", 0.05)]})
    answer = SelfCorrectingRetriever(ret, KNOWN, min_score=0.1).answer("course design")
    assert answer.abstained is True
    assert answer.reason == "below_threshold"
    assert answer.citations == ()


def test_noop_rewrite_does_not_retry() -> None:
    ret = FakeRetriever({"photosynthesis": [(UNKNOWN, 0.9)]})
    answer = SelfCorrectingRetriever(ret, KNOWN).answer("photosynthesis")
    assert answer.abstained is True
    assert answer.reason == "no_supported_leaf"
    assert answer.attempts == 1
    assert answer.rewritten_query is None
    assert len(ret.calls) == 1


def test_citations_never_include_unknown_leaves() -> None:
    ret = FakeRetriever({
        "course design": [
            ("2.3 Design: budget one hour per week", 0.8),
            (UNKNOWN, 0.7),
            ("1.1 Origins: behaviourism", 0.6),
        ],
    })
    answer = SelfCorrectingRetriever(ret, KNOWN).answer("course design")
    assert answer.citations == (
        "2.3 Design: budget one hour per week",
        "1.1 Origins: behaviourism",
    )
    assert UNKNOWN not in answer.citations


def test_attempts_are_bounded() -> None:
    ret = FakeRetriever({
        "What is the topic?": [(UNKNOWN, 0.9)],
        "topic": [(UNKNOWN, 0.9)],
    })
    answer = SelfCorrectingRetriever(ret, KNOWN, max_attempts=2).answer("What is the topic?")
    assert answer.abstained is True
    assert answer.attempts == 2
    assert len(ret.calls) == 2
    assert answer.rewritten_query == "topic"
