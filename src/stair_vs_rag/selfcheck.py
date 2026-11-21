"""Self-checking retrieval (LinkedIn portfolio project 4).

Retrieve, validate the top leaf against the known ToC and a score floor, retry
once with a deterministic stopword-stripped query, otherwise abstain. No LLM
call is used for the check, so it runs offline and the tests stay deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this "
    "to was were what when where which who why with".split()
)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


@dataclass
class SelfCheckAnswer:
    leaf: str | None
    citations: tuple[str, ...]
    abstained: bool
    reason: str
    attempts: int
    rewritten_query: str | None


def _rewrite_query(query: str) -> str:
    """Drop stopwords and edge punctuation; stable, so a no-op means no retry."""
    words = [word.strip("?.!,;:") for word in query.split()]
    return " ".join(w for w in words if w and w.lower() not in _STOPWORDS)


class SelfCorrectingRetriever:
    def __init__(
        self,
        retriever: Retriever,
        known_leaves: frozenset[str] | set[str],
        min_score: float = 0.1,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            msg = f"max_attempts must be >= 1, got {max_attempts}"
            raise ValueError(msg)
        self.retriever = retriever
        self.known_leaves = frozenset(known_leaves)
        self.min_score = min_score
        self.max_attempts = max_attempts

    def answer(self, query: str, k: int = 5) -> SelfCheckAnswer:
        current = query
        rewritten: str | None = None
        attempts = 0
        reason = "no_results"

        while attempts < self.max_attempts:
            attempts += 1
            results = list(self.retriever.search(current, k))
            if not results:
                reason = "no_results"
            elif results[0][0] not in self.known_leaves:
                reason = "no_supported_leaf"
            elif results[0][1] < self.min_score:
                reason = "below_threshold"
            else:
                return SelfCheckAnswer(
                    leaf=results[0][0],
                    citations=tuple(
                        leaf for leaf, _ in results if leaf in self.known_leaves
                    ),
                    abstained=False,
                    reason="supported",
                    attempts=attempts,
                    rewritten_query=rewritten,
                )

            if attempts >= self.max_attempts:
                break
            next_query = _rewrite_query(current)
            if not next_query or next_query == current:
                break
            current = next_query
            rewritten = next_query

        return SelfCheckAnswer(
            leaf=None,
            citations=(),
            abstained=True,
            reason=reason,
            attempts=attempts,
            rewritten_query=rewritten,
        )
