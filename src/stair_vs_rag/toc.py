"""ToC model, parser, and strict leaf matching.

Mirrors the STAIR paper (arXiv 2609.03874) input format: ToC lines like
``leaf 12.9 Step seven: design course structure and learning``.  An entry is
a leaf when the paper marks it with the ``leaf`` prefix; retrieval must
return one of those leaf entries and nothing else (hallucination = any other
output).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_LEAF_RE = re.compile(r"^leaf\s+(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>.*\S)\s*$")
_PLAIN_RE = re.compile(r"^(?P<num>\d+)\.?\s+(?P<title>.*\S)\s*$")


@dataclass
class Entry:
    num: str
    title: str
    is_leaf: bool
    render: str


def parse_toc(text: str) -> list[Entry]:
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LEAF_RE.match(line)
        if m:
            entries.append(Entry(m["num"], m["title"], True, line))
            continue
        m = _PLAIN_RE.match(line)
        if m:
            entries.append(Entry(m["num"], m["title"], False, line))
            continue
        entries.append(Entry("", line, False, line))
    return entries


def leaf_entries(entries: list[Entry]) -> list[Entry]:
    return [e for e in entries if e.is_leaf]


def render_toc(entries: list[Entry]) -> str:
    return "\n".join(e.render for e in entries) + ("\n" if entries else "")


def _norm(s: str) -> str:
    """Lowercase, drop the optional 'leaf' prefix and leader dots, then keep
    only [a-z0-9.]. Makes 'Step five:master' == 'Step five: master' etc."""
    s = s.strip().lower()
    s = re.sub(r"^\.*", "", s)
    s = re.sub(r"^leaf\b\s*", "", s)
    return re.sub(r"[^a-z0-9.]", "", s)


def exact_leaf_match(output: str, entries: list[Entry]) -> str | None:
    """Return the matched leaf's '<num> <title>' or None (hallucinated /
    non-leaf output). Whitespace-insensitive for sloppily generated text."""
    out = _norm(output)
    if not out:
        return None
    for e in entries:
        if not e.is_leaf:
            continue
        norm = _norm(f"{e.num} {e.title}")
        if out == norm or norm.startswith(out):
            return f"{e.num} {e.title}"
    return None


class ToC:
    """Convenience wrapper: parse once, expose the members tests and agents use."""

    def __init__(self, entries: list[Entry]):
        self.entries = entries

    @classmethod
    def from_text(cls, text: str) -> "ToC":
        return cls(parse_toc(text))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

    def __iter__(self):
        return iter(self.entries)
