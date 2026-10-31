"""Synthetic QA generation per leaf section (paper §3.2: an LLM generates the
questions; gold label = the section the text came from)."""
from __future__ import annotations

import re

from .llm import Message, OllamaClient

_QNUM = re.compile(r"^\s*(?:\d+[.)]\s*)?([A-Z].{8,}\?)\s*$")


class GenQueries:
    def __init__(self, client: OllamaClient, n: int = 3):
        self.client = client
        self.n = n

    def for_leaf(self, book: str, leaf: str, section_text: str) -> list[str]:
        q = (f"You are preparing a study guide for the book '{book}'.\n"
             f"Write exactly {self.n} distinct study questions that can be answered "
             f"ONLY by this book section:\n"
             f"Section: {leaf}\nSection text:\n{section_text[:2400]}\n"
             "One question per numbered line. Questions only, no answers, no extra text.")
        reply = self.client.chat([Message("user", q)], timeout=120.0)
        return [m.group(1).strip() for line in reply.splitlines()
                if len(line) < 160 for m in [_QNUM.match(line.strip())] if m]
