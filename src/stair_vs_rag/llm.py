"""Ollama chat client + the STAIR agent (prompt -> leaf, strictly matched)."""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .prompt import build_stair_prompt
from .toc import Entry, exact_leaf_match, parse_toc


@dataclass
class Message:
    role: str
    content: str


class _RequestsTransport:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def post(self, path: str, payload: dict, timeout: float) -> dict:
        r = requests.post(self.base_url + path, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()


class OllamaClient:
    def __init__(self, transport=None, model: str = "gemma2",
                 base_url: str = "http://localhost:11434"):
        self.transport = transport or _RequestsTransport(base_url)
        self.model = model

    def chat(self, messages: list[Message], timeout: float = 90.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        data = self.transport.post("/api/chat", payload, timeout)
        return data["message"]["content"]


class STAIRAgent:
    """Zero-shot STAIR: ToC-aware leaf selection with strict leaf matching.
    (The paper fine-tunes Mistral; this module keeps the same interface so a
    fine-tuned model can slot in later without changing callers.)"""

    def __init__(self, client: OllamaClient):
        self.client = client
        self.hallucinations: list[str] = []

    def pick_leaf(self, book: str, toc_lines: list[str], query: str,
                  timeout: float = 90.0) -> str | None:
        prompt = build_stair_prompt(book, toc_lines, query)
        reply = self.client.chat([Message("user", prompt)], timeout=timeout)
        entries = parse_toc("\n".join(l for l in toc_lines if l))
        leaf = exact_leaf_match(reply, entries)
        if leaf is None:
            self.hallucinations.append(reply)
        return leaf
