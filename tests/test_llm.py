import pytest

import requests

from fixture_toc import FIXTURE_TOC
from stair_vs_rag.prompt import build_stair_prompt
from stair_vs_rag.llm import OllamaClient, STAIRAgent, Message
from stair_vs_rag.toc import parse_toc, exact_leaf_match


class FakeTransport:
    """Transport double for unit tests: canned reply, records calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def post(self, path, payload, timeout):
        self.calls.append((path, payload))
        return {"message": {"content": self.reply}}


def test_llm_client_formats_ollama_chat():
    t = FakeTransport("12.9 Step seven: design course structure and learning")
    client = OllamaClient(transport=t, model="gemma2")
    out = client.chat([Message("user", "hi")], timeout=30)
    assert out == "12.9 Step seven: design course structure and learning"
    path, payload = t.calls[0]
    assert path == "/api/chat" and payload["model"] == "gemma2"
    assert payload["stream"] is False


def test_stair_agent_matches_leaf():
    t = FakeTransport("leaf 12.9 Step seven: design course structure and learning")
    agent = STAIRAgent(OllamaClient(transport=t, model="gemma2"))
    leaf = agent.pick_leaf("Teaching in a Digital Age",
                           [l.render for l in parse_toc(FIXTURE_TOC) if l.is_leaf],
                           "What is the suggested time allocation?")
    assert leaf == "12.9 Step seven: design course structure and learning"


def test_stair_agent_hallucination_is_logged_not_matched():
    t = FakeTransport("Section 9.3.5 Illiberal and hybrid regimes")
    agent = STAIRAgent(OllamaClient(transport=t, model="gemma2"))
    leaf = agent.pick_leaf("Teaching in a Digital Age",
                           [l.render for l in parse_toc(FIXTURE_TOC) if l.is_leaf],
                           "What is the suggested time allocation?")
    assert leaf is None
