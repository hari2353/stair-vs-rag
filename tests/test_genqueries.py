import json

from stair_vs_rag.genqueries import GenQueries
from stair_vs_rag.llm import OllamaClient, Message


class FakeTransport:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def post(self, path, payload, timeout):
        self.calls.append(payload)
        return {"message": {"content": self.reply}}


def test_genqueries_builds_prompt_and_parses_numbered_list():
    t = FakeTransport("1. How should instructors budget design time?\n"
                      "2. Why set learning goals before structuring a course?\n"
                      "3. What is one hour per week used for?\n")
    gen = GenQueries(OllamaClient(transport=t, model="gemma2"))
    out = gen.for_leaf("Teaching in a Digital Age",
                       "12.9 Step seven: design course structure and learning",
                       "Budget one hour per week. Set learning goals first.")
    assert out == [
        "How should instructors budget design time?",
        "Why set learning goals before structuring a course?",
        "What is one hour per week used for?",
    ]
    prompt = t.calls[0]["messages"][0]["content"]
    assert "design course structure" in prompt and "Budget one hour" in prompt


def test_genqueries_skips_garbage_lines():
    t = FakeTransport("How should instructors budget design time?\nSure! Here are questions.\n")
    gen = GenQueries(OllamaClient(transport=t, model="gemma2"))
    assert gen.for_leaf("B", "leaf", "text") == ["How should instructors budget design time?"]


def test_records_json_roundtrip(tmp_path):
    records = {"queries": [{"query": "q", "gold_leaf": "leaf", "book": "B"}]}
    p = tmp_path / "queries.json"
    p.write_text(json.dumps(records))
    assert json.loads(p.read_text()) == records
