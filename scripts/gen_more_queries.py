"""Scale the synthetic query corpus with the paper's §3.2 method (local Ollama).

Generates MORE study questions per leaf section and merges them with the
existing queries into a training file. The original queries.json stays
untouched as the held-out TEST set — the paper-comparable benchmark numbers
(R@1: zero-shot 12%, BM25 68%) were computed on those queries.

    python scripts/gen_more_queries.py            # writes benchmarks/queries-train.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stair_vs_rag.genqueries import GenQueries
from stair_vs_rag.ingest import leaves_texts
from stair_vs_rag.llm import OllamaClient

PDF = "benchmarks/raw/teaching-digital-age.pdf"
TEST = REPO / "benchmarks" / "queries.json"
OUT = REPO / "benchmarks" / "queries-train.json"
BOOK = "teaching-digital-age"
BATCHES_PER_LEAF = 6   # accumulates across runs; OUT rows are kept below
BATCH_N = 10


def main() -> None:
    test_rows = json.loads(TEST.read_text())
    gold_leaves = sorted({r["gold_leaf"] for r in test_rows})
    texts = leaves_texts(PDF)

    # Accumulate: keep every previously generated train row on re-runs.
    rows: list[dict] = []
    seen = {r["query"] for r in test_rows}
    if OUT.exists():
        prior = [r for r in json.loads(OUT.read_text()) if r not in test_rows]
        rows.extend(prior)
        seen |= {r["query"] for r in prior}

    client = OllamaClient(model="gemma2")
    gen = GenQueries(client)
    for leaf in gold_leaves:
        text = texts.get(leaf)
        if not text:
            print(f"no text for {leaf}; skipping", flush=True)
            continue
        got = 0
        for _ in range(BATCHES_PER_LEAF):
            for q in gen.for_leaf(BOOK, leaf, text):
                if q in seen or len(q) < 20:
                    continue
                seen.add(q)
                rows.append({"query": q, "gold_leaf": leaf, "book": BOOK})
                got += 1
        print(f"{leaf} -> +{got} queries", flush=True)

    payload = rows  # train-only: the test rows live in queries.json, never here
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"train file: {len(payload)} train queries (test set untouched in {TEST.name}) -> {OUT}")


if __name__ == "__main__":
    main()
