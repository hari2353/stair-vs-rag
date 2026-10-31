"""End-to-end benchmark: BM25 vs zero-shot STAIR (Gemma 2, ToC prompt).
Paper tie-in: R@1 on the same queries; hallucination = output that matches
no valid leaf. Run with a live Ollama:

    python -m stair_vs_rag.benchmark benchmarks/raw/<book>.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .baselines import BM25Retriever
from .eval import score_prediction
from .genqueries import GenQueries
from .ingest import extract_book, leaves_texts, to_paper_lines
from .llm import OllamaClient, STAIRAgent


def aggregate(rows: list[dict]) -> dict:
    out = {}
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    for system, rs in by_sys.items():
        out[system] = {
            "queries": len(rs),
            "recall_1": sum(x["r1"] for x in rs) / len(rs),
            "hallucination_rate": sum(1 for x in rs if x["hallucinated"]) / len(rs),
        }
    return out


def run(pdf_path: str, out_dir: str = "benchmarks", n_per_leaf: int = 3) -> dict:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    entries, _doc = extract_book(pdf_path)
    leaves = [e.render for e in entries if e.is_leaf]
    texts = leaves_texts(pdf_path)
    toc_lines = to_paper_lines(entries)

    client = OllamaClient(model="gemma2")
    agent = STAIRAgent(client)
    gen = GenQueries(client, n=n_per_leaf)

    queries_path = out / "queries.json"
    queries = json.loads(queries_path.read_text()) if queries_path.exists() else []
    queries = [q for q in queries if q["book"] == Path(pdf_path).stem]
    if not queries:
        stem = Path(pdf_path).stem
        for leaf in leaves:
            text = texts.get(leaf)
            if not text:
                continue
            for q in gen.for_leaf(stem, leaf, text):
                queries.append({"query": q, "gold_leaf": leaf, "book": stem})
        queries_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False))

    retriever = BM25Retriever(dict(texts))
    rows = []
    for q in queries:
        pred, hall = None, False
        leaf = agent.pick_leaf(Path(pdf_path).stem, toc_lines, q["query"])
        pred = leaf
        s = score_prediction(True, pred or "", q["gold_leaf"])
        hall = pred is None
        rows.append({"system": "stair", "r1": s.r1, "hallucinated": hall,
                     "query": q["query"]})
        bm_top = [leaf_id for leaf_id, _ in retriever.search(q["query"], k=1)]
        sb = score_prediction(True, bm_top[0], q["gold_leaf"]) if bm_top else None
        rows.append({"system": "bm25", "r1": sb.r1 if sb else 0,
                     "hallucinated": False, "query": q["query"]})

    agg = aggregate(rows)
    (out / "results.json").write_text(json.dumps({"aggregate": agg, "rows": rows},
                                                 indent=2, ensure_ascii=False))
    return agg
