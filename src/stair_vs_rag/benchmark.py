"""End-to-end benchmark: BM25 vs zero-shot STAIR (Gemma 2, ToC prompt).
Paper tie-in: R@1 on the same queries; hallucination = output that matches
no valid leaf. Run with a live Ollama:

    python -m stair_vs_rag.benchmark benchmarks/raw/<book>.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .baselines import BM25Retriever, ChunkRetriever
from .eval import ndcg_at_k, mrr_at_k, recall_at_k, score_prediction
from .genqueries import GenQueries
from .ingest import extract_book, leaves_texts, to_paper_lines
from .llm import OllamaClient, STAIRAgent


def aggregate(rows: list[dict]) -> dict:
    out = {}
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    for system, rs in by_sys.items():
        n = len(rs)

        def mean(key: str) -> float:
            return sum(float(x.get(key, x.get("r1", 0.0))) for x in rs) / n

        out[system] = {
            "queries": n,
            "recall_1": sum(float(x.get("r1", 0.0)) for x in rs) / n,
            "recall_5": mean("r5"),
            "recall_10": mean("r10"),
            "mrr_10": mean("mrr"),
            "ndcg_10": mean("ndcg"),
            "hallucination_rate": sum(float(x.get("hallucinated", 0.0)) for x in rs) / n,
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
    chunked = ChunkRetriever(dict(texts))
    known = frozenset(texts)
    rows = []
    for q in queries:
        leaf = agent.pick_leaf(Path(pdf_path).stem, toc_lines, q["query"])
        s = score_prediction(True, leaf or "", q["gold_leaf"], known_leaves=known)
        rows.append({"system": "stair", "r1": s.r1, "r5": s.r1, "r10": s.r1,
                     "mrr": float(s.r1), "ndcg": float(s.r1),
                     "hallucinated": not s.valid, "valid": s.valid,
                     "query": q["query"]})
        for system, ret in (("bm25", retriever), ("bm25_chunks", chunked)):
            ranked = [leaf_id for leaf_id, _ in ret.search(q["query"], k=10)]
            rows.append({"system": system,
                         "r1": int(recall_at_k(ranked, q["gold_leaf"], 1)),
                         "r5": int(recall_at_k(ranked, q["gold_leaf"], 5)),
                         "r10": int(recall_at_k(ranked, q["gold_leaf"], 10)),
                         "mrr": mrr_at_k(ranked, q["gold_leaf"], 10),
                         "ndcg": ndcg_at_k(ranked, q["gold_leaf"], 10),
                         "hallucinated": False, "valid": True,
                         "query": q["query"]})

    agg = aggregate(rows)
    (out / "results.json").write_text(json.dumps({"aggregate": agg, "rows": rows},
                                                 indent=2, ensure_ascii=False))
    return agg
