"""Fine-tuned STAIR R@1 eval: adapter vs BM25 vs zero-shot, on the same queries.

 scorers: reuses the repo's own score_prediction (prefix rule) — the exact
 metric that produced the zero-shot and BM25 numbers, so the three systems
 are directly comparable.

    python scripts/eval_finetuned.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from stair_vs_rag.baselines import BM25Retriever
from stair_vs_rag.eval import score_prediction
from stair_vs_rag.ingest import leaves_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--adapter", default="outputs/stair-lora-gemma")
    ap.add_argument("--queries", default="benchmarks/queries.json")
    ap.add_argument("--pdf", default="benchmarks/raw/teaching-digital-age.pdf")
    ap.add_argument("--out", default="benchmarks/finetuned-results.json")
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()

    rows = json.loads(Path(a.queries).read_text())
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, a.adapter).eval()

    def prompt_for(q: str) -> str:
        return (
            "You are a search assistant. Given a user query, answer with the single "
            "table-of-contents leaf path (section number + title) that best answers it.\n\n"
            f"Query: {q}\n\nLeaf:"
        )

    lora = []
    for i, r in enumerate(rows):
        inputs = tok(prompt_for(r["query"]), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        pred = re.sub(r"^[Ll]eaf:\s*", "", pred).strip().rstrip(".")
        hit = score_prediction(True, pred, r["gold_leaf"])
        lora.append({"system": "finetuned-lora", "r1": hit.r1,
                     "hallucinated": hit.hallucinated, "pred": pred,
                     "gold": r["gold_leaf"], "query": r["query"]})
        print(f"[{i + 1}/{len(rows)}] {'HIT' if hit.r1 else 'miss'} "
              f"pred={pred[:45]!r} gold={r['gold_leaf']}", flush=True)

    # BM25 over the same queries, same ingestion as the repo benchmark.
    texts = leaves_texts(a.pdf)
    retriever = BM25Retriever(dict(texts))
    bm = []
    for r in rows:
        top = [leaf_id for leaf_id, _ in retriever.search(r["query"], k=1)]
        s = score_prediction(True, top[0], r["gold_leaf"]) if top else None
        bm.append({"system": "bm25", "r1": s.r1 if s else 0,
                   "hallucinated": False, "query": r["query"]})

    # Zero-shot rows from the existing benchmark output, keyed per query.
    prev = json.loads((REPO / "benchmarks" / "results.json").read_text())
    zero = [x for x in prev["rows"] if x["system"] == "stair"]

    def agg(rs: list[dict]) -> dict:
        n = len(rs) or 1
        return {"queries": len(rs),
                "recall_1": sum(x["r1"] for x in rs) / n,
                "hallucination_rate": sum(1 for x in rs if x["hallucinated"]) / n}

    payload = {"aggregate": {"finetuned-lora": agg(lora),
                             "bm25": agg(bm),
                             "zero-shot-stair": agg(zero)},
               "rows": lora}
    Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
