# stair-vs-rag

Local reproduction of **IBM STAIR (STructure Aware Information Retriever, arXiv:
[2609.03874](https://arxiv.org/abs/2609.03874))** vs. classical RAG baselines,
running fully on this machine (Ollama + Gemma 2, no cloud).

STAIR's idea: instead of cutting a long document into arbitrary chunks, hand
the retriever-LLM the document's **Table of Contents** and have it *generate*
the exact leaf section that answers a query. Retrieval unit = ToC leaf, not a
chunk. Paper numbers (SearchTome, Recall@1): Mistral 13.8 · BM25 59.5 ·
DPR 68.7 · DSI 76.9 · **STAIR 82.6**, hallucination ~0.05%.

## What is reproduced here

| Part | Status |
|------|--------|
| ToC parsing (pymupdf) into the paper's `leaf X.Y` format | ✅ |
| BM25 over leaf-section text (rank_bm25, local) | ✅ |
| Zero-shot STAIR prompt (Table-2 format) on Gemma 2 via Ollama | ✅ |
| Synthetic query generation per leaf (paper §3.2 method) | ✅ |
| Benchmark: R@1/R@5/R@10, MRR@10, nDCG@10 + hallucination rate, results JSON | ✅ |
| Fixed-chunk BM25 baseline (`bm25_chunks`) for structure-vs-chunking comparison | ✅ |
| LoRA fine-tuning of the retriever (paper's core win is *fine-tuned* STAIR) | ✅ QLoRA pipeline (`scripts/finetune_stair.py`) |
| Fine-tuned STAIR **R@1 benchmark** (held-out 50 queries) | ✅ **0.86** — beats BM25 0.68, zero-shot 0.12 (matches paper ordering) |
| Fine-tuned adapter, Gemma-2-2b-it, v2 (432 synthetic queries) | ✅ `outputs/stair-lora-gemma-v2` |
| Fine-tuned adapter, Qwen2.5-1.5B | ✅ `outputs/stair-lora-adapter` |

Not reproduced (yet): fine-tuned DSI, DPR/NV-Embed, Elasticsearch BM25
(rank_bm25 used instead — same scoring family). The core fine-tuned result
IS reproduced: R@1 0.86 on the held-out 50 queries — above BM25 (0.68) and
zero-shot STAIR (0.12), same ordering the paper reports. Method: 432
synthetic study questions generated per §3.2 with local Ollama gemma2
(`scripts/gen_more_queries.py`), Gemma-2-2b-it QLoRA 3 epochs, greedy decode,
repo prefix-rule scorer. The original 50 queries remain a pure held-out test
set — never trained on. A 50-example variant scored R@1 0.0: below ~50
queries per book the adapter only learns the leaf *format*.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
brew services start ollama && ollama pull gemma2   # once
.venv/bin/python -m pytest -q                      # 40 tests
.venv/bin/python -m stair_vs_rag.benchmark benchmarks/raw/teaching-digital-age.pdf

# fine-tuned STAIR (paper §3.2 synthetic queries -> QLoRA -> held-out R@1)
python scripts/gen_more_queries.py                  # scale the train corpus (needs Ollama)
python scripts/finetune_stair.py --model google/gemma-2-2b-it \
    --epochs 3 --queries benchmarks/queries-train.json --out outputs/stair-lora-gemma-v2
python scripts/eval_finetuned.py --adapter outputs/stair-lora-gemma-v2
```

Results land in `benchmarks/results.json` + `benchmarks/queries.json`
(committed so numbers are auditable; raw PDFs are not — see .gitignore).

## Evaluation additions (market-steal, 2026-11-18)

- **Ranking metrics, BEIR-style** — every system row carries `r1`/`r5`/`r10`,
  `mrr` and `ndcg`; the aggregate reports `recall_5`, `recall_10`, `mrr_10`
  and `ndcg_10` next to `recall_1`. Old rows without rank fields default to
  their `r1` (a lower bound), so historical result JSON stays readable.
- **Honest hallucination semantics** — `score_prediction(..., known_leaves=...)`
  flags `hallucinated` only when the output matches **no** real leaf (an
  invented citation, the paper's definition); a wrong-but-real leaf is a
  retrieval miss. Legacy calls without `known_leaves` keep the old behavior.
- **Fixed-chunk RAG baseline** — `ChunkRetriever` splits every leaf into
  120-word chunks with 20-word overlap, runs BM25 over chunks, and returns
  one best score per leaf (`bm25_chunks` system). This is the comparison the
  paper motivates: structure-aware leaves vs arbitrary chunking.

## Layout

```
src/stair_vs_rag/
  toc.py        ToC model, parse/render, strict leaf matching (hallucination gate)
  ingest.py     PDF -> ToC entries + per-leaf text (page ranges)
  prompt.py     verbatim STAIR prompt shape (paper Table 2)
  baselines.py  BM25 leaf retriever + fixed-chunk baseline
  llm.py        Ollama chat client + STAIRAgent (pick_leaf, hallucination log)
  genqueries.py synthetic QA per leaf (gold = source leaf)
  eval.py       per-prediction score, validity, Recall@k/MRR/nDCG
  benchmark.py  aggregate + end-to-end run (stair, bm25, bm25_chunks)
tests/          40 pytest tests incl. synthetic-PDF ToC fixtures
```

Spec-driven rules honored: backward/forward compatibility starts here —
`OllamaClient(transport=...)` takes any transport object, so tests and future
providers (llama.cpp, API models) plug in without changing callers.
