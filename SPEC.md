# Spec: stair-vs-rag — RAG Evaluation Benchmark

## Objective
Benchmark and compare RAG retrieval strategies (stair/stepwise vs flat) on
real-world document corpora. Measures retrieval accuracy, latency, and cost
across different chunking, embedding, and reranking configurations.

## IG-Mined Topics (cross-pollination)
- RAG (T06): chunking, hybrid search, reranking, query transformation, advanced RAG
- Embeddings (T06-02): dimensions, Matryoshka, multilingual, cost
- Vector Index (T06-03): HNSW, IVF-PQ, ScaNN, DiskANN
- RAG Eval (T06-10): RAGAS, golden sets, synthetic questions, retrieval@k
- Semantic Caching (T06-16): exact + semantic + prefix layers
- Context Engineering (T07-11): token budgets, compaction, context editing
- Evaluation (T08): LLM-as-judge, trajectory eval, tool-call accuracy

## Tech Stack
- Python, pyproject.toml, pytest
- Embeddings: sentence-transformers, FAISS
- Evaluation: RAGAS, custom metrics

## Commands
```bash
pytest tests/ -q
```
