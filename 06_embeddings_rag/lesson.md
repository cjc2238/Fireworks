# 06 · Embeddings, Reranking & RAG

**Goal:** build retrieval-augmented generation from first principles, so you can debug any
customer's RAG stack whatever vector database or framework they use.

## Embeddings

An embedding model maps text to a vector, and similar meanings end up close together.

- Endpoint: `POST /inference/v1/embeddings` (OpenAI-compatible)
- Serverless model: **`fireworks/qwen3-embedding-8b`** (40k context, resizable with `dimensions`).
  Others (Qwen3 4B/0.6B, Voyage 4 family) run on dedicated deployments. Legacy:
  `nomic-ai/nomic-embed-text-v1.5`.
- **Qwen3 embeddings are NOT normalized.** Normalize them before cosine similarity, or pass
  `normalize=True`.
- Voyage models expect `input_type="query"` or `"document"` (asymmetric retrieval).

## Reranking

Embeddings are fast but coarse. A **reranker** (cross-encoder) reads the query and each document
*together* and scores relevance much more accurately, but it only scales to tens or hundreds of
candidates.

- Endpoint: `POST /inference/v1/rerank` with `{model, query, documents, top_n, return_documents}`
- Serverless: **`fireworks/qwen3-reranker-8b`**

## The standard RAG pipeline

```
docs ──chunk──► embed ──► vector index
                                │
query ──embed──► top-k (≈20-50) ─┘──► rerank ──► top-n (≈3-5) ──► LLM prompt with citations
```

## What breaks RAG (debugging checklist)

1. **Chunking:** chunks too large dilute meaning, and chunks too small lose context. Overlap helps.
2. **Retrieval misses:** measure *recall@k* on a small labeled set *before* tuning the prompt.
3. **No reranker:** often the cheapest big quality win.
4. **Prompt:** tell the model to answer *only* from context and to say "I don't know" otherwise.
5. **Cold starts:** dedicated embedding deployments that scale to zero return 503 while they
   wake up, so retry them.

## Scripts

```powershell
python 06_embeddings_rag\embeddings_basics.py
python 06_embeddings_rag\rerank.py
python 06_embeddings_rag\rag_from_scratch.py "How do I deploy a LoRA on Fireworks?"
```

`rag_from_scratch.py` indexes **the lesson files in this repo**, so you get a Q&A bot over the
course itself.

## Exercises

1. Create 15 question → correct-file pairs for this repo and measure recall@5 with and without
   the reranker.
2. Try `dimensions=256` vs the full size. How much does recall drop, and how much storage do you
   save?
3. Swap the numpy index for a real vector DB (Chroma, LanceDB, pgvector, Qdrant).
4. Add **citations**: make the model cite `[file#chunk]` and verify each citation exists.
