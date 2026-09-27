"""RAG over this course's own lesson files: chunk -> embed -> retrieve -> rerank -> generate.

Usage:  python 06_embeddings_rag/rag_from_scratch.py "your question"
The index is cached in outputs/rag_index.npz; delete it to rebuild.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import requests

from fwlearn import EMBED_MODEL, INFERENCE_URL, MODEL, RERANK_MODEL, ROOT, auth_headers, client, openai_client

CACHE = ROOT / "outputs" / "rag_index.npz"
CHUNK_CHARS, OVERLAP = 1200, 200


def chunk_docs():
    chunks = []
    for path in sorted(ROOT.glob("*/lesson.md")):
        text = path.read_text(encoding="utf-8")
        for i, start in enumerate(range(0, len(text), CHUNK_CHARS - OVERLAP)):
            chunks.append({"id": f"{path.parent.name}#{i}", "text": text[start:start + CHUNK_CHARS]})
    return chunks


def embed(texts, batch=64):
    oai = openai_client()
    out = []
    for i in range(0, len(texts), batch):
        out += [d.embedding for d in oai.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch]).data]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_index():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return json.loads(str(z["meta"])), z["vecs"]
    chunks = chunk_docs()
    print(f"Embedding {len(chunks)} chunks...")
    vecs = embed([c["text"] for c in chunks])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, vecs=vecs, meta=json.dumps(chunks))
    return chunks, vecs


def retrieve(question, chunks, vecs, k=15, n=4):
    q = embed([question])[0]
    top = np.argsort(-(vecs @ q))[:k]                      # stage 1: vector search
    cands = [chunks[i] for i in top]
    r = requests.post(f"{INFERENCE_URL}/rerank", headers=auth_headers(), timeout=60, json={
        "model": RERANK_MODEL, "query": question, "documents": [c["text"] for c in cands], "top_n": n})
    r.raise_for_status()
    ranked = r.json().get("data") or r.json().get("results")  # stage 2: rerank
    return [cands[x["index"]] for x in ranked]


def answer(question):
    chunks, vecs = load_index()
    ctx = retrieve(question, chunks, vecs)
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in ctx)
    resp = client().chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": (
                "Answer ONLY from the context. Cite sources like [07_production#2]. "
                "If the context doesn't contain the answer, say you don't know.")},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    print("Retrieved:", [c["id"] for c in ctx], "\n")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    answer(" ".join(sys.argv[1:]) or "What's the difference between serverless and dedicated deployments?")
