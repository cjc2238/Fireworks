"""Embed sentences and compare them with cosine similarity."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from fwlearn import EMBED_MODEL, openai_client

sentences = [
    "How do I fine-tune a model on my data?",
    "Steps to train a LoRA adapter with custom examples",
    "What's the weather like in Paris?",
    "GPU autoscaling for inference deployments",
]

oai = openai_client()  # embeddings are OpenAI-compatible
resp = oai.embeddings.create(model=EMBED_MODEL, input=sentences)
vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
print(f"model={EMBED_MODEL} dims={vecs.shape[1]} tokens={resp.usage.total_tokens}")
print(f"norms before normalizing: {np.linalg.norm(vecs, axis=1).round(2)}")

vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # Qwen3 embeddings are unnormalized
sim = vecs @ vecs.T

print("\nCosine similarity matrix:")
for i, s in enumerate(sentences):
    print(f"{s[:45]:<46}", " ".join(f"{x:5.2f}" for x in sim[i]))

# Matryoshka-style resizing: ask for fewer dimensions
small = oai.embeddings.create(model=EMBED_MODEL, input=sentences[:1], dimensions=256)
print(f"\nWith dimensions=256 -> {len(small.data[0].embedding)} dims")
