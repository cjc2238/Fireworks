"""Rerank candidate documents against a query with a cross-encoder."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import requests

from fwlearn import INFERENCE_URL, RERANK_MODEL, auth_headers

query = "What is the capital of France?"
documents = [
    "Paris is the capital and largest city of France, home to the Eiffel Tower and the Louvre.",
    "France is a country in Western Europe known for its wine, cuisine, and history.",
    "The weather in Europe varies significantly between northern and southern regions.",
    "Python is a popular programming language used for web development and data science.",
]

r = requests.post(
    f"{INFERENCE_URL}/rerank",
    headers=auth_headers(),
    json={"model": RERANK_MODEL, "query": query, "documents": documents, "top_n": 3, "return_documents": True},
    timeout=60,
)
r.raise_for_status()
body = r.json()
print("query:", query, "\n")
for item in body.get("data", body.get("results", [])):
    doc = item.get("document") or documents[item["index"]]
    if isinstance(doc, dict):
        doc = doc.get("text", doc)
    print(f"  score={item.get('relevance_score', item.get('score')):.4f}  [{item['index']}] {doc}")
