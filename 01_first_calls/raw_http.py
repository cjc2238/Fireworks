"""The same request with no SDK at all. Shows exactly what goes over the wire,
including Fireworks' performance headers.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import requests

from fwlearn import INFERENCE_URL, MODEL, auth_headers

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is a KV cache? Two sentences."}],
    "max_tokens": 200,
}
r = requests.post(f"{INFERENCE_URL}/chat/completions", headers=auth_headers(), json=payload, timeout=60)
print("HTTP", r.status_code)
r.raise_for_status()

print("\n--- Fireworks response headers ---")
for k, v in r.headers.items():
    if k.lower().startswith("fireworks-") or k.lower().startswith("x-ratelimit"):
        print(f"{k}: {v}")

print("\n--- Body ---")
body = r.json()
print(body["choices"][0]["message"]["content"])
print("\nusage:", json.dumps(body["usage"]))
