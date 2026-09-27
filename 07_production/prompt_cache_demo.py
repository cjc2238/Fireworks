"""Demonstrate prompt caching: a long, stable prefix + session affinity -> cached prompt tokens.

Watch the 'cached' column climb after the first request. Then run with --break to put a
timestamp at the top of the system prompt and watch caching collapse.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time
import uuid

from fwlearn import MODEL, client

BREAK = "--break" in sys.argv

# A long static prefix (~3k tokens): policies, few-shot examples, etc.
POLICY = "\n".join(
    f"Rule {i}: When a customer asks about topic {i}, respond politely, cite policy section {i}, "
    f"and offer escalation if the issue involves billing or data deletion."
    for i in range(1, 120)
)
questions = [
    "How do I reset my password?",
    "Can I get a refund for last month?",
    "Delete all my data please.",
    "What are your support hours?",
]

fw = client()
session = str(uuid.uuid4())
print(f"{'#':<3}{'prompt':>8}{'cached':>8}{'hit%':>7}{'ttft(s)':>9}")
for i, q in enumerate(questions, 1):
    system = POLICY if not BREAK else f"Current time: {time.time()}\n{POLICY}"
    raw = fw.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": q}],
        max_tokens=60,
        extra_headers={"x-session-affinity": session},  # route to the same replica
    )
    h = raw.headers
    prompt = int(h.get("fireworks-prompt-tokens", 0) or 0)
    cached = int(h.get("fireworks-cached-prompt-tokens", 0) or 0)
    ttft = h.get("fireworks-server-time-to-first-token", "?")
    print(f"{i:<3}{prompt:>8}{cached:>8}{(100 * cached / prompt if prompt else 0):>6.0f}%{ttft:>9}")
    raw.parse()

print("\n(If 'cached' stays 0: some models/serverless pools report caching differently; "
      "check the usage dashboard and try another model.)")
