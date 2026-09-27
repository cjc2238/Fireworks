"""Use logprobs to turn an LLM into a classifier that reports its own confidence."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

from fwlearn import MODEL, client

fw = client()
reviews = [
    "Absolutely loved it, would buy again!",
    "It broke after two days. Waste of money.",
    "It's fine I guess. Does what it says.",
]

for text in reviews:
    resp = fw.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Classify sentiment. Answer with exactly one word: positive, negative, or neutral."},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=3,
        reasoning_effort="none",  # skip hidden thinking: fast, short answers
        logprobs=True,
        top_logprobs=5,
    )
    choice = resp.choices[0]
    label = choice.message.content.strip()
    first = choice.logprobs.content[0]  # first generated token
    print(f"\n{text!r}\n  -> {label}  (p={math.exp(first.logprob):.3f})")
    for alt in first.top_logprobs:
        print(f"     {alt.token!r:>12}  p={math.exp(alt.logprob):.3f}")
