"""Compare reasoning_effort levels: quality vs tokens vs latency."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

from fwlearn import BIG_MODEL, banner, client

PUZZLE = (
    "Three boxes are labeled 'apples', 'oranges', and 'mixed'. Every label is wrong. "
    "You may draw one fruit from one box. Which box do you draw from, and how do you relabel them? "
    "Answer in 2 sentences."
)

fw = client()
for effort in ["low", "medium", "high"]:
    banner(f"reasoning_effort={effort}")
    t0 = time.perf_counter()
    try:
        resp = fw.chat.completions.create(
            model=BIG_MODEL,
            messages=[{"role": "user", "content": PUZZLE}],
            reasoning_effort=effort,
            max_tokens=8000,
        )
    except Exception as e:
        print(f"  {BIG_MODEL} rejected this setting: {e}")
        continue
    dt = time.perf_counter() - t0
    msg = resp.choices[0].message
    thinking = getattr(msg, "reasoning_content", None) or ""
    print(f"  latency={dt:.1f}s  completion_tokens={resp.usage.completion_tokens}  "
          f"reasoning_chars={len(thinking)}")
    print("  answer:", (msg.content or "").strip())
