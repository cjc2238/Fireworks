"""Stream tokens as they're generated and measure time-to-first-token (TTFT)."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

from fwlearn import MODEL, client

fw = client()
start = time.perf_counter()
first_token_at = None
n_chunks = 0
usage = None

stream = fw.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a 150-word story about a GPU that dreams."}],
    stream=True,
    reasoning_effort="none",  # skip hidden thinking: fast, short answers
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        n_chunks += 1
        print(chunk.choices[0].delta.content, end="", flush=True)
    # Fireworks sends usage in the final chunk.
    if getattr(chunk, "usage", None):
        usage = chunk.usage

total = time.perf_counter() - start
ttft = (first_token_at or time.perf_counter()) - start
gen_time = total - ttft
print("\n")
print(f"TTFT            : {ttft * 1000:.0f} ms")
print(f"Total time      : {total:.2f} s")
if usage:
    print(f"Output tokens   : {usage.completion_tokens}")
    if gen_time > 0:
        print(f"Output tok/sec  : {usage.completion_tokens / gen_time:.1f}")
