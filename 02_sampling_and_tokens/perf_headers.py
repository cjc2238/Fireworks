"""Read Fireworks' performance metrics: headers (non-streaming) and perf_metrics (streaming)."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, banner, client

fw = client()
msgs = [{"role": "user", "content": "List 5 GPU architectures and one fact about each."}]

banner("Non-streaming: metrics arrive as response headers")
raw = fw.chat.completions.with_raw_response.create(model=MODEL, messages=msgs, max_tokens=300, reasoning_effort="none")
for k, v in raw.headers.items():
    if k.lower().startswith("fireworks-"):
        print(f"  {k}: {v}")
completion = raw.parse()
print("\n  finish_reason:", completion.choices[0].finish_reason)

banner("Streaming: perf_metrics on the final chunk")
stream = fw.chat.completions.create(
    model=MODEL, messages=msgs, max_tokens=300, stream=True, perf_metrics_in_response=True,
    reasoning_effort="none",
)
pm = usage = None
for chunk in stream:
    # perf_metrics and usage can arrive on the finish chunk or on a trailing chunk after it
    pm = getattr(chunk, "perf_metrics", None) or (chunk.model_extra or {}).get("perf_metrics") or pm
    usage = chunk.usage or usage
print("  perf_metrics:", pm)
print("  usage       :", usage)
