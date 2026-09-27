"""A small load tester: TTFT / latency percentiles and throughput at several concurrency levels.

Usage:
  python 07_production/benchmark.py --concurrency 1 4 8 --requests 16
  python 07_production/benchmark.py --model accounts/<acct>/deployments/<id> --prompt-tokens 2000

For serious benchmarking Fireworks publishes a guide and tooling:
https://docs.fireworks.ai/deployments/benchmarking
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import statistics
import time

from fwlearn import MODEL, async_client


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


async def one(fw, sem, model, prompt, max_tokens, effort):
    async with sem:
        t0 = time.perf_counter()
        ttft, out_tokens = None, 0
        stream = await fw.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, stream=True, temperature=0.7, reasoning_effort=effort)
        async for chunk in stream:
            d = chunk.choices[0].delta if chunk.choices else None
            # thinking tokens count as "first token" too: the server has started decoding
            if ttft is None and d and (d.content or getattr(d, "reasoning_content", None)):
                ttft = time.perf_counter() - t0
            if getattr(chunk, "usage", None):
                out_tokens = chunk.usage.completion_tokens
        total = time.perf_counter() - t0
        return ttft or total, total, out_tokens


async def run(model, conc, n, prompt, max_tokens, effort):
    fw = async_client(max_retries=3, timeout=300)
    sem = asyncio.Semaphore(conc)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(fw, sem, model, prompt, max_tokens, effort) for _ in range(n)),
                                   return_exceptions=True)
    wall = time.perf_counter() - t0
    ok = [r for r in results if not isinstance(r, Exception)]
    errs = len(results) - len(ok)
    ttfts, lats, toks = zip(*ok) if ok else ([], [], [])
    per_req_tps = [t / (l - f) for f, l, t in ok if t and l - f > 0.05]
    print(f"{conc:>5} {len(ok):>4} {errs:>4} "
          f"{pct(ttfts, 50) * 1000:>9.0f} {pct(ttfts, 95) * 1000:>9.0f} "
          f"{pct(lats, 50):>8.2f} {pct(lats, 95):>8.2f} "
          f"{(statistics.mean(per_req_tps) if per_req_tps else 0):>9.1f} "
          f"{sum(toks) / wall:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=200, help="approximate prompt length")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--reasoning-effort", default="none", help="none|low|medium|high (model-dependent)")
    a = ap.parse_args()

    prompt = ("word " * max(0, a.prompt_tokens - 30)) + "\nIgnore the filler above. Write a short essay about GPUs."
    print(f"model={a.model} prompt~{a.prompt_tokens}tok max_tokens={a.max_tokens} reasoning_effort={a.reasoning_effort}\n")
    print(f"{'conc':>5} {'ok':>4} {'err':>4} {'ttft_p50':>9} {'ttft_p95':>9} "
          f"{'lat_p50':>8} {'lat_p95':>8} {'tps/req':>9} {'agg_tok/s':>10}")
    for c in a.concurrency:
        asyncio.run(run(a.model, c, a.requests, prompt, a.max_tokens, a.reasoning_effort))


if __name__ == "__main__":
    main()
