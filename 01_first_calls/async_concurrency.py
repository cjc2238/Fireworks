"""Sequential vs concurrent requests. Concurrency is how you get throughput from an API."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio
import time

from fwlearn import MODEL, async_client

N = 10
CONCURRENCY = 5  # cap in-flight requests so you don't trip rate limits
questions = [f"In one sentence, give fun fact #{i} about the number {i}." for i in range(1, N + 1)]


async def ask(fw, sem, q):
    async with sem:
        r = await fw.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=60,
            reasoning_effort="none",
        )
        return r.choices[0].message.content


async def main():
    fw = async_client()

    t0 = time.perf_counter()
    for q in questions[:3]:
        await ask(fw, asyncio.Semaphore(1), q)
    seq = (time.perf_counter() - t0) / 3 * N
    print(f"Sequential (extrapolated to {N}): {seq:.1f}s")

    t0 = time.perf_counter()
    sem = asyncio.Semaphore(CONCURRENCY)
    answers = await asyncio.gather(*(ask(fw, sem, q) for q in questions))
    conc = time.perf_counter() - t0
    print(f"Concurrent (max {CONCURRENCY} in flight): {conc:.1f}s  -> {seq / conc:.1f}x faster\n")
    for a in answers:
        print("-", a.strip())


asyncio.run(main())
