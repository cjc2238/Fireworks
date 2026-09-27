"""Score a model on the held-out triage set. Run it on the base model AND the fine-tuned model.

Usage:  python 10_fine_tuning_sft/evaluate.py <model-id> [--teacher-prompt]
  --teacher-prompt  gives the model the long house-rules prompt (the "prompt engineering" baseline)
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import asyncio
import json
import time

from task import SYSTEM_TEACHER, parse

from fwlearn import MODEL, ROOT, async_client

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model = args[0] if args else MODEL
use_teacher_prompt = "--teacher-prompt" in sys.argv
rows = [json.loads(l) for l in (ROOT / "outputs" / "sft_eval.jsonl").open(encoding="utf-8")]


async def score(fw, sem, row):
    msgs = row["messages"][:-1]
    gold = parse(row["messages"][-1]["content"])
    if use_teacher_prompt:
        msgs = [{"role": "system", "content": SYSTEM_TEACHER}] + msgs[1:]
    async with sem:
        t0 = time.perf_counter()
        r = await fw.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=4000)
        dt = time.perf_counter() - t0
    pred = parse((r.choices[0].message.content or "").strip().splitlines()[-1] if r.choices[0].message.content else "")
    return {f: pred.get(f) == gold.get(f) for f in ("team", "sev", "tag")}, dt, r.usage.prompt_tokens


async def main():
    fw = async_client(max_retries=5, timeout=300)
    sem = asyncio.Semaphore(8)
    res = await asyncio.gather(*(score(fw, sem, r) for r in rows))
    n = len(res)
    print(f"model={model}  prompt={'teacher' if use_teacher_prompt else 'short'}  n={n}")
    for f in ("team", "sev", "tag"):
        print(f"  {f:<5} accuracy: {sum(r[0][f] for r in res) / n:.1%}")
    print(f"  all-3 exact : {sum(all(r[0].values()) for r in res) / n:.1%}")
    print(f"  avg latency : {sum(r[1] for r in res) / n:.2f}s   avg prompt tokens: {sum(r[2] for r in res) / n:.0f}")


asyncio.run(main())
