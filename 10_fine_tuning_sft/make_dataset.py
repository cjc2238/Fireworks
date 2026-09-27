"""Generate a synthetic SFT dataset with a big 'teacher' model (a simple form of distillation).

1. The teacher writes realistic tickets.
2. The teacher labels them using the long house-rules prompt.
3. We save them with the SHORT prompt, so the student learns the rules from examples.

Usage:  python 10_fine_tuning_sft/make_dataset.py [n_tickets=300]
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import asyncio
import json
import random

from task import SYSTEM_SHORT, SYSTEM_TEACHER, TOPICS, parse

from fwlearn import BIG_MODEL, ROOT, async_client

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
random.seed(42)


async def make_one(fw, sem, i):
    async with sem:
        topic = random.choice(TOPICS)
        tone = random.choice(["angry", "polite", "terse", "rambling", "non-native English speaker"])
        t = await fw.chat.completions.create(
            model=BIG_MODEL, temperature=1.0, max_tokens=4000,
            messages=[{"role": "user", "content":
                       f"Write one realistic customer support ticket to an AI inference platform about: {topic}. "
                       f"Tone: {tone}. 1-4 sentences. Output only the ticket text."}])
        ticket = t.choices[0].message.content.strip()
        lab = await fw.chat.completions.create(
            model=BIG_MODEL, temperature=0, max_tokens=4000,
            messages=[{"role": "system", "content": SYSTEM_TEACHER}, {"role": "user", "content": ticket}])
        label = lab.choices[0].message.content.strip().splitlines()[-1]
        if not {"team", "sev", "tag"} <= parse(label).keys():
            return None  # drop malformed labels: data quality first
        return {"messages": [
            {"role": "system", "content": SYSTEM_SHORT},
            {"role": "user", "content": ticket},
            {"role": "assistant", "content": label},
        ]}


async def main():
    fw = async_client(max_retries=4)
    sem = asyncio.Semaphore(8)
    rows = [r for r in await asyncio.gather(*(make_one(fw, sem, i) for i in range(N)),
                                            return_exceptions=True) if isinstance(r, dict)]
    random.shuffle(rows)
    cut = int(len(rows) * 0.85)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    for name, part in [("sft_train.jsonl", rows[:cut]), ("sft_eval.jsonl", rows[cut:])]:
        with (out / name).open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(part):>4} rows -> outputs/{name}")


asyncio.run(main())
