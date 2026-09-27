"""Build a DPO/ORPO dataset: for each question, generate a concise (preferred) and a
verbose (non-preferred) answer. Output: outputs/dpo_pairs.jsonl

Usage:  python 12_training_api/make_dpo_pairs.py
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio
import json

from fwlearn import MODEL, ROOT, async_client

QUESTIONS = [
    "What is a KV cache?", "What is LoRA?", "Why use speculative decoding?",
    "What does temperature do?", "What is prompt caching?", "What is quantization?",
    "What is a deployment shape?", "What is batch inference?", "What is RFT?",
    "What is time to first token?", "What is multi-LoRA serving?", "What is an embedding?",
]
STYLES = {
    "preferred": "Answer in at most 2 crisp sentences. No filler.",
    "non_preferred": "Answer in a long, rambling way with lots of filler, caveats, and repetition.",
}


async def pair(fw, q):
    outs = {}
    for key, style in STYLES.items():
        r = await fw.chat.completions.create(
            model=MODEL, temperature=0.7, max_tokens=600,
            messages=[{"role": "system", "content": style}, {"role": "user", "content": q}])
        outs[key] = [{"role": "assistant", "content": r.choices[0].message.content.strip()}]
    return {"input": {"messages": [{"role": "user", "content": q}]},
            "preferred_output": outs["preferred"], "non_preferred_output": outs["non_preferred"]}


async def main():
    fw = async_client()
    rows = await asyncio.gather(*(pair(fw, q) for q in QUESTIONS))
    out = ROOT / "outputs" / "dpo_pairs.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} pairs -> {out.relative_to(ROOT)}")
    print("Next: firectl dataset create dpo-pairs outputs\\dpo_pairs.jsonl")


asyncio.run(main())
