"""Generate a batch input file: 200 simple arithmetic word problems with known answers,
so the batch output can double as a mini-eval.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import random

from fwlearn import ROOT

random.seed(0)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
inp, answers = out / "batch_input.jsonl", out / "batch_answers.json"

key = {}
with inp.open("w", encoding="utf-8") as f:
    for i in range(200):
        a, b, c = random.randint(2, 99), random.randint(2, 99), random.randint(2, 9)
        q = f"A shop has {a} boxes with {c} apples each, then receives {b} more apples. How many apples total?"
        key[f"q-{i}"] = a * c + b
        f.write(json.dumps({
            "custom_id": f"q-{i}",
            "body": {
                "messages": [
                    {"role": "system", "content": "Solve the problem. End with 'ANSWER: <integer>'."},
                    {"role": "user", "content": q},
                ],
                "max_tokens": 300,
                "temperature": 0,
            },
        }) + "\n")
answers.write_text(json.dumps(key))
print(f"Wrote {inp} and {answers}")
