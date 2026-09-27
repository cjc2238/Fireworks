"""Convert lesson 10's SFT data into RFT format: prompts + ground_truth (no assistant answer).

RFT rows keep only the prompt messages. The label moves into `ground_truth`, which the
evaluator reads. The model has to discover how to earn the reward.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from fwlearn import ROOT

out = ROOT / "outputs"
for split in ("train", "eval"):
    src = out / f"sft_{split}.jsonl"
    if not src.exists():
        sys.exit(f"Missing {src}. Run 10_fine_tuning_sft/make_dataset.py first.")
    dst = out / f"rft_{split}.jsonl"
    n = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for line in f:
            row = json.loads(line)
            *prompt, answer = row["messages"]
            g.write(json.dumps({"messages": prompt, "ground_truth": answer["content"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {dst.relative_to(ROOT)}")
