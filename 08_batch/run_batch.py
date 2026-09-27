"""End-to-end batch job over REST: upload dataset -> create job -> poll -> download -> score.

Usage:  python 08_batch/run_batch.py [model-id]
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re
import time

from fwlearn import MODEL, ROOT, require_account_id
from fwlearn.control import create_dataset, download_dataset, post, wait_for

model = sys.argv[1] if len(sys.argv) > 1 else MODEL
acct = require_account_id()
stamp = time.strftime("%m%d-%H%M%S")
in_id, out_id, job_id = f"batch-in-{stamp}", f"batch-out-{stamp}", f"batch-job-{stamp}"
inp = ROOT / "outputs" / "batch_input.jsonl"
if not inp.exists():
    sys.exit("Run 08_batch/make_batch_input.py first.")

print("1) Uploading input dataset...")
create_dataset(in_id, inp)

print("2) Creating batch job...")
job = post("batchInferenceJobs", {
    "model": model,
    "inputDatasetId": f"accounts/{acct}/datasets/{in_id}",
    "outputDatasetId": f"accounts/{acct}/datasets/{out_id}",
    "inferenceParameters": {"maxTokens": 300, "temperature": 0},
}, batchInferenceJobId=job_id)
print("   created:", job.get("name"))

print("3) Waiting (batch jobs can take minutes to hours)...")
wait_for(f"batchInferenceJobs/{job_id}", ready_states={"JOB_STATE_COMPLETED"}, poll=30)

print("4) Downloading results...")
files = download_dataset(out_id, ROOT / "outputs" / out_id)
print("   files:", [p.name for p in files])

print("5) Scoring against the answer key...")
key = json.loads((ROOT / "outputs" / "batch_answers.json").read_text())
correct = total = 0
for p in files:
    if p.suffix != ".jsonl":
        continue
    for line in p.open(encoding="utf-8"):
        row = json.loads(line)
        cid = row.get("custom_id")
        if cid not in key:
            continue
        resp = row.get("response", row)
        text = json.dumps(resp)
        m = re.findall(r"ANSWER:\s*(-?\d+)", text)
        total += 1
        correct += bool(m) and int(m[-1]) == key[cid]
print(f"   accuracy: {correct}/{total} = {correct / max(total, 1):.1%}")
print("\nTip: open one output line to learn the exact result schema, then tighten this parser.")
