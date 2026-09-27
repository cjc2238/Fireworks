"""Turn passing agent trajectories into an SFT dataset for tool-calling fine-tuning (lesson 10).

Input : outputs/agent_trajectories.jsonl   (written by agent_eval.py)
Output: outputs/agent_sft.jsonl             {"tools": [...], "messages": [...]} per line

Fireworks SFT supports function-calling data: a `tools` array plus assistant messages with
`tool_calls` and `tool` role results. We drop reasoning_content (optional) and dedupe.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

from fwlearn import ROOT

src = ROOT / "outputs" / "agent_trajectories.jsonl"
dst = ROOT / "outputs" / "agent_sft.jsonl"
if not src.exists():
    sys.exit("Run 04e_production_agents/agent_eval.py first.")

KEEP = {"role", "content", "tool_calls", "tool_call_id", "name"}
seen, rows = set(), []
for line in src.open(encoding="utf-8"):
    traj = json.loads(line)
    msgs = []
    for m in traj["messages"]:
        m = {k: v for k, v in m.items() if k in KEEP}
        if m["role"] == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                               for tc in m["tool_calls"]]
            m.setdefault("content", "")
        msgs.append(m)
    key = json.dumps(msgs, sort_keys=True)
    if key not in seen:
        seen.add(key)
        rows.append({"tools": traj["tools"], "messages": msgs})

with dst.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} unique trajectories -> {dst.relative_to(ROOT)}")
n_calls = sum(len(m.get("tool_calls", [])) for r in rows for m in r["messages"])
print(f"{n_calls} tool calls to learn from\n", flush=True)  # flush before the subprocess prints

# reuse lesson 10's validator
subprocess.run([sys.executable, str(ROOT / "10_fine_tuning_sft" / "validate_dataset.py"), str(dst)])
print("\nNext: firectl dataset create agent-sft outputs\\agent_sft.jsonl, then SFT a small model (lesson 10).")
