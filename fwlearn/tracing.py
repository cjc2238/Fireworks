"""JSONL tracing for fwlearn.agent. Plug in with Agent(on_event=Tracer(...)).

Each event is one line: {"run_id", "t" (seconds since start), "kind" (llm|tool|end), ...data}.
JSONL traces are easy to grep, diff, replay, load into pandas, or ship to an observability tool
(see https://docs.fireworks.ai/ecosystem/integrations/mlops-observability).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import ROOT


class Tracer:
    def __init__(self, name: str = "run", out_dir: Path | None = None, echo: bool = False):
        self.run_id = f"{name}-{uuid.uuid4().hex[:8]}"
        self.path = (out_dir or ROOT / "outputs" / "traces") / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.echo = echo

    def __call__(self, kind: str, data: dict):
        ev = {"run_id": self.run_id, "t": round(time.perf_counter() - self.t0, 3), "kind": kind, **data}
        self.events.append(ev)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        if self.echo:
            if kind == "llm":
                calls = ", ".join(n for n, _ in ev["tool_calls"]) or "-> answer"
                print(f"  [{ev['t']:6.2f}s] llm step {ev['step']}: {ev['seconds']}s, "
                      f"{ev['prompt_tokens']}+{ev['completion_tokens']} tok, {calls}")
            elif kind == "tool":
                print(f"  [{ev['t']:6.2f}s]   tool {ev['name']} {'ok' if ev['ok'] else 'ERROR'} ({ev['seconds']}s)")

    def summary(self) -> dict:
        llm = [e for e in self.events if e["kind"] == "llm"]
        tools = [e for e in self.events if e["kind"] == "tool"]
        return {
            "run_id": self.run_id,
            "llm_calls": len(llm),
            "llm_seconds": round(sum(e["seconds"] for e in llm), 2),
            "tool_calls": len(tools),
            "tool_errors": sum(not e["ok"] for e in tools),
            "tool_seconds": round(sum(e["seconds"] for e in tools), 2),
            "prompt_tokens": sum(e["prompt_tokens"] for e in llm),
            "completion_tokens": sum(e["completion_tokens"] for e in llm),
            "wall_seconds": self.events[-1]["t"] if self.events else 0,
            "trace_file": str(self.path.relative_to(ROOT)),
        }
