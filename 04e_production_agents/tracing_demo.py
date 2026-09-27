"""Trace an agent run: live timeline, JSONL trace file, and a summary you'd put on a dashboard."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "04c_building_agents"))

import json

from research_agent import SYSTEM, list_lessons, read_section, search_course

from fwlearn import banner
from fwlearn.agent import Agent
from fwlearn.tracing import Tracer

tracer = Tracer("research", echo=True)
agent = Agent(tools=[list_lessons, search_course, read_section], system=SYSTEM, on_event=tracer)

banner("Timeline")
r = agent.run("Compare prompt caching and batch inference as cost levers. When would you use each?")

banner("Answer")
print(r.text[:1200])

banner("Summary")
s = tracer.summary()
print(json.dumps(s, indent=2))
llm_share = s["llm_seconds"] / max(s["wall_seconds"], 1e-9)
print(f"\nLLM time share: {llm_share:.0%}. Prompt tokens per LLM call: "
      f"{[e['prompt_tokens'] for e in tracer.events if e['kind'] == 'llm']}  <- grows every step")
