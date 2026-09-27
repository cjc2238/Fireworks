"""Plan-and-execute: a reasoning model writes a structured plan; a fast model executes each step
with tools; the planner writes the final report. Big model thinks, small model does.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from pydantic import BaseModel

from fwlearn import BIG_MODEL, MODEL, banner, client
from fwlearn.agent import Agent, tool

# ---- a toy "company data" environment -------------------------------------------------
USAGE = {"2026-07": 1.2e9, "2026-08": 1.9e9, "2026-09": 2.6e9}  # tokens/month
PRICE_PER_M = 0.60


@tool
def get_monthly_tokens(month: str) -> float:
    """Total tokens used in a month, format YYYY-MM. Available: 2026-07, 2026-08, 2026-09."""
    return USAGE[month]


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression, e.g. '(2.6e9-1.9e9)/1.9e9*100'. Use for ALL math."""
    if not set(expression) <= set("0123456789.+-*/()e "):
        raise ValueError("only arithmetic is allowed")
    return eval(expression, {"__builtins__": {}})  # restricted to arithmetic characters above


@tool
def get_price_per_million() -> float:
    """Current blended serverless price in USD per 1M tokens."""
    return PRICE_PER_M


# ---- planner (structured output) ------------------------------------------------------
class Step(BaseModel):
    id: int
    instruction: str


class Plan(BaseModel):
    steps: list[Step]


TASK = ("Compute month-over-month token growth for Aug and Sep 2026, the September bill, and project "
        "October's bill assuming the same growth rate as September.")

fw = client()
tool_list = "\n".join(f"- {t.name}: {t.description}" for t in (get_monthly_tokens, calculate, get_price_per_million))
plan_resp = fw.chat.completions.create(
    model=BIG_MODEL,
    messages=[{"role": "user", "content":
               f"Task: {TASK}\n\nAn executor with these tools will run each step:\n{tool_list}\n\n"
               "Write a short plan (3-7 steps). Each step must be self-contained and executable with the tools. "
               "Reply as JSON with a 'steps' list of {id, instruction}."}],
    response_format={"type": "json_schema", "json_schema": {"name": "Plan", "schema": Plan.model_json_schema()}},
    max_tokens=4000,
)
plan = Plan.model_validate_json(plan_resp.choices[0].message.content)
banner(f"PLAN from {BIG_MODEL.split('/')[-1]}")
for s in plan.steps:
    print(f"{s.id}. {s.instruction}")

# ---- executor (fast model + tools), one fresh context per step ------------------------
executor = Agent(tools=[get_monthly_tokens, calculate, get_price_per_million], model=MODEL, max_steps=6,
                 system="Execute exactly one step. Use tools; report the resulting number(s) briefly.")
results = []
for s in plan.steps:
    context = "\n".join(f"Step {r['id']} result: {r['result']}" for r in results)
    r = executor.run(f"Previous results:\n{context or '(none)'}\n\nNow do step {s.id}: {s.instruction}")
    results.append({"id": s.id, "result": r.text.strip()})
    print(f"\n[step {s.id}] {r.text.strip()[:200]}  ({r.tool_calls} tool calls)")

# ---- planner writes the report ---------------------------------------------------------
banner("REPORT")
final = fw.chat.completions.create(
    model=BIG_MODEL, max_tokens=4000,
    messages=[{"role": "user", "content": f"Task: {TASK}\nStep results:\n{json.dumps(results, indent=1)}\n\n"
                                          "Write a concise report with the key numbers."}])
print(final.choices[0].message.content)
