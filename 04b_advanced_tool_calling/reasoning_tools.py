"""Interleaved thinking: a reasoning model thinks, calls tools, thinks again about the results.
We print the reasoning at each step, and send the FULL assistant message back so the chain survives.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import BIG_MODEL, banner
from fwlearn.agent import Agent, tool

ACCOUNTS = {"acme": {"plan": "enterprise", "monthly_tokens": 2.1e9, "region": "EUROPE"},
            "zeta": {"plan": "developer", "monthly_tokens": 4.0e7, "region": "US"}}


@tool
def get_account(name: str) -> dict:
    """Look up a customer account's plan, monthly token volume, and region."""
    return ACCOUNTS[name.lower()]


@tool
def list_deployment_options(region: str) -> list:
    """List dedicated deployment shapes available in a region, with $/hour and tokens/hour capacity."""
    return [{"shape": "minimal", "usd_per_hour": 3.0, "tokens_per_hour": 4e6},
            {"shape": "throughput", "usd_per_hour": 12.0, "tokens_per_hour": 3e7}]


def show(kind, d):
    if kind == "llm":
        banner(f"step {d['step']}  ({d['seconds']}s, {d['completion_tokens']} completion tokens)")
        if d["tool_calls"]:
            print("calls:", d["tool_calls"])
        if d["content"]:
            print("says :", d["content"][:300])
    elif kind == "tool":
        print(f"  -> {d['name']} returned {d['output'][:150]}")


agent = Agent(
    tools=[get_account, list_deployment_options],
    model=BIG_MODEL,
    reasoning_effort=None,      # let the reasoning model think before each step
    on_event=show,
    extra={"max_tokens": 8000},
)
result = agent.run("Customer 'acme' pays serverless at a blended $0.60 per 1M tokens. Would a dedicated "
                   "deployment in their region be cheaper, running 24/7 (730 h/month)? Show the math.")

banner("Reasoning kept in history (first 300 chars of each assistant turn)")
for m in result.messages:
    if m.get("role") == "assistant" and m.get("reasoning_content"):
        print("-", m["reasoning_content"][:300].replace("\n", " "), "\n")
banner("Final answer")
print(result.text)
print(f"\nsteps={result.steps} tool_calls={result.tool_calls} tokens={result.prompt_tokens}+{result.completion_tokens} "
      f"time={result.seconds}s")
