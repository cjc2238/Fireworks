"""OpenAI Agents SDK on Fireworks: function tools + a handoff.

It measures the intermittent handoff failure from lesson.md (Billing sometimes answers without
using its tool). Each configuration runs N times and we count how often the tool was used,
because a single run can't tell you whether a fix works.

pip install openai-agents
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio

from agents import (Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, ToolCallItem, function_tool,
                    handoff, set_tracing_disabled)
from agents.extensions import handoff_filters
from openai import AsyncOpenAI

from fwlearn import API_KEY, INFERENCE_URL, MODEL

set_tracing_disabled(True)  # the SDK's default tracing exporter sends to OpenAI; we don't want that
N = 5

fw = AsyncOpenAI(api_key=API_KEY, base_url=INFERENCE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=fw)
settings = ModelSettings(temperature=0.7, extra_body={"reasoning_effort": "none"})


@function_tool
def gpu_price(gpu: str) -> str:
    """Hourly price of a GPU type."""
    return {"H100": "$2.90/hr", "H200": "$4.00/hr", "B200": "$5.80/hr"}.get(gpu.upper(), "unknown GPU")


def billing(strong_prompt: bool) -> Agent:
    instructions = ("Answer pricing questions. ALWAYS call gpu_price for any GPU price." if strong_prompt
                    else "You answer pricing questions using tools.")
    return Agent(name="Billing", instructions=instructions, tools=[gpu_price], model=model, model_settings=settings)


def triage(filtered: bool, strong_prompt: bool) -> Agent:
    b = billing(strong_prompt)
    target = handoff(b, input_filter=handoff_filters.remove_all_tools) if filtered else b
    return Agent(name="Triage", instructions="Route pricing questions to Billing.",
                 handoffs=[target], model=model, model_settings=settings)


async def trial(agent) -> bool:
    r = await Runner.run(agent, "How much is an H100 per hour?")
    return r.last_agent.name == "Billing" and any(isinstance(i, ToolCallItem) for i in r.new_items)


async def main():
    print(f"{'billing prompt':<15} {'handoff filter':<15} tool used (of {N})")
    for strong in (False, True):
        for filtered in (False, True):
            ok = await asyncio.gather(*(trial(triage(filtered, strong)) for _ in range(N)))
            print(f"{'strong' if strong else 'weak':<15} {'remove_all_tools' if filtered else 'none':<15} "
                  f"{sum(ok)}/{N}")
    print(f"\nWith N={N} the differences are mostly noise. Raise N to 30+ before concluding which fix works.")


asyncio.run(main())
