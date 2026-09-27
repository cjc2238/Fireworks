# 04c · Building Agents

An **agent** is an LLM in a loop that decides its own next action (a tool call or an answer)
based on what it has seen so far. That's the whole idea. Everything else is engineering around
the loop.

```
          ┌──────────────────────────────────────────────┐
user ──►  │  LLM  ──tool_calls──►  your code runs tools  │ ──► final answer
          │   ▲                           │              │
          │   └──────── tool results ◄────┘              │
          └──────────────────────────────────────────────┘
             state = the messages list (system, user, assistant, tool, ...)
```

## Read the runtime first

Open [`fwlearn/agent.py`](../fwlearn/agent.py). It's about 200 lines, and every lesson from here on
uses it. It implements:

| Concern | How |
|---|---|
| Tools from typed functions | `@tool` → Pydantic model → JSON Schema; the docstring is the description |
| The loop | call the model → run tools → append results → repeat until no tool calls |
| Parallel tools | `ThreadPoolExecutor` over all tool calls in a turn |
| Error recovery | exceptions and validation errors go back to the model *as data* |
| Guardrails | `max_steps`, `token_budget`, loop detection (the same call 3×), result truncation |
| Human-in-the-loop | `@tool(requires_approval=True)` + an `approve(name, args)` callback |
| Reasoning models | the full assistant message (with `reasoning_content`) is kept in history |
| Observability | an `on_event(kind, data)` hook (used for tracing in 04e) |

## Workflow or agent?

Not everything should be an agent. From most control to most autonomy:

| Pattern | The model decides… | Use when |
|---|---|---|
| **Single call** | nothing (you pick the prompt) | classification, extraction |
| **Workflow / chain** | nothing about control flow; *your code* orders the steps | the steps are known in advance |
| **Router** | which branch or specialist | distinct request types |
| **Plan-and-execute** | the plan up front, then executes it | multi-step tasks with a knowable structure |
| **ReAct agent** | every next step | open-ended tasks, exploration |
| **Multi-agent** | who works on what | large tasks with separable skills or permissions |

Rule of thumb: **use the least autonomy that solves the problem.** Every extra step adds latency,
cost and chances to fail. Customers often ask for "agents" when a workflow is what they need, and
that's a valuable conversation to lead.

## The patterns in this lesson

1. **ReAct research agent** (`research_agent.py`): tools to search and read this course; it
   investigates, then answers with citations.
2. **Plan-and-execute** (`plan_and_execute.py`): a reasoning model writes a JSON plan
   (structured output) and a fast model executes each step with tools. Big model for planning,
   small model for doing: a very common way to cut cost.
3. **Router + specialists** (`multi_agent_router.py`): a grammar-constrained router picks a
   specialist agent, and each specialist has *only* its own tools (least privilege). Handoffs pass a
   **clean summary**, not the raw transcript. See the handoff finding in 04d.
4. **Evaluator-optimizer / reflection** (`reflection_loop.py`): the model writes SQL, we
   *execute* it and feed back errors or wrong results until it passes. Verifiable feedback beats
   "please double-check your work".
5. **Human-in-the-loop** (`human_in_the_loop.py`): destructive tools need approval, and a denial
   goes back to the model as data.
6. **Memory** (`memory_agent.py`): a long conversation stays within budget by summarizing old
   turns and keeping durable facts in a scratchpad.

## Context is the scarcest resource

Every step resends the whole history: system prompt, tool schemas, all tool results. A 10-step
agent with 5k-token tool outputs sends over 100k prompt tokens per task. Levers:

- Return **concise, structured** tool results (IDs and fields, not whole documents), and truncate.
- **Summarize** or drop old turns (memory pattern).
- Keep system prompts and tool schemas **stable** so prompt caching hits (lesson 07). Tool schemas
  are part of the prefix, so don't reorder them between calls.
- Use sub-agents: a helper does a messy search in *its own* context and returns only the answer.

## Scripts

```powershell
python 04c_building_agents\research_agent.py "How do I deploy a LoRA, and what does it cost?"
python 04c_building_agents\plan_and_execute.py
python 04c_building_agents\multi_agent_router.py
python 04c_building_agents\reflection_loop.py
python 04c_building_agents\human_in_the_loop.py
python 04c_building_agents\memory_agent.py
```

## Exercises

1. Give the research agent a `run_python(code)` tool that runs in a subprocess with a timeout.
   What safety limits would you add before letting customers use it?
2. Make plan-and-execute **re-plan** when a step fails, instead of carrying on.
3. Add a third specialist to the router (for example "training" questions with a
   `validate_dataset` tool) and measure routing accuracy on 15 labelled prompts.
4. In `reflection_loop.py`, count how many iterations it takes on average, and compare a small
   model with a big one. Is "small model + verifier loop" cheaper than "big model once"?
