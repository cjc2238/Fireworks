# 04d · MCP & Agent Frameworks

## Part 1: The Model Context Protocol (MCP)

**MCP** is an open protocol for plugging tools (and resources and prompts) into LLM apps. Instead
of every app writing its own GitHub/Slack/database integration, a tool provider ships **one MCP
server**, and any MCP-capable *host* (Claude Code, Cursor, agent frameworks, Fireworks' Responses
API) can use it.

```
Host app (your agent) ── MCP client ──JSON-RPC──► MCP server ──► the actual system (DB, API, files)
                                  transports: stdio (local subprocess) | Streamable HTTP (remote)
```

The core methods (JSON-RPC 2.0), which you'll implement yourself in this lesson:

| Method | Direction | Purpose |
|---|---|---|
| `initialize` → `notifications/initialized` | client → server | handshake: protocol version, capabilities |
| `tools/list` | client → server | tool names, descriptions and **`inputSchema`** (JSON Schema, the same thing you send as `tools=`) |
| `tools/call` `{name, arguments}` | client → server | run a tool → `{content: [{type: "text", text}], isError}` |

### Two ways to use MCP with Fireworks

| | **Remote MCP via the Responses API** | **Local MCP bridge (client-side)** |
|---|---|---|
| Who calls the tool | **Fireworks' servers** | **your code** |
| Setup | `tools=[{"type": "mcp", "server_url": "..."}]` | spawn the server, `tools/list` → convert → chat completions |
| Works with | public or reachable HTTP MCP servers | anything, including local files and private networks |
| You see | `mcp_call` + `tool_output` items in `response.output` | every call, in your own loop |
| Script | `remote_mcp_responses.py` | `mini_mcp_server.py` + `mcp_bridge_agent.py` |

> **Why we implement MCP by hand here:** on this machine, Windows Application Control blocks a
> DLL (`win32api`) that the official `mcp` Python SDK loads on Windows. The protocol is simple
> enough that a dependency-free implementation is about 100 lines, and writing it teaches you what
> every framework does under the hood. On an unrestricted machine, use the official SDK
> (`pip install mcp`, `FastMCP`) for real projects.

### MCP security (customers will ask)

- **Tool results are untrusted input.** A web page or document returned by a tool can contain
  instructions ("ignore previous instructions and…"). That's **prompt injection**; see 04e.
- **Tool poisoning:** a malicious server can hide instructions in tool *descriptions*. Only
  connect servers you trust, and review their `tools/list`.
- **Least privilege:** give each agent only the tools it needs. Put read-only and destructive
  tools on different servers, and require approval for writes.
- **Secrets:** pass credentials to servers through environment variables or headers, never in prompts.

## Part 2: Agent frameworks

Fireworks is OpenAI-compatible, so most frameworks work by pointing their OpenAI client at
`https://api.fireworks.ai/inference/v1`. The officially documented integrations are **LangChain /
LangGraph, LlamaIndex, CrewAI, PydanticAI, Strands Agents and AWS AgentCore**
(<https://docs.fireworks.ai/ecosystem/integrations/agent-frameworks>). LiteLLM-based tools use the
`fireworks_ai/<model-id>` prefix.

`agents_sdk_demo.py` uses the **OpenAI Agents SDK** (`pip install openai-agents`) with Fireworks:

```python
fw = AsyncOpenAI(api_key=FIREWORKS_API_KEY, base_url="https://api.fireworks.ai/inference/v1")
model = OpenAIChatCompletionsModel(model="accounts/fireworks/models/...", openai_client=fw)
agent = Agent(name="Billing", tools=[gpu_price], model=model)
```

### A real handoff failure we hit, and what measuring it taught us

With a Triage agent handing off to a Billing agent, one run had **Billing reply "the only tool I
have is a billing transfer"** and never call its pricing tool. The handoff passes the full history,
*including Triage's `transfer_to_billing` tool call*, which plausibly confused the model. A
single rerun with a history filter worked:

```python
from agents import handoff
from agents.extensions import handoff_filters
Agent(..., handoffs=[handoff(billing, input_filter=handoff_filters.remove_all_tools)])
```

It looked fixed. Then `agents_sdk_demo.py` ran every configuration 5 times:

| Billing prompt | Handoff filter | Tool used |
|---|---|---|
| weak | none | 4/5 |
| weak | remove_all_tools | 4/5 |
| strong ("ALWAYS call gpu_price") | none | 5/5 |
| strong | remove_all_tools | 4/5 |

The failure is **intermittent (about 1 in 5) in every configuration**, and 5 runs can't tell the
fixes apart. The "fix" we saw was luck. Takeaways:

1. **One run proves nothing about an agent.** Before claiming a fix, run each variant 20–50
   times at production temperature and compare rates. Customers will show you one failing
   transcript, and your job is to find out how often it happens.
2. Decide what context crosses an agent boundary on principle (clean briefs, as in
   `04c/multi_agent_router.py`), but **verify the effect with measurements.**
3. When the tool *must* be used, don't rely on prompting at all. Force it with `tool_choice`
   in the specialist (lesson 04b), or validate the output and retry.

### Framework or hand-rolled?

| Hand-rolled (`fwlearn.agent`) | Framework |
|---|---|
| You can see and control every token | Handoffs, tracing, guardrails, sessions built in |
| Easy to debug and optimize for caching | Faster to build multi-agent systems |
| You maintain it | Abstractions can hide what's sent (like the bug above) |

Many teams prototype on a framework and hand-roll the hot path. Either way, you need to
understand the loop well enough to debug the framework.

## Scripts

```powershell
python 04d_mcp_and_frameworks\remote_mcp_responses.py   # Fireworks calls DeepWiki's MCP server for you
python 04d_mcp_and_frameworks\mcp_bridge_agent.py       # spawns mini_mcp_server.py, bridges its tools into an agent
python 04d_mcp_and_frameworks\mini_mcp_server.py        # (normally started by the bridge; speaks JSON-RPC on stdio)
python 04d_mcp_and_frameworks\agents_sdk_demo.py        # OpenAI Agents SDK: tools + handoff, failure rate per config
```

## Exercises

1. Add a `read_lesson` tool to `mini_mcp_server.py`. The bridge picks it up automatically,
   with no agent code changes. That's the point of MCP.
2. Register `mini_mcp_server.py` with an MCP host you use (Claude Code: `claude mcp add`), and
   use your course tools from there.
3. Port `04c/multi_agent_router.py` to LangGraph or PydanticAI with Fireworks models. Compare
   lines of code and tokens per request.
4. In `remote_mcp_responses.py`, continue the conversation with `previous_response_id` and count
   how many prompt tokens you save compared with resending history.
