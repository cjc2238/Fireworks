# 04 · Tool Calling & Agents

**Goal:** let the model call your functions, and build the loop that turns that into an agent.

## The protocol (memorize this)

```
1. You send: messages + tools=[JSON-Schema function definitions]
2. Model replies with finish_reason="tool_calls" and message.tool_calls=[{id, function:{name, arguments(JSON string)}}]
3. YOU execute the function(s). The model never runs code itself.
4. You append the assistant message AND one {"role":"tool","tool_call_id":id,"content":result} per call
5. Call again. Repeat until finish_reason == "stop".
```

**`tool_choice`:** `"auto"` (default, model decides), `"none"`, `"required"` (must call at least
one), or `{"type":"function","function":{"name":"..."}}` to force a specific tool.

**Parallel tool calls:** capable models return *several* tool calls in one turn. Run them
concurrently and return all results before the next call.

## Best practices (from the docs and from experience)

- Use **low temperature (0–0.3)**. It reduces hallucinated arguments.
- **Descriptions are your prompt.** The model chooses tools from `description` fields, so write
  them like docs for a new colleague.
- **Validate the arguments.** `arguments` is a JSON *string* and can be malformed. Parse inside a
  try/except and send the error back as the tool result so the model can fix it.
- **Bound the loop** with a max-iterations guard. Agents can loop forever.
- For **reasoning models**, pass the *full* assistant message object back (it includes
  `reasoning_content`). This preserves "interleaved thinking" between tool calls.
- Streaming works with tools, but arguments arrive in pieces that you have to concatenate.

## Scripts

```powershell
python 04_tool_calling\single_tool.py     # one round-trip, shown step by step
python 04_tool_calling\agent_loop.py      # a reusable agent with 3 tools + parallel calls
```

## Exercises

1. Add a `search_docs(query)` tool to `agent_loop.py` that searches the lesson markdown files in
   this repo (simple keyword match is fine). Ask it "How do I deploy a LoRA?"
2. Make a tool raise an exception. Make sure the agent recovers by sending the error text back.
3. Try `tool_choice="required"` on a question that needs no tool. What happens?
4. Swap the model for 2–3 others from `list_models.py`. Compare tool-selection accuracy on 10 test
   questions and write a small scorecard. Customers constantly ask "which model is best for agents?"
   A data-backed answer is what they need.
5. Look at agent frameworks that support Fireworks (LangGraph, CrewAI, OpenAI Agents SDK, and others):
   <https://docs.fireworks.ai/ecosystem/integrations/agent-frameworks>. Port `agent_loop.py` to one.

## Next: the agents track

This lesson covers the protocol. The agents track goes much deeper:
**[04b Advanced tool calling](../04b_advanced_tool_calling/lesson.md)** →
[04c Building agents](../04c_building_agents/lesson.md) →
[04d MCP & frameworks](../04d_mcp_and_frameworks/lesson.md) →
[04e Agents in production](../04e_production_agents/lesson.md)
