# 04b · Advanced Tool Calling

Lesson 04 covered the basic protocol. This lesson covers what decides whether tool calling works
in production: **schema design, control over tool choice, streaming, reasoning models, and
measuring tool-selection accuracy.** Most "the agent is dumb" tickets turn out to be problems in
one of these areas.

## 1. Tool schemas are prompts

The model picks tools from their `name` and `description`, and fills arguments from the JSON Schema.
Treat all three as prompt engineering.

| Do | Why |
|---|---|
| Verb-first, specific names: `search_orders`, not `orders` | Easier to tell apart when there are many tools |
| Descriptions say **when** to use the tool, not just what it does | "Use for ANY question about an order's status or shipping" |
| `enum` / `Literal` for closed sets | The model can't invent a value outside the list |
| Per-argument `description` with an example | "ISO date, e.g. 2026-09-27" |
| Few, meaningful required args, and sensible defaults | Fewer hallucinated arguments |
| Nested objects and arrays when the domain has them | Fireworks supports `$defs` / `$ref`, including recursive refs |
| **Validate the arguments anyway** (Pydantic) | Schema following is strong, not perfect |

Generate schemas from Pydantic rather than writing JSON by hand, so the schema and the
validation code can't drift apart. `fwlearn.agent.tool` does this from type hints:
`Annotated[str, Field(description=...)]` becomes a documented argument and `Literal[...]` becomes an
`enum`.

**Errors as data:** when validation fails, send the validation errors back as the tool result
(`{"ok": false, "error": "invalid arguments", "details": [...]}`). Models fix their own arguments
on the next turn surprisingly well.

## 2. `tool_choice` and its trap

| Value | Behaviour |
|---|---|
| `"auto"` (default) | The model decides |
| `"none"` | No tool calls |
| `"required"` | Must call at least one tool |
| `{"type":"function","function":{"name":"X"}}` | Must call tool X |

**What we observed while building this lesson:** with `get_weather` forced and the user asking for
a joke, the model called `get_weather(city="Seattle")`. With `"required"` it picked "Phoenix".
**Forcing a tool when it doesn't fit makes the model invent arguments.** Use forcing only when
you *know* the tool applies, for example in a pipeline step, or to use a tool as a structured-output
extractor.

`parallel_tool_calls=False` limits the model to one call per turn. Use it when calls depend on
each other or have side effects that must happen in order.

## 3. Streaming tool calls

When streaming, tool calls arrive as **deltas keyed by `index`**. The `id` and `name` appear
once and `arguments` arrives as JSON *fragments* that you concatenate. Only call
`json.loads` after `finish_reason == "tool_calls"`. Streaming lets a UI show
"🔧 calling search_orders…" instantly instead of freezing.

## 4. Reasoning models and tools

Thinking models reason *between* tool calls ("interleaved thinking"). To keep that reasoning chain
intact, **send back the full assistant message**, including `reasoning_content`, not just
`content` and `tool_calls`. `fwlearn.agent` does this by dumping the whole message. Use
`reasoning_history="preserved"` to keep reasoning across user turns too.

Trade-off: thinking before every step is slower and more expensive, but better at multi-step
planning. A common production setup is a reasoning model for planning and a fast model with
`reasoning_effort="none"` for simple tool steps (lesson 04c, plan-and-execute).

## 5. Measure tool selection; don't guess it

"Which model is best for our agent?" is one of the most common customer questions. Answer it with
a **tool-selection eval**: labeled prompts → expected tool (or no tool) plus required arguments →
accuracy per model. Test the confusable cases: similar tools, questions that need no tool, and
multi-tool questions. More tools means lower accuracy, so measure how accuracy changes with the
number of tools.

## Scripts

```powershell
python 04b_advanced_tool_calling\schema_design.py         # Pydantic -> schema, nested args, validation-error recovery
python 04b_advanced_tool_calling\tool_choice_lab.py       # auto/none/required/forced x relevant/irrelevant question
python 04b_advanced_tool_calling\streaming_tools.py       # live tool-call deltas, then a streamed final answer
python 04b_advanced_tool_calling\reasoning_tools.py       # interleaved thinking with a reasoning model
python 04b_advanced_tool_calling\tool_selection_eval.py   # accuracy scorecard; pass model IDs to compare
```

## Exercises

1. Run `schema_design.py` normally, then with `--no-hint`. With the "15-minute steps" hint the
   model fixes the invalid 50 minutes up front; without it you see the validation → repair loop.
   Then delete the other per-argument descriptions and the `Literal` too. How often do arguments break?
1. In our run, **both** models "failed" the refund case the same way: they called
   `get_order_status` before refunding. Is that a model error or a *label* error? Change the eval
   so a case can have several acceptable answers. Deciding what "correct" means is half of eval design.
2. Extend `tool_selection_eval.py` from 8 to 30 tools (add plausible decoys). Plot accuracy against
   tool count for two models.
3. Build a streaming chat UI in the terminal that shows `🔧 tool(args)` lines as they stream.
4. In `reasoning_tools.py`, strip `reasoning_content` before sending history back. Does quality
   change on a harder 4-step task?
