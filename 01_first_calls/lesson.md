# 01 · Your First Calls

**Goal:** make requests to Fireworks in every way a customer might, and understand what comes back.

## The request

A chat completion needs just two things: a `model` and a list of `messages`.

```python
from fireworks import Fireworks
client = Fireworks()                     # reads FIREWORKS_API_KEY
resp = client.chat.completions.create(
    model="accounts/fireworks/models/glm-5p3-flash",
    messages=[{"role": "user", "content": "Say hello in Spanish"}],
)
print(resp.choices[0].message.content)
```

Message roles: `system` (instructions), `user`, `assistant` (the model's earlier replies) and
`tool` (tool results, covered in lesson 04). **The API is stateless.** You resend the whole
conversation every turn. That fact drives cost, latency, and prompt caching (lesson 07).

## The response

| Field | Meaning |
|---|---|
| `choices[0].message.content` | the text |
| `choices[0].finish_reason` | `stop` (natural end), `length` (hit `max_tokens`), `tool_calls` |
| `usage.prompt_tokens / completion_tokens` | what you're billed for |
| `id`, `model`, `created` | useful when you file a support ticket |

**Gotcha:** the default `max_tokens` is 2048. If `finish_reason == "length"`, your answer was
truncated. Customers report this as "the model cuts off", and it's one of the most common support
questions.

## ⚠ Most current models think before they answer

Many models on Fireworks (DeepSeek V4, Kimi K3, GLM 5.x, and others) **reason by default**. They
spend hidden output tokens thinking (returned in `message.reasoning_content`, counted in
`usage.completion_tokens_details.reasoning_tokens`) before writing `content`. What we measured
while building this course:

| Same streaming prompt | TTFT | Output tokens |
|---|---|---|
| default (thinking on) | **15.4 s** | 1,594 |
| `reasoning_effort="none"` | **0.96 s** | 164 |

Two consequences:
- With a small `max_tokens`, the thinking can use up the whole budget and `content` comes back
  **empty** with `finish_reason="length"`. That's a very common confused-customer ticket.
- Pass `reasoning_effort="none"` for fast or simple calls. Some models are *thinking-only* and
  reject `"none"` with a 400 (for example GLM 5.3), so check the model page.

Scripts in this course that want quick plain answers pass `reasoning_effort="none"` explicitly.

## Streaming

With `stream=True` you get chunks as tokens are generated. The user sees the first word within a
few hundred ms instead of waiting for the whole answer. Fireworks includes `usage` in the **final
chunk** (the OpenAI SDK doesn't do this by default, so it's a Fireworks extension).

## Three clients, one API

| Script | Client | Why it matters |
|---|---|---|
| `hello.py` | Fireworks SDK | Native. Fireworks-only params (`top_k`, `min_p`, ...) are plain kwargs. |
| `openai_compat.py` | OpenAI SDK | Migration takes 2 lines. Fireworks-only params go in `extra_body={...}`. |
| `raw_http.py` | `requests` | What's actually on the wire. Essential for debugging. |

## Scripts (run from repo root)

```powershell
python 01_first_calls\hello.py
python 01_first_calls\streaming.py
python 01_first_calls\chat_cli.py          # interactive multi-turn chat, type 'quit' to exit
python 01_first_calls\openai_compat.py
python 01_first_calls\raw_http.py
python 01_first_calls\async_concurrency.py # 10 requests: sequential vs concurrent
```

## Exercises

1. Change `chat_cli.py` so it prints the running token total after each turn. Watch prompt tokens
   grow as the conversation gets longer. Why is that?
2. Set `max_tokens=15` and ask for an essay. Confirm `finish_reason == "length"`.
3. In `async_concurrency.py`, go from 10 to 50 requests. When do you start seeing `429`s? Read
   <https://docs.fireworks.ai/serverless/rate-limits>.
4. Write the same request as a `curl` command (PowerShell: `curl.exe`). Being able to reproduce a
   customer's request from scratch is a core support skill.
5. Call the **Anthropic-compatible** endpoint using the `anthropic` SDK with
   `base_url="https://api.fireworks.ai/inference"`. (See
   <https://docs.fireworks.ai/tools-sdks/anthropic-compatibility>.)
