# 05 · Reasoning, Vision, Audio & the Responses API

## Reasoning models

Models like DeepSeek, Kimi, GLM and Qwen "thinking" variants produce a hidden chain of thought
before answering. On Fireworks:

- The thinking comes back in **`message.reasoning_content`**, separate from `content`. When
  streaming, it's in `delta.reasoning_content`.
- Control it with **`reasoning_effort`**: `"low" | "medium" | "high"`, **or** the
  Anthropic-style `thinking={"type": "enabled", "budget_tokens": N}` (N ≥ 1024). Don't send both.
- **Multi-turn with tools:** pass the full assistant `message` object back so its reasoning is kept
  (interleaved thinking). Use `reasoning_history="preserved"` to keep reasoning across *user*
  turns too.
- Reasoning tokens are **output tokens**, so you pay for them and they add latency. Higher effort
  means better answers on hard problems but is slower and more expensive. Picking the effort level
  per task is a cost conversation you'll have with customers.

## Vision

Send `content` as a list mixing `{"type":"text"}` and `{"type":"image_url"}` parts. Images can be
URLs or `data:image/jpeg;base64,...`.

Limits: **30 images per request**, base64 total **< 10 MB**, each URL **< 5 MB** and must download
within **1.5 s**. Formats: png, jpg/jpeg, gif, bmp, tiff, ppm. Images cost tokens too (see the
"How many tokens per image?" FAQ).

**Common ticket:** "vision call times out / fails on my image URL". Usually the URL is slow or
large. Fix: download it yourself and send base64.

## Audio & video input

Omni models (for example Qwen3 Omni, Nemotron Omni) accept audio and video parts. See
<https://docs.fireworks.ai/guides/video-audio-inputs>.

## The Responses API

`client.responses.create(...)` (OpenAI Responses-compatible) is **stateful**: pass
`previous_response_id` instead of resending history. It supports function tools and **remote MCP
tools that Fireworks executes server-side** (`{"type": "sse", "server_url": ...}`). Set
`store=False` to opt out of storage, but then you can't chain responses.

When to use which:
- **Chat Completions:** maximum control, stateless, works with every SDK and framework.
- **Responses:** less plumbing for multi-turn conversations and hosted MCP tools.

## Scripts

```powershell
python 05_reasoning_vision\reasoning.py
python 05_reasoning_vision\vision.py [path\to\image.jpg]
python 05_reasoning_vision\responses_api.py
```

## Exercises

1. Run `reasoning.py` with each `reasoning_effort`. Record tokens, latency and correctness on 5
   hard logic puzzles, then make a table.
2. Build a **receipt scanner**: photo → vision model → `json_schema` output (lesson 03) → CSV.
3. Stream a reasoning model and print thinking in grey and the answer in white.
4. Use the Responses API for a 3-turn conversation and compare prompt tokens billed against the
   same conversation over Chat Completions.
