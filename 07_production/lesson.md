# 07 · Reliability, Performance & Cost

This lesson separates "I used the API" from "I can run it in production". It's what customers pay
Fireworks engineers to help with.

## Reliability

| Status | Meaning | Retry? |
|---|---|---|
| 400 / 422 | Bad request (bad param, context too long, invalid schema) | ❌ fix the request |
| 401 / 403 | Bad key / no permission for model | ❌ |
| 404 | Wrong model ID, or deployment not found | ❌ (check the ID) |
| 429 | Rate limit | ✅ backoff + jitter |
| 500 / 502 / 504 | Transient server error | ✅ exponential backoff |
| 503 | Overloaded, or a **scaled-to-zero deployment waking up** | ✅ backoff (longer for cold starts) |

- The Fireworks SDK retries **2×** by default, with a **60 s** timeout. Configure with
  `Fireworks(max_retries=..., timeout=...)` or per call with `client.with_options(...)`.
- Suggested timeouts: chat 30–60 s, agentic 5–30 min, long-context 10–30 min.
- Exceptions: `fireworks.RateLimitError`, `fireworks.APIConnectionError`,
  `fireworks.APIStatusError` (with `.status_code`).
- **Fallbacks:** if the primary model keeps failing, fail over to a second model or a
  deployment. Serverless **Priority** tier (`service_tier="priority"`) is less likely to be load-shed.
- The Fireworks dashboard only counts requests that reached the server. Log client-side failures
  (DNS, timeouts) yourself.

## Performance

**Latency = network + queue + prefill (TTFT) + decode (output_tokens / TPS)**

| Lever | Helps | How |
|---|---|---|
| Stream | Perceived latency | `stream=True` |
| Shorter prompts | TTFT, cost | trim context, better retrieval |
| **Prompt caching** | TTFT, cost | stable prefix + session affinity (below) |
| Fewer output tokens | Total latency | `max_tokens`, "be concise", lower `reasoning_effort` |
| Smaller/faster model | Everything | Model cascade (lesson 02) |
| **Fast** serverless mode | TPS | `accounts/fireworks/routers/<model>-fast` |
| **Predicted outputs** | Decode for edits | pass the expected output ([docs](https://docs.fireworks.ai/guides/predicted-outputs)) |
| Dedicated deployment | All (you control batching/hardware) | lesson 09 |
| Client-side | Connection reuse | one long-lived client, HTTP keep-alive, async ([docs](https://docs.fireworks.ai/deployments/client-side-performance-optimization)) |

## Prompt caching (it's automatic, but you have to structure for it)

Fireworks reuses the KV cache for the **longest matching prompt prefix**. The rules:

1. **Static first, dynamic last:** system prompt → tools → few-shot examples → *then* the
   user-specific parts.
2. **No timestamps or IDs in the prefix.** One changed token invalidates everything after it.
3. **Session affinity:** send `x-session-affinity: <session-id>` (or the `user` field) so the
   same conversation lands on the same replica, where its cache lives.
4. `x-prompt-cache-isolation-key` keeps caches separate between tenants.
5. Measure: header `fireworks-cached-prompt-tokens` (and `fireworks-prompt-tokens`).
6. Price: serverless cached input tokens are usually **50% off** (it varies by model).

## Cost

- Serverless: `input_tokens × in_price + cached × cached_price + output_tokens × out_price`
- Batch: 50% off serverless (lesson 08)
- Dedicated: `GPU-hours × $/GPU-hr`, which is independent of tokens. Break-even vs serverless
  depends on utilization (lesson 09).
- Track usage per feature/customer in your own logs. Fireworks also exports usage and cost data
  (see Accounts → Exporting Usage Costs).

## Scripts

```powershell
python 07_production\resilient_client.py
python 07_production\prompt_cache_demo.py
python 07_production\cost_tracker.py
python 07_production\benchmark.py --concurrency 1 4 8 --requests 16
```

## Exercises

1. Break things on purpose: wrong model ID, bad key, `max_tokens=10**7`, invalid schema. Record
   each status code and message. This becomes your personal error-code cheat sheet (compare it with
   <https://docs.fireworks.ai/guides/inference-error-codes>).
2. Use `prompt_cache_demo.py` to show a cache hit rate above 80%, then break it by adding a timestamp
   to the system prompt.
3. Run `benchmark.py` against two models and write a one-page recommendation for a hypothetical
   customer who needs p95 TTFT < 500 ms at 10 concurrent users.
