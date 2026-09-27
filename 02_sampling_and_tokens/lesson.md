# 02 · Sampling, Tokens & Observability

**Goal:** understand *how* a model picks each token, control that choice, and measure performance
the way Fireworks engineers do.

## How generation works (the 60-second version)

1. Your prompt is **tokenized**. A token is roughly ¾ of an English word.
2. **Prefill:** the model processes all prompt tokens in parallel and builds the **KV cache**.
   Prefill cost scales with prompt length and sets **time-to-first-token (TTFT)**.
3. **Decode:** the model generates one token at a time. Each step outputs a probability
   distribution over the whole vocabulary, a *sampler* picks one token, and it's appended.
   Decode speed is **tokens/sec (TPS)**, also called inter-token latency.
4. It stops at an end-of-sequence token, a `stop` string, or `max_tokens`.

Prefill is **compute-bound** and decode is **memory-bandwidth-bound**. Most of Fireworks'
engineering (lesson 14) is about making one or both of these faster.

## Sampling knobs

Fireworks applies each model's recommended defaults (from its HuggingFace `generation_config.json`)
unless you override them.

| Param | Effect | Typical |
|---|---|---|
| `temperature` | Flattens (>1) or sharpens (<1) the distribution. 0 ≈ greedy. | 0–0.3 extraction/tools, 0.7–1.0 creative |
| `top_p` | Keep the smallest set of tokens whose probabilities sum to p | 0.9–0.95 |
| `top_k` | Keep only the k most likely tokens *(Fireworks extension)* | 40–100 |
| `min_p` | Drop tokens below p × (top token prob) *(extension)* | 0.05 |
| `frequency_penalty` / `presence_penalty` | Discourage repetition (OpenAI-style) | 0–1 |
| `repetition_penalty` | Multiplicative penalty over prompt + output *(extension)* | 1.0–1.2 |
| `stop` | Strings that end generation | |
| `n` | Several completions in one request | |
| `logprobs`, `top_logprobs` | Return per-token probabilities | debugging, classifiers, confidence |
| `max_tokens` | Upper bound on output length (default 2048) | |

Other Fireworks extras: `echo`, `return_token_ids`, `raw_output` (shows the exact templated
prompt, which is great for debugging chat templates), `ignore_eos` (benchmarking only),
`logit_bias`, `mirostat_target`/`mirostat_lr`, `perf_metrics_in_response`.

## Observability

- **Non-streaming:** performance data is in **response headers** (`fireworks-prompt-tokens`,
  `fireworks-server-time-to-first-token`, `fireworks-cached-prompt-tokens`,
  `fireworks-sampling-options`, ...). Read them with `.with_raw_response`.
- **Streaming:** pass `perf_metrics_in_response=True` and read `perf_metrics` on the final chunk.

Why it matters: when a customer says "it's slow", you first split the time into
**network → queue → prefill (TTFT) → decode (TPS)**. Each has a different fix.

## Scripts

```powershell
python 02_sampling_and_tokens\temperature_lab.py   # same prompt at several temperatures
python 02_sampling_and_tokens\logprobs_classifier.py
python 02_sampling_and_tokens\perf_headers.py
python 02_sampling_and_tokens\raw_prompt.py        # see the chat template the model actually receives
```

## Exercises

1. In `temperature_lab.py`, add `top_k=1`. Is it the same as `temperature=0`? Why might it
   differ slightly? (Hint: numerics and batching.)
2. Extend `logprobs_classifier.py` to report a *confidence score* and route low-confidence items
   to a bigger model. This "model cascade" pattern saves customers real money.
3. Using `perf_headers.py`, compare TTFT for a 50-token prompt and a 5,000-token prompt. Plot it.
   Then send the long prompt twice and look at `fireworks-cached-prompt-tokens` on the second call.
4. Read the raw prompt from `raw_prompt.py`. Identify the special tokens the chat template adds.
