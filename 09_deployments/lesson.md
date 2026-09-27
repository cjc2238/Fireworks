# 09 · Dedicated Deployments

> 💸 **This lesson costs money.** You pay per GPU-second while replicas are running. Always run
> `teardown` at the end. Set `--min-replica-count 0` so an idle deployment scales to zero.

## Why deploy?

1. **Custom / fine-tuned models** (LoRA or full weights) aren't served on serverless.
2. **Predictable performance:** no noisy neighbours, and you can tune for latency or throughput.
3. **Cost at scale:** at high, steady utilization, GPU-hours beat per-token pricing.
4. **Control:** region (data residency), hardware, quantization, speculative decoding.

## Core concepts

- **Deployment shape:** a pre-validated configuration (GPU type/count, precision, speculation)
  for a model. Built-in shapes: **`default`, `fast`, `throughput`, `minimal`**. Always pass
  `--deployment-shape`, because most failed deployments are ones created without a shape.
  See which shapes a model supports:
  `firectl deployment-shape-version match --model <model-id>`
- **Accelerators:** NVIDIA A100 80GB, H100 80GB, H200 141GB, B200 180GB, B300 288GB; AMD
  MI325X, MI350X.
- **Regions:** `GLOBAL` (default), `US`, `EUROPE`, `APAC`. **Placement is fixed at creation.**
  To move a deployment you have to recreate it.
- **Preemptible (`--preemptible`):** cheaper, but can be interrupted mid-request. Use it only for
  eval/batch work. It can't be toggled after creation.

## Autoscaling

| Flag | Default | Notes |
|---|---|---|
| `--min-replica-count` | 0 | 0 = scale to zero |
| `--max-replica-count` | 1 | |
| `--scale-up-window` | 30s | |
| `--scale-down-window` | 10m | |
| `--scale-to-zero-window` | 1h | min 5m |
| `--load-targets` | `default=0.8` | or `concurrent_requests=5`, `tokens_generated_per_second=150`, `requests_per_second=`, `prompt_tokens_per_second=` |

**Cold starts:** a scaled-to-zero deployment returns **503 `DEPLOYMENT_SCALING_UP`**, and
requests are **not queued**, so clients need retry logic (lesson 07). Deployments with
min-replicas 0 are **auto-deleted after 7 days with no traffic.**

## Speculative decoding

A small **draft model** proposes N tokens and the big model verifies them in one forward pass.
Accepted tokens are "free", so decode speeds up and **output quality is unchanged**. Flags:
`--draft-model`, `--draft-token-count` (start at 4), or `--ngram-speculation-length` (reuses
patterns from the prompt, which is great for code edits and RAG with lots of copying). Draft and
n-gram are mutually exclusive. A poorly matched drafter *slows things down*, so benchmark.

## Querying a deployment

```python
model="accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>"          # by deployment
model="accounts/fireworks/models/<model>#accounts/<ACCOUNT>/deployments/<ID>"  # model on a specific deployment
```

## Walkthrough (≈30–60 min; costs a few dollars on a small model)

```powershell
# 1. Pick a SMALL model and see its shapes
firectl deployment-shape-version match --model accounts/fireworks/models/qwen3-8b

# 2. Create with scale-to-zero
firectl deployment create accounts/fireworks/models/qwen3-8b `
  --deployment-shape minimal --min-replica-count 0 --max-replica-count 1 `
  --scale-to-zero-window 10m --deployment-id learn-qwen3-8b --wait

# 3. Inspect it
python 09_deployments\manage.py list
python 09_deployments\manage.py get learn-qwen3-8b

# 4. Query and benchmark it
python 09_deployments\query_deployment.py learn-qwen3-8b
python 07_production\benchmark.py --model accounts/<ACCOUNT_ID>/deployments/learn-qwen3-8b --concurrency 1 4 16

# 5. TEAR DOWN
firectl deployment delete learn-qwen3-8b
```

(If `qwen3-8b` isn't available, pick any small model from `list_models.py` or the model library.)

## The serverless-vs-dedicated break-even (you'll be asked this)

```
serverless $/month  = tokens_per_month × blended_price_per_token
dedicated  $/month  = replicas × GPUs_per_replica × $/GPU-hr × hours_up
break-even when     tokens_per_month ≈ dedicated $/month ÷ blended price
```

`breakeven.py` computes this from inputs you give it. The deeper point: a dedicated GPU's
*throughput* depends on the prompt/output mix and concurrency, so you have to **benchmark** to get
the tokens-per-GPU-hour number. Don't guess it.

## Exercises

1. Deploy the same model with `minimal` and `throughput` shapes (one after the other, to save
   money). Benchmark both at concurrency 1 and 16. Explain the difference.
2. Hit your scaled-to-zero deployment after it's idle. Capture the 503, then confirm your
   `resilient_client.py` survives it.
3. Make a break-even chart for a customer doing 2B tokens/month.
4. Read [Routers](https://docs.fireworks.ai/deployments/routers) and
   [Reserved capacity](https://docs.fireworks.ai/deployments/reservations). When would you
   recommend each?
