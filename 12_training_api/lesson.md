# 12 · The Training API, Preference Tuning & the Platform Frontier

This lesson covers the advanced surface area: what Fireworks is investing in *now*. Being fluent in
it signals that you're ready for the job, not just that you can use the product.

## 1. Preference tuning: DPO and ORPO

When the model is *correct but not quite right* (too verbose, wrong tone, occasional
hallucination), show it **pairs** of better and worse answers.

```json
{"input": {"messages": [{"role": "user", "content": "Explain LoRA"}]},
 "preferred_output": [{"role": "assistant", "content": "<concise, correct answer>"}],
 "non_preferred_output": [{"role": "assistant", "content": "<rambling answer>"}]}
```

```powershell
firectl dpo-job create --loss-method DPO --base-model <model> --dataset <dataset> --output-model <new-id>
firectl dpo-job create --loss-method ORPO --orpo-lambda 0.1 ...   # ORPO: no reference model needed
```

- **DPO** compares the policy against a frozen reference model.
- **ORPO** combines SFT and preference losses without a reference model, which makes it cheaper.
- Only single-turn examples are supported for now. See `make_dpo_pairs.py`.

## 2. The Training API: your own training loop on Fireworks GPUs

The managed jobs (SFT/DPO/RFT) are fixed recipes. The **Training API** lets you write the loop in
plain Python on your laptop while forward/backward passes run on Fireworks GPUs. It follows a
Tinker-style design:

| Call | What it does |
|---|---|
| `FiretitanServiceClient(...)` | Connect to serverless (or dedicated) training |
| `service.create_lora_training_client(base_model, rank)` | Start a LoRA training run |
| `training_client.forward_backward(datums, loss)` | Compute gradients on a batch of **Datums** (tokens + per-token weights) |
| `training_client.optim_step(AdamParams(...))` | Apply the update |
| `save_weights_for_sampler(name)` → `create_sampling_client(...)` → `sample(...)` | Generate from the current weights, e.g. for RL rollouts |
| `save_state` / `load_state_with_optimizer` | Checkpoint and resume |
| `FireworksClient.promote_session_checkpoint(...)` | Turn a checkpoint into a deployable model |

**Every remote call returns a future. Call `.result()`**, or errors stay silent.

- **Serverless training:** pooled GPUs, per-token billing, LoRA only. Supports SFT, DPO, ORPO, RL,
  distillation and custom losses.
- **Dedicated training:** your own trainer GPUs, time-billed, LoRA **or full-parameter**.
- Install: `pip install "fireworks-ai[training]"` (Python 3.11+).
- Start from the **Cookbook** recipes (SFT, DPO, RL/GRPO, agentic RL, distillation), not a blank
  file: <https://docs.fireworks.ai/fine-tuning/training-api/cookbook/overview>

`training_api_tour.py` has the verified skeleton from the docs, annotated. It needs
tokenizer/datum setup from the cookbook before it will run.

## 3. Bring your own model

```powershell
firectl model create my-model C:\path\to\hf-model\          # config.json + safetensors + tokenizer
firectl model get my-model                                   # wait for READY
firectl deployment create accounts/<ACCOUNT_ID>/models/my-model --deployment-shape default --wait
```

Supported architectures include the Llama, Qwen, DeepSeek, Mistral/Mixtral, Gemma and Phi
families, among others. Related topics: **quantization** (FP8/FP4 precisions, trading a little
quality for a lot of speed; read <https://docs.fireworks.ai/models/quantization> and
<https://docs.fireworks.ai/models/model-quality>).

## 4. Routing and developer products

- **FireRouter** picks a model *per turn* based on your routing preferences, trading quality
  against cost. Model IDs look like `firerouter/<family>`. You pay for the model that serves each
  turn.
- **Fireworks Nexus / FireConnect** connect coding harnesses (Claude Code, Codex, and others), LLM
  gateways and SDKs to Fireworks and other providers, with usage/cost tracking and spend limits.
- **Anthropic-compatible API** at `https://api.fireworks.ai/inference` (the Messages API) and
  NIM compatibility.
- **Agent Skills** (`fireworks-training`) let coding agents drive training jobs. There's also a
  **Docs MCP server**.

## 5. Enterprise topics (expect customer questions)

Zero data retention, data residency, SSO, service accounts, audit logs, BYOB (bring your own
bucket), CMEK, secure RFT, BYOC (bring your own cluster), Microsoft Foundry integration,
quotas, and usage/cost export. Skim each docs page once so you know *what exists* and where to
find it.

## Exercises

1. Build a DPO dataset with `make_dpo_pairs.py` (concise vs verbose answers), train with ORPO, and
   measure the average output length before and after.
2. Work through the Training API **SFT cookbook** end to end, then modify the loss.
3. Upload a small HF model (for example a 0.5B–1B Qwen) with `firectl model create` and deploy it.
4. Write a one-page explainer for a customer: "SFT vs DPO vs RFT vs the Training API: which
   should I use?" (It doubles as interview prep.)
