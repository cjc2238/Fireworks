# 00 · Setup & the Fireworks Mental Model

## What Fireworks is, in one paragraph

Fireworks AI is an **inference and customization platform for open models** (DeepSeek, Kimi, GLM,
Qwen, Llama, gpt-oss, and others). You send requests to an API and Fireworks runs the model on
GPUs it has heavily optimized: its own CUDA kernels (FireAttention), quantization, speculative
decoding and multi-LoRA serving. On top of inference it offers **fine-tuning** (SFT, DPO, RFT, and a
low-level Training API), **batch processing**, and **dedicated deployments** you control. The pitch
to customers: *frontier-quality open models, faster and cheaper than running them yourself, and you
can customize them with your own data.*

## The five nouns you must know

| Concept | What it is | Name format |
|---|---|---|
| **Account** | Your org. Billing and quotas live here. | `accounts/<ACCOUNT_ID>` |
| **Model** | A set of weights: a base model or a LoRA addon | `accounts/fireworks/models/<id>` (public) or `accounts/<you>/models/<id>` (yours) |
| **Deployment** | GPUs running a model just for you | `accounts/<you>/deployments/<id>` |
| **Deployed model** | A specific model loaded into a specific deployment | `<model>#<deployment>` |
| **Dataset** | Uploaded JSONL for training or batch jobs | `accounts/<you>/datasets/<id>` |

Resource IDs are 1–63 characters, lowercase letters, digits and hyphens, and can't start with a digit.

## Two ways to run a model

| | **Serverless** | **Dedicated (on-demand) deployment** |
|---|---|---|
| Setup | None; just call the model ID | Create a deployment (lesson 09) |
| Billing | Per token | Per GPU-second while replicas are up |
| Hardware | Shared, managed by Fireworks | Yours: pick GPU type, count, region, shape |
| Custom/LoRA models | Generally no | Yes |
| Best for | Prototyping, spiky or low volume | Steady high volume, latency SLAs, fine-tunes |

Serverless also has **modes**: Standard (default), **Priority** (`service_tier="priority"`, less
likely to be load-shed), and **Fast** (a `-fast` router model ID aiming for 100+ tokens/sec).

## Two planes

- **Data plane** `https://api.fireworks.ai/inference/v1`: inference. It's OpenAI-compatible
  (and Anthropic-compatible at `/inference`).
- **Control plane** `https://api.fireworks.ai/v1/accounts/...`: create and manage models,
  deployments, datasets and jobs. You drive it through the `firectl` CLI, the SDK, or REST.

Knowing this split is the first step in debugging most customer issues. *"Is this an inference
problem or a resource-management problem?"*

## Setup steps

1. Sign up at <https://fireworks.ai> and create an API key.
2. Follow the setup block in the [root README](../README.md).
3. Run:
   ```powershell
   python 00_setup\check_setup.py     # verifies key, prints your account ID
   python 00_setup\list_models.py     # lists every model live on serverless right now
   ```
4. Put your account ID into `.env` as `FIREWORKS_ACCOUNT_ID`.
5. **Install `firectl`** (you'll need it from lesson 08 on). On Windows:
   ```powershell
   Invoke-WebRequest https://storage.googleapis.com/fireworks-public/firectl/stable/firectl.exe -OutFile firectl.exe
   .\firectl.exe signin
   .\firectl.exe whoami
   ```
   Move `firectl.exe` somewhere on your `PATH` (for example `C:\Users\admin\bin`).

## Exercises

1. Pick 3 models from `list_models.py`. For each, open its page in the
   [model library](https://app.fireworks.ai/models) and note: context length, price per 1M
   input/output tokens, and whether it supports tools, vision or reasoning.
2. In your own words, write down why a customer would move from serverless to a dedicated
   deployment. Give three distinct reasons. (This is a very common interview question.)
3. Explore the dashboard: Usage, API keys, Deployments, Datasets, Fine-tuning. Knowing the UI
   helps you walk customers through it.
