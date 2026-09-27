# 10 · Supervised Fine-Tuning (SFT)

> 💸 Training jobs and the deployment that serves the result cost money. A small LoRA on a small
> model with a few hundred examples is cheap. Check the
> [cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator) before you launch.

## When to fine-tune (and when not to)

| Try first | Fine-tune when |
|---|---|
| Better prompt, few-shot examples | You need a consistent **format/style/voice** that prompting can't hold |
| RAG (for *knowledge*) | You want a **small model to match a big model** on a narrow task (distillation). That's cheaper and faster. |
| Structured outputs | Prompts are huge (long instructions, many examples). Training that in saves tokens on every call. |

**Fine-tuning teaches behaviour, not facts.** For facts, use RAG. Interviewers love this distinction.

## LoRA in one paragraph

Instead of updating all the weights, **LoRA** trains two small low-rank matrices per layer (rank
*r*, default 8 on Fireworks, up to 32). The adapter is megabytes rather than gigabytes. Because of
that, Fireworks can **serve many LoRAs on one base-model deployment** (multi-LoRA), or merge one
into the weights for zero overhead ("live merge").

## Dataset format (JSONL, OpenAI chat format)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- 3 to 3M examples. Loss is computed on **assistant** tokens.
- Optional `weight` (0/1 per message) to skip training on some assistant turns.
- Optional `reasoning_content` on assistant messages to train thinking traces.
- Tool-calling data: include a `tools` array and assistant `tool_calls`.
- Vision SFT: include base64 images (supported on some models).
- Use **Render Samples** in the dashboard to see tokenization and loss masks. It's the first
  debugging step when training "does nothing".

## Hyperparameters that matter

| Param | Default | Guidance |
|---|---|---|
| `--epochs` | 1 | 1–3. More epochs overfit small datasets. |
| `--learning-rate` | model-dependent | Leave it at the default until you have a reason to change it |
| `--lora-rank` | 8 | 16–32 for harder tasks |
| `--batch-size-samples` | 32 | |
| `--max-context-length` | ≥32k | Lower it if you hit OOM |
| `--warm-start-from` | | Continue training an existing LoRA |

## The full loop

```powershell
# 0. Make data (synthetic, from a bigger "teacher" model), then validate it
python 10_fine_tuning_sft\make_dataset.py         # -> outputs\sft_train.jsonl, outputs\sft_eval.jsonl
python 10_fine_tuning_sft\validate_dataset.py outputs\sft_train.jsonl

# 1. Baseline: how good is the small model WITHOUT tuning?
python 10_fine_tuning_sft\evaluate.py accounts/fireworks/models/<small-model>
python 10_fine_tuning_sft\evaluate.py accounts/fireworks/models/<small-model> --teacher-prompt  # "just prompt it better" baseline

# 2. Upload and train (firectl; the UI "Training" tab works too)
firectl dataset create triage-train outputs\sft_train.jsonl
firectl sftj create --base-model accounts/fireworks/models/<small-model> `
  --dataset triage-train --output-model triage-lora-v1 --epochs 2 --lora-rank 16
firectl sftj get <job-id>             # watch until COMPLETED

# 3. Deploy the LoRA (see lesson 09 for the cost rules)
firectl deployment create accounts/<ACCOUNT_ID>/models/triage-lora-v1 --deployment-shape default --min-replica-count 0

# 4. Evaluate the tuned model on the SAME held-out set
python 10_fine_tuning_sft\evaluate.py accounts/<ACCOUNT_ID>/models/triage-lora-v1

# 5. Tear down
firectl deployment delete <deployment-id>
```

### Serving options for LoRAs

| | Live merge (single LoRA) | Multi-LoRA |
|---|---|---|
| Create | `firectl deployment create accounts/<you>/models/<lora>` | base deployment with `--enable-addons` (BF16 shape), then `firectl load-lora <lora> --deployment <id>` |
| Query | `accounts/<you>/models/<lora>` | `accounts/<you>/models/<lora>#accounts/<you>/deployments/<id>` |
| Perf | Same as base model | Some per-request overhead |
| Use | Production, one model | A/B tests, per-customer adapters, many variants |

## Exercises

1. Run the full loop. Report baseline vs tuned accuracy, latency, and $/1k requests compared
   with the big teacher model. **This is a portfolio piece.**
2. Train two variants (rank 8 vs 16, or 1 vs 3 epochs). Serve both with multi-LoRA and compare.
3. Add 20% deliberately bad labels. How much does accuracy drop? (It teaches you why data quality
   comes first.)
4. Read about DPO/ORPO (<https://docs.fireworks.ai/fine-tuning/dpo-fine-tuning>). What data would you
   need to fix a model that's accurate but too verbose?
