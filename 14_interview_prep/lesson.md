# 14 · Interview Prep: Internals, Scenarios & Drills

Technical roles at an inference company test three things: **(1) do you understand how LLM
inference works under the hood, (2) can you debug and solve customer problems on the platform,
and (3) can you communicate trade-offs clearly.** This lesson covers all three.

> Verify company facts (leadership, funding, latest launches) on <https://fireworks.ai/blog> and
> the company's About page shortly before any interview. They change quickly.

---

## Part 1: How fast inference works (know each at "whiteboard" depth)

| Technique | One-line explanation | What it improves |
|---|---|---|
| **Continuous batching** | Add and remove requests from the running batch each decode step instead of waiting for a whole batch to finish | Throughput and GPU utilization |
| **Paged KV cache** | Store the KV cache in fixed-size blocks, like virtual memory, to avoid fragmentation | Concurrency (more sequences fit) |
| **Prompt / prefix caching** | Reuse KV blocks for shared prefixes across requests | TTFT, cost |
| **Custom attention kernels** (Fireworks' *FireAttention*) | Hand-optimized GPU kernels for attention and MoE, tuned for low precision | Latency and throughput |
| **Quantization** (FP8, FP4) | Lower-precision weights/activations/KV. Decode is memory-bound, so fewer bytes means faster decode. | Speed, memory, cost; costs a little quality (measure it) |
| **Speculative decoding** | A draft model or n-grams propose k tokens and the target verifies them in one pass. Output is identical. | Decode latency, especially at low batch sizes |
| **Tensor / pipeline / expert parallelism** | Split one model across GPUs by layer slices, layer groups, or MoE experts | Fitting big models; latency |
| **Disaggregated prefill/decode** | Run compute-bound prefill and memory-bound decode on separate GPU pools | Tail latency under mixed load |
| **Multi-LoRA serving** | One base model in memory, many small adapters batched together | Serving hundreds of fine-tunes cheaply |
| **Constrained decoding** | Mask invalid tokens against a grammar or schema at each step | 100% valid structured output |

**Practice:** explain each one out loud in 60 seconds, then in 5 minutes with a diagram.

### Core mental models

- **Prefill is compute-bound, decode is memory-bandwidth-bound.** Each decode step reads *all* the
  weights to produce one token per sequence. Batching amortizes that read across sequences.
- **The latency/throughput trade-off:** bigger batches give more total tokens/sec but slower
  per-user tokens/sec. Deployment shapes (`fast` vs `throughput`) encode that choice.
- **KV cache size ≈ 2 × layers × kv_heads × head_dim × bytes × tokens.** Long contexts eat GPU
  memory, which is why GQA/MQA/MLA exist.
- **MoE models** (DeepSeek, Kimi, Qwen MoE): the total parameter count sets memory needs, but only
  the *active* parameters per token set compute.

## Part 2: Technical questions to rehearse

1. A customer's TTFT is 3 s on a 20k-token prompt. Walk through how you'd diagnose it and what
   you'd suggest.
2. What's the difference between `json_object` and `json_schema`, and how does constrained
   decoding work?
3. When would you recommend serverless, dedicated, batch or reserved capacity?
4. Explain LoRA. Why can Fireworks serve many LoRAs on one deployment?
5. SFT vs DPO vs RFT: pick one for (a) a brand voice, (b) SQL accuracy, (c) "stop being verbose".
6. How does speculative decoding keep output quality the same? When does it hurt?
7. A deployment returns 503s every morning. Why? (Scale-to-zero cold start. Fix: min replicas
   ≥ 1 during business hours, client retries.)
8. Design a system that serves 500 per-customer fine-tunes. (Multi-LoRA, adapter caching,
   routing, cost model.)
9. How would you evaluate whether FP8 quantization is acceptable for a customer?
10. How do you make prompt caching effective for a multi-tenant chatbot?

## Part 3: Customer scenarios (practice these as role-plays)

For each one: ask clarifying questions → form a hypothesis → decide what to measure →
recommend → follow up.

**Scenario 1: "Your API is slower than OpenAI."**
Ask about the model, prompt/output lengths, streaming, region, concurrency, and whether they measure
TTFT or total time. Measure with `benchmark.py` and the `fireworks-*` headers. Levers: streaming,
the Fast mode model, prompt caching with session affinity, a smaller model, lower
`reasoning_effort`, a dedicated `fast` shape, the closest region, client keep-alive.

**Scenario 2: "Structured output returns garbage / never finishes."**
Likely causes: the schema isn't described in the prompt (the model emits whitespace until it hits
`max_tokens`), truncation (`finish_reason=length`), unsupported schema features (external `$ref`),
or a reasoning model with `json_schema` so there's no reasoning. Walk through the fixes from
lesson 03.

**Scenario 3: "Our fine-tuned model is worse than the base."**
Check the data (duplicates, label noise, wrong roles, assistant weights = 0 → Render Samples), the
epochs (overfitting), the eval (same held-out set? same prompt as training?), and the serving setup
(the right model ID? multi-LoRA vs merged?).

**Scenario 4: "We're getting 429s at launch."**
Serverless rate limits and quotas: backoff with jitter, request a quota increase, Priority tier,
batch for offline work, a dedicated deployment for predictable capacity.

**Scenario 5: "We need EU data residency and zero retention."**
Point to regional deployments (`--region EUROPE`, fixed at creation), zero data retention,
data residency and enterprise docs. Loop in the account team for contracts.

**Scenario 6: "Is it cheaper to self-host on our own H100s?"**
Build the total-cost-of-ownership comparison: GPU cost, utilization, engineering time, kernel
optimizations (tokens per GPU-hour), autoscaling, reliability. Mention BYOC as a middle ground.

**Scenario 7: "Our agent works in the demo but fails in production."**
Ask for traces (you need steps, tool errors and tokens; 04e). Check tool descriptions and schemas
(04b), `tool_choice` misuse (invented arguments), context growth and truncation, the handoff
context between agents (04d), and whether they measured over many runs or just one. Propose an
outcome-based eval suite before changing anything.

**Scenario 8: "Is our agent safe to give email and database access?"**
Prompt injection through tool results, least privilege (split read and write agents),
code-level policy checks, approvals for irreversible actions, and tested guards (04e).

### Agent questions to rehearse

1. Walk through the tool-calling protocol message by message, including parallel calls.
2. Why does forcing `tool_choice` on an unrelated request produce invented arguments?
3. When would you choose a workflow over an agent? Give an example of each.
4. How would you evaluate an agent? Why is judging the transcript not enough?
5. What is MCP? Explain `tools/list` and `tools/call`, and remote vs local execution.
6. How do you reduce the cost and latency of a 10-step agent? (Caching the stable prefix,
   parallel tools, small model for easy steps, concise tool results, step budgets.)
7. How would you use Fireworks fine-tuning to make an agent cheaper? (SFT on successful
   trajectories, RFT with the eval as reward.)

## Part 4: Hands-on drills (time yourself)

- **15 min:** from an empty folder, build a streaming chatbot with tool calling on Fireworks.
- **15 min:** given a customer's failing `curl`, find the bug. (Have a friend break one: wrong
  model path, missing `Bearer`, `max_tokens` too big, bad schema.)
- **30 min:** produce a model-comparison table (quality, latency, cost) for a task you're given.
- **45 min:** SFT data prep → validate → launch a job, explaining each step as you go.

## Part 5: Reading list

- Fireworks blog: engineering posts on FireAttention, speculative decoding, quantization quality,
  RFT, and serving MoE models. <https://fireworks.ai/blog>
- vLLM PagedAttention paper (Kwon et al., 2023): paged KV cache
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
- LoRA (Hu et al., 2021) and S-LoRA / Punica (multi-LoRA serving)
- DPO (Rafailov et al., 2023), GRPO (DeepSeekMath, 2024)
- FlashAttention 1/2/3 (Dao et al.)
- Orca: continuous batching (Yu et al., 2022)
- DistServe / Splitwise (disaggregated prefill/decode)

## Part 6: Your story

Prepare a 2-minute answer to "Why Fireworks?" that ties together:
1. What you built in this course (link to the capstones).
2. A specific Fireworks capability that impressed you, and why it matters to customers.
3. How your background makes you good at the *customer* half of the job.

## Final self-check: are you "expert" level?

- [ ] I can explain every row of the Part 1 table without notes
- [ ] I've used chat, streaming, tools, structured output, vision, embeddings, rerank, batch
- [ ] I've created, benchmarked, autoscaled and deleted a dedicated deployment
- [ ] I've run SFT end to end and measured the improvement
- [ ] I've written and red-teamed an RFT evaluator
- [ ] I can do the serverless vs dedicated break-even math on a whiteboard
- [ ] I've shipped two capstones with READMEs and numbers
- [ ] I can role-play all six scenarios confidently
