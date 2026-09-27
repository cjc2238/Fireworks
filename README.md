# Fireworks AI: From Beginner to Expert (Python)

A hands-on curriculum for learning [Fireworks AI](https://fireworks.ai) well enough to work there as a
technical engineer (Solutions / Forward-Deployed / Support / Developer Relations Engineer).

Every lesson has a **`lesson.md`** (concepts, why it matters, exercises) and **runnable Python scripts**.
Read the lesson, run the scripts, then do the exercises. The exercises matter most: interviewers
will check whether you've *built* things.

> **Verified against the Fireworks docs on 2026-09-25.** Fireworks ships fast. When something
> doesn't match, trust [docs.fireworks.ai](https://docs.fireworks.ai) and the
> [changelog](https://docs.fireworks.ai/updates/changelog). Model IDs in particular change often,
> so they live in `.env` rather than in the code.

---

## Setup (10 minutes)

```powershell
cd C:\Users\admin\Desktop\Fireworks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then paste your API key into .env
python 00_setup\check_setup.py
```

Get an API key at <https://app.fireworks.ai/settings/users/api-keys>. New accounts get a small
free credit. Lessons 01–08 cost cents. Lessons 09–12 use dedicated GPUs and training jobs, which
**cost real money**. Each of those lessons says so up front and shows you how to clean up.

---

## The roadmap

| # | Module | You'll be able to… | Cost |
|---|--------|--------------------|------|
| **Stage 1: Foundations** ||||
| 00 | [Setup & mental model](00_setup/lesson.md) | Explain what Fireworks is, find your account, list live models | free |
| 01 | [Your first calls](01_first_calls/lesson.md) | Chat, stream, hold a conversation, go async; use the Fireworks SDK, OpenAI SDK, and raw HTTP | ¢ |
| 02 | [Sampling, tokens & observability](02_sampling_and_tokens/lesson.md) | Control generation, read logprobs, measure TTFT and tokens/sec | ¢ |
| **Stage 2: Building applications** ||||
| 03 | [Structured outputs](03_structured_outputs/lesson.md) | Get guaranteed-valid JSON with JSON Schema, Pydantic and grammars | ¢ |
| 04 | [Tool calling fundamentals](04_tool_calling/lesson.md) | The tool-call protocol; a first agent loop with parallel tools | ¢ |
| **Stage 2b: Agents track** ||||
| 04b | [Advanced tool calling](04b_advanced_tool_calling/lesson.md) | Design schemas, avoid the `tool_choice` trap, stream tool calls, use reasoning models, eval tool selection | ¢ |
| 04c | [Building agents](04c_building_agents/lesson.md) | ReAct, plan-and-execute, routers, verifier loops, human approval, memory, on a runtime you can read | ¢ |
| 04d | [MCP & agent frameworks](04d_mcp_and_frameworks/lesson.md) | Remote MCP via the Responses API; build an MCP server and client; OpenAI Agents SDK | ¢ |
| 04e | [Agents in production](04e_production_agents/lesson.md) | Tracing, outcome-based evals, prompt-injection defense, trajectories → fine-tuning data | ¢ |
| 05 | [Reasoning, vision & the Responses API](05_reasoning_vision/lesson.md) | Use thinking models, images, and stateful conversations | ¢ |
| 06 | [Embeddings, reranking & RAG](06_embeddings_rag/lesson.md) | Build retrieval-augmented generation from scratch | ¢ |
| **Stage 3: Production engineering** ||||
| 07 | [Reliability, performance & cost](07_production/lesson.md) | Retries, timeouts, concurrency, prompt caching, service tiers | ¢ |
| 08 | [Batch inference](08_batch/lesson.md) | Process thousands of prompts at 50% off | ¢–$ |
| 09 | [Dedicated deployments](09_deployments/lesson.md) | Deploy, autoscale, benchmark and tear down GPUs; choose shapes | $$ |
| **Stage 4: Customization** ||||
| 10 | [Supervised fine-tuning (SFT)](10_fine_tuning_sft/lesson.md) | Prepare data, train a LoRA, deploy it, evaluate it | $$ |
| 11 | [Reinforcement fine-tuning (RFT)](11_rft/lesson.md) | Write evaluators and train with rewards instead of labels | $$$ |
| 12 | [Training API & the platform frontier](12_training_api/lesson.md) | Custom training loops, DPO, distillation, custom models, FireRouter | $$$ |
| **Stage 5: Getting hired** ||||
| 13 | [Capstone projects](13_capstones/lesson.md) | Portfolio projects that show you can do the job | varies |
| 14 | [Interview prep: internals & scenarios](14_interview_prep/lesson.md) | Explain *how* Fireworks is fast; handle customer scenarios | free |

### Weekend schedule (about 8 hours a day)

The whole course fits in one weekend. A few steps (batch jobs, fine-tuning runs, deployments
warming up) run in the background for an hour or more, so **start those early** and keep working
on other lessons while they run.

**Saturday: using the API and building agents**

| Block | Modules | Notes |
|---|---|---|
| Morning (~3 h) | 00–03 | Setup, first calls, sampling, structured outputs. Get very comfortable with the basic API. |
| Midday (~3 h) | 04, 04b–04e | Tool calling and the agents track. Most new customer workloads are agents, so give this block your full attention. |
| Afternoon (~2 h) | 05–07, start 08 | Multimodal work, RAG, production skills. **Start the lesson 08 batch job first thing in this block.** It took about 2 hours in our test run, so check its results at the end of the day. |

**Sunday: deploying, fine-tuning, and getting ready to interview**

| Block | Modules | Notes |
|---|---|---|
| Morning (~3 h) | 09, start 10 | Deploy, benchmark, and **delete** a dedicated deployment. Then generate the SFT dataset and **launch the training job** so it trains while you move on. Budget about $20–50 in credits for Sunday. |
| Midday (~2 h) | 11–12, finish 10 | RFT evaluators and DPO/Training API while training finishes. Then deploy and evaluate your fine-tuned model, and tear the deployment down. |
| Afternoon (~3 h) | 13–14 | Build **one** capstone (A is the best fit for a weekend), publish it on GitHub, and rehearse the interview scenarios out loud. |

Short on time? Do each lesson's scripts and save the exercises for later. The exercises and a
second capstone are what turn "completed the course" into a strong portfolio.

---

## How the code is organized

```
fwlearn/            shared helpers: client(), MODEL, auth_headers(), ...
  agent.py          the ~200-line agent runtime used by the agents track (read it!)
  tracing.py        JSONL tracer for agent runs
  control.py        control-plane REST helpers (datasets, jobs, deployments)
00_setup/ ... 14_interview_prep/
  lesson.md         read this first
  *.py              run from the repo root:  python 01_first_calls\hello.py
outputs/            scripts write artifacts here (git-ignored)
```

Three ways to reach Fireworks from Python, all used in this course:

1. **Fireworks SDK:** `from fireworks import Fireworks`. The native client, with Fireworks-only params as normal kwargs.
2. **OpenAI SDK:** `OpenAI(base_url="https://api.fireworks.ai/inference/v1")`. Most customers start here because it's a drop-in replacement.
3. **Raw HTTP:** `requests.post(...)`. Needed for the control plane and useful for debugging customer issues.

## Official resources to keep open

- Docs: <https://docs.fireworks.ai> (the full page index is at <https://docs.fireworks.ai/llms.txt>)
- Model library: <https://app.fireworks.ai/models>
- Cookbooks: <https://docs.fireworks.ai/examples/cookbooks>
- Courses: <https://docs.fireworks.ai/examples/introduction>
- Blog (engineering deep-dives, which are key for interviews): <https://fireworks.ai/blog>
- Docs MCP server for your coding agent: <https://docs.fireworks.ai/ecosystem/integrations/development-setup>
