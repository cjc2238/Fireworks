# 13 · Capstone Projects

Hiring managers for technical, customer-facing roles want evidence that you can **take a vague
customer problem, build a working solution on the platform, measure it, and explain the
trade-offs.** Build at least **two** of these and publish them on GitHub. Each should have a README
with an architecture diagram, results table, cost analysis and "what I'd do next".

Each capstone reuses code from earlier lessons.

---

## A. "Migrate me off OpenAI" (★ recommended; this is the day job)

**Scenario:** a startup runs a support chatbot on a closed model and wants to cut cost by 70% or
more without losing quality.

1. Build a 100-question eval set with a grading rubric (LLM-as-judge plus exact checks).
2. Benchmark 3–4 Fireworks serverless models for quality, p50/p95 latency and $/1k conversations
   (lessons 02, 07).
3. Add prompt caching and measure the savings.
4. Write the migration guide: the 2-line SDK change, parameter mapping, gotchas (lesson 01
   `extra_body`, lesson 03 schema caveats).
5. **Deliverable:** a recommendation memo with a table like the one below.

| Model | Quality | p95 TTFT | $/1k convos | Notes |
|---|---|---|---|---|

## B. Distill a big model into a small fast one

1. Pick a narrow task (triage, extraction, SQL generation).
2. Teacher (big model) → synthetic dataset → SFT a small model (lesson 10).
3. Optionally, RFT on top with an evaluator (lesson 11).
4. Deploy with multi-LoRA and serve **v1 and v2 side by side** for an A/B test.
5. **Deliverable:** accuracy / latency / cost for teacher, base student and tuned student.

## C. Production RAG service

1. FastAPI service: `/ingest` and `/ask`, with embeddings, a vector DB, reranking and a
   generator with citations (lesson 06).
2. Streaming responses, retries, timeouts, fallbacks (lesson 07).
3. A recall@k eval and an answer-faithfulness eval.
4. A load test at 1/10/50 concurrency (`07_production/benchmark.py` pattern).
5. **Deliverable:** a Dockerized repo and a latency breakdown chart (retrieval vs rerank vs generation).

## D. Agent with a verifiable reward

1. A tool-using agent (lesson 04) for a task with a checkable result, such as answering questions
   over a SQLite DB.
2. An Eval Protocol evaluator that executes SQL and compares results (lesson 11).
3. Train with RFT (remote environment for multi-turn:
   <https://docs.fireworks.ai/fine-tuning/connect-environments>).
4. **Deliverable:** a reward curve plus before/after success rate.

## E. Dedicated deployment capacity plan

1. Choose a customer profile, e.g. "50 RPS peak, 2k-token prompts, 300-token outputs, p95 TTFT < 800 ms".
2. Benchmark 2 shapes on a dedicated deployment (lesson 09) and find the max RPS per replica that
   still meets the SLO.
3. Design the autoscaling config (`--load-targets`, windows, min replicas) and justify it.
4. Compare cost against serverless and batch.
5. **Deliverable:** a capacity-planning doc, which is exactly what solutions engineers write.

---

## Presentation tips

- Lead with the **business outcome** ("cut cost 78% at equal quality"), then the method.
- Always show **how you measured**. Numbers without a method don't convince anyone.
- Include a "failure modes & mitigations" section.
- Record a 3-minute Loom demo per project.
