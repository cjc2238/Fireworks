# 08 · Batch Inference

**Goal:** process large offline workloads (evals, labeling, summarization, synthetic data) at
**50% off serverless per-token prices**, with automatic prompt caching on top.

## When to use batch

| Use batch | Don't |
|---|---|
| Nightly jobs, backfills, evals, data labeling, synthetic data generation | Anything a user is waiting on |
| 1k to millions of requests | A handful of requests (just use async, lesson 01) |
| Results within hours are fine | You need an answer in seconds |

## The workflow

```
1. Write input JSONL:  {"custom_id": "req-1", "body": {"messages": [...], "max_tokens": 100}}
2. Upload it as a Dataset                     (firectl dataset create / REST)
3. Create a batchInferenceJob(model, inputDatasetId, outputDatasetId, inferenceParameters)
4. Poll until JOB_STATE_COMPLETED
5. Download the output dataset and join results on custom_id
```

Using firectl:

```powershell
firectl dataset create batch-demo-input outputs\batch_input.jsonl
firectl batch-inference-job create --model accounts/fireworks/models/<model> --input-dataset-id batch-demo-input
firectl batch-inference-job get <job-id>
firectl dataset download <output-dataset-id>
```

## Limits and gotchas

- Input ≤ 80 GiB, output ≤ 8 GB. Jobs expire after 12/24/48/72 hours.
- `custom_id` must be unique, and results **may come back in any order**. Always join on `custom_id`.
- Not every model supports batch. An unsupported model can sit in `PENDING` forever, so check
  the model page first.
- A job-level `systemPrompt` can't be combined with `prompt_token_ids`. The injected system
  prompt counts toward the context window.
- You can pass `inferenceParameters` (`maxTokens`, `temperature`, `topP`, `topK`, `n`, `extraBody`).

## Scripts

```powershell
python 08_batch\make_batch_input.py          # writes outputs\batch_input.jsonl (200 requests)
python 08_batch\run_batch.py                 # upload -> create job -> poll -> download -> join
```

`run_batch.py` needs `FIREWORKS_ACCOUNT_ID` in `.env`.

## Exercises

1. Use batch to **evaluate** a model: 200 math questions with known answers → accuracy score.
   You'll reuse this harness in lessons 10–11 to compare the base model with a fine-tuned one.
2. Use batch to **generate synthetic training data** for lesson 10 (for example 500 customer-support
   Q&A pairs in a specific brand voice).
3. Estimate the cost of the same job on serverless and on batch, and put it in a table.
