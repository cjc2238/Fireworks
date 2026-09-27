# 11 · Reinforcement Fine-Tuning (RFT)

> 💸 RFT runs many rollouts (inference) plus training, so it costs more than SFT. Start with a
> tiny dataset and a small model (for example `qwen3-4b` / `qwen3-8b`).

## SFT vs RFT

| | SFT | RFT |
|---|---|---|
| You provide | Prompts **+ ideal answers** | Prompts + a **reward function** (evaluator) |
| Model learns | To imitate the answers | To maximize the score, finding its own way there |
| Great for | Format, style, distillation | Verifiable tasks: math, code, SQL, extraction, tool use, agents |
| Risk | Copies label noise | **Reward hacking**: the model exploits holes in your evaluator |

The workflow: **design evaluator → prepare prompts → (connect agent/environment) → train → deploy.**

## Evaluators

An evaluator takes a rollout (the conversation including the model's answer) and returns a score
in **[0, 1]** with a reason. Fireworks uses the open-source **Eval Protocol** (`pip install eval-protocol`).
An evaluator is a pytest-style function decorated with `@evaluation_test`:

```python
from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

@evaluation_test(
    input_dataset=["data.jsonl"],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/<model>", "temperature": 0.0}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def my_eval(row: EvaluationRow, **kwargs) -> EvaluationRow:
    answer = row.messages[-1].content
    row.evaluation_result = EvaluateResult(score=..., reason="...")
    return row
```

### Designing good rewards

- **Partial credit** gives a smoother learning signal than pass/fail.
- **Test the evaluator** on perfect, partially right, wrong, and *malformed or adversarial*
  outputs before training. Unit-test it like production code.
- **Guard against hacking:** penalize over-long outputs and format cheats, and read real rollouts
  during training.
- **Never hard-code secrets** in evaluator code. Use Fireworks secrets and reference them by
  environment variable name.
- Multi-turn agents run in **remote environments** you connect to Fireworks
  (<https://docs.fireworks.ai/fine-tuning/connect-environments>).

## Scripts

```powershell
pip install eval-protocol pytest
python 11_rft\make_rft_dataset.py            # converts lesson 10 data -> outputs\rft_*.jsonl
python -m pytest 11_rft\test_reward.py -q    # unit-test the reward function (no API calls)
python -m pytest 11_rft\test_triage_eval.py -s   # run the evaluator against a live model
```

## Launching RFT

From the directory with your evaluator:

```powershell
cd 11_rft
eval-protocol create rft --base-model accounts/fireworks/models/qwen3-4b
```

Or use `firectl rftj create --help` (base model, dataset, evaluator, output model), or the
dashboard wizard. Monitor reward curves, rollout samples and validation scores (W&B is supported).
Stop when the held-out score plateaus or regresses. Then deploy the LoRA like in lesson 10.

Official quickstarts:
- Single-turn (GSM8K math): <https://docs.fireworks.ai/fine-tuning/quickstart-math>
  (repo: <https://github.com/eval-protocol/quickstart-gsm8k>)
- Remote agent: <https://docs.fireworks.ai/fine-tuning/quickstart-svg-agent>
- All parameters: <https://docs.fireworks.ai/fine-tuning/rft-parameters-reference>

## Exercises

1. Do the official GSM8K quickstart end to end.
2. Run RFT on the triage task with `reward.py`. Compare with your SFT model from lesson 10 on
   `outputs\sft_eval.jsonl`.
3. **Red-team your evaluator:** write 5 outputs that score high but are wrong. Fix the
   evaluator until none of them pass.
4. Write an evaluator for text-to-SQL that *executes* the query against SQLite and compares result
   sets.
