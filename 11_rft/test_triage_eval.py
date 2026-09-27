"""Eval Protocol evaluator for the triage task. Mirrors the official GSM8K quickstart pattern:
https://github.com/eval-protocol/quickstart-gsm8k/blob/main/evaluation.py

Run locally:  python -m pytest 11_rft/test_triage_eval.py -s
This same file is what `eval-protocol create rft` packages as your evaluator.
"""

import os
from pathlib import Path

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

from reward import score_triage

DATASET = str(Path(__file__).resolve().parents[1] / "outputs" / "rft_eval.jsonl")
MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash")


@evaluation_test(
    input_dataset=[DATASET],
    completion_params=[{"model": f"fireworks_ai/{MODEL}", "temperature": 0.0, "max_tokens": 4000}],
    max_dataset_rows=10,
    passed_threshold=0.0,
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def test_triage(row: EvaluationRow, **kwargs) -> EvaluationRow:
    assistant = [m for m in row.messages if getattr(m, "role", None) == "assistant"]
    prediction = str(assistant[-1].content) if assistant and assistant[-1].content else ""
    score, reason = score_triage(prediction.strip().splitlines()[-1] if prediction.strip() else "",
                                 str(row.ground_truth))
    row.evaluation_result = EvaluateResult(score=score, is_score_valid=True, reason=reason)
    return row
