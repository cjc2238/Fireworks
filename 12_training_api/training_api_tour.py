"""An annotated tour of the Fireworks Training API (serverless LoRA training).

WARNING: This is a SKELETON built from the calls documented at
  https://docs.fireworks.ai/fine-tuning/training-api/serverless
It won't run until you fill in `build_datums()` using the cookbook's tokenizer/datum helpers:
  https://docs.fireworks.ai/fine-tuning/training-api/cookbook/sft
It also costs money. Read it first, then run the cookbook version.

Install (Python 3.11+):  pip install "fireworks-ai[training]"
"""

import os

BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"  # check the serverless training model list
MAX_SEQ_LEN = 131072


def build_datums(tokenizer, examples):
    """Turn chat examples into Datums: token ids + per-token loss weights.
    For SFT: weight 0.0 on prompt tokens, 1.0 on assistant tokens.
    See the cookbook for the exact Datum constructor."""
    raise NotImplementedError("Use the cookbook's datum builder here.")


def main():
    import tinker  # installed with fireworks-ai[training]; provides AdamParams
    from fireworks.training.sdk import FireworksClient, FiretitanServiceClient

    # 1) Connect to the serverless training service
    service = FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/training/v1/serverless",
    )
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    print(f"session={service.training_session_id} run={training_client.run_id}")

    tokenizer = None  # cookbook: load the base model's tokenizer
    datums = build_datums(tokenizer, examples=[])

    # 2) One optimizer step. Every call returns a future, so .result() surfaces errors.
    training_client.forward_backward(datums, "importance_sampling").result()
    training_client.optim_step(
        tinker.AdamParams(learning_rate=2.5e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    ).result()

    # 3) Snapshot the weights and sample from exactly that snapshot (this is how RL rollouts work)
    snapshot = training_client.save_weights_for_sampler("step-0001").result().path
    sampler = service.create_sampling_client(model_path=snapshot, tokenizer=tokenizer)
    # completions = sampler.sample(prompt=..., num_samples=8, sampling_params=...).result()
    sampler.close()

    # 4) Checkpoint for resume (weights + optimizer state)
    training_client.save_state("step-0001").result()

    # 5) Promote a checkpoint to a deployable model
    fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
    rows = fw.list_training_session_checkpoints(service.training_session_name)
    target = next(r for r in reversed(rows) if r.get("promotable"))
    fw.promote_session_checkpoint(name=target["name"], output_model_id="my-first-training-api-lora",
                                  base_model=BASE_MODEL)
    print("Promoted. Query it as accounts/<account>/models/my-first-training-api-lora")


if __name__ == "__main__":
    main()
