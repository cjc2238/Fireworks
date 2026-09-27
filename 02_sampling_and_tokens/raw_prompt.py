"""raw_output=True shows the exact text the model saw after the chat template was applied.
Invaluable when debugging "the model ignores my system prompt" tickets.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client

fw = client()
resp = fw.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi!"},
    ],
    max_tokens=20,
    reasoning_effort="none",  # skip hidden thinking: fast, short answers
    raw_output=True,
    return_token_ids=True,
)
choice = resp.choices[0]
raw = getattr(choice, "raw_output", None) or (choice.model_extra or {}).get("raw_output")
print("RAW OUTPUT:\n", raw)
print("\nprompt token ids (first 40):", (getattr(resp, "prompt_token_ids", None) or [])[:40])
