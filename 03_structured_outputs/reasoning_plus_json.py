"""With a reasoning model, json_schema mode disables reasoning. To keep the thinking,
put the schema in the prompt and validate the JSON yourself.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from pydantic import BaseModel, ValidationError

from fwlearn import BIG_MODEL, client


class Answer(BaseModel):
    answer: int
    explanation: str


schema = json.dumps(Answer.model_json_schema(), indent=2)
fw = client()
resp = fw.chat.completions.create(
    model=BIG_MODEL,
    messages=[{
        "role": "user",
        "content": (
            "A train leaves at 3:40pm and arrives at 6:15pm. How many minutes is the trip?\n\n"
            f"Reply with ONLY JSON matching this schema:\n{schema}"
        ),
    }],
    max_tokens=4000,
)
msg = resp.choices[0].message
reasoning = getattr(msg, "reasoning_content", None)
content = msg.content.strip()
if content.startswith("```"):  # strip a markdown fence if the model added one
    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

if reasoning:
    print("REASONING (truncated):\n", reasoning[:800], "\n")
try:
    print("VALIDATED:", Answer.model_validate_json(content))
except ValidationError as e:
    print("Model returned invalid JSON; in production you'd retry or repair.\n", e)
