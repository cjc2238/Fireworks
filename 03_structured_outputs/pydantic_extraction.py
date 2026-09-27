"""Extract structured data with a JSON Schema generated from a Pydantic model."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from typing import Literal

from pydantic import BaseModel, Field

from fwlearn import MODEL, client


class Person(BaseModel):
    name: str
    role: str
    company: str | None = None


class MeetingNotes(BaseModel):
    title: str = Field(description="Short title for the meeting")
    attendees: list[Person]
    decisions: list[str]
    action_items: list[str]
    sentiment: Literal["positive", "neutral", "negative"]


transcript = """
Standup, Tuesday. Priya (eng lead, Acme) said the batch pipeline is 2x faster after moving to
Fireworks batch API. Marco from finance was worried about GPU costs, but agreed once Priya showed
the 50% batch discount. We decided to move the nightly summarization job to batch. Marco will
update the budget sheet; Priya will write the migration doc by Friday. Good energy overall.
"""

fw = client()
resp = fw.chat.completions.create(
    model=MODEL,
    messages=[
        # Always describe the desired output in the prompt too. The schema alone isn't "seen".
        {"role": "system", "content": "Extract meeting notes as JSON matching the provided schema."},
        {"role": "user", "content": transcript},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "MeetingNotes", "schema": MeetingNotes.model_json_schema()},
    },
    temperature=0,
    max_tokens=1000,
)

choice = resp.choices[0]
if choice.finish_reason == "length":
    sys.exit("Output truncated; raise max_tokens.")

notes = MeetingNotes.model_validate_json(choice.message.content)  # typed, validated object
print(notes.model_dump_json(indent=2))
print("\nFirst action item:", notes.action_items[0])
