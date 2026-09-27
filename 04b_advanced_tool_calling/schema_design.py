"""Design a tool schema with Pydantic (nested objects, enums, per-arg docs), validate the model's
arguments, and let the model repair them when validation fails.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from fwlearn import MODEL, banner, client


class Attendee(BaseModel):
    email: Annotated[str, Field(description="Work email, e.g. priya@acme.com")]
    role: Literal["required", "optional"] = "required"


class CreateMeeting(BaseModel):
    """Schedule a meeting. Use whenever the user asks to set up, book, or schedule a meeting."""

    title: Annotated[str, Field(description="Short title, max 60 chars")]
    day: Annotated[date, Field(description="ISO date, e.g. 2026-10-02")]
    duration_minutes: Annotated[int, Field(ge=15, le=240, description="15-240, in 15-minute steps")]
    attendees: Annotated[list[Attendee], Field(min_length=1)]

    @field_validator("duration_minutes")
    @classmethod
    def quarter_hours(cls, v):
        if v % 15:
            raise ValueError("duration must be a multiple of 15 minutes")
        return v


schema = CreateMeeting.model_json_schema()  # nested Attendee lands in $defs, referenced via $ref
if "--no-hint" in sys.argv:
    # Without the "15-minute steps" hint the model usually sends 50, fails validation, and repairs.
    # With the hint, good models fix it up front. Descriptions prevent errors; validation catches the rest.
    schema["properties"]["duration_minutes"]["description"] = "Length in minutes"
TOOLS = [{"type": "function", "function": {
    "name": "create_meeting", "description": CreateMeeting.__doc__, "parameters": schema}}]

banner("Generated schema (note $defs / $ref for the nested model)")
print(json.dumps(schema, indent=1)[:1200])

fw = client()
messages = [
    {"role": "system", "content": "Today is 2026-09-27. Use tools to take actions."},
    # 50 minutes is deliberately invalid (not a multiple of 15) to trigger the repair loop
    {"role": "user", "content": "Book a 50 minute 'Fireworks migration review' next Friday with "
                                "priya@acme.com and optionally marco@acme.com."},
]

for attempt in range(1, 4):
    banner(f"Attempt {attempt}")
    r = fw.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                   temperature=0, reasoning_effort="none")
    msg = r.choices[0].message
    if not msg.tool_calls:
        print("Model replied without a tool call:", msg.content)
        break
    tc = msg.tool_calls[0]
    print("raw arguments:", tc.function.arguments)
    messages.append(msg.model_dump(exclude_none=True))
    try:
        meeting = CreateMeeting.model_validate_json(tc.function.arguments)
        print("VALID ->", meeting)
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps({"ok": True, "meeting_id": "mtg_123"})})
        final = fw.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                           temperature=0, reasoning_effort="none")
        print("\nassistant:", final.choices[0].message.content)
        break
    except Exception as e:
        errors = getattr(e, "errors", lambda **_: str(e))(include_url=False)
        print("INVALID ->", errors)
        # errors as data: the model reads this and fixes its arguments
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(
            {"ok": False, "error": "invalid arguments", "details": errors}, default=str)})
