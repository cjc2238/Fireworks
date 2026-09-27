"""The stateful Responses API: chain turns with previous_response_id instead of resending history."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, openai_client

oai = openai_client()

r1 = oai.responses.create(
    model=MODEL,
    input="My name is Sam and I'm studying for a Fireworks AI interview. Remember that.",
)
print("turn 1:", r1.output_text, "\n")

r2 = oai.responses.create(
    model=MODEL,
    input="What's my name, and what am I preparing for? Then give me one tip.",
    previous_response_id=r1.id,  # the server remembers turn 1
)
print("turn 2:", r2.output_text)
print("\nresponse ids:", r1.id, "->", r2.id)
