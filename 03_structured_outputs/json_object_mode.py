"""json_object mode: guarantees valid JSON, but not any particular shape."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from fwlearn import MODEL, client

fw = client()
resp = fw.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": "Return JSON with keys 'languages' (array of 3 programming languages) and 'reason' (string).",
    }],
    response_format={"type": "json_object"},
    temperature=0,
)
data = json.loads(resp.choices[0].message.content)  # always parses
print(json.dumps(data, indent=2))
