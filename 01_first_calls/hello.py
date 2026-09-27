"""The smallest useful Fireworks program, plus a tour of the response object."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, banner, client, print_usage

fw = client()

resp = fw.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
        {"role": "user", "content": "What is Fireworks AI?"},
    ],
)

banner("Answer")
print(resp.choices[0].message.content)

banner("Anatomy of the response")
print("id            :", resp.id)
print("model         :", resp.model)
print("finish_reason :", resp.choices[0].finish_reason)
print_usage(resp)

banner("Full response as JSON")
print(resp.model_dump_json(indent=2)[:2000])
