"""Fireworks is a drop-in replacement for the OpenAI API: change base_url and api_key.

Most customers migrate this way, so know it cold.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, openai_client

oai = openai_client()  # OpenAI(api_key=FIREWORKS_API_KEY, base_url="https://api.fireworks.ai/inference/v1")

resp = oai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Name three open-weight LLM families."}],
    temperature=0.3,
    # Parameters the OpenAI SDK doesn't know about go in extra_body:
    extra_body={"top_k": 40, "min_p": 0.05},
)
print(resp.choices[0].message.content)
