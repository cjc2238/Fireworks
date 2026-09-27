"""Grammar mode: constrain output with a GBNF grammar. Here, the output can ONLY be one of the
listed ticket categories, followed by a priority 1-5.

GBNF reference: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
Docs: https://docs.fireworks.ai/structured-responses/structured-response-formatting
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client

GRAMMAR = r'''
root     ::= category " | P" priority
category ::= "billing" | "outage" | "feature-request" | "account-access" | "other"
priority ::= [1-5]
'''

tickets = [
    "I was charged twice this month, please refund!!",
    "All my API calls return 503 since 10 minutes ago. Production is down.",
    "Would love a Rust SDK someday.",
]

fw = client()
for t in tickets:
    resp = fw.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Triage the support ticket. Output: <category> | P<priority 1=urgent..5=low>"},
            {"role": "user", "content": t},
        ],
        response_format={"type": "grammar", "grammar": GRAMMAR},
        temperature=0,
        max_tokens=20,
        reasoning_effort="none",
    )
    print(f"{resp.choices[0].message.content:<28} <- {t}")
